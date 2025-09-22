from __future__ import annotations

from typing import List, Tuple

import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, pearsonr, spearmanr

# optional pretty table formatter
try:
    from tabulate import tabulate as _tabulate
except Exception:  # pragma: no cover - optional dep
    _tabulate = None


def build_price_table(price_scan: pl.LazyFrame, window: int) -> pl.DataFrame:
    """
    Build a single continuous price table from multiple 1-second CSVs.
    - Convert microsecond timestamps to integer seconds.
    - Compute log price and 1-second log returns.
    - Compute realized variance (sum of squared returns over a trailing window)
      and realized quarticity (n/3 * sum of quartic returns over the same window).
    """
    return (
        price_scan.with_columns(
            [
                (pl.col("timestamp_us") / 1_000_000).cast(pl.Int64).alias("timestamp"),
                pl.col("open").cast(pl.Float64),
            ]
        )
        .select(["timestamp_us", "timestamp", "open"])  # ensure consistent projection
        .sort("timestamp")
        .with_columns([pl.col("open").log().alias("log_price")])
        .with_columns(
            [(pl.col("log_price") - pl.col("log_price").shift(1)).alias("log_return")]
        )
        .with_columns(
            [
                pl.col("log_return").pow(2).rolling_sum(window_size=window).alias("realized_variance"),
                ((pl.lit(window) / 3) * pl.col("log_return").pow(4).rolling_sum(window_size=window)).alias(
                    "realized_quarticity"
                ),
            ]
        )
        .drop("timestamp_us")
        .drop("open")
        .collect()
    )


def build_resolved_times(resolved: pl.DataFrame) -> pl.DataFrame:
    """
    Prepare the resolved times table with integer-typed round and timestamps.
    """
    return resolved.select(
        [
            pl.col("round").cast(pl.Int64),
            pl.col("round_start_timestamp").cast(pl.Int64),
            pl.col("round_end_timestamp").cast(pl.Int64),
        ]
    )


def build_complete_bids(
    bids_df: pl.DataFrame,
    resolved_times: pl.DataFrame,
    min_round: int,
    max_round: int,
) -> pl.DataFrame:
    """
    Enrich a single bidder's bids with round timestamps for all rounds in [min_round, max_round].
    - Join existing rounds to resolved data to get timestamps.
    - Deduplicate to one bid per round (latest timestamp).
    - Fill missing rounds with reserve amount and timestamps propagated by +60 seconds per gap.
    - Seed leading missing timestamps from resolved times if no prior observed round exists.
    """
    bids_typed = bids_df.with_columns([
        pl.col("round").cast(pl.Int64),
        pl.col("amount").cast(pl.Int64),
    ])

    enriched = bids_typed.join(resolved_times, on="round", how="left").with_columns([
        pl.lit(False).alias("is_censored"),
    ])

    enriched_dedup = enriched.sort(["round", "timestamp"]).unique(
        subset=["round"], keep="last"
    )

    target_rounds = pl.DataFrame({"round": pl.arange(min_round, max_round + 1, eager=True, dtype=pl.Int64)})

    range_times = resolved_times.filter(
        (pl.col("round") >= min_round) & (pl.col("round") <= max_round)
    ).rename({
        "round_start_timestamp": "resolved_start",
        "round_end_timestamp": "resolved_end",
    })

    full = (
        target_rounds
        .join(enriched_dedup.select([
            "round", "bidder", "amount", "round_start_timestamp", "round_end_timestamp", "is_censored"
        ]), on="round", how="left")
        .join(range_times, on="round", how="left")
        .sort("round")
    )

    bidder_label = (
        bids_typed.select(pl.col("bidder").first().alias("bidder")).to_dicts()[0]["bidder"]
        if "bidder" in bids_typed.columns and bids_typed.height > 0 else None
    )

    full = full.with_columns([
        pl.col("amount").is_not_null().alias("has_bid"),
        pl.when(pl.col("amount").is_not_null()).then(pl.col("round")).otherwise(None).forward_fill().alias("prev_round"),
        pl.when(pl.col("round_start_timestamp").is_not_null()).then(pl.col("round_start_timestamp")).otherwise(None).forward_fill().alias("prev_start_ts"),
        pl.when(pl.col("round_end_timestamp").is_not_null()).then(pl.col("round_end_timestamp")).otherwise(None).forward_fill().alias("prev_end_ts"),
    ])

    full = full.with_columns([
        pl.when(pl.col("prev_round").is_not_null()).then(pl.col("round") - pl.col("prev_round")).otherwise(pl.lit(0)).alias("distance"),
    ])

    full = full.with_columns([
        pl.when(pl.col("prev_round").is_not_null()).then(pl.col("prev_start_ts") + pl.col("distance") * 60).otherwise(pl.col("resolved_start")).alias("filled_start"),
        pl.when(pl.col("prev_round").is_not_null()).then(pl.col("prev_end_ts") + pl.col("distance") * 60).otherwise(pl.col("resolved_end")).alias("filled_end"),
    ])

    full = full.with_columns([
        pl.when(pl.col("has_bid")).then(pl.col("amount")).otherwise(pl.lit(10**15)).cast(pl.Int64).alias("amount_final"),
        pl.when(pl.col("has_bid")).then(pl.col("round_start_timestamp")).otherwise(pl.col("filled_start")).cast(pl.Int64).alias("round_start_timestamp_final"),
        pl.when(pl.col("has_bid")).then(pl.col("round_end_timestamp")).otherwise(pl.col("filled_end")).cast(pl.Int64).alias("round_end_timestamp_final"),
        pl.when(pl.col("has_bid")).then(pl.col("is_censored").fill_null(False)).otherwise(pl.lit(True)).alias("is_censored_final"),
        (pl.lit(bidder_label) if bidder_label is not None else pl.col("bidder")).alias("bidder_final"),
    ])

    out = full.select([
        "round",
        pl.col("bidder_final").alias("bidder"),
        pl.col("amount_final").alias("amount"),
        pl.col("round_start_timestamp_final").alias("round_start_timestamp"),
        pl.col("round_end_timestamp_final").alias("round_end_timestamp"),
        pl.col("is_censored_final").alias("is_censored"),
    ])
    return out


def attach_realized_to_bids(bids: pl.DataFrame, price_df: pl.DataFrame) -> pl.DataFrame:
    rv_lookup = price_df.select(["timestamp", "realized_variance", "realized_quarticity"])
    return bids.join(rv_lookup, left_on="round_end_timestamp", right_on="timestamp", how="left")


def prepare_regression_data(bids_complete: pl.DataFrame, wei_per_eth: float) -> tuple[np.ndarray, np.ndarray, List[str], pl.DataFrame]:
    """
    Prepare regression arrays from a complete bids table.
    - Filter rows to those with realized stats available.
    - Scale y to ETH and standardize features.
    - Use features: [realized_variance, realized_quarticity - realized_variance^2]
    Returns (y_eth, X_std, feature_names, df_filtered)
    """
    df = bids_complete.filter(
        pl.col("realized_variance").is_not_null() & pl.col("realized_quarticity").is_not_null()
    )
    y_eth = (df["amount"].cast(pl.Float64).to_numpy()) / wei_per_eth
    rv = df["realized_variance"].cast(pl.Float64).to_numpy()
    rq = df["realized_quarticity"].cast(pl.Float64).to_numpy()
    X_raw = np.column_stack([rv, rq - rv ** 2])
    mu = X_raw.mean(axis=0)
    sd = X_raw.std(axis=0, ddof=1)
    sd[sd == 0] = 1.0
    X_std = (X_raw - mu) / sd
    names_local = ["realized_variance", "realized_quarticity - realized_variance^2"]
    return y_eth, X_std, names_local, df


def print_coefficients(result, names: List[str], tag: str) -> None:
    beta = result.beta
    se = result.se_beta
    tval = beta / se
    pval = 2 * norm.sf(np.abs(tval))
    rows = []
    for i, name in enumerate(names[: len(beta)]):
        rows.append([name, f"{beta[i]:.6e}", f"{se[i]:.6e}", f"{tval[i]:.3f}", f"{pval[i]:.3g}"])

    headers = ["term", "beta", "se", "t", "p"]
    title = f"Tobit coefficients ({tag})"
    if _tabulate is not None:
        print(f"\n=== {title} ===")
        print(_tabulate(rows, headers=headers, tablefmt="github", stralign="right"))
    else:
        print(f"\n=== {title} ===")
        for r in rows:
            print(f"{r[0]:>30}  beta={r[1]}  se={r[2]}  t={r[3]}  p={r[4]}")
    # print model criteria beneath
    crit_rows = [["loglike", f"{result.loglike:.3f}"], ["AIC", f"{result.aic:.2f}"], ["BIC", f"{result.bic:.2f}"]]
    if _tabulate is not None:
        print(_tabulate(crit_rows, headers=["metric", "value"], tablefmt="github", stralign="right"))
    else:
        for m, v in crit_rows:
            print(f"{m:>10}: {v}")


def print_gamma_table(gamma: np.ndarray, se_gamma: np.ndarray, names: List[str], tag: str) -> None:
    rows = []
    for i, name in enumerate(names[: len(gamma)]):
        rows.append([name, f"{gamma[i]:.6e}", f"{se_gamma[i]:.6e}"])
    headers = ["sigma term", "gamma", "se"]
    title = f"Sigma model (log-sigma) ({tag})"
    if _tabulate is not None:
        print(_tabulate(rows, headers=headers, tablefmt="github", stralign="right"))
    else:
        print(f"\n=== {title} ===")
        for r in rows:
            print(f"{r[0]:>30}  gamma={r[1]}  se={r[2]}")


def summarize_residuals(residuals: np.ndarray, tag: str) -> None:
    stats = {
        "n": residuals.size,
        "mean": float(residuals.mean()),
        "std": float(residuals.std(ddof=1)),
        "min": float(np.min(residuals)),
        "p05": float(np.percentile(residuals, 5)),
        "p50": float(np.percentile(residuals, 50)),
        "p95": float(np.percentile(residuals, 95)),
        "max": float(np.max(residuals)),
    }
    rows = [[k, f"{v:.6g}" if isinstance(v, float) else v] for k, v in stats.items()]
    if _tabulate is not None:
        print(_tabulate(rows, headers=[f"Residual stats ({tag})", "value"], tablefmt="github", stralign="right"))
    else:
        print(f"Residual stats ({tag}):", stats)


def compute_r2s(y: np.ndarray, yhat: np.ndarray) -> tuple[float, float]:
    r_pearson, _ = pearsonr(y, yhat)
    r_spearman, _ = spearmanr(y, yhat)
    return float(r_pearson ** 2), float(r_spearman ** 2)


def corr_inputs_with_residuals(X: np.ndarray, residuals: np.ndarray, names: List[str]) -> list[tuple[str, float, float]]:
    out: list[tuple[str, float, float]] = []
    for j, name in enumerate(names):
        r_p, _ = pearsonr(X[:, j], residuals)
        r_s, _ = spearmanr(X[:, j], residuals)
        out.append((name, float(r_p), float(r_s)))
    return out


def print_corr_table(corrs: List[tuple[str, float, float]], tag: str) -> None:
    headers = [f"corr(residual, input) ({tag})", "pearson", "spearman"]
    rows = [[name, f"{rp:.4f}", f"{rs:.4f}"] for name, rp, rs in corrs]
    if _tabulate is not None:
        print(_tabulate(rows, headers=headers, tablefmt="github", stralign="right"))
    else:
        print(f"\n=== Correlations ({tag}) ===")
        for r in rows:
            print(f"{r[0]:>30}  pearson={r[1]}  spearman={r[2]}")


def estimate_conditional_variance_ols(X: np.ndarray, residuals: np.ndarray) -> float:
    """
    Simple proxy for Var(residual | X): regress residual^2 on X with intercept
    and return mean predicted variance. Clipped to non-negative values.
    """
    y2 = residuals ** 2
    X2 = np.column_stack([np.ones(X.shape[0]), X])
    coef, *_ = np.linalg.lstsq(X2, y2, rcond=None)
    y2_hat = X2 @ coef
    y2_hat = np.clip(y2_hat, 0.0, np.inf)
    return float(np.mean(y2_hat))


def print_variance_table(var_uncond: float, var_cond: float, tag: str) -> None:
    rows = [["Var(residual)", f"{var_uncond:.6e}"], ["E[Var(residual|X)]", f"{var_cond:.6e}"]]
    if _tabulate is not None:
        print(_tabulate(rows, headers=[f"Residual variance ({tag})", "value"], tablefmt="github", stralign="right"))
    else:
        print(f"Var(residual)={var_uncond:.6e}, E[Var(residual|X)]={var_cond:.6e}")


def plot_residuals_round_hist(df: pl.DataFrame, residuals: np.ndarray, bidder_tag: str) -> None:
    rounds = df["round"].to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    axes[0].plot(rounds, residuals, lw=0.7)
    axes[0].set_title(f"Residuals vs Round ({bidder_tag})")
    axes[0].set_xlabel("Round")
    axes[0].set_ylabel("Residual (ETH)")
    axes[1].hist(residuals, bins=50, alpha=0.85)
    axes[1].set_title(f"Residual distribution ({bidder_tag})")
    fig.savefig(f"images/residuals_round_and_hist_{bidder_tag}.png", dpi=150)
    plt.close(fig)


def plot_residual_vs_rv(df: pl.DataFrame, residuals: np.ndarray, bidder_tag: str) -> None:
    rv = df["realized_variance"].cast(pl.Float64).to_numpy()
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    ax.scatter(rv, residuals, s=6, alpha=0.6)
    ax.set_title(f"Residual vs RV ({bidder_tag})")
    ax.set_xlabel("Realized Variance (1s, window=60)")
    ax.set_ylabel("Residual (ETH)")
    fig.savefig(f"images/residual_vs_rv_{bidder_tag}.png", dpi=150)
    plt.close(fig)


def plot_yhat_vs_y(mu_hat_arr: np.ndarray, y_arr: np.ndarray, bidder_tag: str) -> None:
    # scatter
    fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
    ax.scatter(mu_hat_arr, y_arr, s=6, alpha=0.6)
    mn = float(min(mu_hat_arr.min(), y_arr.min()))
    mx = float(max(mu_hat_arr.max(), y_arr.max()))
    ax.plot([mn, mx], [mn, mx], "r--", lw=1)
    ax.set_title(f"Predicted vs Actual ({bidder_tag})")
    ax.set_xlabel("Predicted latent y (ETH)")
    ax.set_ylabel("Actual y (ETH)")
    fig.savefig(f"images/scatter_yhat_vs_y_{bidder_tag}.png", dpi=150)
    plt.close(fig)

    # QQ plot (quantile-quantile) between predicted and actual
    qs = np.linspace(0.01, 0.99, 99)
    yhat_q = np.quantile(mu_hat_arr, qs)
    y_q = np.quantile(y_arr, qs)
    fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
    ax.scatter(yhat_q, y_q, s=8, alpha=0.7)
    mn = float(min(yhat_q.min(), y_q.min()))
    mx = float(max(yhat_q.max(), y_q.max()))
    ax.plot([mn, mx], [mn, mx], "r--", lw=1)
    ax.set_title(f"QQ: Predicted vs Actual ({bidder_tag})")
    ax.set_xlabel("Predicted quantiles (ETH)")
    ax.set_ylabel("Actual quantiles (ETH)")
    fig.savefig(f"images/qq_yhat_vs_y_{bidder_tag}.png", dpi=150)
    plt.close(fig)


