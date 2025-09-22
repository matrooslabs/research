"""
bids data only has round of auction, not timestamp. timestamp is saved in auction result.
So one has to add two columns: round_start_timestamp and round_end_timestamp, to the bids data.
some rounds are missing since nobody bids greater than equal to reserve price of 0.001 ETH (10**15 wei)
for missing rounds we fill the bid row with reserve price (this works as censored data)
note that time in price data is in microseconds, so we need to convert it to seconds too
"""

# import libraries
import polars as pl
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import os

##############################################################################
#                                prepare data                                #
##############################################################################

# load price and auction records
# load all price csvs into a single big table (lazy scan with glob)
price_lazy = pl.scan_csv(
    "price/ETHUSDT-1s-*.csv",
    has_header=False,
    new_columns=["timestamp_us", "open"],
)

bids_0x8c6f = pl.read_csv("auction_records/bids_0x8c6f.csv")
bids_0x95c0 = pl.read_csv("auction_records/bids_0x95c0.csv")
auction_resolved = pl.read_csv("auction_records/auction_resolved.csv")


# construct price data with timestamp converted to seconds and compute realized
# stats across all files. This is wrapped into a function for reusability.
RV_WINDOW = 60

def build_price_table(price_scan: pl.LazyFrame, window: int = RV_WINDOW) -> pl.DataFrame:
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
                (pl.col("timestamp_us") / 1_000_000)
                .cast(pl.Int64)
                .alias("timestamp"),
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
                pl.col("log_return")
                .pow(2)
                .rolling_sum(window_size=window)
                .alias("realized_variance"),
                (
                    (pl.lit(window) / 3)
                    * pl.col("log_return").pow(4).rolling_sum(window_size=window)
                ).alias("realized_quarticity"),
            ]
        )
        .drop("timestamp_us")
        .drop("open")
        .collect()
    )

price = build_price_table(price_lazy, window=RV_WINDOW)

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

resolved_times = build_resolved_times(auction_resolved)

MIN_ROUND = 205323
MAX_ROUND = 249961


def build_complete_bids(
    bids_df: pl.DataFrame,
    resolved_times: pl.DataFrame,
    min_round: int,
    max_round: int,
) -> pl.DataFrame:
    # cast types
    bids_typed = bids_df.with_columns(
        [
            pl.col("round").cast(pl.Int64),
            pl.col("amount").cast(pl.Int64),
        ]
    )

    # enrich with resolved timestamps for existing rounds
    enriched = bids_typed.join(resolved_times, on="round", how="left").with_columns(
        [
            pl.lit(False).alias("is_censored"),
        ]
    )

    # deduplicate: keep only one row per round (latest by timestamp)
    enriched_dedup = enriched.sort(
        ["round", "timestamp"]
    ).unique(subset=["round"], keep="last")  # ISO-like strings sort chronologically

    # build full round range
    target_rounds = pl.DataFrame(
        {"round": pl.arange(min_round, max_round + 1, eager=True, dtype=pl.Int64)}
    )

    # resolved times for the full range (used to seed leading gaps)
    range_times = resolved_times.filter(
        (pl.col("round") >= min_round) & (pl.col("round") <= max_round)
    ).rename(
        {
            "round_start_timestamp": "resolved_start",
            "round_end_timestamp": "resolved_end",
        }
    )

    # join enriched onto full range
    full = (
        target_rounds.join(
            enriched_dedup.select(
                [
                    "round",
                    "bidder",
                    "amount",
                    "round_start_timestamp",
                    "round_end_timestamp",
                    "is_censored",
                ]
            ),
            on="round",
            how="left",
        )
        .join(range_times, on="round", how="left")
        .sort("round")
    )

    # bidder label constant
    bidder_label = (
        bids_typed.select(pl.col("bidder").first().alias("bidder")).to_dicts()[0][
            "bidder"
        ]
        if "bidder" in bids_typed.columns and bids_typed.height > 0
        else None
    )

    # flags and previous observed anchors
    full = full.with_columns(
        [
            pl.col("amount").is_not_null().alias("has_bid"),
            pl.when(pl.col("amount").is_not_null())
            .then(pl.col("round"))
            .otherwise(None)
            .forward_fill()
            .alias("prev_round"),
            pl.when(pl.col("round_start_timestamp").is_not_null())
            .then(pl.col("round_start_timestamp"))
            .otherwise(None)
            .forward_fill()
            .alias("prev_start_ts"),
            pl.when(pl.col("round_end_timestamp").is_not_null())
            .then(pl.col("round_end_timestamp"))
            .otherwise(None)
            .forward_fill()
            .alias("prev_end_ts"),
        ]
    )

    full = full.with_columns(
        [
            pl.when(pl.col("prev_round").is_not_null())
            .then(pl.col("round") - pl.col("prev_round"))
            .otherwise(pl.lit(0))
            .alias("distance"),
        ]
    )

    # compute filled timestamps: use prev + 60*distance if prev exists; else seed from resolved
    full = full.with_columns(
        [
            pl.when(pl.col("prev_round").is_not_null())
            .then(pl.col("prev_start_ts") + pl.col("distance") * 60)
            .otherwise(pl.col("resolved_start"))
            .alias("filled_start"),
            pl.when(pl.col("prev_round").is_not_null())
            .then(pl.col("prev_end_ts") + pl.col("distance") * 60)
            .otherwise(pl.col("resolved_end"))
            .alias("filled_end"),
        ]
    )

    # finalize amount, timestamps, censor flag, bidder
    full = full.with_columns(
        [
            pl.when(pl.col("has_bid"))
            .then(pl.col("amount"))
            .otherwise(pl.lit(10**15))
            .cast(pl.Int64)
            .alias("amount_final"),
            pl.when(pl.col("has_bid"))
            .then(pl.col("round_start_timestamp"))
            .otherwise(pl.col("filled_start"))
            .cast(pl.Int64)
            .alias("round_start_timestamp_final"),
            pl.when(pl.col("has_bid"))
            .then(pl.col("round_end_timestamp"))
            .otherwise(pl.col("filled_end"))
            .cast(pl.Int64)
            .alias("round_end_timestamp_final"),
            pl.when(pl.col("has_bid"))
            .then(pl.col("is_censored").fill_null(False))
            .otherwise(pl.lit(True))
            .alias("is_censored_final"),
            (
                pl.lit(bidder_label) if bidder_label is not None else pl.col("bidder")
            ).alias("bidder_final"),
        ]
    )

    # select output columns
    out = full.select(
        [
            "round",
            pl.col("bidder_final").alias("bidder"),
            pl.col("amount_final").alias("amount"),
            pl.col("round_start_timestamp_final").alias("round_start_timestamp"),
            pl.col("round_end_timestamp_final").alias("round_end_timestamp"),
            pl.col("is_censored_final").alias("is_censored"),
        ]
    )

    return out


def attach_realized_to_bids(bids: pl.DataFrame, price_df: pl.DataFrame) -> pl.DataFrame:
    """
    Attach realized statistics from price to bids using round_end_timestamp as key.
    """
    rv_lookup = price_df.select(["timestamp", "realized_variance", "realized_quarticity"])
    return bids.join(rv_lookup, left_on="round_end_timestamp", right_on="timestamp", how="left")


# process each bidder separately
bids_0x8c6f_complete = build_complete_bids(bids_0x8c6f, resolved_times, MIN_ROUND, MAX_ROUND)
bids_0x95c0_complete = build_complete_bids(bids_0x95c0, resolved_times, MIN_ROUND, MAX_ROUND)

bids_0x8c6f_complete = attach_realized_to_bids(bids_0x8c6f_complete, price)
bids_0x95c0_complete = attach_realized_to_bids(bids_0x95c0_complete, price)

# print the first 10 rows of each dataset
print("price data:")
print(price.tail(10))
print("\n\nbids data - 0x8c6f:")
print(bids_0x8c6f_complete.tail(10))
print("\n\nbids data - 0x95c0:")
print(bids_0x95c0_complete.tail(10))

##############################################################################
#                               run regression                               #
##############################################################################

import bid_and_volatility.tobit as tobit

# model: bid_amount ~ beta_var * realized_variance + beta_quar * realized_quarticity + beta_const + error

"""Scale and fit Tobit: y in ETH, features standardized; left bound in ETH."""
RESERVE_WEI = 10**15
WEI_PER_ETH = 10**18
LEFT_ETH = RESERVE_WEI / WEI_PER_ETH


def prepare_regression_data(bids_complete: pl.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str], pl.DataFrame]:
    """
    Prepare regression arrays from a complete bids table.
    - Filter rows to those with realized stats available.
    - Scale y to ETH and standardize features.
    - Use features: [realized_variance, realized_quarticity - realized_variance^2]
      which loosely correspond to mean and variance of IV in microstructure theory.
    Returns (y_eth, X_std, feature_names, df_filtered)
    """
    df = bids_complete.filter(
        pl.col("realized_variance").is_not_null()
        & pl.col("realized_quarticity").is_not_null()
    )
    y_eth = (df["amount"].cast(pl.Float64).to_numpy()) / WEI_PER_ETH
    rv = df["realized_variance"].cast(pl.Float64).to_numpy()
    rq = df["realized_quarticity"].cast(pl.Float64).to_numpy()
    # Feature transform: RV and (RQ - RV^2)
    X_raw = np.column_stack([rv, rq - rv ** 2])
    mu = X_raw.mean(axis=0)
    sd = X_raw.std(axis=0, ddof=1)
    sd[sd == 0] = 1.0
    X_std = (X_raw - mu) / sd
    names_local = ["realized_variance", "realized_quarticity - realized_variance^2"]
    return y_eth, X_std, names_local, df


def print_coefficients(result, names: list[str], tag: str) -> None:
    """Pretty-print coefficients, standard errors, t and p values."""
    beta = result.beta
    se = result.se_beta
    tval = beta / se
    pval = 2 * norm.sf(np.abs(tval))
    print(f"\n=== Tobit coefficients ({tag}) ===")
    for i, name in enumerate(names[: len(beta)]):
        print(f"{name:>20}: beta={beta[i]: .6e}  se={se[i]: .6e}  t={tval[i]: .3f}  p={pval[i]: .3g}")
    print(f"loglike={result.loglike:.3f}  AIC={result.aic:.2f}  BIC={result.bic:.2f}")


def summarize_residuals(residuals: np.ndarray, tag: str) -> None:
    """Print concise residual distribution statistics."""
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
    print(f"Residual stats ({tag}):", stats)

# additional plots: residual vs RV, scatter and QQ of y_hat vs y


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


def run_bidder_regression(bids_complete: pl.DataFrame, bidder_tag: str) -> None:
    """
    Full pipeline per bidder: prepare data, fit Tobit, print coefficients and residual stats,
    and generate robustness plots saved under images/.
    """
    # ensure images dir
    os.makedirs("images", exist_ok=True)

    # prepare regression arrays
    y_eth, X_std, feature_names, df_filtered = prepare_regression_data(bids_complete)

    # fit Tobit with left-censoring at reserve (in ETH)
    import bid_and_volatility.tobit as tobit  # local import to keep top clean

    res = tobit.fit_tobit(y_eth, X_std, left=LEFT_ETH, use_numeric_hessian=True)

    # report
    print_coefficients(res, ["const", *feature_names], bidder_tag)

    # residuals and summaries
    mu_hat_local = tobit.predict_latent(X_std, res.beta, add_intercept=True)
    residuals_local = y_eth - mu_hat_local
    summarize_residuals(residuals_local, bidder_tag)

    # plots
    plot_residuals_round_hist(df_filtered, residuals_local, bidder_tag)
    plot_residual_vs_rv(df_filtered, residuals_local, bidder_tag)
    plot_yhat_vs_y(mu_hat_local, y_eth, bidder_tag)


# Run the regression/plotting pipeline for each bidder
run_bidder_regression(bids_0x8c6f_complete, "0x8c6f")
run_bidder_regression(bids_0x95c0_complete, "0x95c0")
