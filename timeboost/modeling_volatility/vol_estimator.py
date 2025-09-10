import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

# =========================
# 1) Data loading utilities
# =========================

# Binance kline columns (no header)
BINANCE_KLINE_COLS = [
    "open_time_us",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time_us",
    "quote_asset_volume",
    "trades",
    "taker_buy_base",
    "taker_buy_quote",
    "ignore",
]


def load_binance_1s_close(csv_path: str, start=None, end=None) -> pd.Series:
    """
    Read Binance kline CSV (no header), keep timestamp (open_time_us) and close,
    return a UTC-indexed 1-second close series.
    """
    df = pd.read_csv(
        csv_path,
        header=None,
        names=BINANCE_KLINE_COLS,
        usecols=[0, 4],  # open_time_us, close
        dtype={0: "int64", 4: "float64"},
    )
    ts = pd.to_datetime(df["open_time_us"], unit="us", utc=True)
    s = pd.Series(df["close"].values, index=ts).sort_index()
    if start is not None:
        s = s[s.index >= pd.to_datetime(start, utc=True)]
    if end is not None:
        s = s[s.index <= pd.to_datetime(end, utc=True)]
    # If there are gaps/duplicates, you can optionally reindex to a strict 1s grid.
    return s


# ==============================================
# 2) Minute realized variance (RV) from 1-second
# ==============================================


def minute_realized_variance(close_1s: pd.Series) -> pd.Series:
    """
    RV_1m = sum of squared 1s log-returns within each (UTC) minute.
    """
    logp = np.log(close_1s)
    r1s2 = logp.diff().pow(2)
    rv_1m = r1s2.resample("1min").sum().dropna()
    return rv_1m


# =======================================
# 3) Intraday seasonality (minute-of-day)
# =======================================


def make_seasonality_curve(
    rv_1m: pd.Series,
    ref_start: pd.Timestamp,
    ref_end: pd.Timestamp,
    agg: str = "mean",
    eps: float = 1e-18,
) -> pd.Series:
    """
    Build a 1440-length seasonality curve s_tau (minute-of-day -> factor).
    Uses ONLY data between ref_start and ref_end (no look-ahead).
    Normalized to have average 1.0 over the day.
    """
    ref = rv_1m[(rv_1m.index >= ref_start) & (rv_1m.index <= ref_end)]
    if ref.empty:
        raise ValueError("Seasonality reference window has no data.")
    mod = ref.index.hour * 60 + ref.index.minute
    grp = pd.DataFrame({"rv": ref.values, "mod": mod})
    if agg == "median":
        s = grp.groupby("mod")["rv"].median()
    else:
        s = grp.groupby("mod")["rv"].mean()
    # Ensure all 1440 minutes exist; fill missing with global mean
    s = s.reindex(range(1440))
    s = s.fillna(s.mean())
    # Normalize so average factor = 1.0 (keeps overall scale intact)
    s = s / (s.mean() + eps)
    s.name = "seasonality_factor"
    return s


def apply_seasonality(
    rv_1m: pd.Series, s_curve: pd.Series, eps: float = 1e-18
) -> tuple[pd.Series, pd.Series]:
    """
    De-seasonalize: RV_tilde = RV / s_tau, where tau = minute-of-day.
    Also returns the per-timestamp factor series (aligned to rv_1m index).
    """
    mod = rv_1m.index.hour * 60 + rv_1m.index.minute
    s_t = s_curve.reindex(mod).values
    s_series = pd.Series(s_t, index=rv_1m.index, name="s_factor")
    rv_tilde = rv_1m / (s_series + eps)
    rv_tilde.name = "rv_1m_deseasonalized"
    return rv_tilde, s_series


# ============================================
# 4) HAR feature construction (minute horizons)
# ============================================


def har_blocks(
    rv_tilde: pd.Series, windows=(1, 60, 1440), eps: float = 1e-18
) -> pd.DataFrame:
    """
    Build log-HAR regressors on de-seasonalized RV:
        x_t = [1, log(RV^{(1m)}_t), log(RV^{(15m)}_t), log(RV^{(60m)}_t), log(RV^{(1d)}_t)]
    where RV^{(k)}_t is the trailing mean of the last k minutes, inclusive of t.
    """
    X = {}
    for w in windows:
        avg = rv_tilde.rolling(window=w, min_periods=w).mean()
        X[f"log_avg_{w}m"] = np.log(avg + eps)
    X = pd.DataFrame(X, index=rv_tilde.index)
    # Target (one-step-ahead, still de-seasonalized): y_{t+1} = log(RV_{t+1}~ + eps)
    y = np.log(rv_tilde.shift(-1) + eps)
    X["const"] = 1.0
    # Align shapes
    df = pd.concat([X, y.rename("y_next")], axis=1).dropna()
    return df


# ===========================
# 5) Plain OLS (no statistics)
# ===========================


def ols_fit_predict(
    X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame
) -> np.ndarray:
    """
    Fit OLS: beta = (X'X)^{-1} X'y; predict yhat = X_test beta.
    Assumes 'const' column present in X_*.
    """
    cols = X_train.columns
    Xtr = X_train.values
    ytr = y_train.values
    beta, *_ = np.linalg.lstsq(Xtr, ytr, rcond=None)
    yhat = X_test.values @ beta
    return yhat


# =======================================
# 6) Walk-forward daily HAR with no peeking
# =======================================


def walkforward_har_forecast(
    rv_1m: pd.Series,
    seasonality_seed_days: int = 7,
    reg_window_days: int = 7,
    windows=(1, 60, 1440),
    agg_seasonality: str = "mean",
    eps: float = 1e-18,
) -> pd.DataFrame:
    """
    - Build a fixed seasonality curve from the FIRST `seasonality_seed_days`.
      (No look-ahead; simple & robust. You can recompute weekly if desired.)
    - De-seasonalize the entire series using that curve.
    - HAR(1m,15m,60m,1d) OLS refit **once per day** on the last `reg_window_days`.
    - Forecast y_{t+1} for each minute in the out-of-sample period.
    - Map forecasts back to original (re-apply next-minute seasonal factor).
    Returns a DataFrame with realized RV, forecasts, residuals, and losses.
    """
    # Ensure minute index is strictly minutely (resampler already did it)
    rv_1m = rv_1m.sort_index()

    # 6.1 Seasonality curve from the first `seasonality_seed_days`
    first_day = rv_1m.index.normalize()[0]
    ref_start = first_day
    ref_end = (
        first_day + pd.Timedelta(days=seasonality_seed_days) - pd.Timedelta(minutes=1)
    )
    s_curve = make_seasonality_curve(
        rv_1m, ref_start, ref_end, agg=agg_seasonality, eps=eps
    )

    # 6.2 De-seasonalize whole panel (uses fixed s_curve -> no peeking)
    rv_tilde, s_factor = apply_seasonality(rv_1m, s_curve, eps=eps)

    # 6.3 Build HAR design (lags/averages & next-minute target, all de-seasonalized)
    design = har_blocks(rv_tilde, windows=windows, eps=eps)
    # Split into X (without y) and y
    xcols = [c for c in design.columns if c.startswith("log_avg_")] + ["const"]
    X_all = design[xcols]
    y_all = design["y_next"]

    # 6.4 Walk-forward by day: fit on trailing `reg_window_days`, predict next day minute-by-minute
    # Compute day boundaries on the minute index used in X_all/y_all (same as rv_tilde except for dropped NA)
    minutes = X_all.index
    # Unique days as python datetime.date objects to avoid Timestamp-date comparisons
    days = pd.Index(minutes.date).unique()

    # Out-of-sample should start AFTER we have:
    # - 1 day of history for 1440m average,
    # - seasonality seed,
    # - regression window (in days).
    min_start_day = (
        first_day + pd.Timedelta(days=max(1, seasonality_seed_days, reg_window_days))
    ).date()

    forecasts = []
    for d in days:
        if d < min_start_day:
            continue  # Not enough history yet
        day_start = pd.Timestamp(d).tz_localize("UTC")
        day_end = day_start + pd.Timedelta(days=1) - pd.Timedelta(minutes=1)

        # Training window: last `reg_window_days` ending previous day
        train_start = day_start - pd.Timedelta(days=reg_window_days)
        train_end = day_start - pd.Timedelta(minutes=1)
        mask_train = (X_all.index >= train_start) & (X_all.index <= train_end)
        X_train = X_all.loc[mask_train]
        y_train = y_all.loc[mask_train]
        if len(X_train) < 2000:
            # too few points (shouldn't happen with 7d), skip this day
            continue

        # Test minutes: the whole day 'd'
        mask_test = (X_all.index >= day_start) & (X_all.index <= day_end)
        X_test = X_all.loc[mask_test]
        if X_test.empty:
            continue

        # Fit once per day, predict all minutes in the day (yhat are logs of de-seasonalized RV_{t+1})
        yhat_log_de = ols_fit_predict(X_train, y_train, X_test)

        # Map forecasts back to levels (de-seasonalized)
        rvhat_de = np.exp(yhat_log_de) - eps
        rvhat_de = pd.Series(rvhat_de, index=X_test.index, name="rvhat_de")

        # Align to realized next-minute timestamps:
        # Our target y_{t+1} is aligned to X at time t; realized is RV_1m.shift(-1) at index t
        rv_real_next = rv_1m.shift(-1).reindex(X_test.index)

        # Re-apply NEXT-minute seasonality factor (aligned to target index)
        s_next = s_factor.shift(-1).reindex(X_test.index)
        rvhat = rvhat_de * s_next
        rvhat.name = "rvhat"

        # Collect a panel for this day
        df_day = pd.DataFrame(
            {
                "rv_real_next": rv_real_next,
                "rvhat": rvhat,
                "rvhat_de": rvhat_de,
                "s_next": s_next,
            }
        ).dropna()

        # Residuals & losses
        df_day["residual"] = df_day["rvhat"] - df_day["rv_real_next"]
        # QLIKE: rv/rvhat - log(rv/rvhat) - 1  (clip to avoid numerical issues)
        rv_c = np.clip(df_day["rv_real_next"].values, 1e-18, None)
        fh_c = np.clip(df_day["rvhat"].values, 1e-18, None)
        df_day["qlike"] = rv_c / fh_c - np.log(rv_c / fh_c) - 1.0

        forecasts.append(df_day)

    if not forecasts:
        raise RuntimeError(
            "No out-of-sample forecasts produced. Check windows and data coverage."
        )

    out = pd.concat(forecasts).sort_index()
    out["abs_error"] = out["residual"].abs()
    out["sq_error"] = out["residual"].pow(2)
    return out


# =================================
# 7) Convenience: score & summarize
# =================================


def summarize_performance(results: pd.DataFrame) -> pd.Series:
    """
    Basic OOS performance summary.
    """
    s = pd.Series(dtype="float64")
    s["n_obs"] = len(results)
    s["MAE"] = results["abs_error"].mean()
    s["RMSE"] = np.sqrt(results["sq_error"].mean())
    s["mean_QLIKE"] = results["qlike"].mean()
    # Optional: accuracy on volatility (stdev)
    vol_real = np.sqrt(results["rv_real_next"])
    vol_hat = np.sqrt(results["rvhat"])
    s["MAE_vol"] = (vol_hat - vol_real).abs().mean()
    s["RMSE_vol"] = np.sqrt(((vol_hat - vol_real) ** 2).mean())
    return s


# =====================
# 8) End-to-end runner
# =====================

if __name__ == "__main__":
    csv_path = "ETHUSDT-1s-2025-08.csv"  # <-- your file

    # Load 1s close for August
    close_1s = load_binance_1s_close(csv_path)

    # Realized variance per minute
    rv_1m = minute_realized_variance(close_1s)

    # Walk-forward HAR with fixed seasonality from the first 7 days,
    # daily re-fit on last 7 days, and (1m,15m,60m,1d) blocks.
    results = walkforward_har_forecast(
        rv_1m,
        seasonality_seed_days=7,
        reg_window_days=7,
        windows=(1, 60, 1440),
        agg_seasonality="mean",  # or "median" for extra robustness
    )

    # Summaries
    summary = summarize_performance(results)
    print("Out-of-sample summary:")
    print(summary)

    # If you want the “difference between forecast and realized” as a time series:
    results["residual"] = results["rvhat"] - results["rv_real_next"]
    # For quick inspection:
    print("\nSample of forecast vs realized (head):")
    print(results[["rv_real_next", "rvhat", "residual", "qlike"]].head())

    # You can also save the detailed results:
    results.to_csv("har_1m_oos_results.csv")

    # Scatter plot: realized vs forecast
    os.makedirs("images", exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(
        results["rv_real_next"],
        results["rvhat"],
        s=5,
        alpha=0.3,
        edgecolors="none",
    )
    lim_min = 0.0
    lim_max = float(np.nanmax(results[["rv_real_next", "rvhat"]].values))
    ax.plot(
        [lim_min, lim_max],
        [lim_min, lim_max],
        color="red",
        linewidth=1.0,
        label="y = x",
    )
    ax.set_xlabel("Realized RV (next minute)")
    ax.set_ylabel("Forecast RV")
    ax.set_title("HAR 1m Forecast vs Realized")
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    ax.legend()
    fig.tight_layout()
    fig.savefig("images/rv_scatter_har_1m.png", dpi=150)
    plt.close(fig)
