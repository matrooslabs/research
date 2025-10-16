import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Core helpers -----------------------------------------------------------

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


def load_timestamp_close(csv_path: str, start=None, end=None) -> pd.DataFrame:
    df = pd.read_csv(
        csv_path,
        header=None,
        names=BINANCE_KLINE_COLS,
        usecols=[0, 4],  # open_time_us, close
        dtype={0: "int64", 4: "float64"},
    )
    df["timestamp"] = pd.to_datetime(df["open_time_us"], unit="us", utc=True)
    df = df.drop(columns=["open_time_us"]).set_index("timestamp").sort_index()
    if start is not None:
        df = df[df.index >= pd.to_datetime(start, utc=True)]
    if end is not None:
        df = df[df.index <= pd.to_datetime(end, utc=True)]
    return df


def squared_log_returns(close: pd.Series) -> pd.Series:
    return np.log(close).diff().pow(2)


def interval_to_seconds(interval: str) -> float:
    return pd.Timedelta(interval).total_seconds()


def slr_to_daily_vol(slr_sum: pd.Series, interval_seconds: float) -> pd.Series:
    # dailyized vol = sqrt(sum_SLR) * sqrt(86400 / Δt)
    scale = np.sqrt(86400.0 / interval_seconds)
    return np.sqrt(slr_sum) * scale


def make_dailyized_vol_series(
    csv_path: str, interval="5s", start=None, end=None
) -> pd.Series:
    """
    Build dailyized volatility time series for an arbitrary aggregation interval.
    """
    df = load_timestamp_close(csv_path, start=start, end=end)
    slr = squared_log_returns(df["close"]).dropna()
    slr_sum = slr.resample(interval).sum().dropna()
    vol = slr_to_daily_vol(slr_sum, interval_to_seconds(interval))
    return vol


# --- Box plot with means overlay -------------------------------------------


def generate_intervals_5s_to_5min() -> list[str]:
    # 5s, 10s, 15s, ..., 300s (5min)
    out = []
    for s in range(5, 301, 5):
        out.append(f"{s // 60}min" if s % 60 == 0 else f"{s}s")
    return out


def plot_vol_box_by_interval(
    csv_path: str,
    intervals: list[str] | None = None,
    start=None,
    end=None,
    title="Dailyized Volatility Distribution by Aggregation Interval",
    show_means: bool = True,
    annotate_means: bool = False,
):
    """
    Box plot of dailyized volatility for each interval, with optional mean markers.
    """
    if intervals is None:
        intervals = generate_intervals_5s_to_5min()

    # build distributions
    dists, labels, means = [], [], []
    for iv in intervals:
        s = make_dailyized_vol_series(csv_path, interval=iv, start=start, end=end)
        if len(s) == 0:
            continue
        labels.append(iv)
        dists.append(s.values)
        means.append(float(s.mean()))

    plt.figure(figsize=(14, 6))
    bp = plt.boxplot(dists, tick_labels=labels, showfliers=False)
    plt.title(title)
    plt.xlabel("Aggregation Interval")
    plt.ylabel("Dailyized Volatility")
    plt.xticks(rotation=45, ha="right")
    plt.grid(True, axis="y", alpha=0.3)

    # overlay mean markers at box centers (x = 1..N)
    if show_means and len(means) > 0:
        xs = np.arange(1, len(means) + 1)
        plt.plot(xs, means, marker="o", linestyle="none", label="Mean")
        if annotate_means:
            for x, m in zip(xs, means):
                plt.text(x, m, f"{m:.3g}", ha="center", va="bottom")

    if show_means:
        plt.legend()

    plt.tight_layout()


csv_path = "ETHUSDT-1s-2025-08.csv"

# 1) Box plot (whole August) + means overlay
day = 1  # 1 ~ 31

plot_vol_box_by_interval(
    csv_path,
    intervals=generate_intervals_5s_to_5min(),
    start=f"2025-08-{day} 00:00:00",
    end=f"2025-08-{day} 23:59:59",
    show_means=True,
    annotate_means=False,
)
plt.savefig(f"images/vol_box_august_{day}_2025.png")
