import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Column layout for Binance kline CSV (no headers)
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
    """
    Load a Binance kline CSV with no header row and return only timestamp and close.
    Timestamps are converted from microseconds to pandas datetime (UTC).
    """
    df = pd.read_csv(
        csv_path,
        header=None,
        names=BINANCE_KLINE_COLS,
        usecols=[0, 4],  # open_time_us and close
        dtype={0: "int64", 4: "float64"},
    )
    df["timestamp"] = pd.to_datetime(df["open_time_us"], unit="us", utc=True)
    df = df.drop(columns=["open_time_us"])
    df = df.sort_values("timestamp").set_index("timestamp")

    if start is not None:
        df = df[df.index >= pd.to_datetime(start, utc=True)]
    if end is not None:
        df = df[df.index <= pd.to_datetime(end, utc=True)]

    return df


def squared_log_returns(close: pd.Series) -> pd.Series:
    logp = np.log(close)
    r = logp.diff()
    return r.pow(2)


def interval_to_seconds(interval: str) -> float:
    return pd.Timedelta(interval).total_seconds()


def slr_to_daily_vol(slr_sum: pd.Series, interval_seconds: float) -> pd.Series:
    # dailyization: sqrt(sum_SLR) * sqrt(86400 / Δt)
    scale = np.sqrt(86400.0 / interval_seconds)
    return np.sqrt(slr_sum) * scale


def make_dailyized_vol_series(csv_path: str, interval="5s", start=None, end=None):
    df = load_timestamp_close(csv_path, start=start, end=end)
    slr = squared_log_returns(df["close"]).dropna()
    slr_sum = slr.resample(interval).sum().dropna()
    dt_seconds = interval_to_seconds(interval)
    vol_daily = slr_to_daily_vol(slr_sum, dt_seconds)
    return pd.DataFrame({"realized_variance": slr_sum, "daily_volatility": vol_daily})


def plot_multi_interval_vol(
    csv_path: str,
    intervals=("5s", "10s", "30s", "1min", "5min"),
    start=None,
    end=None,
    title="Dailyized Volatility vs Time",
):
    plt.figure(figsize=(12, 6))
    for iv in intervals:
        df_vol = make_dailyized_vol_series(csv_path, interval=iv, start=start, end=end)
        plt.plot(df_vol.index, df_vol["daily_volatility"], label=iv)

    plt.title(title)
    plt.xlabel("Time (UTC)")
    plt.ylabel("Dailyized Volatility")
    plt.legend(title="Interval")
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.gcf().autofmt_xdate()


csv_path = "ETHUSDT-1s-2025-08.csv"
plot_multi_interval_vol(
    csv_path,
    intervals=("5s", "10s", "30s", "1min", "5min"),
    start="2025-08-31 00:00:00",
    end="2025-08-31 23:59:59",
)
plt.savefig("images/multi_interval_vol_august_2025.png")

# save 1m interval vol to csv
df_vol = make_dailyized_vol_series(
    csv_path, interval="1min", start="2025-08-31 00:00:00", end="2025-08-31 23:59:59"
)
df_vol.to_csv("1m_interval_vol_august_2025.csv")
