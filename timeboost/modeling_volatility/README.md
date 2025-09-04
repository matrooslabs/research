## Measuring Intraday Volatility with Realized Variance

This repository explores practical ways to measure and forecast intraday volatility using high-frequency trade data. We use realized variance as a proxy for latent volatility, visualize where it becomes unreliable due to market microstructure, and evaluate a simple HAR-style forecaster.

### Why realized variance?
- **Volatility is latent**: It is not directly observable at high frequency.
- **Realized variance (RV)**: The sum of squared high-frequency log returns over a fixed horizon approximates the quadratic variation of prices. With sufficiently fine sampling, RV is a consistent nonparametric proxy for integrated variance over the horizon.
- **Dailyization**: We often scale RV to an annualized or dailyized volatility for interpretability. In this repo we use dailyized volatility from intraday bins to make cross-interval comparisons.

### Practical difficulty: microstructure noise
- **Problem**: As sampling becomes too fine (e.g., seconds), bid–ask bounce, discreteness, latency, and other microstructure frictions dominate. Squared returns over-count price changes, biasing RV upward.
- **Bias–variance tradeoff**: Coarser intervals reduce microstructure noise but lose information. There exists a “sweet spot” where RV remains informative without being dominated by noise.
- **Goal in this project**: Empirically reveal where RV begins to be unreliable for the ETHUSDT August 2025 sample.

---

## Code overview

### `measure_volatility.py`
Purpose: Build realized variance and dailyized volatility time series at multiple aggregation intervals and visualize their time evolution.

Key pieces:
- `load_timestamp_close(...)`: Reads Binance klines (no header), keeps timestamp and close, converts microsecond timestamps to UTC.
- `squared_log_returns(close)`: Computes squared log-returns.
- Resampling pipeline:
  1. Compute squared log-returns at the native data frequency.
  2. Aggregate by interval with `.resample(interval).sum()` to obtain realized variance.
  3. Dailyize via `sqrt(sum_SLR) * sqrt(86400 / Δt)` to get comparable volatility units across intervals.
- `plot_multi_interval_vol(...)`: Plots dailyized volatility across intervals (e.g., 5s, 10s, 30s, 1min, 5min) for the selected window.
- Outputs: figure saved to `images/multi_interval_vol_august_2025.png` and a CSV `1m_interval_vol_august_2025.csv` with 1-minute RV and dailyized volatility for the day.

Usage sketch:
```bash
poetry run python measure_volatility.py
```

### `interval_vol.py`
Purpose: Diagnose microstructure-driven unreliability by comparing the distribution of dailyized volatility across many aggregation intervals and checking short-lag autocorrelation.

Key pieces:
- Same data-loading and SLR pipeline as above, plus helpers:
  - `generate_intervals_5s_to_5min()`: Produces a dense grid from 5s to 5min in 5-second steps.
  - `plot_vol_box_by_interval(...)`: Box plots of dailyized volatility per interval with optional mean overlays, letting you see how dispersion and typical levels change as the bin gets finer.

Interpretation of results (ETHUSDT August 2025 sample shown in `images/vol_box_august_1_2025.png` and `images/multi_interval_vol_august_2025.png`):
- As bins become very fine (seconds), dailyized volatility levels inflate, consistent with **microstructure noise** biasing realized variance upward.
- The analysis indicates that the **1-minute interval** appears comparatively robust in this dataset—below that, RV starts to deviate materially, echoing the classic overestimation risk at ultra-high frequencies.

Usage sketch:
```bash
poetry run python interval_vol.py
```

### `vol_estimator.py`
Purpose: Build a simple intraday HAR forecaster on de-seasonalized minute-by-minute realized variance and evaluate out-of-sample.

Pipeline:
- Construct 1-second close series and aggregate to 1-minute realized variance (sum of squared 1s log-returns).
- Build a fixed intraday seasonality curve from the first 7 days to de-seasonalize minute RV.
- Create HAR blocks on de-seasonalized RV: trailing means for (1m, 60m, 1440m) with log transforms and a constant.
- Walk-forward, once per day: fit OLS on the last 7 days, forecast next-minute RV for each minute of the next day, then re-apply the next-minute seasonal factor.
- Outputs: `har_1m_oos_results.csv` and `images/rv_scatter_har_1m.png` (scatter of realized next-minute RV vs forecast).

Takeaways:
- The model can struggle to forecast minute-ahead volatility with high fidelity—short-horizon RV is notoriously noisy, even after de-seasonalization and HAR-type smoothing. The scatter plot shows wide dispersion around the 45° line.
- This “failure” is instructive: ultra-short-horizon volatility forecasting is extremely hard, and even simple, well-specified benchmarks will look poor in R².

---

## Solution for modeling "forecast" of agents

Rational-expectations intuition and RV(T+1):
- Under rational expectations, the market’s conditional expectation of next-period variance at time T is the best unbiased forecast, given information at T.
- Using realized variance at T+1 as the outcome aligns with this logic: if players are rational and able to form correct conditional expectations, then the cross-sectional or time-series mean of forecasts should coincide with RV(T+1) in expectation.
- In practice, noise and limited information ensure dispersion around this benchmark; our results visualize precisely this gap.

---

## References (selected)
- Andersen, T. G., and Bollerslev, T. “Answering the Skeptics: Yes, Standard Volatility Models Do Provide Accurate Forecasts.” International Economic Review, 39(4), 1998.
- Andersen, T. G., Bollerslev, T., Diebold, F. X., and Labys, P. “The Distribution of Realized Exchange Rate Volatility.” Journal of the American Statistical Association, 96(453), 2001.
- Barndorff-Nielsen, O. E., and Shephard, N. “Econometric Analysis of Realised Volatility and Its Use in Estimating Stochastic Volatility Models.” Journal of the Royal Statistical Society B, 64(2), 2002.
- Bandi, F. M., and Russell, J. R. “Microstructure Noise, Realized Variance, and Optimal Sampling.” Journal of Econometrics, 2006.
- Hansen, P. R., and Lunde, A. “Realized Variance and Market Microstructure Noise.” Journal of Business & Economic Statistics, 24(2), 2006.

---

## Reproducing the results

### Requirements
- Python 3.12 (managed by Poetry)
- Data file: `ETHUSDT-1s-2025-08.csv` in the project root

### Setup
```bash
# Install dependencies
poetry install

# Spawn a shell with the environment
poetry shell

# dowload price data from binance and store it
```

### Run scripts
```bash
# 1) Multi-interval volatility visualization and 1m export
poetry run python measure_volatility.py

# 2) Interval diagnostics (box plot across 5s..5min for a chosen day)
poetry run python interval_vol.py

# 3) Walk-forward HAR forecasting and scatter of realized vs forecast
poetry run python vol_estimator.py
```

### Outputs
- `images/multi_interval_vol_august_2025.png`: Dailyized vol across intervals
- `images/vol_box_august_1_2025.png`: Box plot of dailyized vol by interval
- `har_1m_oos_results.csv`: Minute-ahead realized vs forecast panel
- `images/rv_scatter_har_1m.png`: Scatter of realized next-minute RV vs forecast
