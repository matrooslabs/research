import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from estimators.sigma_estimators import *

# Load 1-second data using open prices as the price column
df = load_binance_1s_csv("ETHUSDT-1s-2025-08.csv", tz="UTC", price_col="open")

# Build 1-minute realized metrics using a robust estimator and de-seasonalization
deseason = DeSeasonalizer()
est = RealizedKernelEstimator(deseason=deseason, H=8)  # alternatives: NaiveEstimator(), TripowerEstimator(), PreAveragingEstimator()
df_min = build_minute_table(df, est, fit_season_on=("2025-08-01","2025-08-31"))

# Forecast de-seasonalized RV one-step-ahead with HARQ
harq = HARQForecaster()
harq.fit(df_min)
df_min['rv_pred_ds'] = harq.predict(df_min)

# Forecast de-seasonalized RQ one-step-ahead via simple AR(1) on log RQ
lRQ = np.log(df_min['rq_ds'].clip(lower=1e-16)) if 'rq_ds' in df_min else np.log(df_min['rq'].clip(lower=1e-16))
X_rq = pd.DataFrame({'c': 1.0, 'lRQ_t': lRQ}, index=df_min.index)
y_rq = lRQ.shift(-1)
rq_train = X_rq.join(y_rq.to_frame('lRQ_tp1')).dropna()
if len(rq_train) >= 10:
    Xm = rq_train[['c','lRQ_t']].values
    ym = rq_train['lRQ_tp1'].values
    beta = np.linalg.lstsq(Xm, ym, rcond=None)[0]
    lRQ_pred = (X_rq[['c','lRQ_t']].values @ beta)
    rq_pred_ds = np.exp(lRQ_pred)
    df_min['rq_pred_ds'] = pd.Series(rq_pred_ds, index=df_min.index)

# Evaluation helpers
def eval_stats(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    yt = y_true[mask]
    yp = y_pred[mask]
    if len(yt) == 0:
        return {'n': 0, 'mse': np.nan, 'mae': np.nan, 'r2': np.nan}
    err = yt - yp
    mse = float(np.mean(err**2))
    mae = float(np.mean(np.abs(err)))
    sst = float(np.sum((yt - yt.mean())**2))
    r2 = float(1.0 - np.sum(err**2) / sst) if sst > 0 else np.nan
    return {'n': int(len(yt)), 'mse': mse, 'mae': mae, 'r2': r2}

# Align and compute residuals on de-seasonalized scale
df_eval_rv = df_min[[c for c in ['rv_ds','rv_pred_ds'] if c in df_min.columns]].dropna()
df_eval_rq = df_min[[c for c in ['rq_ds','rq_pred_ds'] if c in df_min.columns]].dropna()

# Compare rv_pred(t) to rv_true(t+1) by shifting prediction index forward by 1 minute
if {'rv_ds','rv_pred_ds'}.issubset(df_min.columns):
    rv_true = df_min['rv_ds']
    rv_pred = df_min['rv_pred_ds'].copy()
    rv_pred.index = rv_pred.index + pd.Timedelta(minutes=1)
    rv_eval = pd.concat([
        rv_true.rename('rv_true'),
        rv_pred.rename('rv_pred')
    ], axis=1).dropna()
    rv_stats = eval_stats(rv_eval['rv_true'].values, rv_eval['rv_pred'].values)
    print(f"RV forecast stats (ds): n={rv_stats['n']}, mse={rv_stats['mse']:.4e}, mae={rv_stats['mae']:.4e}, r2={rv_stats['r2']:.3f}")

if {'rq_ds','rq_pred_ds'}.issubset(df_eval_rq.columns):
    rq_stats = eval_stats(df_eval_rq['rq_ds'].values, df_eval_rq['rq_pred_ds'].values)
    print(f"RQ forecast stats (ds): n={rq_stats['n']}, mse={rq_stats['mse']:.4e}, mae={rq_stats['mae']:.4e}, r2={rq_stats['r2']:.3f}")

# Plot residual time series
fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
row = 0
if 'rv_eval' in locals() and len(rv_eval) > 0:
    resid_rv = (rv_eval['rv_true'] - rv_eval['rv_pred']).dropna()
    axes[row].plot(resid_rv.index, resid_rv.values)
    axes[row].set_title("Residual (RV ds): rv_true(t+1) - rv_pred(t)")
    row += 1
if {'rq_ds','rq_pred_ds'}.issubset(df_eval_rq.columns):
    resid_rq = df_eval_rq['rq_ds'] - df_eval_rq['rq_pred_ds']
    axes[row].plot(resid_rq.index, resid_rq.values)
    axes[row].set_title("Residual (RQ ds): rq_ds - rq_pred_ds")
    row += 1

# If only one subplot populated, trim extra axis
if row == 1 and len(axes) == 2:
    fig.delaxes(axes[1])

plt.tight_layout()
plt.savefig("images/rv_rq_residuals.png", dpi=160)
plt.show()