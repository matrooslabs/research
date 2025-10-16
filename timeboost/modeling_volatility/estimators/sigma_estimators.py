# vol_estimators.py
# Minimal, unified pipeline for 1-min RV/IQ/RQ from Binance 1s klines, with de-seasonalization.
# Methods: Naive, Tripower (jump-robust), Pre-averaging (noise-robust), Realized Kernel (noise-robust),
# HARQ forecaster (distributional proxy for mean IV), and a tiny IV/2SLS utility.
# Requires: pandas, numpy

from __future__ import annotations
import math, numpy as np, pandas as pd
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

# ---------- Helpers

def load_binance_1s_csv(path: str, tz: str = "UTC", price_col: str = "close") -> pd.DataFrame:
    """
    Expect columns: open_time(ms or iso), open, high, low, close, volume, ... (standard Binance export)
    Produces df with columns ['ts','close'] at 1-second frequency (may have gaps).
    """
    df = pd.read_csv(path)
    # try to locate a time column
    def _parse_ts(series: pd.Series) -> pd.DatetimeIndex:
        s = series.dropna()
        # Try numeric fast-path with unit detection
        if np.issubdtype(s.dtype, np.number):
            v = int(s.iloc[0])
            # Detect epoch unit by digit length
            digits = len(str(abs(v)))
            if digits >= 19:
                unit = 'ns'
            elif digits >= 16:
                unit = 'us'
            elif digits >= 13:
                unit = 'ms'
            else:
                unit = 's'
            return pd.to_datetime(series, unit=unit, utc=True, errors='coerce')
        # Fallback: let pandas infer
        return pd.to_datetime(series, utc=True, errors='coerce')

    if 'open_time' in df.columns:
        ts = _parse_ts(df['open_time'])
    elif 'timestamp' in df.columns:
        ts = _parse_ts(df['timestamp'])
    else:
        # last resort: try first column with unit detection
        ts = _parse_ts(df.iloc[:,0])
    # choose price column
    chosen = None
    if price_col in df.columns:
        chosen = price_col
    elif 'open' in df.columns:
        chosen = 'open'
    elif 'close' in df.columns:
        chosen = 'close'
    else:
        # fallback for headerless Binance CSV: map common positions
        # 0: open_time, 1: open, 4: close
        if df.shape[1] >= 5 and price_col == 'close':
            df['__price__'] = pd.to_numeric(df.iloc[:,4], errors='coerce')
            chosen = '__price__'
        elif df.shape[1] >= 2 and price_col == 'open':
            df['__price__'] = pd.to_numeric(df.iloc[:,1], errors='coerce')
            chosen = '__price__'
        elif df.shape[1] >= 2:
            df['__price__'] = pd.to_numeric(df.iloc[:,1], errors='coerce')
            chosen = '__price__'
    out = pd.DataFrame({'ts': ts, 'price': pd.to_numeric(df[chosen], errors='coerce')}).dropna()
    out = out.sort_values('ts').drop_duplicates('ts')
    out = out.set_index('ts').tz_convert(tz)
    return out

def log_returns_1s(df: pd.DataFrame) -> pd.DataFrame:
    """Compute 1s log returns from 'price'; keep original index (timezone aware)."""
    r = np.log(df['price']).diff()
    return pd.DataFrame({'r': r}).dropna()

def minute_key(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Floor to minute."""
    return idx.floor('min')

def mu_abs_norm(p: float) -> float:
    """E|Z|^p for Z~N(0,1)."""
    return 2**(p/2) * math.gamma((p+1)/2) / math.sqrt(math.pi)

# ---------- De-seasonalization (minute-of-week multiplicative)

class DeSeasonalizer:
    """
    Multiplicative intraday-seasonality factors for variance (s_var[mow]) and quarticity (s_quar[mow]=s_var^2).
    mow = minute of week (0..10079 for 7*24*60).
    """
    def __init__(self):
        self.s_var = None  # length 10080
        self.s_quar = None

    def fit(self, rv_minute: pd.Series) -> None:
        # rv_minute index must be minute timestamps; use log-mean by minute-of-week; normalize to mean=1
        s = rv_minute.copy()
        s = s[s>0]
        mow = (s.index.dayofweek*24*60 + s.index.hour*60 + s.index.minute).astype(int)
        df = pd.DataFrame({'mow': mow, 'logrv': np.log(s.values)})
        g = df.groupby('mow')['logrv'].mean()
        # exponentiate and normalize
        s_var = np.exp(g)
        s_var = s_var / s_var.mean()
        self.s_var = s_var.reindex(range(7*24*60), fill_value=float(s_var.mean())).values
        self.s_quar = self.s_var**2

    def _factor(self, ts: pd.Index, quarticity: bool=False) -> np.ndarray:
        # Ensure we have DatetimeIndex
        if not isinstance(ts, pd.DatetimeIndex):
            raise TypeError("DeSeasonalizer expects a DatetimeIndex for seasonality application.")
        mow = (ts.dayofweek*24*60 + ts.hour*60 + ts.minute).astype(int)
        fac = self.s_quar if quarticity else self.s_var
        return fac[mow]

    def apply_var(self, x: pd.Series) -> pd.Series:
        """De-seasonalize a variance-like series (divide by s_var)."""
        return x / self._factor(x.index, quarticity=False)

    def reapply_var(self, x: pd.Series) -> pd.Series:
        return x * self._factor(x.index, quarticity=False)

    def apply_quar(self, x: pd.Series) -> pd.Series:
        """De-seasonalize a quarticity-like series (divide by s_quar = s_var^2)."""
        return x / self._factor(x.index, quarticity=True)

    def reapply_quar(self, x: pd.Series) -> pd.Series:
        return x * self._factor(x.index, quarticity=True)

# ---------- Base interface

@dataclass
class MinuteEstimates:
    rv: float       # realized variance over the minute
    iq: float       # integrated quarticity estimate over the minute
    rq: float       # realized quarticity (naive 4th-power estimator) over the minute (for reference)
    extras: Dict[str, float]

class BaseMinuteEstimator:
    """
    Shared interface: compute(df_seconds) -> minute DataFrame with columns ['rv','iq','rq', <extras>].
    df_seconds: index in seconds, column 'r' = log returns.
    """
    name: str = "base"

    def __init__(self, deseason: Optional[DeSeasonalizer]=None):
        self.deseason = deseason

    # ----- estimators must implement this on a numpy 1D array of **1s returns within a minute**.
    def _minute_stats(self, r: np.ndarray) -> MinuteEstimates:
        raise NotImplementedError

    def compute(self, df_seconds: pd.DataFrame) -> pd.DataFrame:
        assert 'r' in df_seconds.columns
        # group second returns by minute
        g = df_seconds['r'].groupby(minute_key(df_seconds.index))
        rows = []
        for tmin, rsec in g:
            r = rsec.values
            if len(r) < 3:  # need minimal data
                continue
            est = self._minute_stats(r)
            rows.append((tmin, est))
        if not rows:
            return pd.DataFrame(columns=['rv','iq','rq'])
        # assemble
        idx = pd.DatetimeIndex([t for t,_ in rows])
        out = pd.DataFrame(
            { 'rv':[e.rv for _,e in rows],
              'iq':[e.iq for _,e in rows],
              'rq':[e.rq for _,e in rows] },
            index=idx
        )
        # extras
        extra_keys = sorted({k for _,e in rows for k in e.extras.keys()})
        for k in extra_keys:
            out[k] = [e.extras.get(k, np.nan) for _,e in rows]

        # de-seasonalize (optional)
        if self.deseason is not None:
            if 'rv' in out:   out['rv_ds'] = self.deseason.apply_var(out['rv'])
            if 'iq' in out:   out['iq_ds'] = self.deseason.apply_quar(out['iq'])
            if 'rq' in out:   out['rq_ds'] = self.deseason.apply_quar(out['rq'])
        return out.sort_index()

# ---------- Concrete estimators

class NaiveEstimator(BaseMinuteEstimator):
    """Naive moment estimators: RV = sum r^2; RQ = (n/3)*sum r^4 (IQ proxy if no noise/jumps)."""
    name = "naive"
    def _minute_stats(self, r: np.ndarray) -> MinuteEstimates:
        n = len(r)
        rv = float(np.sum(r*r))
        rq = float((n/3.0) * np.sum(r**4))
        iq = rq  # in noise-free continuous world, RQ -> IQ
        return MinuteEstimates(rv=rv, iq=iq, rq=rq, extras={'n': n})

class TripowerEstimator(BaseMinuteEstimator):
    """
    Jump-robust: Bipower variance (BV) and Tripower quarticity (TPQ).
    BV ~ IV under jumps; TPQ ~ IQ under jumps.
    """
    name = "tripower"
    def _minute_stats(self, r: np.ndarray) -> MinuteEstimates:
        n = len(r)
        # Bipower variance (need n>=2)
        mu1 = mu_abs_norm(1.0)
        bv = np.nan
        if n >= 2:
            bv = (np.pi/2) * (1.0/(n-1)) * np.sum(np.abs(r[1:]) * np.abs(r[:-1])) / (mu1**2)
        # Tripower quarticity (need n>=3)
        mu43 = mu_abs_norm(4/3)
        tpq = np.nan
        if n >= 3:
            x = np.abs(r)**(4/3)
            tpq = (n * 1.0 / (n-2)) * (1.0 / (mu43**3)) * np.sum(x[2:] * x[1:-1] * x[:-2])
        # Provide also naive RQ as reference
        rq = float((n/3.0) * np.sum(r**4))
        rv = float(np.sum(r*r)) if np.isfinite(bv)==False else float(bv)
        iq = float(tpq) if np.isfinite(tpq) else rq
        return MinuteEstimates(rv=rv, iq=iq, rq=rq, extras={'n': n, 'bv': float(bv) if np.isfinite(bv) else np.nan})

class PreAveragingEstimator(BaseMinuteEstimator):
    """
    Noise-robust via pre-averaging (Jacod et al.). Small-sample, concise implementation.
    Weights g(l/m) = min(x,1-x); window m defaults to 5 seconds.
    Normalizations are approximate but standard in practice for short horizons.
    """
    name = "preavg"
    def __init__(self, deseason: Optional[DeSeasonalizer]=None, m: int = 5):
        super().__init__(deseason)
        self.m = m

    def _preavg(self, r: np.ndarray) -> Tuple[np.ndarray, float, float]:
        m = min(self.m, max(2, len(r)//3))
        if m < 2:  # fallback
            return r.copy(), 1.0, 1.0
        w = np.array([min(l/m, 1 - l/m) for l in range(1, m)], dtype=float)  # length m-1
        # normalize weights
        psi2 = np.sum(w**2)
        psi4 = np.sum((np.diff(np.hstack(([0], w, [0]))) )**2)  # rough bias term proxy
        # pre-averaged increments
        bar = np.array([np.dot(w, r[j+1:j+m]) for j in range(0, len(r)-m+1)])
        return bar, psi2, max(psi4, 1e-12)

    def _minute_stats(self, r: np.ndarray) -> MinuteEstimates:
        n = len(r)
        bar, psi2, psi4 = self._preavg(r)
        if len(bar) == 0:
            rv = float(np.sum(r*r)); rq = float((n/3.0)*np.sum(r**4)); return MinuteEstimates(rv, rq, rq, {'n': n})
        # RV (pre-averaged)
        rv_pa = (1.0 / psi2) * np.sum(bar**2)
        # IQ (pre-averaged quarticity proxy): scale sum bar^4
        iq_pa = (1.0 / (psi2**2)) * np.sum(bar**4)  # concise proxy (works well in practice)
        # naive RQ for reference
        rq = float((n/3.0)*np.sum(r**4))
        return MinuteEstimates(rv=float(rv_pa), iq=float(iq_pa), rq=rq, extras={'n': n, 'm': self.m})

class RealizedKernelEstimator(BaseMinuteEstimator):
    """
    Noise-robust realized kernel RV with Parzen kernel; IQ via pre-averaged tripower on same minute (concise, robust).
    """
    name = "rk"
    def __init__(self, deseason: Optional[DeSeasonalizer]=None, H: Optional[int]=None):
        super().__init__(deseason); self.H = H  # bandwidth in seconds; default set from n

    @staticmethod
    def _parzen(x: float) -> float:
        if x < 0 or x > 1: return 0.0
        if x <= 0.5: return 1 - 6*x*x + 6*x*x*x
        return 2*(1-x)**3

    def _rk_rv(self, r: np.ndarray) -> float:
        n = len(r); H = self.H or max(1, int(np.sqrt(n)))
        r = r - np.mean(r)  # de-mean small-sample stabilizer
        gamma0 = float(np.dot(r, r))
        acovs = [np.dot(r[h:], r[:-h]) for h in range(1, H+1)]
        rv = gamma0
        for h,g in enumerate(acovs, start=1):
            k = self._parzen(h/(H+1))
            rv += 2.0 * k * float(g)
        return max(rv, 0.0)

    def _minute_stats(self, r: np.ndarray) -> MinuteEstimates:
        n = len(r)
        rv_rk = self._rk_rv(r)
        # IQ: use tripower on pre-averaged returns (noise + jump robust, concise)
        bar = np.convolve(np.abs(r)**(4/3), np.ones(3), mode='valid')
        mu43 = mu_abs_norm(4/3)
        tpq = np.nan
        if len(bar) > 0:
            tpq = (n * 1.0 / max(1, (n-2))) * (1.0 / (mu43**3)) * np.sum(
                np.abs(r[2:])**(4/3)*np.abs(r[1:-1])**(4/3)*np.abs(r[:-2])**(4/3)
            )
        rq = float((n/3.0) * np.sum(r**4))
        iq = float(tpq) if np.isfinite(tpq) else rq
        return MinuteEstimates(rv=float(rv_rk), iq=iq, rq=rq, extras={'n': n})

# ---------- HARQ forecaster (direct 1-min-ahead on log RV with RQ interaction)

class HARQForecaster:
    """
    Direct projection of log RV_{t+1} on multi-scale log RV + quarticity interaction (HARQ).
    Features at t: log RV_1m, log RV_5m, log RV_60m, log RV_24h, log RQ_5m, and (log RQ_5m * log RV_1m).
    """
    def __init__(self):
        self.coef = None
        self.used_cols = None

    @staticmethod
    def _rolling_mean(x: pd.Series, k: int) -> pd.Series:
        return x.rolling(k, min_periods=k).mean()

    def _features(self, df_min: pd.DataFrame) -> pd.DataFrame:
        # expects columns: rv_ds (de-seasonalized), rq_ds
        lr = np.log(df_min['rv_ds'].clip(lower=1e-16))
        lRQ5 = np.log(self._rolling_mean(df_min['rq_ds'], 5).clip(lower=1e-16))
        X = pd.DataFrame({
            'c': 1.0,
            'lr_1': lr,
            'lr_5': np.log(self._rolling_mean(df_min['rv_ds'], 5).clip(lower=1e-16)),
            'lr_60': np.log(self._rolling_mean(df_min['rv_ds'], 60).clip(lower=1e-16)),
            'lr_1440': np.log(self._rolling_mean(df_min['rv_ds'], 1440).clip(lower=1e-16)),
            'lRQ5': lRQ5,
        }, index=df_min.index)
        if 'lRQ5' in X.columns and 'lr_1' in X.columns:
            X['int'] = X['lRQ5'] * X['lr_1']
        # Drop any feature columns that are entirely NaN (e.g., if dataset shorter than window)
        all_nan_cols = [c for c in X.columns if X[c].isna().all()]
        if all_nan_cols:
            X = X.drop(columns=all_nan_cols)
        y = lr.shift(-1)  # direct 1-step-ahead
        # Drop any rows where either X has NaNs or y is NaN
        row_ok = X.notna().all(axis=1) & y.notna()
        Xc = X.loc[row_ok]
        yc = y.loc[row_ok]
        return Xc, yc

    def fit(self, df_min: pd.DataFrame) -> None:
        X, y = self._features(df_min)
        # OLS closed form
        XtX = X.values.T @ X.values
        Xty = X.values.T @ y.values
        self.coef = np.linalg.solve(XtX + 1e-8*np.eye(XtX.shape[0]), Xty)
        self.used_cols = list(X.columns)

    def predict(self, df_min: pd.DataFrame) -> pd.Series:
        assert self.coef is not None
        X, _ = self._features(df_min)
        # Ensure we use the same columns as in fit, creating missing columns as zeros if needed
        for c in self.used_cols:
            if c not in X.columns:
                X[c] = 0.0
        X = X[self.used_cols]
        lp = X.values @ self.coef
        # return predictive mean of RV (on de-seasonalized scale); re-apply seasonality if present
        pred_rv_ds = np.exp(lp)
        if 'rv_ds' in df_min.columns and 'rv' in df_min.columns:
            # if user wants seasonality back, multiply by s_var via ratio rv/rv_ds at same timestamps
            pass
        return pd.Series(pred_rv_ds, index=X.index, name='rv_pred_ds')

# ---------- Tiny IV/2SLS utility (E[y|X]=Xb; instruments Z for selected columns of X)

def iv_2sls(y: pd.Series, X: pd.DataFrame, Z: pd.DataFrame, endog_cols: List[str]) -> Dict[str, np.ndarray]:
    """
    Simple 2SLS:
      1) First stage: regress X[endog] on Z to get Xhat[endog].
      2) Second stage: regress y on [Xhat[endog], X[exog]].
    Returns dict with beta, resid, and fitted.
    """
    # First-stage
    Xhat = X.copy()
    Zm = Z.values
    ZtZ_inv = np.linalg.pinv(Zm.T @ Zm)
    for c in endog_cols:
        xm = X[[c]].values
        beta1 = ZtZ_inv @ (Zm.T @ xm)
        Xhat[c] = (Zm @ beta1).ravel()
    # Second-stage
    Xm = Xhat.values
    beta2 = np.linalg.pinv(Xm.T @ Xm) @ (Xm.T @ y.values)
    fitted = Xm @ beta2
    resid = y.values - fitted
    return {'beta': beta2, 'fitted': fitted, 'resid': resid, 'cols': list(X.columns)}

# ---------- Example wiring (kept minimal; call these from your script)

def build_minute_table(
    df_px_1s: pd.DataFrame,
    estimator: BaseMinuteEstimator,
    fit_season_on: Tuple[str, str] = None
) -> pd.DataFrame:
    """
    df_px_1s: index in seconds, column 'price'.
    estimator: one of the classes above (sharing BaseMinuteEstimator interface).
    fit_season_on: optional (start_iso, end_iso) to fit seasonality on a training slice.
    """
    df_r = log_returns_1s(df_px_1s)

    # optional seasonality fit on naive RV (stable and fast)
    if estimator.deseason is not None:
        g = df_r['r'].groupby(minute_key(df_r.index)).apply(lambda x: np.sum(x*x))
        if fit_season_on:
            # Align comparison bounds to the timezone of the index to avoid tz-aware/naive mismatches
            tz = g.index.tz
            start_raw = pd.Timestamp(fit_season_on[0])
            end_raw = pd.Timestamp(fit_season_on[1])
            start = start_raw.tz_localize(tz) if start_raw.tzinfo is None else start_raw.tz_convert(tz)
            end = end_raw.tz_localize(tz) if end_raw.tzinfo is None else end_raw.tz_convert(tz)
            mask = (g.index >= start) & (g.index < end)
            estimator.deseason.fit(g.loc[mask])
        else:
            estimator.deseason.fit(g)

    # compute minute metrics
    df_min = estimator.compute(df_r)

    # convenience columns (ensure present for HARQ)
    if estimator.deseason is not None:
        if 'rv_ds' not in df_min: df_min['rv_ds'] = estimator.deseason.apply_var(df_min['rv'])
        if 'rq_ds' not in df_min: df_min['rq_ds'] = estimator.deseason.apply_quar(df_min['rq'])
    return df_min
