from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union, Dict

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm


ArrayLike = Union[np.ndarray, list]


@dataclass
class TobitResult:
    beta: np.ndarray
    sigma: float
    vcov: np.ndarray
    se_beta: np.ndarray
    se_sigma: float
    loglike: float
    nobs: int
    n_left: int
    n_right: int
    n_uncensored: int
    success: bool
    message: str
    nit: int
    aic: float
    bic: float

    def summary(self) -> Dict[str, object]:
        return {
            "beta": self.beta,
            "sigma": self.sigma,
            "se_beta": self.se_beta,
            "se_sigma": self.se_sigma,
            "loglike": self.loglike,
            "nobs": self.nobs,
            "n_left": self.n_left,
            "n_right": self.n_right,
            "n_uncensored": self.n_uncensored,
            "success": self.success,
            "message": self.message,
            "nit": self.nit,
            "aic": self.aic,
            "bic": self.bic,
        }


def _as_2d(X: ArrayLike, add_intercept: bool) -> np.ndarray:
    X_arr = np.asarray(X, dtype=float)
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(-1, 1)
    if add_intercept:
        ones = np.ones((X_arr.shape[0], 1), dtype=float)
        X_arr = np.hstack([ones, X_arr])
    return X_arr


def _init_params(y: np.ndarray, X: np.ndarray, mask_unc: np.ndarray) -> np.ndarray:
    # OLS on uncensored if available, else all
    if mask_unc.any():
        y0 = y[mask_unc]
        X0 = X[mask_unc]
    else:
        y0 = y
        X0 = X
    beta_ols, *_ = np.linalg.lstsq(X0, y0, rcond=None)
    resid = y0 - X0 @ beta_ols
    sigma0 = float(
        np.std(
            resid,
            ddof=min(len(beta_ols), X0.shape[0]) if X0.shape[0] > len(beta_ols) else 1,
        )
    )
    sigma0 = max(sigma0, 1e-6)
    return np.concatenate([beta_ols, np.array([np.log(sigma0)])])


def _build_masks(
    y: np.ndarray,
    left: Optional[Union[float, np.ndarray]],
    right: Optional[Union[float, np.ndarray]],
    is_left_censored: Optional[np.ndarray],
    is_right_censored: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = y.shape[0]
    if is_left_censored is None:
        if left is None:
            is_left = np.zeros(n, dtype=bool)
        else:
            is_left = y <= (left if np.isscalar(left) else np.asarray(left))
    else:
        is_left = np.asarray(is_left_censored, dtype=bool)

    if is_right_censored is None:
        if right is None:
            is_right = np.zeros(n, dtype=bool)
        else:
            is_right = y >= (right if np.isscalar(right) else np.asarray(right))
    else:
        is_right = np.asarray(is_right_censored, dtype=bool)

    mask_unc = ~(is_left | is_right)

    # normalize left/right into arrays for vectorized ops
    if left is None:
        left_arr = np.full(n, -np.inf)
        left_arr[is_left] = y[is_left]
    else:
        left_arr = (
            np.asarray(left) if not np.isscalar(left) else np.full(n, float(left))
        )
    if right is None:
        right_arr = np.full(n, np.inf)
        right_arr[is_right] = y[is_right]
    else:
        right_arr = (
            np.asarray(right) if not np.isscalar(right) else np.full(n, float(right))
        )

    return is_left, is_right, mask_unc, left_arr, right_arr


def _neg_loglike(
    theta: np.ndarray,
    y: np.ndarray,
    X: np.ndarray,
    is_left: np.ndarray,
    is_right: np.ndarray,
    mask_unc: np.ndarray,
    left_arr: np.ndarray,
    right_arr: np.ndarray,
) -> float:
    p = X.shape[1]
    beta = theta[:p]
    log_sigma = theta[p]
    sigma = np.exp(log_sigma)

    xb = X @ beta
    z_unc = (y - xb) / sigma
    z_left = (left_arr - xb) / sigma
    z_right = (right_arr - xb) / sigma

    ll = np.zeros_like(y, dtype=float)
    if mask_unc.any():
        ll[mask_unc] = norm.logpdf(z_unc[mask_unc]) - np.log(sigma)
    if is_left.any():
        ll[is_left] = norm.logcdf(z_left[is_left])
    if is_right.any():
        ll[is_right] = norm.logsf(z_right[is_right])

    return float(-np.sum(ll))


def fit_tobit(
    y: ArrayLike,
    X: ArrayLike,
    *,
    left: Optional[Union[float, np.ndarray]] = None,
    right: Optional[Union[float, np.ndarray]] = None,
    is_left_censored: Optional[np.ndarray] = None,
    is_right_censored: Optional[np.ndarray] = None,
    add_intercept: bool = True,
    method: str = "BFGS",
    maxiter: int = 1000,
    tol: float = 1e-8,
    use_numeric_hessian: bool = True,
) -> TobitResult:
    """
    Fit a Tobit (censored normal) regression via MLE.

    Parameters
    - y: response (n,)
    - X: design (n, k) or (n,)
    - left/right: scalar or array censoring bounds. If None, will infer from masks
    - is_left_censored / is_right_censored: boolean masks. If None, inferred by comparing to bounds
    - add_intercept: prepend a column of ones to X

    Returns
    - TobitResult with parameters, standard errors, and diagnostics
    """
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    X_arr = _as_2d(X, add_intercept=add_intercept)
    if X_arr.shape[0] != y_arr.shape[0]:
        raise ValueError("X and y must have the same number of rows")

    is_left, is_right, mask_unc, left_arr, right_arr = _build_masks(
        y_arr, left, right, is_left_censored, is_right_censored
    )

    theta0 = _init_params(y_arr, X_arr, mask_unc)

    obj = lambda th: _neg_loglike(
        th, y_arr, X_arr, is_left, is_right, mask_unc, left_arr, right_arr
    )

    res = minimize(
        obj,
        theta0,
        method=method,
        options={"maxiter": maxiter, "gtol": tol},
    )

    # parameters on transformed scale (beta, log_sigma)
    p = X_arr.shape[1]
    beta_hat = res.x[:p]
    log_sigma_hat = float(res.x[p])
    sigma_hat = float(np.exp(log_sigma_hat))

    # covariance on transformed scale
    def _numerical_hessian(func, x, eps: float = 1e-5) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        n = x.size
        H = np.zeros((n, n), dtype=float)
        f0 = func(x)
        for i in range(n):
            ei = np.zeros(n)
            ei[i] = eps
            f_ip = func(x + ei)
            f_im = func(x - ei)
            H[i, i] = (f_ip - 2 * f0 + f_im) / (eps**2)
            for j in range(i + 1, n):
                ej = np.zeros(n)
                ej[j] = eps
                f_pp = func(x + ei + ej)
                f_pm = func(x + ei - ej)
                f_mp = func(x - ei + ej)
                f_mm = func(x - ei - ej)
                H_ij = (f_pp - f_pm - f_mp + f_mm) / (4 * eps**2)
                H[i, j] = H_ij
                H[j, i] = H_ij
        return H

    vcov_theta: np.ndarray
    used_numeric = False
    if use_numeric_hessian:
        try:
            H = _numerical_hessian(obj, res.x)
            vcov_theta = np.linalg.inv(H)
            used_numeric = True
        except Exception:
            used_numeric = False
            vcov_theta = np.full((p + 1, p + 1), np.nan)
    if not use_numeric_hessian or not used_numeric:
        if hasattr(res, "hess_inv"):
            hess_inv = res.hess_inv
            if hasattr(hess_inv, "todense"):
                vcov_theta = np.asarray(hess_inv.todense())
            else:
                vcov_theta = np.asarray(hess_inv)
        else:
            vcov_theta = np.full((p + 1, p + 1), np.nan)

    # delta method to transform (beta, log_sigma) -> (beta, sigma)
    J = np.eye(p + 1)
    J[p, p] = sigma_hat
    vcov = J @ vcov_theta @ J.T

    se_beta = np.sqrt(np.clip(np.diag(vcov)[:p], 0.0, np.inf))
    se_sigma = float(np.sqrt(np.clip(vcov[p, p], 0.0, np.inf)))

    nobs = int(y_arr.shape[0])
    n_left = int(is_left.sum())
    n_right = int(is_right.sum())
    n_unc = int(mask_unc.sum())
    ll = -obj(res.x)
    k_params = p + 1
    aic = 2 * k_params - 2 * ll
    bic = np.log(nobs) * k_params - 2 * ll

    return TobitResult(
        beta=beta_hat,
        sigma=sigma_hat,
        vcov=vcov,
        se_beta=se_beta,
        se_sigma=se_sigma,
        loglike=float(ll),
        nobs=nobs,
        n_left=n_left,
        n_right=n_right,
        n_uncensored=n_unc,
        success=bool(res.success),
        message=str(res.message),
        nit=int(res.nit) if hasattr(res, "nit") else -1,
        aic=float(aic),
        bic=float(bic),
    )


def predict_latent(
    X: ArrayLike, beta: np.ndarray, add_intercept: bool = True
) -> np.ndarray:
    X_arr = _as_2d(X, add_intercept=add_intercept)
    return X_arr @ beta
