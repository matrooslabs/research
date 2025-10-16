from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union, Dict

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.stats import t as student_t


ArrayLike = Union[np.ndarray, list]


# =============================== Num. utilities ===============================

_LOG2PI = np.log(2.0 * np.pi)


def _logphi(z: np.ndarray) -> np.ndarray:
    return -0.5 * (z ** 2) - 0.5 * _LOG2PI


def _logcdf(z: np.ndarray) -> np.ndarray:
    # Stable log CDF
    return norm.logcdf(z)


def _logsf(z: np.ndarray) -> np.ndarray:
    # Stable log survival function
    return norm.logsf(z)


def _mills_left(z: np.ndarray) -> np.ndarray:
    # lambda(z) = phi(z) / Phi(z) in log-domain
    return np.exp(_logphi(z) - _logcdf(z))


def _mills_right(z: np.ndarray) -> np.ndarray:
    # lambda_bar(z) = phi(z) / (1 - Phi(z)) in log-domain
    return np.exp(_logphi(z) - _logsf(z))


def _numerical_hessian(func, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    n = x.size
    H = np.zeros((n, n), dtype=float)
    f0 = func(x)
    step = eps * np.maximum(1.0, np.abs(x))
    for i in range(n):
        ei = np.zeros(n)
        ei[i] = step[i]
        f_ip = func(x + ei)
        f_im = func(x - ei)
        H[i, i] = (f_ip - 2 * f0 + f_im) / (step[i] ** 2)
        for j in range(i + 1, n):
            ej = np.zeros(n)
            ej[j] = step[j]
            f_pp = func(x + ei + ej)
            f_pm = func(x + ei - ej)
            f_mp = func(x - ei + ej)
            f_mm = func(x - ei - ej)
            H_ij = (f_pp - f_pm - f_mp + f_mm) / (4 * step[i] * step[j])
            H[i, j] = H_ij
            H[j, i] = H_ij
    return H


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
        ll[mask_unc] = _logphi(z_unc[mask_unc]) - np.log(sigma)
    if is_left.any():
        ll[is_left] = _logcdf(z_left[is_left])
    if is_right.any():
        ll[is_right] = _logsf(z_right[is_right])

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
        method="L-BFGS-B",
        options={"maxiter": maxiter, "ftol": tol, "gtol": tol, "maxls": 50, "eps": 1e-10},
    )

    # parameters on transformed scale (beta, log_sigma)
    p = X_arr.shape[1]
    beta_hat = res.x[:p]
    log_sigma_hat = float(res.x[p])
    sigma_hat = float(np.exp(log_sigma_hat))

    # covariance on transformed scale
    vcov_theta: np.ndarray
    used_numeric = False
    if use_numeric_hessian:
        try:
            H = _numerical_hessian(obj, res.x, eps=1e-6)
            vcov_theta = np.linalg.pinv(H)
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


# =========================== Heteroskedastic Tobit ===========================


@dataclass
class HeteroTobitResult:
    beta: np.ndarray
    gamma: np.ndarray
    vcov: np.ndarray
    se_beta: np.ndarray
    se_gamma: np.ndarray
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
            "gamma": self.gamma,
            "se_beta": self.se_beta,
            "se_gamma": self.se_gamma,
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


def predict_sigma(Z: ArrayLike, gamma: np.ndarray, add_intercept_sigma: bool = True) -> np.ndarray:
    Z_arr = _as_2d(Z, add_intercept=add_intercept_sigma)
    log_sigma_i = Z_arr @ gamma
    return np.exp(log_sigma_i)


def fit_tobit_hetero(
    y: ArrayLike,
    X: ArrayLike,
    Z: Optional[ArrayLike] = None,
    *,
    left: Optional[Union[float, np.ndarray]] = None,
    right: Optional[Union[float, np.ndarray]] = None,
    is_left_censored: Optional[np.ndarray] = None,
    is_right_censored: Optional[np.ndarray] = None,
    add_intercept: bool = True,
    add_intercept_sigma: bool = True,
    method: str = "BFGS",
    maxiter: int = 1000,
    tol: float = 1e-8,
    use_numeric_hessian: bool = True,
) -> HeteroTobitResult:
    """
    Heteroskedastic Tobit via MLE.

    Mean: y*_i = X_i @ beta + u_i
    Var:  u_i ~ N(0, sigma_i^2),  log(sigma_i) = Z_i @ gamma
    """
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    X_arr = _as_2d(X, add_intercept=add_intercept)
    if Z is None:
        Z_arr = np.ones((X_arr.shape[0], 1), dtype=float) if add_intercept_sigma else np.empty((X_arr.shape[0], 0))
    else:
        Z_arr = _as_2d(Z, add_intercept=add_intercept_sigma)

    if X_arr.shape[0] != y_arr.shape[0] or Z_arr.shape[0] != y_arr.shape[0]:
        raise ValueError("X, Z, and y must have the same number of rows")

    is_left, is_right, mask_unc, left_arr, right_arr = _build_masks(
        y_arr, left, right, is_left_censored, is_right_censored
    )

    # initialize from homoskedastic
    theta0_homo = _init_params(y_arr, X_arr, mask_unc)
    beta0 = theta0_homo[:-1]
    log_sigma0 = float(theta0_homo[-1])
    gamma0 = np.zeros(Z_arr.shape[1], dtype=float)
    if gamma0.size > 0:
        gamma0[0] = log_sigma0
    theta0 = np.concatenate([beta0, gamma0])

    p = X_arr.shape[1]
    q = Z_arr.shape[1]

    def nll(theta: np.ndarray) -> float:
        beta = theta[:p]
        gamma = theta[p : p + q]
        xb = X_arr @ beta
        log_sigma_i = Z_arr @ gamma
        sigma_i = np.exp(log_sigma_i)

        z_unc = (y_arr - xb) / sigma_i
        z_left = (left_arr - xb) / sigma_i
        z_right = (right_arr - xb) / sigma_i

        ll = np.zeros_like(y_arr, dtype=float)
        if mask_unc.any():
            ll[mask_unc] = norm.logpdf(z_unc[mask_unc]) - log_sigma_i[mask_unc]
        if is_left.any():
            ll[is_left] = norm.logcdf(z_left[is_left])
        if is_right.any():
            ll[is_right] = norm.logsf(z_right[is_right])
        return float(-np.sum(ll))

    res = minimize(
        nll,
        theta0,
        method="L-BFGS-B",
        options={"maxiter": maxiter, "ftol": tol, "gtol": tol, "maxls": 50, "eps": 1e-10},
    )


# ===================== Heteroskedastic Tobit with Student-t ====================


@dataclass
class HeteroTobitTResult:
    beta: np.ndarray
    gamma: np.ndarray
    nu: float
    vcov: np.ndarray
    se_beta: np.ndarray
    se_gamma: np.ndarray
    se_nu: float
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
            "gamma": self.gamma,
            "nu": self.nu,
            "se_beta": self.se_beta,
            "se_gamma": self.se_gamma,
            "se_nu": self.se_nu,
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


def fit_tobit_hetero_t(
    y: ArrayLike,
    X: ArrayLike,
    Z: Optional[ArrayLike] = None,
    *,
    left: Optional[Union[float, np.ndarray]] = None,
    right: Optional[Union[float, np.ndarray]] = None,
    is_left_censored: Optional[np.ndarray] = None,
    is_right_censored: Optional[np.ndarray] = None,
    add_intercept: bool = True,
    add_intercept_sigma: bool = True,
    method: str = "L-BFGS-B",
    maxiter: int = 2000,
    tol: float = 1e-9,
    use_numeric_hessian: bool = True,
) -> HeteroTobitTResult:
    """
    Heteroskedastic Tobit with Student-t residuals via MLE.

    Mean: y*_i = X_i @ beta + u_i
    Scale: sigma_i = exp(Z_i @ gamma)
    Residual: u_i / sigma_i ~ t_{nu}
    """
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    X_arr = _as_2d(X, add_intercept=add_intercept)
    if Z is None:
        Z_arr = np.ones((X_arr.shape[0], 1), dtype=float) if add_intercept_sigma else np.empty((X_arr.shape[0], 0))
    else:
        Z_arr = _as_2d(Z, add_intercept=add_intercept_sigma)

    if X_arr.shape[0] != y_arr.shape[0] or Z_arr.shape[0] != y_arr.shape[0]:
        raise ValueError("X, Z, and y must have the same number of rows")

    is_left, is_right, mask_unc, left_arr, right_arr = _build_masks(
        y_arr, left, right, is_left_censored, is_right_censored
    )

    # initialize from normal heteroskedastic model
    theta0_homo = _init_params(y_arr, X_arr, mask_unc)
    beta0 = theta0_homo[:-1]
    log_sigma0 = float(theta0_homo[-1])
    gamma0 = np.zeros(Z_arr.shape[1], dtype=float)
    if gamma0.size > 0:
        gamma0[0] = log_sigma0
    # nu parameterization: nu = 2 + exp(eta) ensures nu > 2
    eta0 = np.log(8.0)  # nu ~ 2 + e^eta => start around nu ~ 2 + 8 = 10
    theta0 = np.concatenate([beta0, gamma0, np.array([eta0], dtype=float)])

    p = X_arr.shape[1]
    q = Z_arr.shape[1]

    def nll(theta: np.ndarray) -> float:
        beta = theta[:p]
        gamma = theta[p : p + q]
        eta = float(theta[p + q])
        nu = 2.0 + np.exp(eta)
        xb = X_arr @ beta
        log_sigma_i = Z_arr @ gamma
        sigma_i = np.exp(log_sigma_i)

        z_unc = (y_arr - xb) / sigma_i
        z_left = (left_arr - xb) / sigma_i
        z_right = (right_arr - xb) / sigma_i

        ll = np.zeros_like(y_arr, dtype=float)
        if mask_unc.any():
            ll[mask_unc] = student_t.logpdf(z_unc[mask_unc], df=nu) - log_sigma_i[mask_unc]
        if is_left.any():
            ll[is_left] = student_t.logcdf(z_left[is_left], df=nu)
        if is_right.any():
            ll[is_right] = student_t.logsf(z_right[is_right], df=nu)
        return float(-np.sum(ll))

    res = minimize(
        nll,
        theta0,
        method=method,
        options={"maxiter": maxiter, "ftol": tol, "gtol": tol, "maxls": 50, "eps": 1e-10},
    )

    theta_hat = res.x
    beta_hat = theta_hat[:p]
    gamma_hat = theta_hat[p : p + q]
    eta_hat = float(theta_hat[p + q])
    nu_hat = float(2.0 + np.exp(eta_hat))

    # numeric Hessian for variance-covariance on (beta, gamma, eta)
    if use_numeric_hessian:
        try:
            H = _numerical_hessian(nll, theta_hat, eps=1e-6)
            vcov_theta = np.linalg.pinv(H)
        except Exception:
            vcov_theta = np.full((p + q + 1, p + q + 1), np.nan)
    else:
        if hasattr(res, "hess_inv"):
            hess_inv = res.hess_inv
            vcov_theta = np.asarray(hess_inv.todense()) if hasattr(hess_inv, "todense") else np.asarray(hess_inv)
        else:
            vcov_theta = np.full((p + q + 1, p + q + 1), np.nan)

    se_all = np.sqrt(np.clip(np.diag(vcov_theta), 0.0, np.inf))
    se_beta = se_all[:p]
    se_gamma = se_all[p : p + q]
    se_eta = float(se_all[p + q])
    # delta method: nu = 2 + exp(eta) => dnu/deta = exp(eta)
    se_nu = float(np.exp(eta_hat) * se_eta)

    nobs = int(y_arr.shape[0])
    n_left = int(is_left.sum())
    n_right = int(is_right.sum())
    n_unc = int(mask_unc.sum())
    ll = -nll(theta_hat)
    k_params = p + q + 1
    aic = 2 * k_params - 2 * ll
    bic = np.log(nobs) * k_params - 2 * ll

    return HeteroTobitTResult(
        beta=beta_hat,
        gamma=gamma_hat,
        nu=nu_hat,
        vcov=vcov_theta,
        se_beta=se_beta,
        se_gamma=se_gamma,
        se_nu=se_nu,
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

    theta_hat = res.x
    beta_hat = theta_hat[:p]
    gamma_hat = theta_hat[p : p + q]

    # numeric Hessian for variance-covariance
    if use_numeric_hessian:
        try:
            H = _numerical_hessian(nll, theta_hat, eps=1e-6)
            vcov_theta = np.linalg.pinv(H)
        except Exception:
            vcov_theta = np.full((p + q, p + q), np.nan)
    else:
        if hasattr(res, "hess_inv"):
            hess_inv = res.hess_inv
            vcov_theta = np.asarray(hess_inv.todense()) if hasattr(hess_inv, "todense") else np.asarray(hess_inv)
        else:
            vcov_theta = np.full((p + q, p + q), np.nan)

    se_all = np.sqrt(np.clip(np.diag(vcov_theta), 0.0, np.inf))
    se_beta = se_all[:p]
    se_gamma = se_all[p : p + q]

    nobs = int(y_arr.shape[0])
    n_left = int(is_left.sum())
    n_right = int(is_right.sum())
    n_unc = int(mask_unc.sum())
    ll = -nll(theta_hat)
    k_params = p + q
    aic = 2 * k_params - 2 * ll
    bic = np.log(nobs) * k_params - 2 * ll

    return HeteroTobitResult(
        beta=beta_hat,
        gamma=gamma_hat,
        vcov=vcov_theta,
        se_beta=se_beta,
        se_gamma=se_gamma,
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
