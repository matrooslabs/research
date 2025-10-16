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
import os
from bid_and_volatility.utils import (
    build_price_table,
    build_resolved_times,
    build_complete_bids,
    attach_realized_to_bids,
    prepare_regression_data,
)
import bid_and_volatility.tobit as tobit

##############################################################################
#                                prepare data                                #
##############################################################################

# load price and auction records
# load all price csvs into a single big table (lazy scan with glob)
price_lazy = pl.scan_csv(
    "data/price/ETHUSDT-1s-*.csv",
    has_header=False,
    new_columns=["timestamp_us", "open"],
)

bids_0x8c6f = pl.read_csv("data/auction_records/bids_0x8c6f.csv")
bids_0x95c0 = pl.read_csv("data/auction_records/bids_0x95c0.csv")
auction_resolved = pl.read_csv("data/auction_records/auction_resolved.csv")


RV_WINDOW = 60
price = build_price_table(price_lazy, window=RV_WINDOW)

resolved_times = build_resolved_times(auction_resolved)

MIN_ROUND = 205323
MAX_ROUND = 249961


from bid_and_volatility.utils import build_complete_bids as _build_complete_bids

# process each bidder separately
bids_0x8c6f_complete = _build_complete_bids(bids_0x8c6f, resolved_times, MIN_ROUND, MAX_ROUND)
bids_0x95c0_complete = _build_complete_bids(bids_0x95c0, resolved_times, MIN_ROUND, MAX_ROUND)

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
#                     run regression: tobit with RV only                     #
##############################################################################

# TODO: implement this

##############################################################################
#                    run regression: tobit with RV and RQ                    #
##############################################################################

# model: bid_amount ~ beta_var * realized_variance + beta_quar * realized_quarticity + beta_const + error

"""Fit Tobit: y in ETH, features unstandardized; left bound in ETH."""
RESERVE_WEI = 10**15
WEI_PER_ETH = 10**18
LEFT_ETH = RESERVE_WEI / WEI_PER_ETH


from bid_and_volatility.utils import prepare_regression_data as _prepare_regression_data


def _print_t_stats(tag: str, names: list[str], beta: np.ndarray, se_beta: np.ndarray) -> None:
    tvals = beta[: len(se_beta)] / se_beta
    header = f"t-statistics ({tag})"
    print(f"\n=== {header} ===")
    for i, name in enumerate(names[: len(tvals)]):
        print(f"{name:>30}  t={tvals[i]: .3f}")


def _print_loss_table(tag: str, rows: list[tuple[str, float, float]]) -> None:
    # rows: (model_name, mse, qlike)
    print(f"\n=== Loss comparison (MSE and QLIKE) ({tag}) ===")
    print(f"{'model':>20}  {'MSE':>14}  {'QLIKE':>14}")
    for name, mse, qlike in rows:
        print(f"{name:>20}  {mse:14.6e}  {qlike:14.6e}")


def run_bidder_regression(bids_complete: pl.DataFrame, bidder_tag: str) -> None:
    """
    Full pipeline per bidder: prepare data, fit Tobit, and print only t-stats
    and loss comparisons (MSE, QLIKE). Plots are skipped.
    """

    # prepare regression arrays (use raw, unstandardized features)
    y_eth, _X_ignored, feature_names, df_filtered = _prepare_regression_data(bids_complete, WEI_PER_ETH)
    rv_arr = df_filtered["realized_variance"].cast(pl.Float64).to_numpy()
    rq_arr = df_filtered["realized_quarticity"].cast(pl.Float64).to_numpy()
    X_raw = np.column_stack([rv_arr, rq_arr - rv_arr ** 2])

    # fit Tobit with left-censoring at reserve (in ETH)
    import bid_and_volatility.tobit as tobit  # local import to keep top clean

    res = tobit.fit_tobit(y_eth, X_raw, left=LEFT_ETH, use_numeric_hessian=True)

    # report t-stats only for homoskedastic mean model
    _print_t_stats(f"{bidder_tag} (Tobit, homoskedastic)", ["const", *feature_names], res.beta, res.se_beta)

    # predictions for loss metrics
    mu_hat_local = tobit.predict_latent(X_raw, res.beta, add_intercept=True)
    sigma2_local = float(res.sigma ** 2)

    # ================= Additional requested models and stats =================
    # 1) Tobit with only RV as input
    rv_only = X_raw[:, [0]]  # column 0 is RV
    res_rv_only = tobit.fit_tobit(y_eth, rv_only, left=LEFT_ETH, use_numeric_hessian=True)
    _print_t_stats(f"{bidder_tag} (RV only)", ["const", feature_names[0]], res_rv_only.beta, res_rv_only.se_beta)
    mu_rv = tobit.predict_latent(rv_only, res_rv_only.beta, add_intercept=True)
    sigma2_rv = float(res_rv_only.sigma ** 2)

    # 2) Heteroskedastic Tobit with both RV and (RQ - RV^2) for mean and variance
    #    - Mean: X_raw (two columns)
    #    - Variance: Z = [log(RV), log(RQ - RV^2)] with intercept for sigma
    #      We stabilize logs with a small epsilon. No standardization.
    iv_var_arr = rq_arr - rv_arr ** 2
    eps = 1e-20
    Z_raw = np.column_stack([
        np.log(np.maximum(rv_arr, eps)),
        np.log(np.maximum(iv_var_arr, eps)),
    ])
    # Try heteroskedastic normal Tobit
    loss_rows: list[tuple[str, float, float]] = []
    eps_var = 1e-20
    mse_homo = float(np.mean((y_eth - mu_hat_local) ** 2))
    qlike_homo = float(np.mean(np.log(np.maximum(sigma2_local, eps_var)) + (y_eth - mu_hat_local) ** 2 / np.maximum(sigma2_local, eps_var)))
    loss_rows.append(("Homoskedastic", mse_homo, qlike_homo))
    mse_rv = float(np.mean((y_eth - mu_rv) ** 2))
    qlike_rv = float(np.mean(np.log(np.maximum(sigma2_rv, eps_var)) + (y_eth - mu_rv) ** 2 / np.maximum(sigma2_rv, eps_var)))
    loss_rows.append(("RV only", mse_rv, qlike_rv))

    res_het = None
    try:
        res_het = tobit.fit_tobit_hetero(
            y_eth,
            X_raw,
            Z=Z_raw,
            left=LEFT_ETH,
            add_intercept=True,
            add_intercept_sigma=True,
            use_numeric_hessian=True,
        )
    except Exception as e:
        print(f"[warn] heteroskedastic Tobit failed: {e}")

    if res_het is not None and hasattr(res_het, "beta") and hasattr(res_het, "gamma"):
        _print_t_stats(f"{bidder_tag} (hetero mean)", ["const", *feature_names], res_het.beta, res_het.se_beta)
        t_gamma = res_het.gamma / res_het.se_gamma
        print(f"\n=== t-statistics (sigma model) ({bidder_tag}) ===")
        for nm, tv in zip(["const_sigma", *feature_names], t_gamma):
            print(f"{nm:>30}  t={tv: .3f}")

        mu_hat_het = tobit.predict_latent(X_raw, res_het.beta, add_intercept=True)
        sigma_i = np.exp(np.column_stack([
            np.ones(Z_raw.shape[0]), Z_raw
        ]) @ res_het.gamma)
        sigma2_i = sigma_i ** 2
        resid_het = y_eth - mu_hat_het
        mse_het = float(np.mean(resid_het ** 2))
        qlike_het = float(np.mean(np.log(np.maximum(sigma2_i, eps_var)) + resid_het ** 2 / np.maximum(sigma2_i, eps_var)))
        loss_rows.append(("Heteroskedastic", mse_het, qlike_het))
    else:
        print("[info] Skipping heteroskedastic (normal) section due to failed fit.")

    # Heteroskedastic Tobit with Student-t residuals
    res_het_t = None
    try:
        res_het_t = tobit.fit_tobit_hetero_t(
            y_eth,
            X_raw,
            Z=Z_raw,
            left=LEFT_ETH,
            add_intercept=True,
            add_intercept_sigma=True,
            use_numeric_hessian=True,
        )
    except Exception as e:
        print(f"[warn] heteroskedastic-t Tobit failed: {e}")

    if res_het_t is not None and hasattr(res_het_t, "beta") and hasattr(res_het_t, "gamma"):
        _print_t_stats(f"{bidder_tag} (hetero-t mean)", ["const", *feature_names], res_het_t.beta, res_het_t.se_beta)
        t_gamma_t = res_het_t.gamma / res_het_t.se_gamma
        print(f"\n=== t-statistics (sigma model, t) ({bidder_tag}) ===")
        for nm, tv in zip(["const_sigma", *feature_names], t_gamma_t):
            print(f"{nm:>30}  t={tv: .3f}")

        mu_hat_t = tobit.predict_latent(X_raw, res_het_t.beta, add_intercept=True)
        sigma_t_i = np.exp(np.column_stack([
            np.ones(Z_raw.shape[0]), Z_raw
        ]) @ res_het_t.gamma)
        resid_t = y_eth - mu_hat_t
        nu = float(res_het_t.nu)
        var_t_i = (nu / (nu - 2.0)) * (sigma_t_i ** 2)
        mse_t = float(np.mean(resid_t ** 2))
        qlike_t = float(np.mean(np.log(np.maximum(var_t_i, eps_var)) + resid_t ** 2 / np.maximum(var_t_i, eps_var)))
        loss_rows.append(("Heteroskedastic-t", mse_t, qlike_t))
    else:
        print("[info] Skipping heteroskedastic (Student-t) section due to failed fit.")

    _print_loss_table(bidder_tag, loss_rows)


# Run the regression/plotting pipeline for each bidder
# run_bidder_regression(bids_0x8c6f_complete, "0x8c6f")
# run_bidder_regression(bids_0x95c0_complete, "0x95c0")

##############################################################################
#            run regression: heteroskedastic tobit with RV and RQ            #
##############################################################################

# TODO: implement this