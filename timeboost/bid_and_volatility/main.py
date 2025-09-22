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
    print_coefficients,
    summarize_residuals,
    compute_r2s,
    corr_inputs_with_residuals,
    plot_residuals_round_hist,
    plot_residual_vs_rv,
    plot_yhat_vs_y,
)
import bid_and_volatility.tobit as tobit

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

"""Scale and fit Tobit: y in ETH, features standardized; left bound in ETH."""
RESERVE_WEI = 10**15
WEI_PER_ETH = 10**18
LEFT_ETH = RESERVE_WEI / WEI_PER_ETH


from bid_and_volatility.utils import prepare_regression_data as _prepare_regression_data


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
    y_eth, X_std, feature_names, df_filtered = _prepare_regression_data(bids_complete, WEI_PER_ETH)

    # fit Tobit with left-censoring at reserve (in ETH)
    import bid_and_volatility.tobit as tobit  # local import to keep top clean

    res = tobit.fit_tobit(y_eth, X_std, left=LEFT_ETH, use_numeric_hessian=True)

    # report (heteroskedastic block will follow)
    print_coefficients(res, ["const", *feature_names], f"{bidder_tag} (Tobit, homoskedastic)")

    # residuals and summaries, plus R^2 variants and input-residual correlations
    mu_hat_local = tobit.predict_latent(X_std, res.beta, add_intercept=True)
    residuals_local = y_eth - mu_hat_local
    summarize_residuals(residuals_local, f"{bidder_tag} (homo)")
    r2_p, r2_s = compute_r2s(y_eth, mu_hat_local)
    print(f"R^2 (Pearson)={r2_p:.4f}, R^2 (Spearman)={r2_s:.4f}")
    from bid_and_volatility.utils import print_corr_table, estimate_conditional_variance_ols, print_variance_table
    corrs = corr_inputs_with_residuals(X_std, residuals_local, feature_names)
    print_corr_table(corrs, f"{bidder_tag} (homo)")
    var_uncond = float(np.var(residuals_local, ddof=1))
    var_cond = estimate_conditional_variance_ols(X_std, residuals_local)
    print_variance_table(var_uncond, var_cond, f"{bidder_tag} (homo)")

    # plots for homoskedastic fit
    plot_residuals_round_hist(df_filtered, residuals_local, bidder_tag)
    plot_residual_vs_rv(df_filtered, residuals_local, bidder_tag)
    plot_yhat_vs_y(mu_hat_local, y_eth, bidder_tag)

    # ================= Additional requested models and stats =================
    # 1) Tobit with only RV as input
    rv_only = X_std[:, [0]]  # after standardization, column 0 is RV
    res_rv_only = tobit.fit_tobit(y_eth, rv_only, left=LEFT_ETH, use_numeric_hessian=True)
    print_coefficients(res_rv_only, ["const", feature_names[0]], f"{bidder_tag} (RV only)")
    mu_rv = tobit.predict_latent(rv_only, res_rv_only.beta, add_intercept=True)
    r2p_rv, r2s_rv = compute_r2s(y_eth, mu_rv)
    print(f"R^2 (Pearson)={r2p_rv:.4f}, R^2 (Spearman)={r2s_rv:.4f}")
    resid_rv = y_eth - mu_rv
    summarize_residuals(resid_rv, f"{bidder_tag} (RV only)")
    corrs_rv = corr_inputs_with_residuals(rv_only, resid_rv, [feature_names[0]])
    print_corr_table(corrs_rv, f"{bidder_tag} (RV only)")
    var_uncond_rv = float(np.var(resid_rv, ddof=1))
    var_cond_rv = estimate_conditional_variance_ols(rv_only, resid_rv)
    print_variance_table(var_uncond_rv, var_cond_rv, f"{bidder_tag} (RV only)")

    # 2) Heteroskedastic Tobit with both RV and (RQ - RV^2) for mean and variance
    #    - Mean: X_std (two columns)
    #    - Variance: Z = [log(RV), log(RQ - RV^2)] with intercept for sigma
    #      We stabilize logs with a small epsilon and standardize Z for numerics.
    rv_arr = df_filtered["realized_variance"].cast(pl.Float64).to_numpy()
    rq_arr = df_filtered["realized_quarticity"].cast(pl.Float64).to_numpy()
    iv_var_arr = rq_arr - rv_arr ** 2
    eps = 1e-20
    Z_raw = np.column_stack([
        np.log(np.maximum(rv_arr, eps)),
        np.log(np.maximum(iv_var_arr, eps)),
    ])
    mu_Z = Z_raw.mean(axis=0)
    sd_Z = Z_raw.std(axis=0, ddof=1)
    sd_Z[sd_Z == 0] = 1.0
    Z_std = (Z_raw - mu_Z) / sd_Z
    res_het = tobit.fit_tobit_hetero(
        y_eth,
        X_std,
        Z=Z_std,
        left=LEFT_ETH,
        add_intercept=True,
        add_intercept_sigma=True,
        use_numeric_hessian=True,
    )
    from bid_and_volatility.utils import print_gamma_table
    print_coefficients(res_het, ["const", *feature_names], f"{bidder_tag} (hetero mean)")
    print_gamma_table(res_het.gamma, res_het.se_gamma, ["const_sigma", *feature_names], f"{bidder_tag}")

    # predictions and stats for heteroskedastic model
    mu_hat_het = tobit.predict_latent(X_std, res_het.beta, add_intercept=True)
    r2p_het, r2s_het = compute_r2s(y_eth, mu_hat_het)
    print(f"R^2 (Pearson)={r2p_het:.4f}, R^2 (Spearman)={r2s_het:.4f}")
    resid_het = y_eth - mu_hat_het
    summarize_residuals(resid_het, f"{bidder_tag} (hetero)")
    corrs_het = corr_inputs_with_residuals(X_std, resid_het, feature_names)
    print_corr_table(corrs_het, f"{bidder_tag} (hetero)")
    var_uncond_het = float(np.var(resid_het, ddof=1))
    var_cond_het = estimate_conditional_variance_ols(X_std, resid_het)
    print_variance_table(var_uncond_het, var_cond_het, f"{bidder_tag} (hetero)")


# Run the regression/plotting pipeline for each bidder
run_bidder_regression(bids_0x8c6f_complete, "0x8c6f")
run_bidder_regression(bids_0x95c0_complete, "0x95c0")

##############################################################################
#            run regression: heteroskedastic tobit with RV and RQ            #
##############################################################################

# TODO: implement this