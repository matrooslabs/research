suppressPackageStartupMessages({
    library(readr)
    library(dplyr)
    library(crch)
})

# Output file for results
OUT_DIR <- "outputs"
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)
REPORT_PATH <- file.path(OUT_DIR, "lr_test_summary.txt")

# Likelihood-ratio test helper
lr_test <- function(fit_reduced, fit_full) {
    ll_r <- as.numeric(logLik(fit_reduced))
    ll_f <- as.numeric(logLik(fit_full))
    df_r <- attr(logLik(fit_reduced), "df")
    df_f <- attr(logLik(fit_full), "df")
    stat <- 2 * (ll_f - ll_r)
    df_diff <- df_f - df_r
    pval <- pchisq(stat, df = df_diff, lower.tail = FALSE)
    list(ll_reduced = ll_r, ll_full = ll_f, df_reduced = df_r, df_full = df_f, df_diff = df_diff, lr_stat = stat, p_value = pval)
}

# Fit both models and run LR test for a single bidder file
run_for_bidder <- function(data_path, bidder_label) {
    eps <- 1e-12
    LEFT_CENSOR <- 1e15

    df0 <- readr::read_csv(data_path, show_col_types = FALSE)

    required_cols <- c("amount", "is_censored", "realized_variance", "realized_quarticity")
    miss <- setdiff(required_cols, names(df0))
    if (length(miss) > 0) stop(sprintf("%s missing required columns: %s", bidder_label, paste(miss, collapse = ", ")))

    df <- df0 %>%
        filter(!is.na(amount), !is.na(is_censored), !is.na(realized_variance), !is.na(realized_quarticity))

    if (!all(df$is_censored %in% c(0, 1))) stop("`is_censored` must be 0/1")

    # Feature engineering (original scale)
    df <- df %>% mutate(
        RV = pmax(as.numeric(realized_variance), eps),
        RQ = pmax(as.numeric(realized_quarticity), eps),
        RQmRV2 = pmax(RQ - RV^2, eps),
        lRV = log(RV),
        lRQmRV2 = log(RQmRV2)
    )

    # Numerical scaling
    amount_unit <- 1e15
    rv_unit <- 1e-6
    rq_unit <- 1e-12

    df <- df %>% mutate(
        amount_s = amount / amount_unit,
        RV_s = RV / rv_unit,
        RQmRV2_s = RQmRV2 / rq_unit,
        lRV_s = log(RV_s),
        lRQmRV2_s = log(RQmRV2_s)
    )

    left_vec_s <- rep(LEFT_CENSOR, nrow(df)) / amount_unit
    right_vec_s <- rep(Inf, nrow(df)) / amount_unit

    set.seed(42)
    # Reduced model: RV only (location and scale)
    fit_r <- crch::crch(
        formula = amount_s ~ RV_s,
        scale.formula = ~lRV_s,
        data = df,
        left = left_vec_s,
        right = right_vec_s,
        dist = "student"
    )

    # Full model: RV + (RQ - RV^2) with corresponding scale terms
    fit_f <- crch::crch(
        formula = amount_s ~ RV_s + RQmRV2_s,
        scale.formula = ~ lRV_s + lRQmRV2_s,
        data = df,
        left = left_vec_s,
        right = right_vec_s,
        dist = "student"
    )

    test <- lr_test(fit_r, fit_f)

    # Info criteria
    aic_r <- AIC(fit_r)
    bic_r <- BIC(fit_r)
    aic_f <- AIC(fit_f)
    bic_f <- BIC(fit_f)
    d_aic <- aic_f - aic_r
    d_bic <- bic_f - bic_r

    # McFadden R^2 vs intercept-only null (location ~ 1, scale ~ 1)
    fit_null <- crch::crch(
        formula = amount_s ~ 1,
        scale.formula = ~1,
        data = df,
        left = left_vec_s,
        right = right_vec_s,
        dist = "student"
    )
    ll_null <- as.numeric(logLik(fit_null))
    r2_mcf_r <- if (is.finite(ll_null) && ll_null != 0) 1 - as.numeric(logLik(fit_r)) / ll_null else NA_real_
    r2_mcf_f <- if (is.finite(ll_null) && ll_null != 0) 1 - as.numeric(logLik(fit_f)) / ll_null else NA_real_

    cat("\n==============================\n")
    cat(sprintf("Bidder: %s\n", bidder_label))
    cat("Models compared: RV_only (reduced) vs RV_RQmRV2 (full)\n")
    cat(sprintf("LogLik (reduced): %.3f  | df: %d\n", test$ll_reduced, test$df_reduced))
    cat(sprintf("LogLik (full):    %.3f  | df: %d\n", test$ll_full, test$df_full))
    cat(sprintf("LR stat: %.3f  | df diff: %d  | p-value: %.4g\n", test$lr_stat, test$df_diff, test$p_value))
    cat(sprintf("AIC (reduced): %.3f  | BIC (reduced): %.3f\n", aic_r, bic_r))
    cat(sprintf("AIC (full):    %.3f  | BIC (full):    %.3f\n", aic_f, bic_f))
    cat(sprintf("Delta AIC (full - reduced): %.3f\n", d_aic))
    cat(sprintf("Delta BIC (full - reduced): %.3f\n", d_bic))
    cat(sprintf("McFadden R^2 (reduced vs null): %.6f\n", r2_mcf_r))
    cat(sprintf("McFadden R^2 (full vs null):    %.6f\n", r2_mcf_f))

    # Also append the same block to a summary text file
    cat("\n==============================\n", file = REPORT_PATH, append = TRUE)
    cat(sprintf("Bidder: %s\n", bidder_label), file = REPORT_PATH, append = TRUE)
    cat("Models compared: RV_only (reduced) vs RV_RQmRV2 (full)\n", file = REPORT_PATH, append = TRUE)
    cat(sprintf("LogLik (reduced): %.3f  | df: %d\n", test$ll_reduced, test$df_reduced), file = REPORT_PATH, append = TRUE)
    cat(sprintf("LogLik (full):    %.3f  | df: %d\n", test$ll_full, test$df_full), file = REPORT_PATH, append = TRUE)
    cat(sprintf("LR stat: %.3f  | df diff: %d  | p-value: %.4g\n", test$lr_stat, test$df_diff, test$p_value), file = REPORT_PATH, append = TRUE)
    cat(sprintf("AIC (reduced): %.3f  | BIC (reduced): %.3f\n", aic_r, bic_r), file = REPORT_PATH, append = TRUE)
    cat(sprintf("AIC (full):    %.3f  | BIC (full):    %.3f\n", aic_f, bic_f), file = REPORT_PATH, append = TRUE)
    cat(sprintf("Delta AIC (full - reduced): %.3f\n", d_aic), file = REPORT_PATH, append = TRUE)
    cat(sprintf("Delta BIC (full - reduced): %.3f\n", d_bic), file = REPORT_PATH, append = TRUE)
    cat(sprintf("McFadden R^2 (reduced vs null): %.6f\n", r2_mcf_r), file = REPORT_PATH, append = TRUE)
    cat(sprintf("McFadden R^2 (full vs null):    %.6f\n", r2_mcf_f), file = REPORT_PATH, append = TRUE)
    invisible(test)
}

# Entrypoint: two bidders
run <- function() {
    base <- file.path("data")
    bidders <- c("0x8c6f", "0x95c0")
    # Initialize the summary file with a header
    if (file.exists(REPORT_PATH)) file.remove(REPORT_PATH)
    cat(sprintf("LR test summary generated on %s\n", format(Sys.time(), "%Y-%m-%d %H:%M:%S")), file = REPORT_PATH)
    for (b in bidders) {
        path <- file.path(base, sprintf("bids_%s_processed.csv", b))
        if (!file.exists(path)) {
            warning(sprintf("Data file not found for %s: %s", b, path))
            next
        }
        run_for_bidder(path, b)
    }
}

if (sys.nframe() == 0) {
    run()
}
