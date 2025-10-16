# scripts/run_crch_tobit.R
# Heteroskedastic Tobit with Student's t errors in crch
# amount ~ beta0 + beta_mean * RV + beta_var * RQ + eps
# log(scale) = gamma0 + gamma1*log(RV) + gamma2*log(RQ)
# Censoring: left-censored at reserve. Supports constant or row-specific limits.

suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(ggplot2)
  library(crch)
  library(scales)
  library(cowplot)
})

# -----------------------------
# Paths & I/O
# -----------------------------
DATA_PATH <- file.path("data", "bids_0x8c6f_processed.csv")
OUT_DIR <- "outputs"
FIG_DIR <- file.path(OUT_DIR, "figures")
MODEL_PATH <- file.path(OUT_DIR, "crch_tobit_model.rds")
REPORT_PATH <- file.path(OUT_DIR, "model_summary_RV_RQmRV2.txt")

dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)
dir.create(FIG_DIR, showWarnings = FALSE, recursive = TRUE)

df0 <- readr::read_csv(DATA_PATH, show_col_types = FALSE)

# -----------------------------
# Basic checks
# -----------------------------
required_cols <- c(
  "round", "bidder", "amount", "round_start_timestamp", "round_end_timestamp",
  "is_censored", "realized_variance", "realized_quarticity"
)
miss <- setdiff(required_cols, names(df0))
if (length(miss) > 0) stop("Missing required columns: ", paste(miss, collapse = ", "))

# Drop rows with missing key fields
df <- df0 %>%
  filter(
    !is.na(amount),
    !is.na(realized_variance),
    !is.na(realized_quarticity),
    !is.na(is_censored)
  )

# -----------------------------
# Censoring setup
# -----------------------------
# Option A (constant left-censor limit): set LEFT_CENSOR manually if you have a known reserve.
# Example: LEFT_CENSOR <- 0.1
LEFT_CENSOR <- 1e15

# Option B (row-specific limits): if your data encodes censored observations with amount==reserve when is_censored==1,
# we can pass a per-row 'left' vector. This is robust when reserve differs by row.
# We'll auto-detect if censored rows share (nearly) a common value; if so, we'll use a constant limit.
eps <- 1e-12

if (all(df$is_censored %in% c(0, 1))) {
  if (is.na(LEFT_CENSOR)) {
    # Try to infer a constant reserve from censored rows:
    cens_vals <- unique(df$amount[df$is_censored == 1])
    if (length(cens_vals) == 1L) {
      LEFT_CENSOR <- cens_vals[1]
      message(sprintf("Detected constant left-censor at amount = %g", LEFT_CENSOR))
      left_vec <- rep(LEFT_CENSOR, nrow(df))
    } else {
      # Use row-specific censor points (amount where censored, -Inf otherwise)
      message("Using row-specific left-censor points from the data (varying reserves).")
      left_vec <- ifelse(df$is_censored == 1, df$amount, -Inf)
    }
  } else {
    message(sprintf("Using user-specified constant left-censor at %g", LEFT_CENSOR))
    left_vec <- rep(LEFT_CENSOR, nrow(df))
  }
} else {
  stop("`is_censored` must be 0/1.")
}

# Right-censoring not used here; set to +Inf
right_vec <- rep(Inf, nrow(df))

# -----------------------------
# Feature engineering
# -----------------------------
df <- df %>%
  mutate(
    RV            = pmax(as.numeric(realized_variance), eps),
    RQ            = pmax(as.numeric(realized_quarticity), eps),
    RQ_minus_RV2  = pmax(RQ - RV^2, eps),
    lRV           = log(RV),
    lRQ           = log(RQ),
    lRQ_minus_RV2 = log(RQ_minus_RV2)
  )

# Also materialize a human-readable column name as requested
df[["RQ-RV^2"]] <- df$RQ_minus_RV2

# ---- NUMERICAL SCALING (add this) ----
# Choose stable units (adjust if your data differs)
amount_unit <- 1e15 # typical magnitude ~ 1e15 wei
rv_unit <- 1e-6 # typical magnitude ~ 1e-6
rq_unit <- 1e-12 # typical magnitude ~ 1e-12

df <- df %>%
  mutate(
    amount_s = amount / amount_unit,
    RV_s = RV / rv_unit,
    RQ_s = RQ / rq_unit,
    RQmRV2_s = RQ_minus_RV2 / rq_unit,
    lRV_s = log(RV_s),
    lRQ_s = log(RQ_s),
    lRQmRV2_s = log(RQmRV2_s)
  )

# Scale the censor limits in the same way as amount
left_vec_s <- left_vec / amount_unit
right_vec_s <- right_vec / amount_unit

# -----------------------------
# Fit heteroskedastic Tobit (Student's t)
# -----------------------------
# Location/mean equation: amount ~ RV + RQ
# Scale equation: log(sigma) ~ lRV + lRQ
# Dist: "student" gives t-errors; df is handled internally by crch (estimated or default).
# IMPORTANT: pass left/right for censoring. crch supports vectorized cut points.
set.seed(42)
fit <- crch::crch(
  formula       = amount_s ~ RV_s + RQmRV2_s,
  scale.formula = ~ lRV_s + lRQmRV2_s,
  data          = df,
  left          = left_vec_s,
  right         = right_vec_s,
  dist          = "student"
)


saveRDS(fit, MODEL_PATH)

# Write a human-readable summary
sink(REPORT_PATH)
cat("===== Heteroskedastic Tobit (crch) with Student's t errors =====\n\n")
print(summary(fit))
cat("\n\n-- Notes --\n")
cat("* Location (mean) model: amount ~ RV + (RQ - RV^2)\n")
cat("* Scale model: log(sigma) ~ log(RV) + log(RQ - RV^2)\n")
cat("* Censoring: left = 1e15; right = +Inf\n")
cat("* Distribution: Student's t (robust to outliers / heavy tails)\n")
sink()

message("Model fitted. Summary saved to: ", REPORT_PATH)

# -----------------------------
# Predictions & residual-like diagnostics
# -----------------------------
df$pred_mu_s <- as.numeric(predict(fit, newdata = df, type = "location"))
df$pred_sig_s <- as.numeric(predict(fit, newdata = df, type = "scale"))
message(sprintf("Debug: calling response predict with left length=%d, right length=%d", length(left_vec_s), length(right_vec_s)))
df$pred_resp_s <- as.numeric(predict(
  fit,
  newdata = df,
  type = "response",
  left = left_vec_s,
  right = right_vec_s
))

# Back to original amount units for plots/interpretation
df$pred_mu <- df$pred_mu_s * amount_unit
df$pred_resp <- df$pred_resp_s * amount_unit

# Observed-only Pearson R^2 (use uncensored rows only)
obs_idx <- df$is_censored == 0
if (any(obs_idx)) {
  r_obs <- suppressWarnings(cor(df$amount[obs_idx], df$pred_resp[obs_idx], use = "complete.obs"))
  r2_obs <- as.numeric(r_obs)^2
} else {
  r2_obs <- NA_real_
}

# Append metric to report
sink(REPORT_PATH, append = TRUE)
cat(sprintf(
  "\nObserved-only Pearson R^2 (y vs yhat on uncensored): %s\n",
  ifelse(is.na(r2_obs), "NA", formatC(r2_obs, digits = 6, format = "f"))
))
sink()


df$std_latent_resid <- (df$amount_s - df$pred_mu_s) / df$pred_sig_s
# Censoring statistics (rows and rounds)
total_rows <- nrow(df)
num_cens_rows <- sum(df$is_censored == 1, na.rm = TRUE)
num_uncens_rows <- sum(df$is_censored == 0, na.rm = TRUE)
total_rounds <- length(unique(df$round))
num_cens_rounds <- length(unique(df$round[df$is_censored == 1]))
num_uncens_rounds <- length(unique(df$round[df$is_censored == 0]))

sink(REPORT_PATH, append = TRUE)
cat(sprintf("Rows: total=%d, censored=%d, uncensored=%d\n", total_rows, num_cens_rows, num_uncens_rows))
cat(sprintf("Rounds: total=%d, censored=%d, uncensored=%d\n", total_rounds, num_cens_rounds, num_uncens_rounds))
sink()

# -----------------------------
# Plots
# -----------------------------
# 1) Observed vs Predicted (response scale), color by censoring
p1 <- ggplot(df, aes(x = pred_resp, y = amount, color = factor(is_censored))) +
  geom_point(alpha = 0.65) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
  labs(
    title = "Observed vs Predicted (response scale, Tobit)",
    x = "Predicted response (original units)",
    y = "Observed amount (original units)",
    color = "Censored"
  ) +
  theme_minimal()



# 2) Estimated scale (sigma) vs log RV (and color by log (RQ - RV^2))
p2 <- ggplot(df, aes(x = lRV, y = pred_sig_s, color = lRQ_minus_RV2)) +
  geom_point(alpha = 0.65) +
  labs(
    title = "Estimated scale (σ) over covariates",
    x = "log(RV)",
    y = "σ (predicted)",
    color = "log(RQ - RV^2)"
  ) +
  theme_minimal()

# 3) Latent standardized residuals distribution
p3 <- ggplot(df, aes(x = std_latent_resid)) +
  geom_histogram(bins = 60) +
  labs(
    title = "Std. latent residuals (quick check)",
    x = "(amount - μ̂)/σ̂",
    y = "Count"
  ) +
  theme_minimal()

# 4) Partial dependence-style lines: vary RV across quantiles, hold RQ at median
mk_grid <- function(n = 50, df_ref = df) {
  rv_seq_s <- seq(quantile(df_ref$RV_s, 0.01), quantile(df_ref$RV_s, 0.99), length.out = n)
  rqmr_med_s <- median(df_ref$RQmRV2_s)
  data.frame(
    RV_s      = rv_seq_s,
    RQmRV2_s  = rqmr_med_s,
    lRV_s     = log(rv_seq_s),
    lRQmRV2_s = log(rqmr_med_s)
  )
}

grid <- mk_grid()
grid$mu_line <- as.numeric(predict(fit, newdata = grid, type = "location"))
grid$sigma_line <- as.numeric(predict(fit, newdata = grid, type = "scale"))
grid$resp_line <- as.numeric(predict(fit, newdata = grid, type = "location"))


p4a <- ggplot(grid, aes(x = RV_s, y = mu_line)) +
  geom_line() +
  scale_x_continuous(labels = label_number(scale_cut = cut_short_scale())) +
  labs(
    title = "Partial effect on latent mean μ (hold RQ-RV^2 at median)",
    x = "RV (scaled)", y = "μ̂"
  ) +
  theme_minimal()

p4b <- ggplot(grid, aes(x = RV_s, y = sigma_line)) +
  geom_line() +
  scale_x_continuous(labels = label_number(scale_cut = cut_short_scale())) +
  labs(
    title = "Partial effect on scale σ (hold RQ-RV^2 at median)",
    x = "RV (scaled)", y = "σ̂"
  ) +
  theme_minimal()

# Save plots
ggsave(file.path(FIG_DIR, "obs_vs_pred.png"), p1, width = 7.5, height = 5.2, dpi = 150)
ggsave(file.path(FIG_DIR, "sigma_vs_logRV.png"), p2, width = 7.5, height = 5.2, dpi = 150)
ggsave(file.path(FIG_DIR, "std_latent_resid_hist.png"), p3, width = 7.5, height = 5.2, dpi = 150)
ggsave(file.path(FIG_DIR, "partial_mu_vs_RV.png"), p4a, width = 7.5, height = 5.2, dpi = 150)
ggsave(file.path(FIG_DIR, "partial_sigma_vs_RV.png"), p4b, width = 7.5, height = 5.2, dpi = 150)

# Combined dashboard (optional)
dash <- cowplot::plot_grid(p1, p2, p3, ncol = 1, rel_heights = c(1.1, 1, 1))
ggsave(file.path(FIG_DIR, "dashboard_basic.png"), dash, width = 8, height = 12, dpi = 150)

message("All plots saved to ./outputs/figures")
