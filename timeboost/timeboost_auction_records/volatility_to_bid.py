import polars as pl

# calculate realized variance as sum of squared log 1-second returns

# calculate realized variance of variance as sum of quartic log 1-second returns

# regressor:bid (potentially normalized by TVL) at each round, can be censored due to the reserve price

# variables: E[\int_0^T sigma^2_t dt], Var(\int_0^T sigma^2_t dt), constant

# model: tobit