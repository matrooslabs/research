import polars as pl
from pathlib import Path
from datetime import datetime, timezone

# Paths
ROOT_DIR = Path(__file__).parent
ONCHAIN_DIR = ROOT_DIR / "onchain_data"
PRICE_DIR = ROOT_DIR / "price_tables"

SWAPS_GLOB = "timeboosted_swaps_*.csv"
AUCTION_RESOLVED_PATH = ONCHAIN_DIR / "auction_resolved.tsv"
PRICES_PATH = PRICE_DIR / "prices_consolidated.tsv"
OUTPUT_PATH = ROOT_DIR / "auction_resolved_with_markouts.csv"

# Supported token addresses (case-insensitive); ensure lowercase for joins
ETH_ADDRESS = "0x82aF49447D8a07e3bd95BD0d56f35241523fBab1"
BTC_ADDRESS = "0x2f2a2543B76A4166549F7aaB2e75Bef0aefC5B0f"
USDC_ADDRESS = "0xaf88d065e77c8cC2239327C5EDb3A432268e5831"
USDCe_ADDRESS = "0xFF970A61A04b1cA14834A43f5dE4533eBDDB5CC8"
USDT_ADDRESS = "0xFd086bC7CD5C481DCC9C85ebE478A1C0b69FCbb9"
SUPPORTED_ADDRESSES = {ETH_ADDRESS, BTC_ADDRESS, USDC_ADDRESS, USDCe_ADDRESS, USDT_ADDRESS}

# Lowercased variants for joins/filters
ETH_ADDR_L = ETH_ADDRESS.lower()
BTC_ADDR_L = BTC_ADDRESS.lower()
USDC_ADDR_L = USDC_ADDRESS.lower()
USDCe_ADDR_L = USDCe_ADDRESS.lower()
USDT_ADDR_L = USDT_ADDRESS.lower()
SUPPORTED_ADDRS_LOWER = {ETH_ADDR_L, BTC_ADDR_L, USDC_ADDR_L, USDCe_ADDR_L, USDT_ADDR_L}

# Markout horizon in seconds
MARKOUT_SECONDS = 5
WEI_PER_ETH = 10**18

# range of dates
START_DATETIME = "2025-06-01 00:00:00"
END_DATETIME = "2025-08-31 23:59:59"


# Convert to epoch seconds (UTC)
START_TS = int(datetime.strptime(START_DATETIME, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp())
END_TS = int(datetime.strptime(END_DATETIME, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp())


def get_swap_paths():
	"""Return all swap CSV files matching the glob in ONCHAIN_DIR."""
	files = sorted(ONCHAIN_DIR.glob(SWAPS_GLOB))
	if not files:
		raise FileNotFoundError(f"No swap files found matching: {ONCHAIN_DIR / SWAPS_GLOB}")
	return files


def read_swaps() -> pl.DataFrame:
	"""Read swaps CSVs and normalize types/addresses."""
	# Ensure there is at least one file; pattern read uses glob directly
	_ = get_swap_paths()
	pattern = str(ONCHAIN_DIR / SWAPS_GLOB)
	lf = pl.scan_csv(pattern, infer_schema_length=0)
	# Normalize and keep rows where either leg is supported (others will price as 0)
	lf = (
		lf.with_columns([
			pl.col("timestamp").cast(pl.Int64),
			pl.col("tx_hash").cast(pl.String),
			pl.col("token_in").str.to_lowercase().alias("token_in_addr"),
			pl.col("token_out").str.to_lowercase().alias("token_out_addr"),
			pl.col("gas_used").cast(pl.Int64),
			pl.col("effective_gas_price").cast(pl.Int64),
			pl.col("amount_in").cast(pl.Float64),
			pl.col("amount_out").cast(pl.Float64),
		])
		.filter(
			pl.col("token_in_addr").is_in(list(SUPPORTED_ADDRS_LOWER))
			|
			pl.col("token_out_addr").is_in(list(SUPPORTED_ADDRS_LOWER))
		)
	)
	return lf.collect()


def read_prices_for_supported() -> pl.DataFrame:
	"""Read consolidated prices and filter to supported token addresses only."""
	prices = pl.read_csv(
		PRICES_PATH,
		separator="\t",
		infer_schema_length=0,
		try_parse_dates=False,
	)
	prices = (
		prices.with_columns([
			pl.col("timestamp").cast(pl.Int64),
			pl.col("address").str.to_lowercase().alias("address"),
			pl.col("price").cast(pl.Float64),
		])
		.filter(pl.col("address").is_in(list(SUPPORTED_ADDRS_LOWER)))
	)
	# Augment prices: USDC.e mirrors USDC; USDT is always 1.0
	usdc_prices = prices.filter(pl.col("address") == USDC_ADDR_L)
	usdce_prices = usdc_prices.with_columns(pl.lit(USDCe_ADDR_L).alias("address")).select(["timestamp", "price", "address"])
	# Build USDT prices on the union of available timestamps
	all_ts = prices.select(pl.col("timestamp")).unique()
	usdt_prices = all_ts.with_columns([
		pl.lit(USDT_ADDR_L).alias("address"),
		pl.lit(1.0).alias("price"),
	]).select(["timestamp", "price", "address"])
	# Ensure base prices also use the same order
	prices = prices.select(["timestamp", "price", "address"])
	prices = pl.concat([prices, usdce_prices, usdt_prices])
	return prices


def compute_per_swap_markout(swaps: pl.DataFrame, prices: pl.DataFrame) -> pl.DataFrame:
	"""
	Compute per-swap markout using only t+5s prices:
	markout = price_out(t+5)*amount_out - price_in(t+5)*amount_in.
	Missing prices contribute 0.
	"""
	# Compute the t+5 timestamp
	swaps_plus = swaps.with_columns((pl.col("timestamp") + MARKOUT_SECONDS).alias("timestamp_plus"))
	# Join price_out at t+5
	p_tp_out = prices.rename({"price": "price_out_tp"})
	swaps_out = swaps_plus.join(
		p_tp_out,
		left_on=["token_out_addr", "timestamp_plus"],
		right_on=["address", "timestamp"],
		how="left",
	)
	# Join price_in at t+5
	p_tp_in = prices.rename({"price": "price_in_tp"})
	swaps_io = swaps_out.join(
		p_tp_in,
		left_on=["token_in_addr", "timestamp_plus"],
		right_on=["address", "timestamp"],
		how="left",
	)
	# Compute USD terms and markout
	swaps_markout = swaps_io.with_columns(
		(
			pl.col("amount_out") * pl.col("price_out_tp").fill_null(0.0)
			-
			pl.col("amount_in") * pl.col("price_in_tp").fill_null(0.0)
		).alias("swap_markout_usd")
	)
	return swaps_markout


def compute_per_tx_markout_minus_gas(swaps_with_markout: pl.DataFrame, prices: pl.DataFrame) -> pl.DataFrame:
	"""Aggregate per tx_hash: sum markouts and subtract one gas cost using ETH price at (tx_timestamp+5)."""
	# Aggregate per transaction (without gas)
	per_tx = (
		swaps_with_markout.group_by("tx_hash").agg([
			pl.col("timestamp").min().alias("tx_timestamp"),
			pl.col("swap_markout_usd").sum().alias("tx_markout_usd"),
			pl.col("gas_used").first().alias("gas_used_first"),
			pl.col("effective_gas_price").first().alias("effective_gas_price_first"),
		])
	)
	# Gas in ETH
	per_tx = per_tx.with_columns([
		(pl.col("gas_used_first") * pl.col("effective_gas_price_first") / WEI_PER_ETH).alias("gas_cost_eth"),
		(pl.col("tx_timestamp") + MARKOUT_SECONDS).alias("tx_timestamp_plus"),
	])
	# ETH price at tx_timestamp+5
	eth_prices = (
		prices.filter(pl.col("address") == ETH_ADDR_L)
		.select(["timestamp", "price"]).rename({"price": "eth_price_tp"})
	)
	per_tx = per_tx.join(eth_prices, left_on="tx_timestamp_plus", right_on="timestamp", how="left")
	# Subtract gas once per tx
	per_tx = per_tx.with_columns([
		(pl.col("gas_cost_eth") * pl.col("eth_price_tp").fill_null(0.0)).alias("gas_cost_usd"),
		(
			pl.col("tx_markout_usd")
			-
			pl.col("gas_cost_eth") * pl.col("eth_price_tp").fill_null(0.0)
		).alias("tx_markout_minus_gas_usd"),
	])
	return per_tx


def read_auction_resolved() -> pl.DataFrame:
	"""Read auction_resolved.tsv (TSV) with necessary columns."""
	ar = pl.read_csv(
		AUCTION_RESOLVED_PATH,
		separator="\t",
		infer_schema_length=0,
		try_parse_dates=False,
	)
	# Ensure integer timestamp columns
	ar = ar.with_columns([
		pl.col("round_start_timestamp").cast(pl.Int64),
		pl.col("round_end_timestamp").cast(pl.Int64),
	])
	# Keep rows whose round_end_timestamp is within [START_TS, END_TS]
	ar = ar.filter(
		(pl.col("round_end_timestamp") >= pl.lit(START_TS)) &
		(pl.col("round_end_timestamp") <= pl.lit(END_TS))
	)
	# Add row id to map back after join_asof
	ar = ar.with_row_index(name="__row_id__")
	return ar


def map_transactions_to_rounds(per_tx: pl.DataFrame, ar: pl.DataFrame) -> pl.DataFrame:
	"""
	For each transaction, find the auction_resolved row where
	round_start_timestamp <= tx_timestamp <= round_end_timestamp (inclusive).
	We use an asof join on start time, then filter by end time inclusively.
	Returns a DataFrame with columns [__row_id__, tx_markout_minus_gas_usd].
	"""
	# Sort for asof join
	per_tx_sorted = per_tx.sort("tx_timestamp")
	ar_sorted = ar.sort("round_start_timestamp")
	joined = per_tx_sorted.join_asof(
		ar_sorted,
		left_on="tx_timestamp",
		right_on="round_start_timestamp",
		strategy="backward",
	)
	# Keep only those where tx_timestamp <= round_end_timestamp (inclusive)
	mapped = joined.filter(pl.col("tx_timestamp") <= pl.col("round_end_timestamp"))
	return mapped.select(["__row_id__", "tx_markout_minus_gas_usd"])


def compute_agg_markout_per_round() -> pl.DataFrame:
	"""End-to-end computation of aggregate markout per auction_resolved row."""
	swaps = read_swaps()
	prices = read_prices_for_supported()
	# Per-swap markout (only t+5s)
	swaps_markout = compute_per_swap_markout(swaps, prices)
	# Per-tx aggregation minus gas using ETH price at tx+5
	per_tx = compute_per_tx_markout_minus_gas(swaps_markout, prices)
	# Map per-tx to auction rounds and sum per row
	ar = read_auction_resolved()
	mapped = map_transactions_to_rounds(per_tx, ar)
	summed = mapped.group_by("__row_id__").agg(pl.col("tx_markout_minus_gas_usd").sum().alias("agg_markout"))
	# Join back to auction_resolved and fill missing with 0.0
	out_full = ar.join(summed, on="__row_id__", how="left").with_columns(
		pl.col("agg_markout").fill_null(0.0)
	).drop("__row_id__")
	# Join ETH price at round_start_timestamp for USD conversions
	eth_at_start = (
		prices
		.filter(pl.col("address") == ETH_ADDR_L)
		.select(["timestamp", "price"]) 
		.rename({"price": "eth_price_at_round_start"})
	)
	out_full = out_full.join(
		eth_at_start,
		left_on="round_start_timestamp",
		right_on="timestamp",
		how="left",
	)
	# Compute bid/payment in USD
	out_full = out_full.with_columns([
		(
			pl.col("first_price_amount").cast(pl.Float64) / WEI_PER_ETH *
			pl.col("eth_price_at_round_start").fill_null(0.0)
		).alias("bid_in_usd"),
		(
			pl.col("price").cast(pl.Float64) / WEI_PER_ETH *
			pl.col("eth_price_at_round_start").fill_null(0.0)
		).alias("payment_in_usd"),
	])
	# Keep only requested columns
	out = out_full.select([
		"round",
		"first_price_bidder",
		"first_price_express_lane_controller",
		"first_price_amount",
		"price",
		"round_start_timestamp",
		"round_end_timestamp",
		"agg_markout",
		"bid_in_usd",
		"payment_in_usd",
	])
	return out


def main() -> None:
	# Ensure at least one swaps file exists
	_ = get_swap_paths()
	if not AUCTION_RESOLVED_PATH.exists():
		raise FileNotFoundError(f"Missing auction_resolved file: {AUCTION_RESOLVED_PATH}")
	if not PRICES_PATH.exists():
		raise FileNotFoundError(f"Missing prices file: {PRICES_PATH}")

	result = compute_agg_markout_per_round()
	# Write CSV at project root
	result.write_csv(OUTPUT_PATH)
	print(f"Wrote output: {OUTPUT_PATH}")


if __name__ == "__main__":
	main()
