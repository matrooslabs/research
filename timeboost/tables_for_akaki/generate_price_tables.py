import polars as pl
from pathlib import Path
from typing import Iterator, Tuple


BINANCE_DIR = Path(__file__).parent / "binance_klines"
OUTPUT_DIR = Path(__file__).parent / "price_tables"


def detect_timestamp_divisor(example_ts: int) -> int:
	"""
	Detect the appropriate divisor to convert the provided timestamp to seconds.

	Binance aggregated klines (1s interval here) sometimes appear as:
	- milliseconds: ~1.7e12
	- microseconds: ~1.7e15

	We map by magnitude:
	- >= 1e15 -> microseconds -> divide by 1_000_000
	- >= 1e12 -> milliseconds -> divide by 1_000
	- else -> assume already seconds -> divide by 1
	"""
	if example_ts >= 1_000_000_000_000_000:
		return 1_000_000
	if example_ts >= 1_000_000_000_000:
		return 1_000
	return 1


def process_file_to_df(csv_path: Path, address: str) -> pl.DataFrame | None:
	"""
	Transform one Binance kline CSV into a Polars DataFrame with columns
	[timestamp, price, address]. The source has no header; we read columns 0 and 4.
	"""
	try:
		# Detect timestamp unit from the first row of column 0
		head_df = pl.read_csv(
			csv_path,
			has_header=False,
			columns=[0],
			n_rows=1,
			infer_schema_length=1,
			ignore_errors=True,
		)
		if head_df.height == 0:
			return None
		ts_raw = int(head_df.to_series()[0])
		divisor = detect_timestamp_divisor(ts_raw)

		# Read only needed columns: 0=open_time, 1=open
		df = pl.read_csv(
			csv_path,
			has_header=False,
			columns=[0, 1],
			new_columns=["timestamp_raw", "open_price"],
			ignore_errors=False,
		)
		out = (
			df.select([
				(pl.col("timestamp_raw").cast(pl.Int64) // divisor).alias("timestamp"),
				pl.col("open_price").cast(pl.Float64).round(8).alias("price"),
			])
			.with_columns(pl.lit(address).alias("address"))
		)
		return out
	except Exception:
		return None


def pair_from_filename(path: Path) -> str:
	# Example filename: BTCUSDT-1s-2025-04.csv -> pair is BTCUSDT
	name = path.name
	return name.split("-")[0]


def address_for_pair(pair: str) -> str | None:
	upper = pair.upper()
	if upper.startswith("ETH"):
		return "0x82aF49447D8a07e3bd95BD0d56f35241523fBab1"
	if upper.startswith("BTC"):
		return "0x2f2a2543B76A4166549F7aaB2e75Bef0aefC5B0f"
	if upper.startswith("USDC"):
		return "0xaf88d065e77c8cC2239327C5EDb3A432268e5831"
	return None


def build_consolidated_table() -> None:
	OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
	out_path = OUTPUT_DIR / "prices_consolidated.tsv"

	# Start fresh each run for deterministic output
	if out_path.exists():
		out_path.unlink()

	# Create empty file with header using Polars to ensure correct order/types
	pl.DataFrame({
		"timestamp": pl.Series([], dtype=pl.Int64),
		"price": pl.Series([], dtype=pl.Float64),
		"address": pl.Series([], dtype=pl.String),
	}).write_csv(out_path, separator="\t", include_header=True)

	# Append transformed chunks per source file
	for entry in sorted(BINANCE_DIR.glob("*.csv")):
		pair = pair_from_filename(entry)
		address = address_for_pair(pair)
		if address is None:
			continue
		df = process_file_to_df(entry, address)
		if df is None or df.height == 0:
			continue
		with out_path.open("a") as f:
			df.write_csv(f, separator="\t", include_header=False)


def main() -> None:
	if not BINANCE_DIR.exists():
		raise FileNotFoundError(f"Missing directory: {BINANCE_DIR}")
	build_consolidated_table()
	print(f"Wrote consolidated price table to {OUTPUT_DIR}")


if __name__ == "__main__":
	main()


