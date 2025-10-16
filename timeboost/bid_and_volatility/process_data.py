import os
import glob
from typing import List

import polars as pl

from bid_and_volatility.utils import (
    build_price_table,
    build_resolved_times,
    build_complete_bids,
    attach_realized_to_bids,
)


PRICE_GLOB = "data/price/ETHUSDT-1s-*.csv"
AUCTION_DIR = "data/auction_records"
BIDS_PATTERN = os.path.join(AUCTION_DIR, "bids_*.csv")
OUTPUT_SUFFIX = "_processed.csv"
RV_WINDOW = 60


def _list_bid_files() -> List[str]:
    files = sorted(glob.glob(BIDS_PATTERN))
    return [f for f in files if os.path.isfile(f)]


def process_all() -> None:
    # Build price table (log returns, realized variance/quarticity)
    price_lazy = pl.scan_csv(
        PRICE_GLOB,
        has_header=False,
        new_columns=["timestamp_us", "open"],
    )
    price = build_price_table(price_lazy, window=RV_WINDOW)

    # Load auction resolved times: provides round start/end timestamps
    resolved = pl.read_csv(os.path.join(AUCTION_DIR, "auction_resolved.csv"))
    resolved_times = build_resolved_times(resolved)

    # Determine full round span from resolved data
    min_round = int(resolved_times["round"].min())
    max_round = int(resolved_times["round"].max())

    # Process each bids_*.csv
    bid_files = _list_bid_files()
    if not bid_files:
        print("No bid files found.")
        return

    for path in bid_files:
        if os.path.basename(path) == "auction_resolved.csv":
            continue

        print(f"Processing {path} ...")
        bids = pl.read_csv(path)

        # Fill missing rounds with reserve price and generate timestamps
        complete = build_complete_bids(bids, resolved_times, min_round, max_round)

        # Attach realized variance and quarticity at round end
        complete_with_rv = attach_realized_to_bids(complete, price)

        # Save to CSV with suffix
        base, ext = os.path.splitext(path)
        out_path = f"{base}{OUTPUT_SUFFIX}"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        complete_with_rv.write_csv(out_path)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    process_all()


