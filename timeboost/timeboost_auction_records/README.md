# timeboost_auction_records

This module fetches Timeboost Express Lane `AuctionResolved` events from Arbitrum, decodes them, writes an append‑only TSV on disk, and optionally loads them into ClickHouse.

## Overview
- Fetch logs with `eth_getLogs` in parallel across block ranges
- Decode the `AuctionResolved(bool,uint64,address,address,uint256,uint256,uint64,uint64)` event
- Write decoded rows to a tab‑separated file `auction_resolved.tsv` (with header)
- Optionally deduplicate by `round` across runs
- Load into ClickHouse via `load_auction_resolved.sh`

## Prerequisites
- Python 3.10+
- Dependencies (install in this folder):
  - `pip install -r requirements.txt` (or use the project’s `pyproject.toml`/`poetry`)
- A working Arbitrum RPC endpoint (HTTP)
  - Export `ARBITRUM_RPC_URL` or pass `--rpc-url`
- Optional: ClickHouse client `clickhouse` in `$PATH` for loading into DB

## Event and ABI
- Contract address (default): `0x5fcb496a31b7AE91e7c9078Ec662bd7A55cd3079`
- Topic0 (keccak): `0x7f5bdabbd27a8fc572781b177055488d7c6729a2bade4f57da9d200f31c15d47`
- ABI: `ABIs/ExpressLaneAuction.json` (not required by the fetcher; used for reference)

## How it works
1) The script partitions the requested block window into chunks (default 10,000 blocks).
2) Multiple async workers query `eth_getLogs` concurrently with a global RPS limit.
3) Each log is normalized and then decoded into the following typed fields:
   - `is_multi_bid_auction` (bool)
   - `round` (uint64)
   - `first_price_bidder` (indexed `address`)
   - `first_price_express_lane_controller` (indexed `address`)
   - `first_price_amount` (uint256)
   - `price` (uint256)
   - `round_start_timestamp` (uint64, seconds)
   - `round_end_timestamp` (uint64, seconds)
   Plus metadata: `block_number`, `log_index`, `transaction_hash`, `contract_address`.
4) Rows are batched and flushed to a TSV file with a header. On startup, the script can optionally rewrite the TSV to unique `round` values to avoid duplicates across runs.

## Storage format
- Output path (default): `auction_resolved.tsv`
- Format: Tab‑Separated With Names (header row)
- Column order (exact):
  1. `block_number`
  2. `log_index`
  3. `transaction_hash`
  4. `contract_address`
  5. `is_multi_bid_auction`
  6. `round`
  7. `first_price_bidder`
  8. `first_price_express_lane_controller`
  9. `first_price_amount`
  10. `price`
  11. `round_start_timestamp`
  12. `round_end_timestamp`

## Usage
Fetch logs from a date to latest:
```bash
python get_auction_resolved_logs.py \
  --from-datetime "2025-04-01 00:00:00Z" \
  --out auction_resolved.tsv \
  --workers 16 \
  --chunk-size 10000 \
  --rps 64
```

Fetch logs by explicit block range:
```bash
python get_auction_resolved_logs.py \
  --from-block 199000000 \
  --to-block 200000000 \
  --out auction_resolved.tsv
```

Override RPC URL and contract address if needed:
```bash
python get_auction_resolved_logs.py \
  --rpc-url "$ARBITRUM_RPC_URL" \
  --address 0x5fcb496a31b7AE91e7c9078Ec662bd7A55cd3079
```

Disable on‑startup deduplication (keeps file as‑is):
```bash
python get_auction_resolved_logs.py --no-dedupe-rounds-in-file
```

Key flags:
- `--from-datetime` or `--from-block`: defines start
- `--to-block`: end (default latest)
- `--chunk-size`: blocks per worker job (default 10k)
- `--workers`: number of parallel workers (default 16)
- `--rps`: global requests per second limit (default 64)
- `--out`: TSV output path (default `auction_resolved.tsv`)
- `--dedupe-rounds-in-file/--no-dedupe-rounds-in-file`: dedupe control

## ClickHouse loading
A helper script is provided to load the TSV into ClickHouse:

```bash
./load_auction_resolved.sh auction_resolved.tsv
```

Environment variables (defaults shown):
```bash
CLICKHOUSE_HOST=matroos.xyz
CLICKHOUSE_PORT=9000
CLICKHOUSE_USER=brontes
CLICKHOUSE_PASSWORD=brontes
CLICKHOUSE_DB=timeboost
CLICKHOUSE_TABLE=auction
```

The loader will:
- Ensure the database exists
- Truncate the target table
- Stream‑insert the TSV with header using `FORMAT TabSeparatedWithNames`

### Suggested ClickHouse schema
Make sure your table matches the TSV columns and types. Example:
```sql
CREATE TABLE IF NOT EXISTS timeboost.auction (
  block_number UInt64,
  log_index UInt32,
  transaction_hash String,
  contract_address String,
  is_multi_bid_auction Nullable(UInt8),
  round UInt64,
  first_price_bidder String,
  first_price_express_lane_controller String,
  first_price_amount UInt256,
  price UInt256,
  round_start_timestamp UInt64,
  round_end_timestamp UInt64
) ENGINE = ReplacingMergeTree()
ORDER BY (round);
```

Notes:
- `UInt256` requires recent ClickHouse versions; if unavailable, store as `Decimal(76,0)` or `String`.
- `is_multi_bid_auction` is `Nullable` because some logs may not decode cleanly.

## Troubleshooting
- Connection errors: verify `ARBITRUM_RPC_URL` and your network connectivity
- Empty results: check block range, address, and topic0
- Duplicates: keep `--dedupe-rounds-in-file` enabled or run it once and disable for speed
- Performance: tune `--workers`, `--rps`, and `--chunk-size` based on RPC limits

## License
MIT (see repository root if provided).
