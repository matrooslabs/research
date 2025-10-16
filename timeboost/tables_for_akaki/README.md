## Tables for Akaki — Timeboost Markout Pipeline

### Purpose
Compute per-auction aggregate markouts for Arbitrum Timeboost rounds by joining on-chain swaps with price data. Outputs a single CSV with per-round PnL-like metrics and convenience USD values for bids and payments.

### Outputs
- `auction_resolved_with_markouts.csv`: one row per auction round with:
  - round metadata (`round`, `first_price_bidder`, `first_price_express_lane_controller`, timestamps)
  - `agg_markout`: sum of per-tx markouts minus gas
  - `bid_in_usd`: `first_price_amount` converted to USD at round start
  - `payment_in_usd`: `price` converted to USD at round start

### Data Requirements (to reproduce)
Place these files in the given locations:
- `onchain_data/auction_resolved.tsv` (TSV): auction rounds with at least
  - `round`, `first_price_bidder`, `first_price_express_lane_controller`, `first_price_amount`, `price`, `round_start_timestamp`, `round_end_timestamp`
- `onchain_data/timeboosted_swaps_*.csv` (CSV, one or many): Timeboosted swaps with at least
  - `timestamp`, `tx_hash`, `token_in`, `token_out`, `gas_used`, `effective_gas_price`, `amount_in`, `amount_out`
- `price_tables/prices_consolidated.tsv` (TSV): per-second prices with columns
  - `timestamp`, `address`, `price`

Supported token addresses (priced): ETH, BTC, USDC, USDC.e, USDT (USDT assumed 1.0; USDC.e mirrors USDC). Update constants in `compute_agg_markouts.py` if needed.

### How It Works
1. Read and union all swaps `onchain_data/timeboosted_swaps_*.csv` (lazy scan).
2. Filter to supported token addresses and compute per-swap markout using prices at t+5 seconds:
   - `swap_markout_usd = price_out(t+5)*amount_out - price_in(t+5)*amount_in`
3. Aggregate per transaction and subtract one gas cost using ETH price at (tx_timestamp+5).
4. Map each transaction to the auction round where `round_start_timestamp <= tx_timestamp <= round_end_timestamp`.
5. Sum per-round to get `agg_markout` and join back to `auction_resolved.tsv`.
6. Convert `first_price_amount` and `price` to USD at `round_start_timestamp` → `bid_in_usd`, `payment_in_usd`.
7. Filter rounds by end time within `[START_DATETIME, END_DATETIME]`.

### Configure Date Range
Edit in `compute_agg_markouts.py`:
```
START_DATETIME = "YYYY-MM-DD 00:00:00"
END_DATETIME   = "YYYY-MM-DD 23:59:59"
```
Times are interpreted in UTC and converted to epoch seconds.

### Run
Using Poetry:
```
poetry install
poetry run python compute_agg_markouts.py
```
The result is written to `auction_resolved_with_markouts.csv`.

### Notes
- The pipeline is memory-conscious by lazily scanning swap CSVs.
- Missing prices are treated as 0 in markout computations.
- ETH price lookups for gas and USD conversions use exact second matches. Ensure your consolidated prices cover all required timestamps.


