#!/usr/bin/env bash
set -euo pipefail

# ── CONFIG ───────────────────────────────────────────────────────────────────
CLICKHOUSE_HOST=${CLICKHOUSE_HOST:-matroos.xyz}
CLICKHOUSE_PORT=${CLICKHOUSE_PORT:-9000}
CLICKHOUSE_USER=${CLICKHOUSE_USER:-brontes}
CLICKHOUSE_PASSWORD=${CLICKHOUSE_PASSWORD:-brontes}

# Destination
CLICKHOUSE_DB=${CLICKHOUSE_DB:-cex_kosunghun}
CLICKHOUSE_TABLE=${CLICKHOUSE_TABLE:-prices_1s}


TSV_FILE=${1:-price_tables/prices_consolidated.tsv}

if [[ ! -f "$TSV_FILE" ]]; then
	echo "Error: TSV file not found: $TSV_FILE" >&2
	exit 1
fi

echo "⟳ Loading '$TSV_FILE' → ${CLICKHOUSE_DB}.${CLICKHOUSE_TABLE}…"

# Ensure database exists
clickhouse client \
	--host     "$CLICKHOUSE_HOST" \
	--port     "$CLICKHOUSE_PORT" \
	--user     "$CLICKHOUSE_USER" \
	${CLICKHOUSE_PASSWORD:+--password="$CLICKHOUSE_PASSWORD"} \
	--query="CREATE DATABASE IF NOT EXISTS ${CLICKHOUSE_DB}"

# Create table if not exists: timestamp, price, address
clickhouse client \
	--host     "$CLICKHOUSE_HOST" \
	--port     "$CLICKHOUSE_PORT" \
	--user     "$CLICKHOUSE_USER" \
	${CLICKHOUSE_PASSWORD:+--password="$CLICKHOUSE_PASSWORD"} \
	--query="CREATE TABLE IF NOT EXISTS ${CLICKHOUSE_DB}.${CLICKHOUSE_TABLE} (
		timestamp UInt64,
		price Float64,
		address String
	) ENGINE = MergeTree
	ORDER BY (address, timestamp)"

# Import with header row
cat "$TSV_FILE" \
| clickhouse client \
	--host     "$CLICKHOUSE_HOST" \
	--port     "$CLICKHOUSE_PORT" \
	--user     "$CLICKHOUSE_USER" \
	${CLICKHOUSE_PASSWORD:+--password="$CLICKHOUSE_PASSWORD"} \
	--query="INSERT INTO ${CLICKHOUSE_DB}.${CLICKHOUSE_TABLE} FORMAT TabSeparatedWithNames"


echo "✔ Done."
