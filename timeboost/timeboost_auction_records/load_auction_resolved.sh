#!/usr/bin/env bash
set -euo pipefail

# ── CONFIG ───────────────────────────────────────────────────────────────────
CLICKHOUSE_HOST=${CLICKHOUSE_HOST:-matroos.xyz}
CLICKHOUSE_PORT=${CLICKHOUSE_PORT:-9000}
CLICKHOUSE_USER=${CLICKHOUSE_USER:-brontes}
CLICKHOUSE_PASSWORD=${CLICKHOUSE_PASSWORD:-brontes}

# Destination
CLICKHOUSE_DB=${CLICKHOUSE_DB:-timeboost}
CLICKHOUSE_TABLE=${CLICKHOUSE_TABLE:-auction}

# TSV file to import (default: auction_resolved.tsv)
TSV_FILE=${1:-auction_resolved.tsv}

if [[ ! -f "$TSV_FILE" ]]; then
  echo "Error: TSV file not found: $TSV_FILE" >&2
  exit 1
fi

echo "⟳ Loading '$TSV_FILE' → ${CLICKHOUSE_DB}.${CLICKHOUSE_TABLE}…"

# Ensure database exists (table should already exist per schema you created)
clickhouse client \
  --host     "$CLICKHOUSE_HOST" \
  --port     "$CLICKHOUSE_PORT" \
  --user     "$CLICKHOUSE_USER" \
  ${CLICKHOUSE_PASSWORD:+--password="$CLICKHOUSE_PASSWORD"} \
  --query="CREATE DATABASE IF NOT EXISTS ${CLICKHOUSE_DB}"

# Import with header row
cat "$TSV_FILE" \
| clickhouse client \
    --host     "$CLICKHOUSE_HOST" \
    --port     "$CLICKHOUSE_PORT" \
    --user     "$CLICKHOUSE_USER" \
    ${CLICKHOUSE_PASSWORD:+--password="$CLICKHOUSE_PASSWORD"} \
    --query="INSERT INTO ${CLICKHOUSE_DB}.${CLICKHOUSE_TABLE} FORMAT TabSeparatedWithNames"

echo "✔ Done."


