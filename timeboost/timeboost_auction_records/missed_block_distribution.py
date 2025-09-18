import os
from typing import Dict, List, Tuple
from bisect import bisect_right

try:
    from dotenv import load_dotenv  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    load_dotenv = None  # type: ignore

import polars as pl
from web3 import Web3
import matplotlib.pyplot as plt


def initialize_web3() -> Web3:
    if load_dotenv is not None:
        try:
            load_dotenv()
        except Exception:
            pass

    rpc_url = os.environ.get("ARBITRUM_RPC_URL")
    if not rpc_url:
        raise RuntimeError(
            "Environment variable ARBITRUM_RPC_URL is not set. "
            "Export it or place it in a .env file."
        )

    w3 = Web3(Web3.HTTPProvider(rpc_url, request_kwargs={"timeout": 60}))
    if not w3.is_connected():
        raise RuntimeError("Failed to connect to ARBITRUM_RPC_URL provider")
    return w3


def read_auction_rounds(tsv_path: str) -> Tuple[List[int], List[int], int]:
    df = pl.read_csv(
        tsv_path,
        separator="\t",
        columns=["round_start_timestamp", "round_end_timestamp"],
        schema_overrides={"round_start_timestamp": pl.Int64, "round_end_timestamp": pl.Int64},
    ).sort("round_start_timestamp")

    start_list: List[int] = df["round_start_timestamp"].to_list()
    end_list: List[int] = df["round_end_timestamp"].to_list()

    if not start_list or not end_list:
        raise RuntimeError("Auction rounds file is empty or missing required columns")

    max_duration = max(e - s for s, e in zip(start_list, end_list))
    return start_list, end_list, int(max_duration)


def read_missing_blocks(csv_path: str) -> List[Tuple[int, int, int]]:
    df = pl.read_csv(
        csv_path,
        columns=["missing_start", "missing_end", "missing_count"],
        schema_overrides={"missing_start": pl.Int64, "missing_end": pl.Int64, "missing_count": pl.Int64},
    )
    rows: List[Tuple[int, int, int]] = list(
        zip(df["missing_start"].to_list(), df["missing_end"].to_list(), df["missing_count"].to_list())
    )
    return rows


def get_block_timestamp(w3: Web3, block_number: int, cache: Dict[int, int]) -> int:
    if block_number in cache:
        return cache[block_number]
    block = w3.eth.get_block(block_number)
    ts = int(block["timestamp"])  # type: ignore[index]
    cache[block_number] = ts
    return ts


def find_round_index_for_time(start_list: List[int], end_list: List[int], t: int) -> int:
    # Index of last round with start <= t
    idx = bisect_right(start_list, t) - 1
    if idx >= 0 and t <= end_list[idx]:  # inclusive end
        return idx
    return -1


def compute_distribution_simple(
    w3: Web3,
    missing_rows: List[Tuple[int, int, int]],
    start_list: List[int],
    end_list: List[int],
    max_duration: int,
) -> Tuple[List[int], int, int]:
    # Histogram over integer relative positions [0, max_duration]
    histogram: List[int] = [0 for _ in range(max_duration + 1)]
    total_missing_blocks: int = 0
    total_matched_blocks: int = 0

    ts_cache: Dict[int, int] = {}

    for block_start, block_end, missing_count in missing_rows:
        # Fetch timestamps for the range endpoints
        t_start = get_block_timestamp(w3, block_start, ts_cache)
        t_end = get_block_timestamp(w3, block_end, ts_cache)
        if t_end < t_start:
            t_start, t_end = t_end, t_start

        total_missing_blocks += int(missing_count)

        if missing_count <= 0:
            continue

        if missing_count == 1:
            ts = t_start
            ridx = find_round_index_for_time(start_list, end_list, ts)
            if ridx != -1:
                pos = ts - start_list[ridx]
                if 0 <= pos <= max_duration:
                    histogram[pos] += 1
                    total_matched_blocks += 1
            continue

        # Interpolate timestamps for each missing block inclusively across [t_start, t_end]
        span = t_end - t_start
        for j in range(missing_count):
            # Inclusive endpoints mapping: j=0 -> t_start, j=missing_count-1 -> t_end
            # Use round() to get integer seconds
            if span <= 0:
                ts = t_start
            else:
                ts = int(round(t_start + (span * (j / float(missing_count - 1)))))

            ridx = find_round_index_for_time(start_list, end_list, ts)
            if ridx == -1:
                continue
            pos = ts - start_list[ridx]
            if 0 <= pos <= max_duration:
                histogram[pos] += 1
                total_matched_blocks += 1

    return histogram, total_missing_blocks, total_matched_blocks


def summarize_distribution(histogram: List[float]) -> Tuple[float, float]:
    total = float(sum(histogram))
    if total <= 0:
        return 0.0, 0.0
    mean = sum(pos * c for pos, c in enumerate(histogram)) / total
    var = sum(((pos - mean) ** 2) * c for pos, c in enumerate(histogram)) / total
    std = var ** 0.5
    return mean, std


def main() -> None:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    missing_csv = os.path.join(base_dir, "missed_blocks.csv")
    rounds_tsv = os.path.join(base_dir, "onchain_data", "auction_resolved.tsv")

    w3 = initialize_web3()

    start_list, end_list, max_duration = read_auction_rounds(rounds_tsv)
    missing_rows = read_missing_blocks(missing_csv)

    histogram, total_missing, total_matched = compute_distribution_simple(
        w3=w3,
        missing_rows=missing_rows,
        start_list=start_list,
        end_list=end_list,
        max_duration=max_duration,
    )

    # Convert to float for stats
    hist_float = [float(c) for c in histogram]
    mean, std = summarize_distribution(hist_float)

    print("Computed relative position distribution of missed blocks within auction rounds")
    print(f"Total missing blocks (from CSV): {int(total_missing):,}")
    print(f"Total matched (within any round): {int(total_matched):,} ({(100.0*total_matched/max(total_missing,1.0)):.2f}%)")
    print(f"Mean relative position (sec): {mean:.3f}")
    print(f"Std dev (sec): {std:.3f}")

    # Emit per-position counts and save to CSV
    out_df = pl.DataFrame({
        "position": list(range(len(histogram))),
        "count": histogram,
    })
    out_path = os.path.join(base_dir, "missed_block_position_distribution.csv")
    out_df.write_csv(out_path)
    print(f"Saved per-position distribution to: {out_path}")

    # Print a simple text histogram
    max_count = max(histogram) if histogram else 0
    scale = 50 / max_count if max_count > 0 else 0
    print("\nHistogram (position: count | bar)")
    for pos, cnt in enumerate(histogram):
        if cnt == 0:
            bar = ""
        else:
            bar = "#" * max(1, int(round(cnt * scale)))
        print(f"{pos:2d}: {cnt:8d} | {bar}")

    # Save matplotlib histogram figure
    positions = list(range(len(histogram)))
    counts = histogram
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(positions, counts, width=1.0, color="C0")
    ax.set_xlabel("Relative position (sec)")
    ax.set_ylabel("Missed blocks")
    ax.set_title("Distribution of missed blocks by position within auction round")
    ax.set_xlim(-0.5, len(positions) - 0.5)
    fig.tight_layout()
    out_png = os.path.join(base_dir, "missed_block_position_hist.png")
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Saved histogram figure to: {out_png}")


if __name__ == "__main__":
    main()


