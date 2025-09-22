import os
from typing import Dict, Iterable, List, Tuple
from bisect import bisect_right

try:
    from dotenv import load_dotenv  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    load_dotenv = None  # type: ignore

import polars as pl
from web3 import Web3
import matplotlib.pyplot as plt


Interval = Tuple[int, int]


def initialize_web3() -> Web3:
    """Create and return a connected Web3 instance.

    Behavior:
    - Optionally loads environment variables from a .env file (if python-dotenv
      is available).
    - Reads ARBITRUM_RPC_URL from the environment and establishes an HTTP
      connection to the provider.
    - Raises a RuntimeError if the variable is missing or the connection fails.

    Returns:
        A connected Web3 instance configured with the ARBITRUM RPC URL.
    """
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
    """Read auction rounds from a TSV and return start/end timestamps and max duration.

    The TSV is expected to contain at least two integer columns:
    - round_start_timestamp
    - round_end_timestamp (inclusive)

    The rows are sorted by start time to enable binary searches. The function
    also computes the maximum round duration in seconds, which determines the
    histogram size for positions.

    Args:
        tsv_path: Absolute path to the auction rounds TSV file.

    Returns:
        (start_list, end_list, max_duration)
            start_list: List of per-round start timestamps (sorted ascending).
            end_list: List of per-round inclusive end timestamps.
            max_duration: Maximum value of (end - start) across all rounds.
    """
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


def read_recorded_blocks(csv_path: str) -> Iterable[int]:
    """Read recorded block numbers from CSV.

    Expects a CSV with a single integer column named "block_number". Returns a
    Python list for simplicity. If the file is large and memory is a concern,
    this can be adapted to an iterator/streaming approach, but for typical
    use-cases a list is convenient and fast.

    Args:
        csv_path: Absolute path to the recorded blocks CSV file.

    Returns:
        A list of block numbers (possibly with duplicates or unsorted).
    """
    df = pl.read_csv(csv_path, columns=["block_number"], schema_overrides={"block_number": pl.Int64})
    return df["block_number"].to_list()


def derive_missing_streaks(
    recorded_blocks: Iterable[int],
    range_start: int,
    range_end: int,
) -> List[Interval]:
    """Compute contiguous missing block intervals within a closed range.

    This function avoids constructing the full set of all block numbers in the
    interval. Instead, it processes the sorted unique recorded blocks and walks
    a cursor from range_start to range_end, emitting a streak each time there is
    a gap between the cursor and the next recorded block.

    Args:
        recorded_blocks: Iterable of observed block numbers (may be unsorted and
            contain duplicates). Only values within [range_start, range_end] are
            considered.
        range_start: First block number (inclusive) of the interval to inspect.
        range_end: Last block number (inclusive) of the interval to inspect.

    Returns:
        A list of (start_block, end_block) tuples representing contiguous
        missing streaks, inclusive at both ends. Empty if nothing is missing.
    """
    # Assume recorded_blocks can be large; process sorted unique
    sorted_blocks = sorted(set(b for b in recorded_blocks if range_start <= b <= range_end))
    streaks: List[Interval] = []

    cursor = range_start
    for b in sorted_blocks:
        if b > cursor:
            # Missing from cursor to b-1
            streaks.append((cursor, b - 1))
        cursor = b + 1

    if cursor <= range_end:
        streaks.append((cursor, range_end))

    return [(s, e) for (s, e) in streaks if s <= e]


def get_block_timestamp(w3: Web3, block_number: int, cache: Dict[int, int]) -> int:
    """Fetch the timestamp for a given block number with memoization.

    A small in-memory cache is used to avoid repeating RPC calls when different
    streaks share a boundary block. The timestamp is returned as an integer
    seconds-since-epoch.

    Args:
        w3: Connected Web3 instance.
        block_number: L2 block number whose timestamp to retrieve.
        cache: Dict mapping block_number -> timestamp for memoization.

    Returns:
        The integer timestamp for the given block.
    """
    if block_number in cache:
        return cache[block_number]
    block = w3.eth.get_block(block_number)
    ts = int(block["timestamp"])  # type: ignore[index]
    cache[block_number] = ts
    return ts


def find_round_index_for_time(start_list: List[int], end_list: List[int], t: int) -> int:
    """Find the index of the auction round that contains timestamp t.

    Uses binary search over the sorted start_list to locate the last round with
    start <= t and then checks whether t <= end for that round. Round ends are
    treated as inclusive.

    Args:
        start_list: Sorted list of round start timestamps.
        end_list: List of round inclusive end timestamps (same length/order as starts).
        t: Timestamp to locate.

    Returns:
        The index of the containing round, or -1 if t is not within any round.
    """
    idx = bisect_right(start_list, t) - 1
    if idx >= 0 and t <= end_list[idx]:
        return idx
    return -1


def compute_distribution_from_streaks(
    w3: Web3,
    streaks: List[Interval],
    start_list: List[int],
    end_list: List[int],
    max_duration: int,
) -> Tuple[List[int], int, int]:
    """Convert missing block streaks into a histogram of relative positions.

    For each contiguous missing streak [first_block, last_block], we query the
    timestamps of the first and last blocks and then linearly interpolate a
    timestamp for each missing block in between. Each synthetic timestamp is
    mapped to its containing auction round (if any), and the integer offset from
    round start (in seconds) is counted into a histogram bucket.

    Notes and assumptions:
    - Round ends are inclusive; if a timestamp equals round_end_timestamp, it
      counts toward the last position of that round.
    - Linear interpolation assumes roughly constant seconds-per-block over the
      streak. This is an approximation but adequate for distributional stats.
    - Only two RPC calls per streak (first and last blocks) are made.

    Args:
        w3: Connected Web3 instance.
        streaks: List of inclusive missing ranges (first_block, last_block).
        start_list: Sorted list of round start timestamps.
        end_list: List of round inclusive end timestamps.
        max_duration: Maximum (end - start) among all rounds; sets histogram size.

    Returns:
        (histogram, total_missing, total_matched)
            histogram: List where index is relative second offset and value is
                the count of missing blocks observed at that offset.
            total_missing: Total number of missing blocks across all streaks.
            total_matched: Number of those blocks that fell within any round.
    """
    histogram: List[int] = [0 for _ in range(max_duration + 1)]
    total_missing = 0
    total_matched = 0
    ts_cache: Dict[int, int] = {}

    for first_block, last_block in streaks:
        count = (last_block - first_block + 1)
        total_missing += count

        t_start = get_block_timestamp(w3, first_block, ts_cache)
        t_end = get_block_timestamp(w3, last_block, ts_cache)
        if t_end < t_start:
            t_start, t_end = t_end, t_start

        if count == 1:
            ts = t_start
            ridx = find_round_index_for_time(start_list, end_list, ts)
            if ridx != -1:
                pos = ts - start_list[ridx]
                if 0 <= pos <= max_duration:
                    histogram[pos] += 1
                    total_matched += 1
            continue

        span = max(0, t_end - t_start)
        for j in range(count):
            if span == 0:
                ts = t_start
            else:
                ts = int(round(t_start + (span * (j / float(count - 1)))))

            ridx = find_round_index_for_time(start_list, end_list, ts)
            if ridx == -1:
                continue
            pos = ts - start_list[ridx]
            if 0 <= pos <= max_duration:
                histogram[pos] += 1
                total_matched += 1

    return histogram, total_missing, total_matched


def summarize_distribution(histogram: List[int]) -> Tuple[float, float]:
    """Compute mean and standard deviation of positions, weighted by counts.

    Treats the histogram index as the position in seconds and the value as the
    weight. Returns population standard deviation (not sample std).

    Args:
        histogram: Per-position counts.

    Returns:
        (mean, std) where both are floats measured in seconds.
    """
    total = float(sum(histogram))
    if total <= 0:
        return 0.0, 0.0
    mean = sum(pos * c for pos, c in enumerate(histogram)) / total
    var = sum(((pos - mean) ** 2) * c for pos, c in enumerate(histogram)) / total
    std = var ** 0.5
    return mean, std


def main() -> None:
    """Entry point: derive missing streaks, build distribution, and save outputs.

    Steps:
    1) Read auction rounds and prepare search structures and max duration.
    2) Read block numbers from recorded_blocks.csv and compute the missing
       streaks within the closed interval [interval_start, interval_end].
    3) For each streak, query timestamps for its endpoints, interpolate per-block
       timestamps, map to rounds, and accumulate a histogram of relative
       positions.
    4) Print summary stats, save CSV and a PNG histogram, and print an ASCII bar
       histogram to the console for quick inspection.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    rounds_tsv = os.path.join(base_dir, "onchain_data", "auction_resolved.tsv")
    recorded_csv = os.path.join(base_dir, "recorded_blocks.csv")

    # Range to evaluate
    interval_start = 342607969
    interval_end = 374383982

    w3 = initialize_web3()
    start_list, end_list, max_duration = read_auction_rounds(rounds_tsv)

    recorded_blocks = read_recorded_blocks(recorded_csv)
    streaks = derive_missing_streaks(recorded_blocks, interval_start, interval_end)

    histogram, total_missing, total_matched = compute_distribution_from_streaks(
        w3=w3,
        streaks=streaks,
        start_list=start_list,
        end_list=end_list,
        max_duration=max_duration,
    )

    mean, std = summarize_distribution(histogram)

    print("Computed relative position distribution from missing streaks (manual derivation)")
    print(f"Interval: [{interval_start}, {interval_end}] total blocks: {interval_end - interval_start + 1:,}")
    print(f"Total missing blocks (derived): {int(total_missing):,}")
    print(f"Total matched (within any round): {int(total_matched):,} ({(100.0*total_matched/max(total_missing,1.0)):.2f}%)")
    print(f"Mean relative position (sec): {mean:.3f}")
    print(f"Std dev (sec): {std:.3f}")

    # Save CSV
    out_df = pl.DataFrame({
        "position": list(range(len(histogram))),
        "count": histogram,
    })
    out_csv = os.path.join(base_dir, "missed_block_position_distribution_alt.csv")
    out_df.write_csv(out_csv)
    print(f"Saved per-position distribution to: {out_csv}")

    # Save figure
    positions = list(range(len(histogram)))
    counts = histogram
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(positions, counts, width=1.0, color="C1")
    ax.set_xlabel("Relative position (sec)")
    ax.set_ylabel("Missed blocks")
    ax.set_title("Distribution of missed blocks by position (manual derivation)")
    ax.set_xlim(-0.5, len(positions) - 0.5)
    fig.tight_layout()
    out_png = os.path.join(base_dir, "missed_block_position_hist_alt.png")
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Saved histogram figure to: {out_png}")

    # Print ASCII histogram
    max_count = max(histogram) if histogram else 0
    scale = 50 / max_count if max_count > 0 else 0
    print("\nHistogram (position: count | bar)")
    for pos, cnt in enumerate(histogram):
        bar = "" if cnt == 0 else "#" * max(1, int(round(cnt * scale)))
        print(f"{pos:2d}: {cnt:8d} | {bar}")


if __name__ == "__main__":
    main()


