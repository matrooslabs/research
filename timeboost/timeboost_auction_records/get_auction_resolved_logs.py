import argparse
import asyncio
import json
import os
import sys
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple
from datetime import datetime, timezone
from dotenv import load_dotenv
import aiohttp
from web3 import Web3
import polars as pl

load_dotenv()

DECODED_COLUMNS: List[str] = [
    "block_number",
    "log_index",
    "transaction_hash",
    "contract_address",
    "is_multi_bid_auction",
    "round",
    "first_price_bidder",
    "first_price_express_lane_controller",
    "first_price_amount",
    "price",
    "round_start_timestamp",
    "round_end_timestamp",
]


def ensure_rpc_url(cmd_arg_url: Optional[str]) -> str:
    url = cmd_arg_url or os.getenv("ARBITRUM_RPC_URL")
    if not url:
        print("ERROR: Provide --rpc-url or set ARBITRUM_RPC_URL in env.", file=sys.stderr)
        sys.exit(1)
    return url


def build_web3(rpc_url: str) -> Web3:
    w3 = Web3(Web3.HTTPProvider(rpc_url))
    if not w3.is_connected():
        print("ERROR: Could not connect to RPC at", rpc_url, file=sys.stderr)
        sys.exit(2)
    return w3


class AsyncRateLimiter:
    def __init__(self, max_calls_per_sec: int) -> None:
        self.max_calls = int(max_calls_per_sec)
        self.period = 1.0
        self.calls: Deque[float] = deque()
        self.lock = asyncio.Lock()

    async def acquire(self) -> None:
        loop = asyncio.get_running_loop()
        async with self.lock:
            now = loop.time()
            while True:
                while self.calls and now - self.calls[0] >= self.period:
                    self.calls.popleft()
                if len(self.calls) < self.max_calls:
                    self.calls.append(now)
                    return
                sleep_for = self.period - (now - self.calls[0])
                await asyncio.sleep(max(0.0, sleep_for))
                now = loop.time()


async def json_rpc_post(
    session: aiohttp.ClientSession,
    url: str,
    payload: dict,
    limiter: Optional[AsyncRateLimiter],
) -> dict:
    if limiter is not None:
        await limiter.acquire()
    async with session.post(url, json=payload) as resp:
        resp.raise_for_status()
        return await resp.json()


async def eth_get_logs(
    session: aiohttp.ClientSession,
    url: str,
    params: dict,
    limiter: Optional[AsyncRateLimiter],
) -> List[dict]:
    payload = {"jsonrpc": "2.0", "method": "eth_getLogs", "params": [params], "id": 1}
    data = await json_rpc_post(session, url, payload, limiter)
    if "error" in data:
        raise RuntimeError(data["error"])
    return data.get("result", [])


def _convert_hex_to_int(hex_str: int | str) -> int:
    if isinstance(hex_str, str) and hex_str.startswith("0x"):
        return int(hex_str, 16)
    return int(hex_str)


def _normalize_log_for_output(log: dict) -> dict:
    topics = log.get("topics", []) or []
    topic0 = topics[0] if len(topics) > 0 else None
    topic1 = topics[1] if len(topics) > 1 else None
    topic2 = topics[2] if len(topics) > 2 else None
    topic3 = topics[3] if len(topics) > 3 else None
    return {
        "block_number": _convert_hex_to_int(log.get("blockNumber", 0)),
        "log_index": _convert_hex_to_int(log.get("logIndex", 0)),
        "transaction_hash": log.get("transactionHash"),
        "contract_address": log.get("address"),
        "data": log.get("data", "0x"),
        "topic0": topic0,
        "topic1": topic1,
        "topic2": topic2,
        "topic3": topic3,
    }


def _topic_to_bool(topic_hex: Optional[str]) -> Optional[bool]:
    if not isinstance(topic_hex, str) or not topic_hex.startswith("0x"):
        return None
    try:
        return int(topic_hex, 16) != 0
    except Exception:
        return None


def _topic_to_address(topic_hex: Optional[str]) -> Optional[str]:
    if not isinstance(topic_hex, str) or not topic_hex.startswith("0x"):
        return None
    try:
        # topics are 32-byte values; address is right-most 20 bytes
        hex_no_prefix = topic_hex[2:].rjust(64, "0")
        addr_hex = hex_no_prefix[-40:]
        return "0x" + addr_hex
    except Exception:
        return None


def _split_32byte_words(data_hex: str) -> List[int]:
    if not isinstance(data_hex, str) or not data_hex.startswith("0x"):
        return []
    hex_no_prefix = data_hex[2:]
    # pad to multiple of 64 chars (32 bytes)
    if len(hex_no_prefix) % 64 != 0:
        hex_no_prefix = hex_no_prefix.ljust(((len(hex_no_prefix) // 64) + 1) * 64, "0")
    words: List[int] = []
    for i in range(0, len(hex_no_prefix), 64):
        chunk = hex_no_prefix[i : i + 64]
        try:
            words.append(int(chunk, 16) if chunk else 0)
        except Exception:
            words.append(0)
    return words


def decode_auction_resolved_from_flat_log(log_flat: dict, expected_topic0: str) -> Optional[dict]:
    # Ensure correct event signature
    if str(log_flat.get("topic0")).lower() != str(expected_topic0).lower():
        return None

    # Indexed fields via topics
    is_multi = _topic_to_bool(log_flat.get("topic1"))
    first_price_bidder = _topic_to_address(log_flat.get("topic2"))
    first_price_ctrl = _topic_to_address(log_flat.get("topic3"))

    # Non-indexed fields in data (packed as 32-byte words):
    # uint64 round, uint256 firstPriceAmount, uint256 price, uint64 roundStartTimestamp, uint64 roundEndTimestamp
    words = _split_32byte_words(str(log_flat.get("data", "0x")))
    if len(words) < 5:
        return None
    round_num = int(words[0])
    first_price_amount = int(words[1])
    price = int(words[2])
    round_start_ts = int(words[3])
    round_end_ts = int(words[4])

    return {
        "block_number": int(log_flat.get("block_number", 0)),
        "log_index": int(log_flat.get("log_index", 0)),
        "transaction_hash": str(log_flat.get("transaction_hash", "")),
        "contract_address": str(log_flat.get("contract_address", "")),
        "is_multi_bid_auction": bool(is_multi) if is_multi is not None else None,
        "round": round_num,
        "first_price_bidder": first_price_bidder,
        "first_price_express_lane_controller": first_price_ctrl,
        "first_price_amount": first_price_amount,
        "price": price,
        "round_start_timestamp": round_start_ts,
        "round_end_timestamp": round_end_ts,
    }


def _parse_iso_datetime_utc(dt_text: str) -> int:
    text = dt_text.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    text = text.replace(" ", "T") if "T" not in text else text
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def _get_block_timestamp(w3: Web3, block_number: int) -> int:
    blk = w3.eth.get_block(block_number)
    return int(blk["timestamp"])  # type: ignore[index]


def find_block_at_or_after_timestamp(w3: Web3, target_ts: int) -> int:
    earliest_num = int(w3.eth.get_block("earliest")["number"])  # type: ignore[index]
    latest_num = int(w3.eth.get_block("latest")["number"])  # type: ignore[index]

    earliest_ts = _get_block_timestamp(w3, earliest_num)
    if target_ts <= earliest_ts:
        return earliest_num
    latest_ts = _get_block_timestamp(w3, latest_num)
    if target_ts >= latest_ts:
        return latest_num

    low = earliest_num
    high = latest_num
    while low < high:
        mid = (low + high) // 2
        mid_ts = _get_block_timestamp(w3, mid)
        if mid_ts < target_ts:
            low = mid + 1
        else:
            high = mid
    return low


async def fetch_range_logs(
    session: aiohttp.ClientSession,
    rpc_url: str,
    target_addr: str,
    topics: List[str],
    start_b: int,
    end_b: int,
    limiter: Optional[AsyncRateLimiter],
) -> List[dict]:
    if end_b < start_b:
        return []
    params = {
        "fromBlock": hex(start_b),
        "toBlock": hex(end_b),
        "address": Web3.to_checksum_address(target_addr),
        "topics": topics,
    }
    try:
        logs = await eth_get_logs(session, rpc_url, params, limiter)
        if not logs:
            return []
        return [_normalize_log_for_output(l) for l in logs]
    except Exception as e:
        size = end_b - start_b + 1
        print(f"WARN: get_logs failed for [{start_b}, {end_b}] (size={size}): {e}")
        if size <= 1000:
            return []
        mid = (start_b + end_b) // 2
        left = await fetch_range_logs(
            session,
            rpc_url,
            target_addr,
            topics,
            start_b,
            mid,
            limiter,
        )
        right = await fetch_range_logs(
            session,
            rpc_url,
            target_addr,
            topics,
            mid + 1,
            end_b,
            limiter,
        )
        return left + right


async def fetch_worker(
    name: int,
    session: aiohttp.ClientSession,
    rpc_url: str,
    target_addr: str,
    topics: List[str],
    jobs: "asyncio.Queue[Tuple[int,int]]",
    out_queue: "asyncio.Queue[List[dict]]",
    limiter: Optional[AsyncRateLimiter],
) -> None:
    while True:
        start_b, end_b = await jobs.get()
        try:
            logs_flat = await fetch_range_logs(
                session,
                rpc_url,
                target_addr,
                topics,
                start_b,
                end_b,
                limiter,
            )
            if logs_flat:
                decoded = []
                expected_t0 = get_auction_resolved_topic0()
                for lf in logs_flat:
                    row = decode_auction_resolved_from_flat_log(lf, expected_t0)
                    if row is not None:
                        decoded.append(row)
                if decoded:
                    print(f"+ {len(decoded)} events in [{start_b}, {end_b}]")
                    await out_queue.put(decoded)
                else:
                    print(f". no events in [{start_b}, {end_b}]")
            else:
                print(f". no logs in [{start_b}, {end_b}]")
        finally:
            jobs.task_done()


async def writer_worker_tsv(
    out_path: str, queue: "asyncio.Queue[List[dict]]", batch_size: int
) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    buffer: List[dict] = []
    wrote_header = os.path.exists(out_path) and os.path.getsize(out_path) > 0
    while True:
        rows = await queue.get()
        if rows is None:  # type: ignore[arg-type]
            if buffer:
                df = pl.DataFrame(buffer).select(DECODED_COLUMNS)
                with open(out_path, "ab") as f:
                    df.write_csv(f, separator="\t", include_header=not wrote_header)
                wrote_header = True
            queue.task_done()
            break
        buffer.extend(rows)
        if len(buffer) >= batch_size:
            df = pl.DataFrame(buffer).select(DECODED_COLUMNS)
            with open(out_path, "ab") as f:
                df.write_csv(f, separator="\t", include_header=not wrote_header)
            wrote_header = True
            buffer.clear()
        queue.task_done()


def get_auction_resolved_topic0() -> str:
    # keccak256("AuctionResolved(bool,uint64,address,address,uint256,uint256,uint64,uint64)")
    # 0x7f5bdabbd27a8fc572781b177055488d7c6729a2bade4f57da9d200f31c15d47
    return "0x7f5bdabbd27a8fc572781b177055488d7c6729a2bade4f57da9d200f31c15d47"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch AuctionResolved logs via eth_getLogs in parallel and save decoded TSV"
    )
    parser.add_argument(
        "--rpc-url",
        type=str,
        default=None,
        help="RPC HTTP URL (defaults to $ARBITRUM_RPC_URL)",
    )
    parser.add_argument(
        "--address",
        type=str,
        default="0x5fcb496a31b7AE91e7c9078Ec662bd7A55cd3079",
        help="Contract address",
    )
    parser.add_argument(
        "--from-block",
        type=int,
        default=None,
        help="Start block number (inclusive). If omitted, uses --from-datetime",
    )
    parser.add_argument(
        "--to-block",
        type=int,
        default=None,
        help="End block number (inclusive). Default: latest",
    )
    parser.add_argument(
        "--from-datetime",
        type=str,
        default="2025-04-01 00:00:00Z",
        help=(
            "UTC datetime for start (e.g. '2025-04-01 00:00:00Z'). Used when --from-block is not provided"
        ),
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=10_000,
        help="Block chunk size (default 10k)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Number of parallel workers (default 16)",
    )
    parser.add_argument(
        "--rps",
        type=int,
        default=50,
        help="Global max requests per second across all workers (default 100)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="auction_resolved.tsv",
        help="Output TSV path",
    )
    parser.add_argument(
        "--flush-every",
        type=int,
        default=5000,
        help="Writer batch size before flushing TSV (default 5000)",
    )

    args = parser.parse_args()

    rpc_url = ensure_rpc_url(args.rpc_url)
    w3 = build_web3(rpc_url)
    latest = int(w3.eth.get_block("latest")["number"])  # type: ignore[index]

    if args.from_block is not None:
        start_block = int(args.from_block)
    else:
        target_ts = _parse_iso_datetime_utc(str(args.from_datetime))
        start_block = find_block_at_or_after_timestamp(w3, target_ts)

    end_block = int(args.to_block) if args.to_block is not None else latest

    if start_block > end_block:
        print("ERROR: from-block must be <= to-block", file=sys.stderr)
        sys.exit(3)

    topic0 = get_auction_resolved_topic0()
    topics = [topic0]

    async def run_async() -> None:
        jobs: asyncio.Queue[Tuple[int, int]] = asyncio.Queue()
        out_queue: asyncio.Queue[List[dict]] = asyncio.Queue()

        cur = int(start_block)
        chunk = int(args.chunk_size)
        while cur <= end_block:
            to_b = min(cur + chunk - 1, end_block)
            await jobs.put((cur, to_b))
            cur = to_b + 1

        limiter = AsyncRateLimiter(max_calls_per_sec=int(args.rps))
        async with aiohttp.ClientSession() as session:
            writer_task = asyncio.create_task(
                writer_worker_tsv(args.out, out_queue, int(args.flush_every))
            )
            workers = [
                asyncio.create_task(
                    fetch_worker(
                        i,
                        session,
                        rpc_url,
                        args.address,
                        topics,
                        jobs,
                        out_queue,
                        limiter,
                    )
                )
                for i in range(int(args.workers))
            ]

            await jobs.join()
            await out_queue.put(None)  # type: ignore[arg-type]
            await writer_task
            for t in workers:
                t.cancel()

    asyncio.run(run_async())


if __name__ == "__main__":
    main()


