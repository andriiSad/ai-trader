"""WhiteBIT REST API candle collector.

Fetches candle data from WhiteBIT public API and stores as CSV.
Idempotent: re-running fetches only new data.
"""

from __future__ import annotations

import asyncio
import csv
import logging
import time
from pathlib import Path

import requests

log = logging.getLogger(__name__)

WHITEBIT_KLINE_URL = "https://whitebit.com/api/v1/public/kline/{market}/{interval}"
WHITEBIT_FUNDING_URL = "https://whitebit.com/api/v4/public/funding-history/{market}"

INTERVAL_SECONDS = {
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "2h": 7200,
    "4h": 14400,
    "6h": 21600,
    "12h": 43200,
    "1d": 86400,
    "1w": 604800,
    "1M": 2592000,
}

CSV_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]
FUNDING_CSV_COLUMNS = ["timestamp", "funding_rate"]

MAX_LIMIT = 1440
FUNDING_MAX_LIMIT = 100
REQUEST_DELAY = 0.2


def fetch_candles(
    pair: str,
    interval: str,
    start_ts: int,
    end_ts: int,
    limit: int = MAX_LIMIT,
) -> list[list]:
    """Fetch candles from WhiteBIT REST API.

    Args:
        pair: Trading pair (e.g. "ETH_USDT").
        interval: Candle interval (e.g. "4h").
        start_ts: Start timestamp (Unix seconds).
        end_ts: End timestamp (Unix seconds).
        limit: Max number of candles per request.

    Returns:
        List of candle arrays: [time, open, close, high, low, volume].

    Raises:
        requests.HTTPError: On API error responses.
        requests.ConnectionError: On network failures.
    """
    url = WHITEBIT_KLINE_URL.format(market=pair, interval=interval)
    params = {"start": start_ts, "end": end_ts, "limit": limit}

    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()

    return resp.json()


def reorder_candle(candle: list) -> list[str]:
    """Reorder WhiteBIT candle [time, open, close, high, low, volume]
    to standard OHLCV [timestamp, open, high, low, close, volume]."""
    return [
        str(candle[0]),
        str(candle[1]),
        str(candle[3]),
        str(candle[4]),
        str(candle[2]),
        str(candle[5]),
    ]


def save_candles(candles: list[list], pair: str, interval: str, data_dir: str) -> None:
    """Save candles to CSV file, creating directory and header if needed.

    Args:
        candles: List of WhiteBIT candle arrays.
        pair: Trading pair name.
        interval: Candle interval.
        data_dir: Root data directory.
    """
    csv_path = Path(data_dir) / pair / f"{interval}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    write_header = not csv_path.exists() or csv_path.stat().st_size == 0

    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(CSV_COLUMNS)
        for candle in candles:
            writer.writerow(reorder_candle(candle))


def get_last_timestamp(csv_path: Path) -> int | None:
    """Get the last timestamp from a CSV file.

    Returns None if file doesn't exist, is empty, or has only a header.
    """
    if not csv_path.exists():
        return None
    try:
        with open(csv_path, "rb") as f:
            size = csv_path.stat().st_size
            if size < 50:
                return None
            f.seek(max(0, size - 500))
            tail = f.read().decode(errors="replace")
            lines = tail.strip().split("\n")
            if len(lines) < 2:
                return None
            last = lines[-1].strip()
            if not last or last.startswith("timestamp"):
                return None
            return int(last.split(",")[0])
    except (ValueError, IndexError, OSError):
        return None


def load_existing_data(pair: str, interval: str, data_dir: str) -> int | None:
    """Load existing data info for a pair/interval.

    Returns the last timestamp if data exists, None otherwise.
    """
    csv_path = Path(data_dir) / pair / f"{interval}.csv"
    return get_last_timestamp(csv_path)


def run_candles(pair: str, interval: str, data_dir: str = "data") -> int:
    """Fetch and store candles for a pair/interval.

    Idempotent: if data already exists, only fetches newer candles.

    Args:
        pair: Trading pair (e.g. "ETH_USDT").
        interval: Candle interval (e.g. "4h").
        data_dir: Root data directory.

    Returns:
        Number of new candles fetched.
    """
    interval_sec = INTERVAL_SECONDS.get(interval, 14400)
    csv_path = Path(data_dir) / pair / f"{interval}.csv"

    last_ts = get_last_timestamp(csv_path)
    start_ts = last_ts + interval_sec if last_ts is not None else int(time.time()) - 30 * 86400

    end_ts = int(time.time())

    if start_ts >= end_ts:
        log.info("%s %s: already up to date", pair, interval)
        return 0

    log.info("%s %s: fetching from %d to %d", pair, interval, start_ts, end_ts)

    total_fetched = 0
    current_start = start_ts

    while current_start < end_ts:
        candles = fetch_candles(pair, interval, current_start, end_ts)

        if not candles:
            break

        save_candles(candles, pair, interval, data_dir)
        total_fetched += len(candles)

        log.info(
            "%s %s: fetched %d candles (total: %d)",
            pair,
            interval,
            len(candles),
            total_fetched,
        )

        last_candle_ts = int(candles[-1][0])
        current_start = last_candle_ts + interval_sec

        if current_start < end_ts:
            time.sleep(REQUEST_DELAY)

    log.info("%s %s: complete — %d candles saved", pair, interval, total_fetched)
    return total_fetched


def pair_to_futures_market(pair: str) -> str:
    """Convert trading pair to WhiteBIT futures market format.

    'BTC_USDT' -> 'BTC_PERP'
    """
    base = pair.split("_")[0]
    return f"{base}_PERP"


def fetch_funding_rates(
    pair: str,
    limit: int = FUNDING_MAX_LIMIT,
    offset: int = 0,
    start_date: int | None = None,
    end_date: int | None = None,
) -> list[dict]:
    """Fetch funding rate history from WhiteBIT public API.

    Args:
        pair: Trading pair (e.g. "BTC_USDT").
        limit: Number of records (max 100).
        offset: Pagination offset.
        start_date: Start timestamp (Unix seconds).
        end_date: End timestamp (Unix seconds).

    Returns:
        List of funding rate records.
    """
    market = pair_to_futures_market(pair)
    url = WHITEBIT_FUNDING_URL.format(market=market)
    params: dict = {"limit": limit, "offset": offset}
    if start_date is not None:
        params["startDate"] = start_date
    if end_date is not None:
        params["endDate"] = end_date

    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()

    return resp.json()


def save_funding_rates(records: list[dict], pair: str, data_dir: str) -> None:
    """Save funding rate records to CSV.

    Args:
        records: List of funding rate API response records.
        pair: Trading pair name.
        data_dir: Root data directory.
    """
    csv_path = Path(data_dir) / pair / "funding.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    write_header = not csv_path.exists() or csv_path.stat().st_size == 0

    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(FUNDING_CSV_COLUMNS)
        for record in records:
            writer.writerow([record["fundingTime"], record["fundingRate"]])


def get_last_funding_timestamp(csv_path: Path) -> int | None:
    """Get the last timestamp from a funding CSV file.

    Returns None if file doesn't exist, is empty, or has only a header.
    """
    if not csv_path.exists():
        return None
    try:
        with open(csv_path, "rb") as f:
            size = csv_path.stat().st_size
            if size < 30:
                return None
            f.seek(max(0, size - 500))
            tail = f.read().decode(errors="replace")
            lines = tail.strip().split("\n")
            if len(lines) < 2:
                return None
            last = lines[-1].strip()
            if not last or last.startswith("timestamp"):
                return None
            return int(last.split(",")[0])
    except (ValueError, IndexError, OSError):
        return None


def run_funding(pair: str, data_dir: str = "data") -> int:
    """Fetch and store funding rates for a pair.

    Idempotent: if data already exists, only fetches newer records.

    Args:
        pair: Trading pair (e.g. "BTC_USDT").
        data_dir: Root data directory.

    Returns:
        Number of new funding records fetched.
    """
    csv_path = Path(data_dir) / pair / "funding.csv"
    last_ts = get_last_funding_timestamp(csv_path)

    total_fetched = 0
    offset = 0

    while True:
        start_date = (last_ts + 1) if last_ts is not None else None
        records = fetch_funding_rates(
            pair, limit=FUNDING_MAX_LIMIT, offset=offset, start_date=start_date
        )

        if not records:
            break

        save_funding_rates(records, pair, data_dir)
        total_fetched += len(records)

        log.info("%s funding: fetched %d records (total: %d)", pair, len(records), total_fetched)

        if len(records) < FUNDING_MAX_LIMIT:
            break

        offset += FUNDING_MAX_LIMIT
        time.sleep(REQUEST_DELAY)

    log.info("%s funding: complete — %d records saved", pair, total_fetched)
    return total_fetched


WHITEBIT_WS_URL = "wss://ws.whitebit.com/ws"

DURATION_SECONDS = {
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "2h": 7200,
    "4h": 14400,
    "6h": 21600,
    "12h": 43200,
    "1d": 86400,
    "1w": 604800,
}

ORDERBOOK_CSV_COLUMNS = ["timestamp", "side", "price", "quantity"]


def parse_ws_depth_message(msg: dict) -> dict | None:
    """Parse a WhiteBIT WebSocket depth_update message.

    Args:
        msg: Raw WebSocket message dict.

    Returns:
        Dict with keys: bids, asks, is_full, pair, timestamp.
        None if message is not a depth_update or is malformed.
    """
    if msg.get("method") != "depth_update":
        return None
    params = msg.get("params")
    if not isinstance(params, list) or len(params) < 3:
        return None
    is_full = params[0]
    data = params[1]
    pair = params[2]
    if not isinstance(data, dict):
        return None
    return {
        "bids": data.get("bids", []),
        "asks": data.get("asks", []),
        "is_full": is_full,
        "pair": pair,
        "timestamp": data.get("timestamp"),
    }


def save_orderbook_snapshot(
    bids: list[list[str]],
    asks: list[list[str]],
    pair: str,
    output_dir: str,
    timestamp: int | float,
) -> None:
    """Save order book snapshot as CSV.

    Args:
        bids: List of [price, quantity] pairs.
        asks: List of [price, quantity] pairs.
        pair: Trading pair name.
        output_dir: Root output directory.
        timestamp: Snapshot timestamp (Unix seconds).
    """
    csv_path = Path(output_dir) / pair / "orderbook" / f"{int(timestamp)}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(ORDERBOOK_CSV_COLUMNS)
        for price, qty in bids:
            writer.writerow([int(timestamp), "bid", price, qty])
        for price, qty in asks:
            writer.writerow([int(timestamp), "ask", price, qty])


async def run_orderbook(pair: str, duration: str, output_dir: str = "data") -> int:
    """Connect to WhiteBIT WebSocket and capture order book snapshots.

    Subscribes to L2 depth updates for the given pair and saves
    a snapshot every 60 seconds.

    Args:
        pair: Trading pair (e.g. "BTC_USDT").
        duration: Collection duration (e.g. "1h", "30m").
        output_dir: Root output directory.

    Returns:
        Number of snapshots saved.
    """
    import json

    import websockets

    duration_sec = DURATION_SECONDS.get(duration, 3600)
    snapshots_saved = 0
    last_snapshot_ts = 0

    start_time = time.time()

    async with websockets.connect(WHITEBIT_WS_URL) as ws:
        subscribe_msg = json.dumps(
            {
                "id": 1,
                "method": "depth_subscribe",
                "params": [pair, 10, "0"],
            }
        )
        await ws.send(subscribe_msg)

        while time.time() - start_time < duration_sec:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
            except TimeoutError:
                continue
            except Exception:
                await asyncio.sleep(1)
                continue

            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue

            parsed = parse_ws_depth_message(msg)
            if parsed is None:
                continue

            if not parsed["is_full"]:
                continue

            now = int(parsed["timestamp"])
            if now - last_snapshot_ts < 60:
                continue

            save_orderbook_snapshot(
                parsed["bids"],
                parsed["asks"],
                pair,
                output_dir,
                now,
            )
            snapshots_saved += 1
            last_snapshot_ts = now
            log.info("%s orderbook: saved snapshot %d (ts=%d)", pair, snapshots_saved, now)

    log.info("%s orderbook: complete — %d snapshots saved", pair, snapshots_saved)
    return snapshots_saved


def build_parser():
    """Build the CLI argument parser."""
    import argparse

    parser = argparse.ArgumentParser(description="WhiteBIT candle data collector for signal-lab")
    subparsers = parser.add_subparsers(dest="command", required=True)

    candles_parser = subparsers.add_parser("candles", help="Fetch candle data")
    candles_parser.add_argument("--pair", required=True, help="Trading pair (e.g. ETH_USDT)")
    candles_parser.add_argument(
        "--interval", required=True, help="Candle interval (e.g. 4h, 1h, 1d)"
    )
    candles_parser.add_argument(
        "--output-dir", default="data", help="Output directory (default: data)"
    )

    funding_parser = subparsers.add_parser("funding", help="Fetch funding rate data")
    funding_parser.add_argument("--pair", required=True, help="Trading pair (e.g. BTC_USDT)")
    funding_parser.add_argument(
        "--output-dir", default="data", help="Output directory (default: data)"
    )

    orderbook_parser = subparsers.add_parser("orderbook", help="Collect order book snapshots")
    orderbook_parser.add_argument("--pair", required=True, help="Trading pair (e.g. BTC_USDT)")
    orderbook_parser.add_argument(
        "--duration", required=True, help="Collection duration (e.g. 1h, 30m)"
    )
    orderbook_parser.add_argument(
        "--output-dir", default="data", help="Output directory (default: data)"
    )

    return parser
