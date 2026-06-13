#!/usr/bin/env python3
"""WhiteBIT candle data collector for ML trading research."""

import argparse
import asyncio
import csv
import json
import logging
import signal
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import websockets
import yaml
from websockets.exceptions import ConnectionClosed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

WS_URL = "wss://api.whitebit.com/ws"
PING_INTERVAL = 50
RECONNECT_DELAY = 5

INTERVAL_MAP = {
    "1s": 1,
    "2s": 2,
    "3s": 3,
    "5s": 5,
    "10s": 10,
    "12s": 12,
    "15s": 15,
    "20s": 20,
    "30s": 30,
    "1m": 60,
    "2m": 120,
    "3m": 180,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "2h": 7200,
    "4h": 14400,
    "6h": 21600,
    "12h": 43200,
    "1d": 86400,
    "2d": 172800,
    "3d": 259200,
    "1w": 604800,
    "1mo": 2592000,
}

CSV_HEADER = ["timestamp", "open", "high", "low", "close", "volume", "deal"]


def parse_interval(s):
    """Parse interval string (e.g. '1m') or raw int to seconds."""
    if s in INTERVAL_MAP:
        return INTERVAL_MAP[s]
    try:
        val = int(s)
        if val <= 0:
            raise ValueError
        return val
    except (ValueError, TypeError):
        raise argparse.ArgumentTypeError(f"Invalid interval: {s}")


def load_config(path):
    """Load YAML config file."""
    with open(path) as f:
        return yaml.safe_load(f)


def build_config(args):
    """Load config from YAML, then override with CLI args."""
    config = load_config(args.config)

    if args.pairs:
        config["pairs"] = args.pairs
    if args.intervals:
        config["intervals"] = args.intervals
    if args.start_date:
        config["start_date"] = args.start_date
    if args.output_dir:
        config["output_dir"] = args.output_dir

    return config


def get_csv_path(data_dir, pair, interval):
    """Get CSV file path for a pair and interval."""
    return Path(data_dir) / pair / f"{interval}.csv"


def ensure_csv(path):
    """Create CSV file with header if it doesn't exist."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADER)


def get_last_timestamp(path):
    """Get the last timestamp from a CSV file. Returns None if empty or missing."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            size = path.stat().st_size
            if size < 50:
                return None
            f.seek(max(0, size - 200))
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


def reorder_candle(c):
    """Reorder WhiteBIT candle [time, open, close, high, low, vol, deal, market]
    to standard OHLCV [timestamp, open, high, low, close, volume, deal]."""
    return [c[0], c[1], c[3], c[4], c[2], c[5], c[6]]


def append_candles(path, candles):
    """Append candles to CSV file, reordering to OHLCV format."""
    path = Path(path)
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        for c in candles:
            writer.writerow(reorder_candle(c))


def update_or_append(path, candle, last_ts_map):
    """Update last row if same timestamp, otherwise append.
    Returns the timestamp written."""
    path = Path(path)
    key = str(path)
    ts = candle[0]
    row = reorder_candle(candle)

    if last_ts_map.get(key) == ts:
        lines = path.read_text().splitlines()
        if len(lines) > 1:
            lines[-1] = ",".join(str(v) for v in row)
            path.write_text("\n".join(lines) + "\n")
    else:
        with open(path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)
        last_ts_map[key] = ts

    return ts


def make_parser():
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="WhiteBIT candle data collector for ML trading research"
    )
    parser.add_argument(
        "command",
        choices=["backfill", "live"],
        help="backfill: fetch historical data; live: stream real-time data",
    )
    parser.add_argument("--config", default="config.yaml", help="YAML config file path")
    parser.add_argument("--pairs", nargs="+", help="Override trading pairs")
    parser.add_argument("--intervals", nargs="+", help="Override intervals (e.g. 1m 5m)")
    parser.add_argument("--start-date", help="Override backfill start date (YYYY-MM-DD)")
    parser.add_argument("--output-dir", help="Override output directory")
    return parser


# --- Backfiller ---


class Backfiller:
    def __init__(self, config):
        self.config = config
        self._req_id = 0
        self._pending = {}

    def _next_id(self):
        self._req_id += 1
        return self._req_id

    async def _keepalive(self, ws):
        while True:
            await asyncio.sleep(PING_INTERVAL)
            try:
                await ws.send(json.dumps({"id": 0, "method": "ping", "params": []}))
                log.debug("ping sent")
            except Exception:
                break

    async def _fetch_candles(self, ws, pair, start_ts, end_ts, interval_sec):
        req_id = self._next_id()
        msg = json.dumps(
            {
                "id": req_id,
                "method": "candles_request",
                "params": [pair, start_ts, end_ts, interval_sec],
            }
        )
        await ws.send(msg)

        while True:
            raw = await ws.recv()
            data = json.loads(raw)
            if data.get("id") == req_id:
                if data.get("error"):
                    raise RuntimeError(f"candles_request error: {data['error']}")
                return data.get("result", [])
            if data.get("method") == "ping":
                await ws.send(json.dumps({"id": 0, "method": "ping", "params": []}))

    def _determine_start(self, csv_path, interval_sec):
        last_ts = get_last_timestamp(csv_path)
        if last_ts is not None:
            return last_ts + interval_sec

        start_date = self.config.get("start_date")
        if start_date:
            dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=UTC)
            return int(dt.timestamp())

        return int((datetime.now(UTC) - timedelta(days=30)).timestamp())

    async def run(self):
        pairs = self.config["pairs"]
        intervals = self.config["intervals"]
        data_dir = self.config.get("output_dir", "data")
        chunk_size = self.config.get("backfill_chunk_size", 500)
        delay = self.config.get("backfill_delay", 0.5)

        async with websockets.connect(WS_URL) as ws:
            keepalive_task = asyncio.ensure_future(self._keepalive(ws))
            try:
                for pair in pairs:
                    for interval in intervals:
                        interval_sec = parse_interval(interval)
                        csv_path = get_csv_path(data_dir, pair, interval)
                        ensure_csv(csv_path)

                        start_ts = self._determine_start(csv_path, interval_sec)
                        now = int(time.time())

                        if start_ts >= now:
                            log.info(f"{pair} {interval}: already up to date")
                            continue

                        log.info(f"{pair} {interval}: backfilling from {start_ts} to {now}")
                        total_fetched = 0
                        chunk_num = 0

                        current_start = start_ts
                        while current_start < now:
                            chunk_num += 1
                            chunk_seconds = chunk_size * interval_sec
                            current_end = min(current_start + chunk_seconds, now)

                            candles = await self._fetch_candles(
                                ws, pair, current_start, current_end, interval_sec
                            )

                            if candles:
                                append_candles(csv_path, candles)
                                total_fetched += len(candles)

                            log.info(
                                f"{pair} {interval}: chunk {chunk_num} done, "
                                f"fetched {len(candles)} candles (total: {total_fetched})"
                            )

                            current_start = current_end
                            if current_start < now:
                                await asyncio.sleep(delay)

                        log.info(f"{pair} {interval}: complete — {total_fetched} candles saved")
            finally:
                keepalive_task.cancel()
                try:
                    await keepalive_task
                except asyncio.CancelledError:
                    pass


# --- LiveCollector ---


class LiveCollector:
    def __init__(self, config):
        self.config = config
        self._running = True
        self._last_ts_map = {}

    async def run(self):
        pairs = self.config["pairs"]
        intervals = self.config["intervals"]
        data_dir = self.config["output_dir"]

        self._last_ts_map = {}
        for pair in pairs:
            for interval in intervals:
                csv_path = get_csv_path(data_dir, pair, interval)
                ensure_csv(csv_path)
                ts = get_last_timestamp(csv_path)
                if ts is not None:
                    self._last_ts_map[str(csv_path)] = ts

        log.info(
            "Live streaming %d pairs × %d intervals",
            len(pairs),
            len(intervals),
        )

        tasks = [self._stream_interval(interval) for interval in intervals]
        await asyncio.gather(*tasks)

    async def _stream_interval(self, interval_str):
        interval_sec = parse_interval(interval_str)
        pairs = self.config["pairs"]
        data_dir = self.config["output_dir"]

        while self._running:
            log.info("[%s] Connecting to %s", interval_str, WS_URL)
            keepalive = None
            try:
                async with websockets.connect(WS_URL) as ws:
                    keepalive = asyncio.create_task(self._keepalive(ws))

                    for pair in pairs:
                        await self._subscribe(ws, pair, interval_sec)
                    log.info("[%s] Subscribed to %d pairs", interval_str, len(pairs))

                    async for raw in ws:
                        if not self._running:
                            break
                        msg = json.loads(raw)
                        if msg.get("method") != "candles_update":
                            continue
                        candle = msg["params"][0]
                        market = candle[7]
                        csv_path = get_csv_path(data_dir, market, interval_str)
                        log.info(
                            "[%s] %s t=%s",
                            interval_str,
                            market,
                            candle[0],
                        )
                        update_or_append(csv_path, candle, self._last_ts_map)
            except (ConnectionClosed, ConnectionError) as e:
                log.warning("[%s] Connection lost: %s", interval_str, e)
            except Exception:
                log.exception("[%s] Unexpected error", interval_str)
            finally:
                if keepalive is not None:
                    keepalive.cancel()
                    try:
                        await keepalive
                    except asyncio.CancelledError:
                        pass

            if self._running:
                self._refresh_last_ts_map(interval_str)
                self._check_gaps(interval_str, interval_sec)
                log.info("[%s] Reconnecting in %ds", interval_str, RECONNECT_DELAY)
                await asyncio.sleep(RECONNECT_DELAY)

    async def _keepalive(self, ws):
        try:
            while True:
                await asyncio.sleep(PING_INTERVAL)
                await ws.send(json.dumps({"id": 0, "method": "ping", "params": []}))
        except asyncio.CancelledError:
            pass

    async def _subscribe(self, ws, pair, interval_sec):
        sub_id = id(ws) + interval_sec
        msg = json.dumps(
            {"id": sub_id, "method": "candles_subscribe", "params": [pair, interval_sec]}
        )
        await ws.send(msg)
        await asyncio.sleep(0.1)

    def stop(self):
        self._running = False

    def _refresh_last_ts_map(self, interval_str):
        pairs = self.config["pairs"]
        data_dir = self.config["output_dir"]
        for pair in pairs:
            csv_path = get_csv_path(data_dir, pair, interval_str)
            ts = get_last_timestamp(csv_path)
            if ts is not None:
                self._last_ts_map[str(csv_path)] = ts

    def _check_gaps(self, interval_str, interval_sec):
        now = int(time.time())
        pairs = self.config["pairs"]
        data_dir = self.config["output_dir"]
        threshold = 2 * interval_sec
        for pair in pairs:
            csv_path = get_csv_path(data_dir, pair, interval_str)
            key = str(csv_path)
            last_ts = self._last_ts_map.get(key)
            if last_ts is not None:
                gap = now - last_ts
                if gap > threshold:
                    log.warning(
                        "[%s] %s: gap of %ds detected (last_ts=%d, now=%d). "
                        "Consider running `scraper.py backfill`.",
                        interval_str,
                        pair,
                        gap,
                        last_ts,
                        now,
                    )


# --- Entrypoint ---


def main():
    parser = make_parser()
    args = parser.parse_args()
    config = build_config(args)

    if args.command == "backfill":
        backfiller = Backfiller(config)
        asyncio.run(backfiller.run())
    elif args.command == "live":
        collector = LiveCollector(config)

        def on_signal(sig, frame):
            log.info("Shutting down...")
            collector.stop()

        signal.signal(signal.SIGINT, on_signal)
        signal.signal(signal.SIGTERM, on_signal)

        asyncio.run(collector.run())


if __name__ == "__main__":
    main()
