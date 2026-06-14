#!/usr/bin/env python3
"""CLI entry point for WhiteBIT candle collection.

Usage:
    python collect.py candles --pair ETH_USDT --interval 4h
    python collect.py candles --pair BTC_USDT --interval 1h
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.collect import build_parser, run_candles, run_funding, run_orderbook

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "candles":
        count = run_candles(args.pair, args.interval, args.output_dir)
        print(f"Fetched {count} new candles for {args.pair} {args.interval}")
    elif args.command == "funding":
        count = run_funding(args.pair, args.output_dir)
        print(f"Fetched {count} new funding records for {args.pair}")
    elif args.command == "orderbook":
        import asyncio

        count = asyncio.run(run_orderbook(args.pair, args.duration, args.output_dir))
        print(f"Collected {count} orderbook snapshots for {args.pair}")


if __name__ == "__main__":
    main()
