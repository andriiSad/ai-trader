#!/usr/bin/env python3
"""Data exploration dashboard for ML trading research."""

import argparse

from src.explore_charts import build_dashboard


def main():
    parser = argparse.ArgumentParser(description="Generate data exploration dashboard")
    parser.add_argument("--data-dir", default="data", help="Root data directory")
    parser.add_argument(
        "--pairs", nargs="+", default=["BTC_USDT", "ETH_USDT"], help="Trading pairs"
    )
    parser.add_argument(
        "--output", default="data/reports/data_exploration.html", help="Output HTML path"
    )
    args = parser.parse_args()
    build_dashboard(args.data_dir, args.pairs, args.output)


if __name__ == "__main__":
    main()
