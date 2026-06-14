#!/usr/bin/env python3
"""Backtester CLI — config-driven backtest engine and dashboard launcher."""

import argparse
import sys
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser(description="Backtester")
    parser.add_argument(
        "command",
        choices=["run", "dashboard"],
        help="Command to execute",
    )
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config file")
    args = parser.parse_args()

    config_path = Path(args.config)
    with open(config_path) as f:
        config = yaml.safe_load(f)

    if args.command == "run":
        from src.pipeline import run_backtest

        base_dir = config_path.parent
        config["predictions_path"] = str(base_dir / config["predictions_path"])
        config["candles_dir"] = str(base_dir / config["candles_dir"])
        config["output"]["results_path"] = str(base_dir / config["output"]["results_path"])
        if "tear_sheet_path" in config["output"]:
            config["output"]["tear_sheet_path"] = str(
                base_dir / config["output"]["tear_sheet_path"]
            )

        results = run_backtest(config)
        print(f"Backtest complete. Results saved to {config['output']['results_path']}")
        for pair, metrics in results["overall"].items():
            print(
                f"  {pair}: total_return={metrics['total_return']:.4f}, sharpe={metrics['sharpe']:.4f}, trades={metrics['num_trades']}"
            )
        sys.exit(0)
    elif args.command == "dashboard":
        print("Run: streamlit run projects/backtester/src/dashboard.py")
        sys.exit(0)


if __name__ == "__main__":
    main()
