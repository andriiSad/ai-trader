#!/usr/bin/env python3
"""ML Trading Trainer — LightGBM walk-forward pipeline."""

import argparse
import logging
import sys

import yaml
from src.pipeline import run_evaluate, run_features, run_full_pipeline, run_train, run_visualize

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main():
    parser = argparse.ArgumentParser(description="ML Trading Trainer")
    parser.add_argument(
        "command",
        choices=["run", "features", "train", "evaluate", "visualize"],
        help="Pipeline command to execute",
    )
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config file")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    try:
        if args.command == "run":
            run_full_pipeline(config)
        elif args.command == "features":
            df = run_features(config)
            print(f"Features ready: {len(df)} rows, {len(df.columns)} columns")
        elif args.command == "train":
            df = run_features(config)
            results = run_train(config, df)
            print(f"Training complete. Overall accuracy: {results['overall_accuracy']:.3f}")
        elif args.command == "evaluate":
            df = run_features(config)
            results = run_train(config, df)
            path = run_evaluate(config, results)
            print(f"Results saved to {path}")
        elif args.command == "visualize":
            run_visualize(config)
            print("Dashboards generated.")
    except Exception as e:
        logging.error("Pipeline failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
