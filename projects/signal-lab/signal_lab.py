#!/usr/bin/env python3
"""Signal Lab — crypto data download and signal generation."""

import argparse
import logging
import os

import yaml
from src.data import download_candles, load_candles
from src.features import generate_features
from src.labels import generate_labels
from src.metrics import print_fold_summary
from src.pipeline import print_summary, run_pipeline
from src.scaler import fit_scaler, transform
from src.walk_forward import split

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)

MODEL_CHOICES = ["lr", "lgbm", "lstm"]


def _run_train_lr(df, config):
    from src.models.logistic import train_logistic

    feat_df = generate_features(df)
    label_df = generate_labels(df)
    merged = feat_df.merge(label_df, on="timestamp", how="inner")

    feature_cols = [c for c in merged.columns if c not in ("timestamp", "label")]

    accuracies = []
    for fold_idx, (train_df, test_df) in enumerate(split(merged)):
        scaler = fit_scaler(train_df[feature_cols])
        X_train_scaled = transform(train_df[feature_cols], scaler)
        X_test_scaled = transform(test_df[feature_cols], scaler)

        y_train = train_df.loc[X_train_scaled.index, "label"].values
        y_test = test_df.loc[X_test_scaled.index, "label"].values
        X_train = X_train_scaled.values
        X_test = X_test_scaled.values

        result = train_logistic(X_train, y_train, X_test, y_test)
        print_fold_summary("Logistic Regression", fold_idx, result["metrics"])
        accuracies.append(result["metrics"]["accuracy"])

    if accuracies:
        avg = sum(accuracies) / len(accuracies)
        print(f"\nAverage accuracy across {len(accuracies)} folds: {avg:.4f}")


def _run_train_lgbm(df, config):
    from src.models.lightgbm_model import train_lightgbm

    feat_df = generate_features(df)
    label_df = generate_labels(df)
    merged = feat_df.merge(label_df, on="timestamp", how="inner")

    feature_cols = [c for c in merged.columns if c not in ("timestamp", "label")]

    accuracies = []
    for fold_idx, (train_df, test_df) in enumerate(split(merged)):
        scaler = fit_scaler(train_df[feature_cols])
        X_train_scaled = transform(train_df[feature_cols], scaler)
        X_test_scaled = transform(test_df[feature_cols], scaler)

        y_train = train_df.loc[X_train_scaled.index, "label"].values
        y_test = test_df.loc[X_test_scaled.index, "label"].values
        X_train = X_train_scaled.values
        X_test = X_test_scaled.values

        result = train_lightgbm(X_train, y_train, X_test, y_test)
        print_fold_summary("LightGBM", fold_idx, result["metrics"])
        accuracies.append(result["metrics"]["accuracy"])

        if fold_idx == 0 and result.get("feature_importance"):
            print("\n  Feature Importance (top 10):")
            for name, imp in result["feature_importance"][:10]:
                print(f"    {name}: {imp}")

    if accuracies:
        avg = sum(accuracies) / len(accuracies)
        print(f"\nAverage accuracy across {len(accuracies)} folds: {avg:.4f}")


def _run_train_lstm(df, config):
    from src.models.lstm import train_lstm

    feat_df = generate_features(df)
    label_df = generate_labels(df)
    merged = feat_df.merge(label_df, on="timestamp", how="inner")

    feature_cols = [c for c in merged.columns if c not in ("timestamp", "label")]

    accuracies = []
    for fold_idx, (train_df, test_df) in enumerate(split(merged)):
        scaler = fit_scaler(train_df[feature_cols])
        X_train_scaled = transform(train_df[feature_cols], scaler)
        X_test_scaled = transform(test_df[feature_cols], scaler)

        y_train = train_df.loc[X_train_scaled.index, "label"].values
        y_test = test_df.loc[X_test_scaled.index, "label"].values
        X_train = X_train_scaled.values
        X_test = X_test_scaled.values

        result = train_lstm(X_train, y_train, X_test, y_test)
        print_fold_summary("LSTM", fold_idx, result["metrics"])
        accuracies.append(result["metrics"]["accuracy"])

    if accuracies:
        avg = sum(accuracies) / len(accuracies)
        print(f"\nAverage accuracy across {len(accuracies)} folds: {avg:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Signal Lab")
    parser.add_argument(
        "command",
        choices=["download", "features", "train", "run"],
        help="Command to execute",
    )
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config file")
    parser.add_argument(
        "--model",
        choices=MODEL_CHOICES,
        default=None,
        help="Model to train (default: all)",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.command == "download":
        pair = config["pair"]
        interval = config["interval"]
        start_date = config.get("start_date")
        data_dir = config.get("data_dir", "data")

        logging.info(f"Downloading {pair} {interval} candles...")
        df = download_candles(pair, interval, start_date, data_dir)
        csv_path = os.path.join(data_dir, pair, f"{interval}.csv")
        logging.info(f"Saved {len(df)} candles to {csv_path}")

    elif args.command == "features":
        pair = config["pair"]
        interval = config["interval"]
        data_dir = config.get("data_dir", "data")

        logging.info(f"Loading candles for {pair} {interval}...")
        df = load_candles(pair, interval, data_dir)

        logging.info("Generating features...")
        feat_df = generate_features(df)

        logging.info("Generating labels...")
        label_df = generate_labels(df)

        merged = feat_df.merge(label_df, on="timestamp", how="inner")

        out_dir = os.path.join(data_dir, pair)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "features.parquet")
        merged.to_parquet(out_path, index=False)
        logging.info(f"Saved {len(merged)} rows to {out_path}")

    elif args.command == "train":
        pair = config["pair"]
        interval = config["interval"]
        data_dir = config.get("data_dir", "data")

        logging.info(f"Loading candles for {pair} {interval}...")
        df = load_candles(pair, interval, data_dir)

        models_to_run = [args.model] if args.model else ["lr", "lgbm", "lstm"]
        for m in models_to_run:
            if m == "lr":
                logging.info("Running walk-forward with Logistic Regression...")
                _run_train_lr(df, config)
            elif m == "lstm":
                logging.info("Running walk-forward with LSTM...")
                _run_train_lstm(df, config)
            elif m == "lgbm":
                logging.info("Running walk-forward with LightGBM...")
                _run_train_lgbm(df, config)

    elif args.command == "run":
        pair = config["pair"]
        interval = config["interval"]
        data_dir = config.get("data_dir", "data")

        logging.info(f"Loading candles for {pair} {interval}...")
        df = load_candles(pair, interval, data_dir)

        logging.info("Generating features...")
        feat_df = generate_features(df)
        logging.info("Generating labels...")
        label_df = generate_labels(df)
        merged = feat_df.merge(label_df, on="timestamp", how="inner")
        logging.info(f"Merged dataset: {len(merged)} rows")

        models = [args.model] if args.model else None
        results = run_pipeline(merged, models=models)
        print_summary(results)


if __name__ == "__main__":
    main()
