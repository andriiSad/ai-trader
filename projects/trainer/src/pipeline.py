"""Full pipeline orchestrator: features → train → evaluate → visualize."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from .data_loader import load_all_pairs
from .features import generate_features
from .labels import generate_labels
from .result_charts import build_results_dashboard
from .trainer import run_walk_forward
from .walk_forward import walk_forward_splits

logger = logging.getLogger(__name__)


def run_features(config: dict) -> pd.DataFrame:
    """Load data, generate features + labels, return model-ready DataFrame."""
    data_dir = config["data_dir"]
    pairs = config["pairs"]
    interval = config.get("interval", "5m")
    horizon = config.get("features", {}).get("prediction_horizon", 5)
    threshold = config.get("features", {}).get("label_threshold", 0.0)

    pair_dfs = load_all_pairs(data_dir, pairs, interval)

    frames: list[pd.DataFrame] = []
    for pair, df in pair_dfs.items():
        if df.empty:
            logger.warning("No data for %s, skipping", pair)
            continue
        df = df.copy()
        df["pair"] = pair
        labels = generate_labels(df, horizon=horizon, threshold=threshold)
        featured = generate_features(df)
        featured["label"] = labels.reindex(featured.index).values
        featured = featured.dropna(subset=["label"])
        frames.append(featured)

    if not frames:
        raise RuntimeError("No data loaded for any pair")

    combined = pd.concat(frames, ignore_index=True)
    logger.info("Features ready: %d rows, %d columns", len(combined), len(combined.columns))
    return combined


def run_train(config: dict, df: pd.DataFrame) -> dict:
    """Run walk-forward training, return evaluation results."""
    wf_config = config.get("walk_forward", {})
    splits = walk_forward_splits(
        df,
        train_window=wf_config.get("train_window", "3ME"),
        test_window=wf_config.get("test_window", "1ME"),
        step=wf_config.get("step", "1ME"),
    )

    if not splits:
        raise RuntimeError("No walk-forward splits generated")

    logger.info("Generated %d walk-forward splits", len(splits))

    feature_cols = [c for c in df.columns if c not in ("timestamp", "pair", "label")]
    results = run_walk_forward(df, feature_cols, splits)
    logger.info("Training complete. Overall accuracy: %.3f", results["overall_accuracy"])
    return results


def run_evaluate(config: dict, results: dict) -> str:
    """Save evaluation results to JSON, return path."""
    results_path = config.get("output", {}).get(
        "results_path", "data/reports/evaluation_results.json"
    )
    out = Path(results_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Evaluation results saved to %s", out)
    return str(out)


def run_visualize(config: dict) -> None:
    """Build results dashboard + data exploration dashboard."""
    results_path = config.get("output", {}).get(
        "results_path", "data/reports/evaluation_results.json"
    )
    results_html = str(Path(results_path).parent / "training_results.html")
    build_results_dashboard(results_path, results_html)
    logger.info("Training results dashboard saved to %s", results_html)

    from .explore_charts import build_dashboard

    data_dir = config["data_dir"]
    pairs = config["pairs"]
    exploration_html = str(Path(results_path).parent / "data_exploration.html")
    build_dashboard(data_dir, pairs, exploration_html)
    logger.info("Data exploration dashboard saved to %s", exploration_html)


def run_full_pipeline(config: dict) -> None:
    """Run all steps in sequence: features → train → evaluate → visualize."""
    logger.info("=== Step 1/4: Feature Engineering ===")
    df = run_features(config)

    logger.info("=== Step 2/4: Walk-Forward Training ===")
    results = run_train(config, df)

    logger.info("=== Step 3/4: Evaluate & Save ===")
    results_path = run_evaluate(config, results)

    logger.info("=== Step 4/4: Visualize ===")
    run_visualize(config)

    logger.info("Pipeline complete. Results: %s", results_path)
