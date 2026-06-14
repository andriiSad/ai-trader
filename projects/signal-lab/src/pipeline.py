from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from src.scaler import fit_scaler, transform
from src.walk_forward import split
from src.wandb_logger import (
    finish_run,
    init_run,
    log_feature_importance,
    log_metrics,
)

logger = logging.getLogger(__name__)

MODEL_NAMES = {"lr": "Logistic Regression", "lgbm": "LightGBM", "lstm": "LSTM"}


def _train_model(model_key: str, X_train, y_train, X_test, y_test) -> dict:
    if model_key == "lr":
        from src.models.logistic import train_logistic

        return train_logistic(X_train, y_train, X_test, y_test)
    elif model_key == "lgbm":
        from src.models.lightgbm_model import train_lightgbm

        return train_lightgbm(X_train, y_train, X_test, y_test)
    elif model_key == "lstm":
        from src.models.lstm import train_lstm

        return train_lstm(X_train, y_train, X_test, y_test)
    else:
        raise ValueError(f"Unknown model: {model_key}")


def run_pipeline(
    merged: pd.DataFrame,
    models: list[str] | None = None,
    wandb_project: str = "signal-lab",
) -> dict[str, Any]:
    if models is None:
        models = ["lr", "lgbm", "lstm"]

    feature_cols = [c for c in merged.columns if c not in ("timestamp", "label")]

    results: dict[str, Any] = {}

    for model_key in models:
        model_name = MODEL_NAMES.get(model_key, model_key)
        logger.info(f"Training {model_name}...")
        model_results: list[dict[str, Any]] = []

        for fold_idx, (train_df, test_df) in enumerate(split(merged)):
            logger.info(f"  Fold {fold_idx}")

            scaler = fit_scaler(train_df[feature_cols])
            X_train = transform(train_df[feature_cols], scaler).values
            X_test = transform(test_df[feature_cols], scaler).values
            y_train = train_df["label"].values[: len(X_train)]
            y_test = test_df["label"].values[: len(X_test)]

            init_run(
                project=wandb_project,
                model_name=model_key,
                fold_idx=fold_idx,
                config={"model": model_key, "fold": fold_idx},
            )

            result = _train_model(model_key, X_train, y_train, X_test, y_test)
            metrics = result["metrics"]

            log_metrics(metrics, step=fold_idx)

            if "feature_importance" in result:
                names = [fi[0] for fi in result["feature_importance"]]
                importances = [fi[1] for fi in result["feature_importance"]]
                log_feature_importance(names, importances)

            finish_run()

            model_results.append({"fold": fold_idx, "metrics": metrics})
            logger.info(
                f"  Fold {fold_idx}: accuracy={metrics['accuracy']:.4f}, f1={metrics['f1']:.4f}"
            )

        if model_results:
            avg_acc = sum(r["metrics"]["accuracy"] for r in model_results) / len(model_results)
            avg_f1 = sum(r["metrics"]["f1"] for r in model_results) / len(model_results)
            logger.info(f"{model_name}: avg accuracy={avg_acc:.4f}, avg f1={avg_f1:.4f}")

        results[model_key] = {
            "model_name": model_name,
            "folds": model_results,
            "avg_accuracy": (
                sum(r["metrics"]["accuracy"] for r in model_results) / len(model_results)
                if model_results
                else 0.0
            ),
            "avg_f1": (
                sum(r["metrics"]["f1"] for r in model_results) / len(model_results)
                if model_results
                else 0.0
            ),
        }

    return results


def print_summary(results: dict[str, Any]) -> None:
    print(f"\n{'=' * 60}")
    print(f"{'Model':<25} {'Avg Accuracy':>14} {'Avg F1':>10} {'Folds':>6}")
    print(f"{'-' * 60}")
    for _key, info in results.items():
        name = info["model_name"]
        acc = info["avg_accuracy"]
        f1 = info["avg_f1"]
        n_folds = len(info["folds"])
        print(f"{name:<25} {acc:>14.4f} {f1:>10.4f} {n_folds:>6}")
    print(f"{'=' * 60}")
