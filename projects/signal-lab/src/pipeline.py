from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from src.scaler import fit_scaler, transform
from src.walk_forward import split

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

            train_features = train_df[feature_cols]
            test_features = test_df[feature_cols]

            scaler = fit_scaler(train_features)
            X_train_scaled = transform(train_features, scaler)
            X_test_scaled = transform(test_features, scaler)

            y_train = train_df.loc[X_train_scaled.index, "label"].values
            y_test = test_df.loc[X_test_scaled.index, "label"].values
            X_train = X_train_scaled.values
            X_test = X_test_scaled.values

            result = _train_model(model_key, X_train, y_train, X_test, y_test)
            metrics = result["metrics"]

            fold_data: dict[str, Any] = {"fold": fold_idx, "metrics": metrics}
            if "feature_importance" in result:
                fold_data["feature_importance"] = result["feature_importance"]

            model_results.append(fold_data)
            logger.info(
                f"  Fold {fold_idx}: accuracy={metrics['accuracy']:.4f}, f1={metrics['f1']:.4f}"
            )

        avg_acc = (
            sum(r["metrics"]["accuracy"] for r in model_results) / len(model_results)
            if model_results
            else 0.0
        )
        avg_f1 = (
            sum(r["metrics"]["f1"] for r in model_results) / len(model_results)
            if model_results
            else 0.0
        )
        logger.info(f"{model_name}: avg accuracy={avg_acc:.4f}, avg f1={avg_f1:.4f}")

        results[model_key] = {
            "model_name": model_name,
            "folds": model_results,
            "avg_accuracy": avg_acc,
            "avg_f1": avg_f1,
        }

    _upload_to_wandb(results, wandb_project)

    return results


def _upload_to_wandb(results: dict[str, Any], project: str) -> None:
    logger.info("Uploading results to wandb...")
    from src.wandb_logger import (
        finish_run,
        init_run,
        log_feature_importance,
        log_metrics,
    )

    for model_key, info in results.items():
        init_run(
            project=project,
            model_name=model_key,
            config={
                "model": model_key,
                "avg_accuracy": info["avg_accuracy"],
                "avg_f1": info["avg_f1"],
                "n_folds": len(info["folds"]),
            },
        )

        for fold_data in info["folds"]:
            fold_idx = fold_data["fold"]
            metrics = fold_data["metrics"]
            log_metrics(
                {
                    "fold": fold_idx,
                    "accuracy": metrics["accuracy"],
                    "f1": metrics["f1"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                },
                step=fold_idx,
            )

            if "feature_importance" in fold_data:
                fi = fold_data["feature_importance"]
                log_feature_importance([f[0] for f in fi], [f[1] for f in fi])

        log_metrics(
            {
                "avg_accuracy": info["avg_accuracy"],
                "avg_f1": info["avg_f1"],
            }
        )
        finish_run()

    _upload_comparison_table(results, project)


def _upload_comparison_table(results: dict[str, Any], project: str) -> None:
    try:
        import wandb

        wandb.init(
            project=project,
            name="model-comparison",
            settings=wandb.Settings(_disable_stats=True, _disable_meta=True),
        )
        columns = ["Model", "Avg Accuracy", "Avg F1", "Folds"]
        data = []
        for _key, info in results.items():
            data.append(
                [info["model_name"], info["avg_accuracy"], info["avg_f1"], len(info["folds"])]
            )
        table = wandb.Table(columns=columns, data=data)
        wandb.log({"model_comparison": table})
        wandb.log(
            {
                "accuracy_chart": wandb.plot.bar(
                    table, "Model", "Avg Accuracy", title="Accuracy by Model"
                )
            }
        )
        wandb.log({"f1_chart": wandb.plot.bar(table, "Model", "Avg F1", title="F1 by Model")})
        wandb.finish()
    except Exception as e:
        logger.warning(f"wandb comparison table failed: {e}")


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
