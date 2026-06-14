from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def init_run(
    project: str = "signal-lab",
    model_name: str = "model",
    config: dict[str, Any] | None = None,
) -> None:
    try:
        import wandb

        if wandb.run is not None:
            wandb.finish()
        wandb.init(
            project=project,
            name=model_name,
            config=config or {},
            settings=wandb.Settings(
                _disable_stats=True,
                _disable_meta=True,
            ),
        )
    except Exception as e:
        logger.warning(f"wandb init_run failed (no-op): {e}")


def log_metrics(metrics: dict[str, Any], step: int | None = None) -> None:
    try:
        import wandb

        wandb.log(metrics, step=step)
    except Exception as e:
        logger.warning(f"wandb log_metrics failed (no-op): {e}")


def log_confusion_matrix(y_true, y_pred, fold_idx: int) -> None:
    try:
        import numpy as np
        import wandb
        cm = np.array(wandb.confusion_matrix(y_true=y_true, preds=y_pred))
        wandb.log({f"confusion_matrix_fold_{fold_idx}": cm}, step=fold_idx)
    except Exception:
        pass


def log_feature_importance(feature_names: list[str], importances: list[float]) -> None:
    try:
        import wandb

        data = [[name, imp] for name, imp in zip(feature_names, importances, strict=True)]
        table = wandb.Table(columns=["feature", "importance"], data=data)
        wandb.log(
            {
                "feature_importance": wandb.plot.bar(
                    table, "feature", "importance", title="Feature Importance"
                )
            }
        )
    except Exception as e:
        logger.warning(f"wandb log_feature_importance failed (no-op): {e}")


def finish_run() -> None:
    try:
        import wandb

        wandb.finish()
    except Exception as e:
        logger.warning(f"wandb finish_run failed (no-op): {e}")
