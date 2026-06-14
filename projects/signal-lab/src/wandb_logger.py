from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def init_run(
    project: str = "signal-lab",
    model_name: str = "model",
    fold_idx: int = 0,
    config: dict[str, Any] | None = None,
) -> None:
    try:
        import wandb

        wandb.init(
            project=project,
            name=f"{model_name}-fold-{fold_idx}",
            config=config or {},
            reinit=True,
        )
    except Exception as e:
        logger.warning(f"wandb init_run failed (no-op): {e}")


def log_metrics(metrics: dict[str, Any], step: int | None = None) -> None:
    try:
        import wandb

        log_dict: dict[str, Any] = {}
        for key in ("accuracy", "f1", "precision", "recall", "loss"):
            if key in metrics:
                log_dict[key] = metrics[key]

        if "confusion_matrix" in metrics:
            cm = metrics["confusion_matrix"]
            log_dict["confusion_matrix"] = wandb.Table(
                columns=["predicted_0", "predicted_1"],
                data=cm,
            )

        wandb.log(log_dict, step=step)
    except Exception as e:
        logger.warning(f"wandb log_metrics failed (no-op): {e}")


def log_artifact(
    name: str, artifact_type: str, path: str, metadata: dict[str, Any] | None = None
) -> None:
    try:
        import wandb

        artifact = wandb.Artifact(name=name, type=artifact_type, metadata=metadata or {})
        artifact.add_file(path)
        wandb.log_artifact(artifact)
    except Exception as e:
        logger.warning(f"wandb log_artifact failed (no-op): {e}")


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
