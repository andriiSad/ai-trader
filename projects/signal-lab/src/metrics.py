from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def compute_metrics(
    y_true: np.ndarray | list,
    y_pred: np.ndarray | list,
    y_prob: np.ndarray | list | None = None,
) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def print_fold_summary(model_name: str, fold_idx: int, metrics: dict) -> None:
    print(f"\n{'=' * 50}")
    print(f"  {model_name} — Fold {fold_idx}")
    print(f"{'=' * 50}")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  F1:        {metrics['f1']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  Confusion Matrix: {metrics['confusion_matrix']}")
    print(f"{'=' * 50}")
