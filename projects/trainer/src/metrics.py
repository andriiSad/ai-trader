"""Evaluation metrics for binary classification."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_metrics(y_true, y_pred_prob, threshold: float = 0.5) -> dict:
    """Compute classification metrics from probabilities.

    Parameters
    ----------
    y_true : array-like
        Ground-truth binary labels.
    y_pred_prob : array-like
        Predicted probabilities.
    threshold : float
        Decision threshold for hard predictions.

    Returns
    -------
    dict
        Keys: accuracy, precision, recall, f1, confusion_matrix, sample_count.
    """
    y_true = np.asarray(y_true)
    y_pred_prob = np.asarray(y_pred_prob)
    y_pred = (y_pred_prob >= threshold).astype(int)

    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "sample_count": len(y_true),
    }


def compute_pair_metrics(
    df: pd.DataFrame, predictions: np.ndarray, pairs: list[str]
) -> dict[str, dict]:
    """Compute per-pair metrics from predictions aligned with DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``pair`` and ``label`` columns.
    predictions : np.ndarray
        Predicted probabilities, same length as ``df``.
    pairs : list[str]
        Pair names to compute metrics for.

    Returns
    -------
    dict[str, dict]
        Mapping of pair name → metrics dict (from ``compute_metrics``).
    """
    result: dict[str, dict] = {}
    for pair in pairs:
        mask = df["pair"] == pair
        if mask.sum() == 0:
            continue
        y_true = df.loc[mask, "label"].values
        y_pred = predictions[mask.values]
        pair_metrics = compute_metrics(y_true, y_pred)
        result[pair] = pair_metrics
    return result
