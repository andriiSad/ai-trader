"""LightGBM training pipeline with walk-forward evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import compute_metrics, compute_pair_metrics

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

DEFAULT_PARAMS: dict = {
    "objective": "binary",
    "metric": "binary_logloss",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
    "seed": 42,
}


def train_fold(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    params: dict | None = None,
) -> dict:
    """Train LightGBM on one fold, return predictions and metrics.

    Parameters
    ----------
    X_train, y_train : training data
    X_test, y_test : test data
    params : dict, optional
        LightGBM parameters. Defaults to DEFAULT_PARAMS.

    Returns
    -------
    dict
        Keys: predictions, hard_predictions, model, feature_importance.
    """
    if lgb is None:
        raise ImportError("lightgbm is not installed")

    if params is None:
        params = DEFAULT_PARAMS.copy()
    else:
        merged = DEFAULT_PARAMS.copy()
        merged.update(params)
        params = merged

    train_data = lgb.Dataset(X_train, label=y_train)
    model = lgb.train(params, train_data, num_boost_round=200)

    predictions = model.predict(X_test)
    hard_predictions = (predictions >= 0.5).astype(int)

    importance = model.feature_importance(importance_type="gain")
    feature_names = X_train.columns.tolist()
    feature_importance = dict(zip(feature_names, importance.tolist(), strict=True))

    return {
        "predictions": predictions,
        "hard_predictions": hard_predictions,
        "model": model,
        "feature_importance": feature_importance,
    }


def run_walk_forward(df: pd.DataFrame, feature_cols: list[str], splits: list[dict]) -> dict:
    """Run full walk-forward evaluation.

    Parameters
    ----------
    df : pd.DataFrame
        Model-ready DataFrame with features, label, timestamp, and optionally pair.
    feature_cols : list[str]
        Feature column names.
    splits : list[dict]
        Output from ``walk_forward_splits``.

    Returns
    -------
    dict
        Evaluation results matching evaluation_results.json format.
    """
    per_fold: list[dict] = []
    all_predictions = np.zeros(len(df))
    all_indices = np.array([], dtype=int)
    feature_importance_accum: dict[str, float] = {}
    pairs = sorted(df["pair"].unique().tolist()) if "pair" in df.columns else []
    prediction_rows: list[dict] = []

    for fold in splits:
        train_mask = fold["train_mask"]
        test_mask = fold["test_mask"]

        X_train = df.loc[train_mask, feature_cols]
        y_train = df.loc[train_mask, "label"]
        X_test = df.loc[test_mask, feature_cols]
        y_test = df.loc[test_mask, "label"]

        result = train_fold(X_train, y_train, X_test, y_test)

        test_indices = np.where(test_mask.values)[0]
        all_predictions[test_indices] = result["predictions"]
        all_indices = np.concatenate([all_indices, test_indices])

        test_df = df.loc[test_mask]
        for i, idx in enumerate(test_df.index):
            row = {
                "timestamp": test_df.loc[idx, "timestamp"],
                "pair": test_df.loc[idx, "pair"] if "pair" in test_df.columns else "",
                "probability": float(result["predictions"][i]),
                "hard_prediction": int(result["hard_predictions"][i]),
                "label": int(test_df.loc[idx, "label"]),
                "fold_id": fold["fold_id"],
                "fold_test_start": fold["test_start"],
                "fold_test_end": fold["test_end"],
            }
            prediction_rows.append(row)

        metrics = compute_metrics(y_test.values, result["predictions"])
        fold_result = {
            "fold_id": fold["fold_id"],
            "train_start": str(fold["train_start"]),
            "train_end": str(fold["train_end"]),
            "test_start": str(fold["test_start"]),
            "test_end": str(fold["test_end"]),
            "train_samples": int(train_mask.sum()),
            "test_samples": int(test_mask.sum()),
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "confusion_matrix": metrics["confusion_matrix"],
            "sample_count": metrics["sample_count"],
        }
        per_fold.append(fold_result)

        for feat, imp in result["feature_importance"].items():
            feature_importance_accum[feat] = feature_importance_accum.get(feat, 0.0) + imp

    overall_mask = np.zeros(len(df), dtype=bool)
    overall_mask[all_indices] = True
    overall_y = df.loc[overall_mask, "label"].values
    overall_pred = all_predictions[all_indices]
    overall_metrics = compute_metrics(overall_y, overall_pred)

    per_pair = {}
    if pairs:
        pair_df = df.loc[overall_mask].copy()
        per_pair = compute_pair_metrics(pair_df, overall_pred, pairs)

    num_folds = len(per_fold) if per_fold else 1
    feature_importance_list = sorted(
        [{"feature": k, "importance": v / num_folds} for k, v in feature_importance_accum.items()],
        key=lambda x: x["importance"],
        reverse=True,
    )

    return {
        "overall_accuracy": overall_metrics["accuracy"],
        "overall_precision": overall_metrics["precision"],
        "overall_recall": overall_metrics["recall"],
        "overall_f1": overall_metrics["f1"],
        "per_fold": per_fold,
        "per_pair": per_pair,
        "feature_importance": feature_importance_list,
        "prediction_rows": prediction_rows,
    }
