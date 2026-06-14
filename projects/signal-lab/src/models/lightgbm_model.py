from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.metrics import compute_metrics


def train_lightgbm(
    X_train: np.ndarray | pd.DataFrame,
    y_train: np.ndarray | pd.Series,
    X_test: np.ndarray | pd.DataFrame,
    y_test: np.ndarray | pd.Series,
) -> dict:
    model = lgb.LGBMClassifier(
        n_estimators=100,
        num_leaves=31,
        learning_rate=0.05,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=5,
        random_state=42,
        objective="binary",
        metric="binary_logloss",
        verbose=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = compute_metrics(y_test, y_pred, y_prob)

    feature_importance = sorted(
        zip(model.feature_name_, model.feature_importances_, strict=True),
        key=lambda x: x[1],
        reverse=True,
    )

    return {
        "model": model,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "metrics": metrics,
        "feature_importance": feature_importance,
    }
