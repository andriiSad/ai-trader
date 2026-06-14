from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.metrics import compute_metrics


def train_logistic(
    X_train: np.ndarray | pd.DataFrame,
    y_train: np.ndarray | pd.Series,
    X_test: np.ndarray | pd.DataFrame,
    y_test: np.ndarray | pd.Series,
) -> dict:
    model = LogisticRegression(l1_ratio=0, C=1.0, max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = compute_metrics(y_test, y_pred, y_prob)

    return {
        "model": model,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "metrics": metrics,
    }
