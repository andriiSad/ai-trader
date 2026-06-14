import numpy as np
from src.models.lightgbm_model import train_lightgbm


def _synthetic_data(n: int = 20, features: int = 3, seed: int = 42):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n, features))
    w = rng.normal(0, 1, features)
    prob = 1 / (1 + np.exp(-X @ w))
    y = (prob > 0.5).astype(int)
    split = n * 4 // 5
    return X[:split], y[:split], X[split:], y[split:]


def test_train_lightgbm_returns_required_keys():
    X_train, y_train, X_test, y_test = _synthetic_data()
    result = train_lightgbm(X_train, y_train, X_test, y_test)
    assert "model" in result
    assert "y_pred" in result
    assert "y_prob" in result
    assert "metrics" in result
    assert "feature_importance" in result


def test_train_lightgbm_predictions_shape():
    X_train, y_train, X_test, y_test = _synthetic_data()
    result = train_lightgbm(X_train, y_train, X_test, y_test)
    assert len(result["y_pred"]) == len(y_test)
    assert len(result["y_prob"]) == len(y_test)


def test_train_lightgbm_feature_importance_sorted_descending():
    X_train, y_train, X_test, y_test = _synthetic_data()
    result = train_lightgbm(X_train, y_train, X_test, y_test)
    importances = [imp for _, imp in result["feature_importance"]]
    assert importances == sorted(importances, reverse=True)
