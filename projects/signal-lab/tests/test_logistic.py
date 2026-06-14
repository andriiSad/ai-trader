import numpy as np
from src.metrics import compute_metrics
from src.models.logistic import train_logistic


def _synthetic_data(n: int = 20, features: int = 3, seed: int = 42):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n, features))
    w = rng.normal(0, 1, features)
    prob = 1 / (1 + np.exp(-X @ w))
    y = (prob > 0.5).astype(int)
    split = n * 4 // 5
    return X[:split], y[:split], X[split:], y[split:]


def test_train_logistic_returns_required_keys():
    X_train, y_train, X_test, y_test = _synthetic_data()
    result = train_logistic(X_train, y_train, X_test, y_test)
    assert "model" in result
    assert "y_pred" in result
    assert "y_prob" in result
    assert "metrics" in result


def test_train_logistic_predictions_shape():
    X_train, y_train, X_test, y_test = _synthetic_data()
    result = train_logistic(X_train, y_train, X_test, y_test)
    assert len(result["y_pred"]) == len(y_test)
    assert len(result["y_prob"]) == len(y_test)


def test_compute_metrics_keys():
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1])
    y_prob = np.array([0.1, 0.9, 0.4, 0.2, 0.8])
    metrics = compute_metrics(y_true, y_pred, y_prob)
    assert "accuracy" in metrics
    assert "f1" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "confusion_matrix" in metrics


def test_compute_metrics_known_values():
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 0, 0, 1])
    y_prob = np.array([0.9, 0.3, 0.2, 0.8])
    metrics = compute_metrics(y_true, y_pred, y_prob)
    assert metrics["accuracy"] == 0.5
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["f1"] == 0.5
    assert metrics["confusion_matrix"] == [[1, 1], [1, 1]]


def test_compute_metrics_perfect():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.9, 0.8])
    metrics = compute_metrics(y_true, y_pred, y_prob)
    assert metrics["accuracy"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
