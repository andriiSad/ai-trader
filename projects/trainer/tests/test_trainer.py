"""Tests for LightGBM training pipeline."""

import numpy as np
import pandas as pd
import pytest
from src.trainer import train_fold


@pytest.fixture
def train_test_data():
    rng = np.random.default_rng(42)
    n_train, n_test = 200, 50
    n_features = 5
    X_train = pd.DataFrame(
        rng.standard_normal((n_train, n_features)), columns=[f"f{i}" for i in range(n_features)]
    )
    y_train = pd.Series(rng.integers(0, 2, n_train))
    X_test = pd.DataFrame(
        rng.standard_normal((n_test, n_features)), columns=[f"f{i}" for i in range(n_features)]
    )
    y_test = pd.Series(rng.integers(0, 2, n_test))
    return X_train, y_train, X_test, y_test


def test_train_fold_returns_expected_keys(train_test_data):
    X_train, y_train, X_test, y_test = train_test_data
    result = train_fold(X_train, y_train, X_test, y_test)
    assert "predictions" in result
    assert "hard_predictions" in result
    assert "model" in result
    assert "feature_importance" in result


def test_predictions_length(train_test_data):
    X_train, y_train, X_test, y_test = train_test_data
    result = train_fold(X_train, y_train, X_test, y_test)
    assert len(result["predictions"]) == len(X_test)
    assert len(result["hard_predictions"]) == len(X_test)


def test_predictions_are_probabilities(train_test_data):
    X_train, y_train, X_test, y_test = train_test_data
    result = train_fold(X_train, y_train, X_test, y_test)
    preds = result["predictions"]
    assert (preds >= 0).all()
    assert (preds <= 1).all()


def test_hard_predictions_binary(train_test_data):
    X_train, y_train, X_test, y_test = train_test_data
    result = train_fold(X_train, y_train, X_test, y_test)
    hp = result["hard_predictions"]
    unique_vals = set(hp.tolist())
    assert unique_vals.issubset({0, 1})


def test_feature_importance_has_all_features(train_test_data):
    X_train, y_train, X_test, y_test = train_test_data
    result = train_fold(X_train, y_train, X_test, y_test)
    fi = result["feature_importance"]
    assert set(fi.keys()) == set(X_train.columns)
    assert all(v >= 0 for v in fi.values())


def test_custom_params_override(train_test_data):
    X_train, y_train, X_test, y_test = train_test_data
    custom = {"num_leaves": 15, "learning_rate": 0.1}
    result = train_fold(X_train, y_train, X_test, y_test, params=custom)
    assert len(result["predictions"]) == len(X_test)
