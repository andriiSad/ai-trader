import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import run_pipeline


def _synthetic_merged(n: int = 2000, n_features: int = 5, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="4h")
    data = {"timestamp": dates.astype(np.int64).values // 10**3}
    for i in range(n_features):
        data[f"feat_{i}"] = rng.normal(0, 1, n)
    data["label"] = rng.integers(0, 2, n)
    return pd.DataFrame(data)


def _mock_train_result(X_train, y_train, X_test, y_test) -> dict:
    rng = np.random.default_rng(0)
    y_pred = rng.integers(0, 2, len(y_test))
    y_prob = rng.uniform(0, 1, len(y_test))
    return {
        "model": MagicMock(),
        "y_pred": y_pred,
        "y_prob": y_prob,
        "metrics": {
            "accuracy": 0.55,
            "f1": 0.50,
            "precision": 0.52,
            "recall": 0.48,
            "confusion_matrix": [[10, 5], [8, 12]],
        },
    }


@pytest.fixture(autouse=True)
def _mock_wandb(monkeypatch):
    wandb_mod = ModuleType("wandb")
    wandb_mod.init = MagicMock()
    wandb_mod.log = MagicMock()
    wandb_mod.finish = MagicMock()
    wandb_mod.Table = MagicMock(return_value=MagicMock())
    wandb_mod.plot = MagicMock()
    wandb_mod.plot.bar = MagicMock(return_value=MagicMock())
    wandb_mod.Artifact = MagicMock(return_value=MagicMock())

    monkeypatch.setitem(sys.modules, "wandb", wandb_mod)


def test_run_pipeline_returns_model_keys(monkeypatch):
    monkeypatch.setattr(
        "src.pipeline._train_model",
        lambda model_key, X_train, y_train, X_test, y_test: _mock_train_result(
            X_train, y_train, X_test, y_test
        ),
    )
    merged = _synthetic_merged()
    results = run_pipeline(merged, models=["lr"])
    assert "lr" in results


def test_run_pipeline_model_selection(monkeypatch):
    call_log = []

    def _tracking_train(model_key, X_train, y_train, X_test, y_test):
        call_log.append(model_key)
        return _mock_train_result(X_train, y_train, X_test, y_test)

    monkeypatch.setattr("src.pipeline._train_model", _tracking_train)
    merged = _synthetic_merged()
    run_pipeline(merged, models=["lr"])
    assert all(m == "lr" for m in call_log)
    assert len(call_log) > 0


def test_run_pipeline_returns_fold_results(monkeypatch):
    monkeypatch.setattr(
        "src.pipeline._train_model",
        lambda model_key, X_train, y_train, X_test, y_test: _mock_train_result(
            X_train, y_train, X_test, y_test
        ),
    )
    merged = _synthetic_merged()
    results = run_pipeline(merged, models=["lr"])
    assert "folds" in results["lr"]
    assert len(results["lr"]["folds"]) > 0
    assert "metrics" in results["lr"]["folds"][0]


def test_run_pipeline_avg_metrics(monkeypatch):
    monkeypatch.setattr(
        "src.pipeline._train_model",
        lambda model_key, X_train, y_train, X_test, y_test: _mock_train_result(
            X_train, y_train, X_test, y_test
        ),
    )
    merged = _synthetic_merged()
    results = run_pipeline(merged, models=["lr"])
    assert "avg_accuracy" in results["lr"]
    assert "avg_f1" in results["lr"]
    assert results["lr"]["avg_accuracy"] == pytest.approx(0.55)


def test_run_pipeline_multiple_models(monkeypatch):
    monkeypatch.setattr(
        "src.pipeline._train_model",
        lambda model_key, X_train, y_train, X_test, y_test: _mock_train_result(
            X_train, y_train, X_test, y_test
        ),
    )
    merged = _synthetic_merged()
    results = run_pipeline(merged, models=["lr", "lgbm"])
    assert "lr" in results
    assert "lgbm" in results
    assert len(results) == 2


def test_run_pipeline_default_models(monkeypatch):
    monkeypatch.setattr(
        "src.pipeline._train_model",
        lambda model_key, X_train, y_train, X_test, y_test: _mock_train_result(
            X_train, y_train, X_test, y_test
        ),
    )
    merged = _synthetic_merged()
    results = run_pipeline(merged, models=None)
    assert "lr" in results
    assert "lgbm" in results
    assert "lstm" in results


def test_wandb_logger_noop_when_not_initialized():
    from src.wandb_logger import finish_run, log_feature_importance, log_metrics

    log_metrics({"accuracy": 0.5, "f1": 0.4})
    log_feature_importance(["a", "b"], [0.5, 0.5])
    finish_run()
