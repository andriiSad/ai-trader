"""Tests for pipeline module."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from src.pipeline import run_evaluate, run_features, run_full_pipeline, run_train


def _make_sample_config(tmpdir: str) -> dict:
    """Generate a minimal config pointing to synthetic data."""
    return {
        "pairs": ["TEST_PAIR"],
        "interval": "5m",
        "data_dir": tmpdir,
        "features": {"prediction_horizon": 5, "label_threshold": 0.0},
        "walk_forward": {"train_window": "3h", "test_window": "1h", "step": "1h"},
        "model": {"num_leaves": 31, "learning_rate": 0.05},
        "output": {"results_path": str(Path(tmpdir) / "reports" / "evaluation_results.json")},
    }


def _write_sample_csv(path: Path, n: int = 500) -> None:
    """Write a synthetic candle CSV."""
    rng = np.random.default_rng(42)
    timestamps = np.arange(1704067200, 1704067200 + n * 300, 300)[:n]
    close = 100 + np.cumsum(rng.normal(0, 0.5, n))
    high = close + abs(rng.normal(0, 0.3, n))
    low = close - abs(rng.normal(0, 0.3, n))
    opn = close + rng.normal(0, 0.1, n)
    volume = rng.uniform(1, 100, n)
    deal = rng.uniform(100, 10000, n)

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": opn,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "deal": deal,
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


class TestRunFeatures:
    def test_returns_dataframe_with_expected_columns(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "TEST_PAIR" / "5m.csv"
            _write_sample_csv(csv_path)

            config = _make_sample_config(tmpdir)
            df = run_features(config)

            assert isinstance(df, pd.DataFrame)
            assert "label" in df.columns
            assert "timestamp" in df.columns
            assert "pair" in df.columns
            assert len(df) > 0

    def test_raises_on_empty_data(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = _make_sample_config(tmpdir)
            with pytest.raises(RuntimeError, match="No data loaded"):
                run_features(config)


class TestRunTrain:
    def test_returns_results_dict(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "TEST_PAIR" / "5m.csv"
            _write_sample_csv(csv_path, n=500)

            config = _make_sample_config(tmpdir)
            df = run_features(config)
            results = run_train(config, df)

            assert "overall_accuracy" in results
            assert "per_fold" in results
            assert "feature_importance" in results
            assert isinstance(results["per_fold"], list)


class TestRunEvaluate:
    def test_saves_json_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = _make_sample_config(tmpdir)
            results = {
                "overall_accuracy": 0.55,
                "per_fold": [],
                "per_pair": {},
                "feature_importance": [],
            }
            path = run_evaluate(config, results)
            assert Path(path).exists()
            with open(path) as f:
                data = json.load(f)
            assert data["overall_accuracy"] == 0.55


class TestRunFullPipeline:
    def test_completes_without_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "TEST_PAIR" / "5m.csv"
            _write_sample_csv(csv_path, n=500)

            config = _make_sample_config(tmpdir)
            run_full_pipeline(config)

            results_path = Path(tmpdir) / "reports" / "evaluation_results.json"
            assert results_path.exists()
