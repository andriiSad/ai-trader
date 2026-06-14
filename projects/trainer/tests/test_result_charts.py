"""Tests for result_charts module."""

import json
import tempfile
from pathlib import Path

import plotly.graph_objects as go
from src.result_charts import (
    accuracy_bar_chart,
    build_results_dashboard,
    confusion_matrix_chart,
    equity_curve_chart,
    feature_importance_chart,
    fold_timeline_chart,
    pair_accuracy_chart,
)


def _make_sample_results() -> dict:
    """Generate synthetic evaluation results for testing."""
    return {
        "overall_accuracy": 0.55,
        "overall_precision": 0.56,
        "overall_recall": 0.54,
        "overall_f1": 0.55,
        "per_fold": [
            {
                "fold_id": 0,
                "train_start": "2024-01-01",
                "train_end": "2024-04-01",
                "test_start": "2024-04-01",
                "test_end": "2024-05-01",
                "train_samples": 1000,
                "test_samples": 300,
                "accuracy": 0.52,
                "precision": 0.53,
                "recall": 0.51,
                "f1": 0.52,
                "confusion_matrix": [[80, 70], [74, 76]],
                "sample_count": 300,
            },
            {
                "fold_id": 1,
                "train_start": "2024-02-01",
                "train_end": "2024-05-01",
                "test_start": "2024-05-01",
                "test_end": "2024-06-01",
                "train_samples": 1000,
                "test_samples": 300,
                "accuracy": 0.58,
                "precision": 0.59,
                "recall": 0.57,
                "f1": 0.58,
                "confusion_matrix": [[90, 60], [66, 84]],
                "sample_count": 300,
            },
        ],
        "per_pair": {
            "BTC_USDT": {
                "accuracy": 0.56,
                "precision": 0.57,
                "recall": 0.55,
                "f1": 0.56,
                "confusion_matrix": [[85, 65], [70, 80]],
                "sample_count": 300,
            },
            "ETH_USDT": {
                "accuracy": 0.54,
                "precision": 0.55,
                "recall": 0.53,
                "f1": 0.54,
                "confusion_matrix": [[85, 65], [73, 77]],
                "sample_count": 300,
            },
        },
        "feature_importance": [
            {"feature": "rsi_14", "importance": 150.5},
            {"feature": "macd_line", "importance": 120.3},
            {"feature": "bb_bandwidth", "importance": 95.1},
            {"feature": "volatility_10", "importance": 80.2},
            {"feature": "return_lag_1", "importance": 70.0},
        ],
    }


class TestAccuracyBarChart:
    def test_returns_figure(self):
        results = _make_sample_results()
        fig = accuracy_bar_chart(results["per_fold"])
        assert isinstance(fig, go.Figure)

    def test_has_bar_trace(self):
        results = _make_sample_results()
        fig = accuracy_bar_chart(results["per_fold"])
        assert len(fig.data) >= 1


class TestFeatureImportanceChart:
    def test_returns_figure(self):
        results = _make_sample_results()
        fig = feature_importance_chart(results["feature_importance"])
        assert isinstance(fig, go.Figure)

    def test_top_n_limits_features(self):
        results = _make_sample_results()
        fig = feature_importance_chart(results["feature_importance"], top_n=3)
        assert isinstance(fig, go.Figure)


class TestConfusionMatrixChart:
    def test_returns_figure(self):
        cm = [[80, 70], [74, 76]]
        fig = confusion_matrix_chart(cm, 0)
        assert isinstance(fig, go.Figure)

    def test_has_heatmap_trace(self):
        cm = [[80, 70], [74, 76]]
        fig = confusion_matrix_chart(cm, 0)
        assert len(fig.data) == 1


class TestEquityCurveChart:
    def test_returns_figure(self):
        results = _make_sample_results()
        fig = equity_curve_chart(results["per_fold"])
        assert isinstance(fig, go.Figure)

    def test_has_scatter_trace(self):
        results = _make_sample_results()
        fig = equity_curve_chart(results["per_fold"])
        assert len(fig.data) >= 1


class TestPairAccuracyChart:
    def test_returns_figure_with_data(self):
        results = _make_sample_results()
        fig = pair_accuracy_chart(results["per_pair"])
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 4

    def test_returns_figure_empty(self):
        fig = pair_accuracy_chart({})
        assert isinstance(fig, go.Figure)


class TestFoldTimelineChart:
    def test_returns_figure(self):
        results = _make_sample_results()
        fig = fold_timeline_chart(results["per_fold"])
        assert isinstance(fig, go.Figure)

    def test_has_traces_per_fold(self):
        results = _make_sample_results()
        fig = fold_timeline_chart(results["per_fold"])
        assert len(fig.data) == len(results["per_fold"])


class TestBuildResultsDashboard:
    def test_creates_html_file(self):
        results = _make_sample_results()
        with tempfile.TemporaryDirectory() as tmpdir:
            results_path = Path(tmpdir) / "evaluation_results.json"
            output_path = Path(tmpdir) / "training_results.html"

            with open(results_path, "w") as f:
                json.dump(results, f)

            build_results_dashboard(str(results_path), str(output_path))
            assert output_path.exists()
            content = output_path.read_text()
            assert "Training Results Dashboard" in content
            assert len(content) > 100

    def test_creates_parent_directories(self):
        results = _make_sample_results()
        with tempfile.TemporaryDirectory() as tmpdir:
            results_path = Path(tmpdir) / "evaluation_results.json"
            output_path = Path(tmpdir) / "subdir" / "training_results.html"

            with open(results_path, "w") as f:
                json.dump(results, f)

            build_results_dashboard(str(results_path), str(output_path))
            assert output_path.exists()
