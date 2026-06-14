"""Tests for save_predictions function."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from src.pipeline import save_predictions


def _make_predictions_rows(n_per_fold: int = 10, n_folds: int = 2) -> list[dict]:
    """Generate synthetic prediction rows mimicking run_walk_forward output."""
    rng = np.random.default_rng(42)
    rows = []
    base_ts = pd.Timestamp("2024-01-01")
    for fold_id in range(n_folds):
        test_start = base_ts + pd.Timedelta(days=fold_id * 30)
        test_end = test_start + pd.Timedelta(days=30)
        for i in range(n_per_fold):
            ts = test_start + pd.Timedelta(hours=i * 2)
            rows.append(
                {
                    "timestamp": ts,
                    "pair": "BTC_USDT" if i % 2 == 0 else "ETH_USDT",
                    "probability": rng.uniform(0, 1),
                    "hard_prediction": rng.integers(0, 2),
                    "label": rng.integers(0, 2),
                    "fold_id": fold_id,
                    "fold_test_start": test_start,
                    "fold_test_end": test_end,
                }
            )
    return rows


class TestSavePredictions:
    def test_creates_csv_file(self):
        rows = _make_predictions_rows()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "predictions.csv")
            save_predictions(rows, path)
            assert Path(path).exists()

    def test_csv_columns(self):
        rows = _make_predictions_rows()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "predictions.csv")
            save_predictions(rows, path)
            df = pd.read_csv(path)
            expected_cols = [
                "timestamp",
                "pair",
                "probability",
                "hard_prediction",
                "label",
                "fold_id",
                "fold_test_start",
                "fold_test_end",
            ]
            assert list(df.columns) == expected_cols

    def test_row_count_matches_input(self):
        rows = _make_predictions_rows(n_per_fold=15, n_folds=3)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "predictions.csv")
            save_predictions(rows, path)
            df = pd.read_csv(path)
            assert len(df) == 15 * 3

    def test_ordered_by_timestamp_ascending(self):
        rows = _make_predictions_rows(n_per_fold=20, n_folds=3)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "predictions.csv")
            save_predictions(rows, path)
            df = pd.read_csv(path)
            timestamps = pd.to_datetime(df["timestamp"])
            assert (timestamps.diff().dropna() >= pd.Timedelta(0)).all()

    def test_fold_metadata_present(self):
        rows = _make_predictions_rows(n_per_fold=5, n_folds=2)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "predictions.csv")
            save_predictions(rows, path)
            df = pd.read_csv(path)
            assert df["fold_id"].nunique() == 2
            assert df["fold_test_start"].notna().all()
            assert df["fold_test_end"].notna().all()

    def test_all_folds_and_pairs_included(self):
        rows = _make_predictions_rows(n_per_fold=10, n_folds=3)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "predictions.csv")
            save_predictions(rows, path)
            df = pd.read_csv(path)
            assert set(df["fold_id"].unique()) == {0, 1, 2}
            assert set(df["pair"].unique()) == {"BTC_USDT", "ETH_USDT"}

    def test_creates_parent_directories(self):
        rows = _make_predictions_rows(n_per_fold=2, n_folds=1)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "nested" / "dir" / "predictions.csv")
            save_predictions(rows, path)
            assert Path(path).exists()

    def test_empty_rows_writes_header_only(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "predictions.csv")
            save_predictions([], path)
            df = pd.read_csv(path)
            assert len(df) == 0
            expected_cols = [
                "timestamp",
                "pair",
                "probability",
                "hard_prediction",
                "label",
                "fold_id",
                "fold_test_start",
                "fold_test_end",
            ]
            assert list(df.columns) == expected_cols
