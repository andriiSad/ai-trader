"""Tests for walk-forward split engine."""

import numpy as np
import pandas as pd
import pytest
from src.walk_forward import walk_forward_splits


@pytest.fixture
def synthetic_df():
    """12 months of daily data starting 2024-01-01."""
    dates = pd.date_range("2024-01-01", periods=365, freq="D")
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "timestamp": dates,
            "close": 100 + np.cumsum(rng.standard_normal(365) * 0.5),
            "label": rng.integers(0, 2, 365),
        }
    )


def test_splits_count_3m_1m_1m(synthetic_df):
    folds = walk_forward_splits(synthetic_df, train_window="3ME", test_window="1ME", step="1ME")
    assert len(folds) >= 5


def test_train_precedes_test(synthetic_df):
    folds = walk_forward_splits(synthetic_df, train_window="3ME", test_window="1ME", step="1ME")
    for fold in folds:
        assert fold["train_end"] <= fold["test_start"], (
            f"Fold {fold['fold_id']}: train_end={fold['train_end']} > test_start={fold['test_start']}"
        )


def test_no_data_leakage(synthetic_df):
    folds = walk_forward_splits(synthetic_df, train_window="3ME", test_window="1ME", step="1ME")
    for fold in folds:
        train_mask = fold["train_mask"]
        test_mask = fold["test_mask"]
        overlap = train_mask & test_mask
        assert overlap.sum() == 0, f"Fold {fold['fold_id']} has {overlap.sum()} overlapping rows"


def test_masks_cover_correct_dates(synthetic_df):
    folds = walk_forward_splits(synthetic_df, train_window="3ME", test_window="1ME", step="1ME")
    ts = synthetic_df["timestamp"]
    for fold in folds:
        train_dates = ts[fold["train_mask"]]
        test_dates = ts[fold["test_mask"]]
        assert train_dates.min() >= fold["train_start"]
        assert train_dates.max() < fold["train_end"]
        assert test_dates.min() >= fold["test_start"]
        assert test_dates.max() < fold["test_end"]


def test_fold_ids_sequential(synthetic_df):
    folds = walk_forward_splits(synthetic_df, train_window="3ME", test_window="1ME", step="1ME")
    ids = [f["fold_id"] for f in folds]
    assert ids == list(range(len(folds)))


def test_folds_slide_forward(synthetic_df):
    folds = walk_forward_splits(synthetic_df, train_window="3ME", test_window="1ME", step="1ME")
    for i in range(1, len(folds)):
        assert folds[i]["train_start"] > folds[i - 1]["train_start"]
        assert folds[i]["test_start"] > folds[i - 1]["test_start"]


def test_short_data_returns_empty():
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    df = pd.DataFrame({"timestamp": dates, "close": range(10), "label": [0, 1] * 5})
    folds = walk_forward_splits(df, train_window="3ME", test_window="1ME", step="1ME")
    assert len(folds) == 0


def test_unix_timestamp_conversion():
    dates = pd.date_range("2024-01-01", periods=400, freq="D")
    unix_ts = dates.view("i8") // 10**6  # datetime64[us] → seconds
    df = pd.DataFrame(
        {
            "timestamp": unix_ts,
            "close": range(400),
            "label": [0, 1] * 200,
        }
    )
    folds = walk_forward_splits(df, train_window="3ME", test_window="1ME", step="1ME")
    assert len(folds) >= 1


def test_train_and_test_masks_have_nonzero_rows(synthetic_df):
    folds = walk_forward_splits(synthetic_df, train_window="3ME", test_window="1ME", step="1ME")
    for fold in folds:
        assert fold["train_mask"].sum() > 0
        assert fold["test_mask"].sum() > 0


def test_single_fold_with_custom_windows():
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    df = pd.DataFrame({"timestamp": dates, "close": range(100), "label": [0, 1] * 50})
    folds = walk_forward_splits(df, train_window="1ME", test_window="1ME", step="1ME")
    assert len(folds) >= 1


def test_legacy_m_offset_alias():
    """Legacy '3M' alias should be auto-converted to '3ME'."""
    dates = pd.date_range("2024-01-01", periods=365, freq="D")
    df = pd.DataFrame({"timestamp": dates, "close": range(365), "label": [0, 1] * 182 + [0]})
    folds = walk_forward_splits(df, train_window="3M", test_window="1M", step="1M")
    assert len(folds) >= 1
