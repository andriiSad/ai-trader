"""Tests for label generation module."""

import pandas as pd
import pytest
from src.labels import generate_labels


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "close": [
                100.0,
                102.0,
                101.0,
                105.0,
                103.0,
                110.0,
                108.0,
                107.0,
                109.0,
                112.0,
                115.0,
                113.0,
                111.0,
                108.0,
                106.0,
            ]
        }
    )


def test_label_up(sample_df):
    labels = generate_labels(sample_df, horizon=1, threshold=0.0)
    # close[0]=100 → close[1]=102 → up → 1
    assert labels.iloc[0] == 1.0
    # close[1]=102 → close[2]=101 → down → 0
    assert labels.iloc[1] == 0.0


def test_label_down(sample_df):
    labels = generate_labels(sample_df, horizon=1, threshold=0.0)
    assert labels.iloc[2] == 1.0  # 101→105 = up


def test_threshold(sample_df):
    labels = generate_labels(sample_df, horizon=1, threshold=0.01)
    # close[0]=100 → close[1]=102, return=0.02 > 0.01 → 1
    assert labels.iloc[0] == 1.0
    # close[2]=101 → close[3]=105, return≈0.0396 > 0.01 → 1
    assert labels.iloc[2] == 1.0
    # close[1]=102 → close[2]=101, return≈-0.0098, not > 0.01 → 0
    assert labels.iloc[1] == 0.0


def test_last_horizon_nan(sample_df):
    horizon = 3
    labels = generate_labels(sample_df, horizon=horizon)
    assert labels.iloc[-horizon:].isna().all()
    assert not labels.iloc[:-horizon].isna().any()


def test_label_length(sample_df):
    labels = generate_labels(sample_df, horizon=5)
    assert len(labels) == len(sample_df)


def test_determinism(sample_df):
    labels1 = generate_labels(sample_df, horizon=5, threshold=0.0)
    labels2 = generate_labels(sample_df, horizon=5, threshold=0.0)
    pd.testing.assert_series_equal(labels1, labels2)
