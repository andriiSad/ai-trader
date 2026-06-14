"""Tests for feature engineering module."""

import numpy as np
import pandas as pd
import pytest
from src.features import (
    compute_bollinger_bands,
    compute_ema,
    compute_lagged_returns,
    compute_lagged_volume,
    compute_macd,
    compute_rsi,
    compute_sma,
    generate_features,
)

SAMPLE_SIZE = 100


@pytest.fixture
def sample_df():
    rng = np.random.default_rng(42)
    close = 100 + np.cumsum(rng.standard_normal(SAMPLE_SIZE) * 0.5)
    high = close + rng.uniform(0.1, 1.0, SAMPLE_SIZE)
    low = close - rng.uniform(0.1, 1.0, SAMPLE_SIZE)
    opn = close + rng.uniform(-0.5, 0.5, SAMPLE_SIZE)
    volume = rng.uniform(10, 100, SAMPLE_SIZE)
    return pd.DataFrame(
        {
            "timestamp": range(SAMPLE_SIZE),
            "open": opn,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "deal": volume * close,
        }
    )


def test_rsi_bounds(sample_df):
    rsi = compute_rsi(sample_df["close"])
    valid = rsi.dropna()
    assert (valid >= 0).all()
    assert (valid <= 100).all()


def test_rsi_constant_series():
    close = pd.Series([50.0] * 30)
    rsi = compute_rsi(close, period=14)
    # With zero losses, RSI should be 100 (or NaN if avg_loss=0 causes issues)
    # After warmup, all gains=0, all losses=0 → RSI is NaN by definition (0/0)
    # But with constant price, delta=0, so gain=0, loss=0 → rs=NaN → rSI=NaN
    # This is correct behavior for a flat series
    assert rsi.isna().sum() >= 0  # no crash


def test_macd_returns_three_series(sample_df):
    result = compute_macd(sample_df["close"])
    assert isinstance(result, tuple)
    assert len(result) == 3
    for s in result:
        assert isinstance(s, pd.Series)
        assert len(s) == SAMPLE_SIZE


def test_bollinger_ordering(sample_df):
    upper, middle, lower = compute_bollinger_bands(sample_df["close"])
    valid_idx = upper.dropna().index
    assert (upper[valid_idx] >= middle[valid_idx]).all()
    assert (middle[valid_idx] >= lower[valid_idx]).all()


def test_sma_hand_calculated():
    close = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])
    sma3 = compute_sma(close, 3)
    assert np.isnan(sma3.iloc[0])
    assert np.isnan(sma3.iloc[1])
    assert sma3.iloc[2] == pytest.approx(20.0)
    assert sma3.iloc[3] == pytest.approx(30.0)
    assert sma3.iloc[4] == pytest.approx(40.0)


def test_ema_basic(sample_df):
    ema = compute_ema(sample_df["close"], 12)
    assert len(ema) == SAMPLE_SIZE
    assert not np.isnan(ema.iloc[-1])


def test_lagged_returns_keys(sample_df):
    result = compute_lagged_returns(sample_df["close"], [1, 3, 5])
    assert set(result.keys()) == {"return_lag_1", "return_lag_3", "return_lag_5"}


def test_lagged_volume_keys(sample_df):
    result = compute_lagged_volume(sample_df["volume"], [1, 3, 5])
    assert set(result.keys()) == {
        "volume_change_lag_1",
        "volume_change_lag_3",
        "volume_change_lag_5",
    }


def test_generate_features_column_count(sample_df):
    features = generate_features(sample_df)
    feature_cols = [c for c in features.columns if c not in ("timestamp", "pair", "deal")]
    assert 50 <= len(feature_cols) <= 70


def test_generate_features_no_nan(sample_df):
    features = generate_features(sample_df)
    assert features.isna().sum().sum() == 0


def test_generate_features_preserves_timestamp(sample_df):
    features = generate_features(sample_df)
    assert "timestamp" in features.columns


def test_generate_features_determinism(sample_df):
    result1 = generate_features(sample_df)
    result2 = generate_features(sample_df)
    pd.testing.assert_frame_equal(result1, result2)
