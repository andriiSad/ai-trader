"""Tests for explore_charts module."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from src.explore_charts import (
    candlestick_chart,
    correlation_heatmap,
    feature_distributions,
    return_distribution,
    volume_profile,
)


def _make_sample_df(n: int = 200) -> pd.DataFrame:
    """Generate synthetic candle data for testing."""
    rng = np.random.default_rng(42)
    timestamps = pd.date_range("2024-01-01", periods=n, freq="5min")
    close = 100 + np.cumsum(rng.normal(0, 0.5, n))
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": close + rng.normal(0, 0.1, n),
            "high": close + abs(rng.normal(0, 0.3, n)),
            "low": close - abs(rng.normal(0, 0.3, n)),
            "close": close,
            "volume": rng.uniform(1, 100, n),
            "deal": rng.uniform(100, 10000, n),
        }
    )


class TestCandlestickChart:
    def test_returns_figure(self):
        df = _make_sample_df()
        fig = candlestick_chart(df, "BTC_USDT")
        assert isinstance(fig, go.Figure)

    def test_has_two_subplots(self):
        df = _make_sample_df()
        fig = candlestick_chart(df, "BTC_USDT")
        assert len(fig.data) == 2


class TestVolumeProfile:
    def test_returns_figure(self):
        df = _make_sample_df()
        fig = volume_profile(df, "BTC_USDT")
        assert isinstance(fig, go.Figure)


class TestReturnDistribution:
    def test_returns_figure_default_periods(self):
        df = _make_sample_df()
        fig = return_distribution(df, "BTC_USDT")
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 2

    def test_custom_periods(self):
        df = _make_sample_df()
        fig = return_distribution(df, "BTC_USDT", periods=[1, 3, 10])
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 3


class TestCorrelationHeatmap:
    def test_returns_figure(self):
        df = _make_sample_df()
        fig = correlation_heatmap(df, "BTC_USDT")
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1


class TestFeatureDistributions:
    def test_returns_figure(self):
        df = _make_sample_df()
        fig = feature_distributions(df, "BTC_USDT")
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 3
