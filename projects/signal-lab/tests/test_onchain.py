import numpy as np
import pandas as pd
import pytest
from src.features.onchain import (
    MockNetflowAdapter,
    NetflowAdapter,
    generate,
)


def _make_ohlcv(n: int = 50) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    close = 100 + np.cumsum(rng.standard_normal(n) * 0.5)
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="4h"),
            "open": close - 0.1,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": rng.uniform(10, 100, n),
            "deal": np.arange(n),
        }
    )


class TestNetflowAdapterInterface:
    def test_base_class_is_abstract(self):
        adapter = NetflowAdapter()
        with pytest.raises(NotImplementedError):
            adapter.fetch(pd.Series(dtype="datetime64[ns]"))


class TestMockNetflowAdapter:
    def test_generates_correct_number_of_rows(self):
        adapter = MockNetflowAdapter(seed=42)
        timestamps = pd.Series(pd.date_range("2024-01-01", periods=30, freq="4h"))
        result = adapter.fetch(timestamps)
        assert len(result) == 30

    def test_output_columns(self):
        adapter = MockNetflowAdapter(seed=42)
        timestamps = pd.Series(pd.date_range("2024-01-01", periods=10, freq="4h"))
        result = adapter.fetch(timestamps)
        assert list(result.columns) == ["timestamp", "netflow"]

    def test_no_nans_in_raw_data(self):
        adapter = MockNetflowAdapter(seed=42)
        timestamps = pd.Series(pd.date_range("2024-01-01", periods=50, freq="4h"))
        result = adapter.fetch(timestamps)
        assert not result["netflow"].isna().any()

    def test_deterministic_with_seed(self):
        timestamps = pd.Series(pd.date_range("2024-01-01", periods=20, freq="4h"))
        adapter1 = MockNetflowAdapter(seed=42)
        adapter2 = MockNetflowAdapter(seed=42)
        result1 = adapter1.fetch(timestamps)
        result2 = adapter2.fetch(timestamps)
        pd.testing.assert_frame_equal(result1, result2)

    def test_timestamps_match_input(self):
        adapter = MockNetflowAdapter(seed=42)
        timestamps = pd.Series(pd.date_range("2024-01-01", periods=15, freq="4h"))
        result = adapter.fetch(timestamps)
        pd.testing.assert_series_equal(result["timestamp"], timestamps, check_names=False)

    def test_distribution_properties(self):
        adapter = MockNetflowAdapter(seed=42)
        timestamps = pd.Series(pd.date_range("2024-01-01", periods=1000, freq="4h"))
        result = adapter.fetch(timestamps)
        assert abs(result["netflow"].mean()) < 500
        assert 500 < result["netflow"].std() < 3000


class TestOnchainGenerate:
    def test_output_columns(self):
        df = _make_ohlcv(50)
        result = generate(df, seed=42)
        expected_cols = ["timestamp", "netflow_raw", "netflow_cum_3", "netflow_zscore_20"]
        assert list(result.columns) == expected_cols

    def test_no_nans(self):
        df = _make_ohlcv(50)
        result = generate(df, seed=42)
        assert not result["netflow_raw"].isna().any()

    def test_no_infs(self):
        df = _make_ohlcv(50)
        result = generate(df, seed=42)
        assert not np.isinf(result.select_dtypes(include=[np.number])).any().any()

    def test_preserves_all_rows(self):
        df = _make_ohlcv(50)
        result = generate(df, seed=42)
        assert len(result) == 50

    def test_rows_equal_input(self):
        df = _make_ohlcv(100)
        result = generate(df, seed=42)
        assert len(result) == len(df)

    def test_timestamp_preserved(self):
        df = _make_ohlcv(50)
        result = generate(df, seed=42)
        assert "timestamp" in result.columns
        assert result["timestamp"].is_unique

    def test_deterministic(self):
        df = _make_ohlcv(50)
        result1 = generate(df, seed=42)
        result2 = generate(df, seed=42)
        pd.testing.assert_frame_equal(result1, result2)

    def test_composable_with_ohlcv(self):
        from src.features import compose
        from src.features.ohlcv import generate as ohlcv_generate

        df = _make_ohlcv(100)
        ohlcv_features = ohlcv_generate(df)
        onchain_features = generate(df, seed=42)
        result = compose(ohlcv_features, onchain_features)
        assert "netflow_raw" in result.columns
        assert "rsi_14" in result.columns
        assert "timestamp" in result.columns
        assert not result.isna().any().any()

    def test_generate_from_modules_integration(self):
        from src.features import generate_from_modules

        df = _make_ohlcv(100)
        result = generate_from_modules(df, ["ohlcv", "onchain"])
        assert "netflow_raw" in result.columns
        assert "netflow_cum_3" in result.columns
        assert "netflow_zscore_20" in result.columns
        assert "rsi_14" in result.columns
