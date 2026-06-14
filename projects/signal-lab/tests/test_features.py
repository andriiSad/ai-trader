import numpy as np
import pandas as pd
import pytest
from src.features import (
    atr_14,
    bb_pct_b,
    compose,
    ema_cross_9_21,
    generate_features,
    generate_from_modules,
    macd_hist,
    obv,
    return_lag_1,
    return_lag_5,
    rsi_14,
    sma_cross_5_20,
    volume_ratio,
)
from src.features.ohlcv import generate as ohlcv_generate


def _make_ohlcv(n: int = 50) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    close = 100 + np.cumsum(rng.standard_normal(n) * 0.5)
    return pd.DataFrame(
        {
            "timestamp": np.arange(n),
            "open": close - 0.1,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": rng.uniform(10, 100, n),
            "deal": np.arange(n),
        }
    )


class TestRSI14:
    def test_range(self):
        df = _make_ohlcv(100)
        rsi = rsi_14(df).dropna()
        assert (rsi >= 0).all() and (rsi <= 100).all()

    def test_length(self):
        df = _make_ohlcv(50)
        rsi = rsi_14(df)
        assert len(rsi) == 50


class TestMACDHist:
    def test_length(self):
        df = _make_ohlcv(50)
        hist = macd_hist(df)
        assert len(hist) == 50

    def test_no_nans_after_warmup(self):
        df = _make_ohlcv(100)
        hist = macd_hist(df)
        assert not hist.iloc[30:].isna().any()


class TestBBPctB:
    def test_length(self):
        df = _make_ohlcv(50)
        pct_b = bb_pct_b(df)
        assert len(pct_b) == 50

    def test_values_near_middle(self):
        df = _make_ohlcv(100)
        pct_b = bb_pct_b(df).dropna()
        assert pct_b.mean() == pytest.approx(0.5, abs=0.5)


class TestATR14:
    def test_positive(self):
        df = _make_ohlcv(50)
        atr = atr_14(df).dropna()
        assert (atr >= 0).all()

    def test_length(self):
        df = _make_ohlcv(50)
        atr = atr_14(df)
        assert len(atr) == 50


class TestOBV:
    def test_length(self):
        df = _make_ohlcv(50)
        result = obv(df)
        assert len(result) == 50

    def test_monotonic_close_increases_obv_up(self):
        df = pd.DataFrame(
            {
                "timestamp": range(5),
                "open": [10, 11, 12, 13, 14],
                "high": [11, 12, 13, 14, 15],
                "low": [9, 10, 11, 12, 13],
                "close": [10, 11, 12, 13, 14],
                "volume": [100, 100, 100, 100, 100],
                "deal": [0, 1, 2, 3, 4],
            }
        )
        result = obv(df)
        assert result.iloc[-1] > result.iloc[0]


class TestSMACross:
    def test_ratio_near_one(self):
        df = _make_ohlcv(100)
        ratio = sma_cross_5_20(df).dropna()
        assert 0.9 < ratio.mean() < 1.1

    def test_length(self):
        df = _make_ohlcv(50)
        ratio = sma_cross_5_20(df)
        assert len(ratio) == 50


class TestEMACross:
    def test_ratio_near_one(self):
        df = _make_ohlcv(100)
        ratio = ema_cross_9_21(df).dropna()
        assert 0.9 < ratio.mean() < 1.1

    def test_length(self):
        df = _make_ohlcv(50)
        ratio = ema_cross_9_21(df)
        assert len(ratio) == 50


class TestVolumeRatio:
    def test_length(self):
        df = _make_ohlcv(50)
        ratio = volume_ratio(df)
        assert len(ratio) == 50

    def test_positive(self):
        df = _make_ohlcv(50)
        ratio = volume_ratio(df).dropna()
        assert (ratio > 0).all()


class TestReturnLag1:
    def test_first_is_nan(self):
        df = _make_ohlcv(10)
        ret = return_lag_1(df)
        assert np.isnan(ret.iloc[0])

    def test_length(self):
        df = _make_ohlcv(50)
        ret = return_lag_1(df)
        assert len(ret) == 50


class TestReturnLag5:
    def test_first_5_nan(self):
        df = _make_ohlcv(10)
        ret = return_lag_5(df)
        assert np.isnan(ret.iloc[0])
        assert np.isnan(ret.iloc[4])

    def test_length(self):
        df = _make_ohlcv(50)
        ret = return_lag_5(df)
        assert len(ret) == 50


class TestGenerateFeatures:
    def test_output_columns(self):
        df = _make_ohlcv(100)
        result = generate_features(df)
        expected = [
            "timestamp",
            "rsi_14",
            "macd_hist",
            "bb_pct_b",
            "atr_14",
            "obv",
            "sma_cross_5_20",
            "ema_cross_9_21",
            "volume_ratio",
            "return_lag_1",
            "return_lag_5",
        ]
        assert list(result.columns) == expected

    def test_no_nans(self):
        df = _make_ohlcv(100)
        result = generate_features(df)
        assert not result.isna().any().any()

    def test_no_infs(self):
        df = _make_ohlcv(100)
        result = generate_features(df)
        assert not np.isinf(result.select_dtypes(include=[np.number])).any().any()

    def test_rows_less_than_input(self):
        df = _make_ohlcv(100)
        result = generate_features(df)
        assert len(result) <= len(df)

    def test_timestamp_preserved(self):
        df = _make_ohlcv(100)
        result = generate_features(df)
        assert "timestamp" in result.columns


class TestOhlcvGenerate:
    def test_output_columns(self):
        df = _make_ohlcv(100)
        result = ohlcv_generate(df)
        expected = [
            "timestamp",
            "rsi_14",
            "macd_hist",
            "bb_pct_b",
            "atr_14",
            "obv",
            "sma_cross_5_20",
            "ema_cross_9_21",
            "volume_ratio",
            "return_lag_1",
            "return_lag_5",
        ]
        assert list(result.columns) == expected

    def test_no_nans(self):
        df = _make_ohlcv(100)
        result = ohlcv_generate(df)
        assert not result.isna().any().any()

    def test_no_infs(self):
        df = _make_ohlcv(100)
        result = ohlcv_generate(df)
        assert not np.isinf(result.select_dtypes(include=[np.number])).any().any()

    def test_timestamp_keyed(self):
        df = _make_ohlcv(100)
        result = ohlcv_generate(df)
        assert "timestamp" in result.columns
        assert result["timestamp"].is_unique


class TestCompose:
    def test_single_dataframe(self):
        df = _make_ohlcv(100)
        feat = ohlcv_generate(df)
        result = compose(feat)
        pd.testing.assert_frame_equal(result, feat)

    def test_merge_two(self):
        df = _make_ohlcv(100)
        feat_a = ohlcv_generate(df)
        feat_b = df[["timestamp"]].copy()
        feat_b["custom_feature"] = 1.0
        result = compose(feat_a, feat_b)
        assert "custom_feature" in result.columns
        assert "rsi_14" in result.columns
        assert "timestamp" in result.columns

    def test_inner_join_drops_non_overlapping(self):
        df = _make_ohlcv(100)
        feat_a = ohlcv_generate(df)
        feat_b = df[["timestamp"]].head(50).copy()
        feat_b["extra"] = 1.0
        result = compose(feat_a, feat_b)
        assert len(result) <= 50

    def test_cleans_infinities(self):
        df = _make_ohlcv(100)
        feat = ohlcv_generate(df)
        dirty = df[["timestamp"]].copy()
        dirty["inf_col"] = np.inf
        result = compose(feat, dirty)
        assert not np.isinf(result.select_dtypes(include=[np.number])).any().any()

    def test_cleans_nans(self):
        df = _make_ohlcv(100)
        feat = ohlcv_generate(df)
        dirty = df[["timestamp"]].copy()
        dirty["nan_col"] = np.nan
        result = compose(feat, dirty)
        assert not result.isna().any().any()

    def test_no_args_raises(self):
        with pytest.raises(ValueError, match="At least one"):
            compose()


class TestGenerateFromModules:
    def test_single_module(self):
        df = _make_ohlcv(100)
        result = generate_from_modules(df, ["ohlcv"])
        expected = generate_features(df)
        pd.testing.assert_frame_equal(result, expected)

    def test_columns_match_single_module(self):
        df = _make_ohlcv(100)
        result = generate_from_modules(df, ["ohlcv"])
        assert "timestamp" in result.columns
        assert "rsi_14" in result.columns
        assert len(result.columns) == 11


class TestCrossAsset:
    @pytest.fixture
    def btc_eth_data(self, tmp_path):
        rng = np.random.default_rng(42)
        n = 100
        ts = pd.date_range("2024-01-01", periods=n, freq="4h")
        btc_close = 40000 + np.cumsum(rng.standard_normal(n) * 100)
        eth_close = 2500 + np.cumsum(rng.standard_normal(n) * 10)

        btc_df = pd.DataFrame(
            {
                "timestamp": ts,
                "open": btc_close - 50,
                "high": btc_close + 100,
                "low": btc_close - 100,
                "close": btc_close,
                "volume": rng.uniform(100, 1000, n),
                "deal": np.arange(n),
            }
        )
        eth_df = pd.DataFrame(
            {
                "timestamp": ts,
                "open": eth_close - 5,
                "high": eth_close + 10,
                "low": eth_close - 10,
                "close": eth_close,
                "volume": rng.uniform(100, 1000, n),
            }
        )
        eth_dir = tmp_path / "ETH_USDT"
        eth_dir.mkdir()
        eth_df.to_csv(eth_dir / "4h.csv", index=False)
        return btc_df, tmp_path

    def test_output_columns(self, btc_eth_data):
        from src.features.cross_asset import generate

        btc_df, data_dir = btc_eth_data
        result = generate(btc_df, data_dir=str(data_dir), interval="4h")
        expected_cols = ["timestamp", "btc_eth_corr_14", "btc_eth_corr_28", "btc_eth_ratio"]
        assert list(result.columns) == expected_cols

    def test_no_nans(self, btc_eth_data):
        from src.features.cross_asset import generate

        btc_df, data_dir = btc_eth_data
        result = generate(btc_df, data_dir=str(data_dir), interval="4h")
        assert not result.isna().any().any()

    def test_output_length(self, btc_eth_data):
        from src.features.cross_asset import generate

        btc_df, data_dir = btc_eth_data
        result = generate(btc_df, data_dir=str(data_dir), interval="4h")
        assert len(result) <= len(btc_df)
        assert len(result) == len(btc_df) - 28

    def test_corr_values_bounded(self, btc_eth_data):
        from src.features.cross_asset import generate

        btc_df, data_dir = btc_eth_data
        result = generate(btc_df, data_dir=str(data_dir), interval="4h")
        assert (result["btc_eth_corr_14"] >= -1).all()
        assert (result["btc_eth_corr_14"] <= 1).all()
        assert (result["btc_eth_corr_28"] >= -1).all()
        assert (result["btc_eth_corr_28"] <= 1).all()

    def test_ratio_positive(self, btc_eth_data):
        from src.features.cross_asset import generate

        btc_df, data_dir = btc_eth_data
        result = generate(btc_df, data_dir=str(data_dir), interval="4h")
        assert (result["btc_eth_ratio"] > 0).all()

    def test_perfect_correlation(self, tmp_path):
        from src.features.cross_asset import generate

        n = 60
        ts = pd.date_range("2024-01-01", periods=n, freq="4h")
        rng = np.random.default_rng(99)
        returns = rng.standard_normal(n) * 0.01
        btc_close = 40000 * (1 + pd.Series(returns)).cumprod().values
        eth_close = 2500 * (1 + pd.Series(returns)).cumprod().values

        btc_df = pd.DataFrame(
            {
                "timestamp": ts,
                "open": btc_close,
                "high": btc_close,
                "low": btc_close,
                "close": btc_close,
                "volume": np.ones(n),
                "deal": np.arange(n),
            }
        )
        eth_df = pd.DataFrame(
            {
                "timestamp": ts,
                "open": eth_close,
                "high": eth_close,
                "low": eth_close,
                "close": eth_close,
                "volume": np.ones(n),
            }
        )
        eth_dir = tmp_path / "ETH_USDT"
        eth_dir.mkdir()
        eth_df.to_csv(eth_dir / "4h.csv", index=False)

        result = generate(btc_df, data_dir=str(tmp_path), interval="4h")
        assert result["btc_eth_corr_14"].iloc[-1] == pytest.approx(1.0, abs=0.01)
        assert result["btc_eth_corr_28"].iloc[-1] == pytest.approx(1.0, abs=0.01)

    def test_missing_eth_data_raises(self, tmp_path):
        from src.features.cross_asset import generate

        btc_df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=50, freq="4h"),
                "close": np.ones(50),
                "open": np.ones(50),
                "high": np.ones(50),
                "low": np.ones(50),
                "volume": np.ones(50),
                "deal": np.arange(50),
            }
        )
        with pytest.raises(FileNotFoundError):
            generate(btc_df, data_dir=str(tmp_path), interval="4h")

    def test_timestamp_keyed(self, btc_eth_data):
        from src.features.cross_asset import generate

        btc_df, data_dir = btc_eth_data
        result = generate(btc_df, data_dir=str(data_dir), interval="4h")
        assert result["timestamp"].is_unique
