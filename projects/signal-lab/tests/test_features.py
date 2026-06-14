import numpy as np
import pandas as pd
import pytest
from src.features import (
    atr_14,
    bb_pct_b,
    ema_cross_9_21,
    generate_features,
    macd_hist,
    obv,
    return_lag_1,
    return_lag_5,
    rsi_14,
    sma_cross_5_20,
    volume_ratio,
)


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
