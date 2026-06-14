"""Tests for data_loader module."""

import tempfile

import pandas as pd
from src.data_loader import CANDLE_COLUMNS, load_all_pairs, load_candles


class TestLoadCandles:
    def test_loads_valid_csv(self):
        csv_content = "timestamp,open,high,low,close,volume,deal\n1778794200,81415.05,81495.09,81414.25,81485.81,2.866543,233470.07967153\n1778794500,81460.08,81521.93,81431.5,81480.09,6.736121,548884.20346675\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv_content)
            f.flush()
            df = load_candles(f.name, "TEST_PAIR")

        assert not df.empty
        assert list(df.columns) == CANDLE_COLUMNS
        assert df.shape[0] == 2
        assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])

    def test_missing_file_returns_empty_df(self):
        df = load_candles("/nonexistent/path.csv", "MISSING")
        assert df.empty
        assert list(df.columns) == CANDLE_COLUMNS

    def test_handles_missing_columns_gracefully(self):
        csv_content = (
            "timestamp,open,high,low,close\n1778794200,81415.05,81495.09,81414.25,81485.81\n"
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv_content)
            f.flush()
            df = load_candles(f.name, "PARTIAL")

        assert not df.empty
        assert "volume" in df.columns
        assert "deal" in df.columns
        assert df["volume"].iloc[0] == 0.0


class TestLoadAllPairs:
    def test_loads_multiple_pairs(self, tmp_path):
        for pair in ["BTC_USDT", "ETH_USDT"]:
            pair_dir = tmp_path / pair
            pair_dir.mkdir()
            csv_path = pair_dir / "5m.csv"
            csv_path.write_text(
                "timestamp,open,high,low,close,volume,deal\n1778794200,100,110,90,105,1.0,100.0\n"
            )

        result = load_all_pairs(str(tmp_path), ["BTC_USDT", "ETH_USDT"])
        assert len(result) == 2
        assert not result["BTC_USDT"].empty
        assert not result["ETH_USDT"].empty

    def test_missing_pair_returns_empty(self, tmp_path):
        result = load_all_pairs(str(tmp_path), ["NONEXISTENT"])
        assert len(result) == 1
        assert result["NONEXISTENT"].empty
