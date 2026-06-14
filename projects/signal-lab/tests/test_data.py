import os
import tempfile
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from src.data import download_candles, load_candles

EXPECTED_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume", "deal"]


def _make_fake_candles(count: int, start_time: int = 1609459200000) -> list[list]:
    """Generate fake Binance kline data."""
    candles = []
    for i in range(count):
        open_time = start_time + i * 4 * 60 * 60 * 1000  # 4h intervals
        candles.append(
            [
                open_time,
                "30000.00",  # open
                "31000.00",  # high
                "29000.00",  # low
                "30500.00",  # close
                "100.50",  # volume
                open_time + 4 * 60 * 60 * 1000 - 1,  # close_time
                "3050000.00",  # quote_asset_volume
                500,  # number_of_trades (deal)
                "50.25",  # taker_buy_base
                "1525000.00",  # taker_buy_quote
                "0",  # ignore
            ]
        )
    return candles


class TestCsvSchema:
    def test_columns_match_spec(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_client = MagicMock()
            mock_client.get_klines.return_value = _make_fake_candles(5)

            with patch("src.data.Client", return_value=mock_client):
                df = download_candles("BTCUSDT", "4h", None, tmpdir)

            assert list(df.columns) == EXPECTED_COLUMNS

    def test_dtypes_are_numeric(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_client = MagicMock()
            mock_client.get_klines.return_value = _make_fake_candles(5)

            with patch("src.data.Client", return_value=mock_client):
                df = download_candles("BTCUSDT", "4h", None, tmpdir)

            for col in ["open", "high", "low", "close", "volume"]:
                assert pd.api.types.is_float_dtype(df[col]), f"{col} should be float"
            assert pd.api.types.is_integer_dtype(df["deal"]), "deal should be int"


class TestDownloadCandles:
    def test_returns_dataframe(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_client = MagicMock()
            mock_client.get_klines.return_value = _make_fake_candles(10)

            with patch("src.data.Client", return_value=mock_client):
                df = download_candles("BTCUSDT", "4h", None, tmpdir)

            assert isinstance(df, pd.DataFrame)

    def test_csv_saved_to_correct_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_client = MagicMock()
            mock_client.get_klines.return_value = _make_fake_candles(5)

            with patch("src.data.Client", return_value=mock_client):
                download_candles("BTCUSDT", "4h", None, tmpdir)

            csv_path = os.path.join(tmpdir, "BTCUSDT", "4h.csv")
            assert os.path.exists(csv_path)

    def test_csv_has_all_rows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_client = MagicMock()
            mock_client.get_klines.return_value = _make_fake_candles(20)

            with patch("src.data.Client", return_value=mock_client):
                df = download_candles("BTCUSDT", "4h", None, tmpdir)

            assert len(df) == 20

    def test_pagination_multiple_requests(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_client = MagicMock()
            # First call returns 1000 candles, second returns 500
            mock_client.get_klines.side_effect = [
                _make_fake_candles(1000, start_time=1609459200000),
                _make_fake_candles(500, start_time=1609459200000 + 1000 * 4 * 60 * 60 * 1000),
            ]

            with patch("src.data.Client", return_value=mock_client):
                df = download_candles("BTCUSDT", "4h", None, tmpdir)

            assert len(df) == 1500
            assert mock_client.get_klines.call_count == 2

    def test_empty_response_returns_empty_dataframe(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_client = MagicMock()
            mock_client.get_klines.return_value = []

            with patch("src.data.Client", return_value=mock_client):
                df = download_candles("BTCUSDT", "4h", None, tmpdir)

            assert len(df) == 0
            assert list(df.columns) == EXPECTED_COLUMNS

    def test_pair_formatting_underscore_removed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_client = MagicMock()
            mock_client.get_klines.return_value = _make_fake_candles(5)

            with patch("src.data.Client", return_value=mock_client):
                download_candles("BTC_USDT", "4h", None, tmpdir)

            csv_path = os.path.join(tmpdir, "BTC_USDT", "4h.csv")
            assert os.path.exists(csv_path)


class TestLoadCandles:
    @pytest.fixture()
    def csv_dir(self, tmp_path):
        pair_dir = tmp_path / "BTC_USDT"
        pair_dir.mkdir()
        csv_path = pair_dir / "4h.csv"
        csv_path.write_text(
            "timestamp,open,high,low,close,volume,deal\n"
            "1609459200000,30000.0,31000.0,29000.0,30500.0,100.5,500\n"
            "1609473600000,30500.0,31500.0,29500.0,31000.0,200.0,600\n"
        )
        return tmp_path

    def test_load_returns_dataframe(self, csv_dir):
        df = load_candles("BTC_USDT", "4h", str(csv_dir))
        assert isinstance(df, pd.DataFrame)

    def test_load_columns_match_spec(self, csv_dir):
        df = load_candles("BTC_USDT", "4h", str(csv_dir))
        assert list(df.columns) == EXPECTED_COLUMNS

    def test_load_row_count(self, csv_dir):
        df = load_candles("BTC_USDT", "4h", str(csv_dir))
        assert len(df) == 2

    def test_load_dtypes(self, csv_dir):
        df = load_candles("BTC_USDT", "4h", str(csv_dir))
        assert pd.api.types.is_integer_dtype(df["timestamp"])
        assert pd.api.types.is_integer_dtype(df["deal"])
        for col in ["open", "high", "low", "close", "volume"]:
            assert pd.api.types.is_float_dtype(df[col]), f"{col} should be float"

    def test_load_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_candles("BTC_USDT", "4h", str(tmp_path))
