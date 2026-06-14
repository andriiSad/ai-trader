import csv
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from src.collect import (
    fetch_candles,
    get_last_timestamp,
    load_existing_data,
    run_candles,
    save_candles,
)

WHITEBIT_CANDLE_FORMAT = [
    1718000000,  # time
    "3500.00",  # open
    "3550.00",  # close
    "3600.00",  # high
    "3400.00",  # low
    "100.5",  # volume
]


def _make_whitebit_candles(count: int, start_time: int = 1718000000) -> list[list]:
    """Generate fake WhiteBIT REST API candle data."""
    candles = []
    for i in range(count):
        ts = start_time + i * 4 * 3600
        candles.append(
            [
                ts,
                f"{3500.0 + i * 10:.2f}",
                f"{3550.0 + i * 10:.2f}",
                f"{3600.0 + i * 10:.2f}",
                f"{3400.0 + i * 10:.2f}",
                f"{100.0 + i:.1f}",
            ]
        )
    return candles


class TestFetchCandles:
    def test_returns_list_of_candles(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = _make_whitebit_candles(5)

        with patch("src.collect.requests.get", return_value=mock_response):
            candles = fetch_candles("ETH_USDT", "4h", 1718000000, 1718200000)

        assert len(candles) == 5

    def test_handles_empty_response(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = []

        with patch("src.collect.requests.get", return_value=mock_response):
            candles = fetch_candles("ETH_USDT", "4h", 1718000000, 1718200000)

        assert candles == []

    def test_handles_api_error(self):
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.raise_for_status.side_effect = Exception("Rate limited")

        with patch("src.collect.requests.get", return_value=mock_response):
            with pytest.raises(Exception, match="Rate limited"):
                fetch_candles("ETH_USDT", "4h", 1718000000, 1718200000)

    def test_handles_network_error(self):
        import requests

        with patch("src.collect.requests.get", side_effect=requests.ConnectionError("timeout")):
            with pytest.raises(requests.ConnectionError):
                fetch_candles("ETH_USDT", "4h", 1718000000, 1718200000)


class TestSaveCandles:
    def test_creates_csv_with_header(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            candles = _make_whitebit_candles(3)
            save_candles(candles, "ETH_USDT", "4h", tmpdir)

            csv_path = Path(tmpdir) / "ETH_USDT" / "4h.csv"
            assert csv_path.exists()

            with open(csv_path) as f:
                reader = csv.reader(f)
                header = next(reader)
            assert header == ["timestamp", "open", "high", "low", "close", "volume"]

    def test_writes_correct_column_order(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            candles = _make_whitebit_candles(1)
            save_candles(candles, "ETH_USDT", "4h", tmpdir)

            csv_path = Path(tmpdir) / "ETH_USDT" / "4h.csv"
            with open(csv_path) as f:
                reader = csv.reader(f)
                next(reader)  # skip header
                row = next(reader)

            # WhiteBIT format: [time, open, close, high, low, volume]
            # CSV format: timestamp, open, high, low, close, volume
            assert row[0] == "1718000000"  # timestamp
            assert row[1] == "3500.00"  # open
            assert row[2] == "3600.00"  # high (was position 3 in API)
            assert row[3] == "3400.00"  # low (was position 4 in API)
            assert row[4] == "3550.00"  # close (was position 2 in API)
            assert row[5] == "100.0"  # volume

    def test_writes_multiple_candles(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            candles = _make_whitebit_candles(10)
            save_candles(candles, "ETH_USDT", "4h", tmpdir)

            csv_path = Path(tmpdir) / "ETH_USDT" / "4h.csv"
            with open(csv_path) as f:
                lines = f.readlines()
            assert len(lines) == 11  # header + 10 rows


class TestGetLastTimestamp:
    def test_returns_none_for_missing_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ts = get_last_timestamp(Path(tmpdir) / "nonexistent.csv")
            assert ts is None

    def test_returns_none_for_empty_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "empty.csv"
            csv_path.write_text("timestamp,open,high,low,close,volume\n")
            ts = get_last_timestamp(csv_path)
            assert ts is None

    def test_returns_last_timestamp(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test.csv"
            csv_path.write_text(
                "timestamp,open,high,low,close,volume\n"
                "1718000000,3500.0,3600.0,3400.0,3550.0,100.5\n"
                "1718014400,3550.0,3650.0,3450.0,3600.0,200.0\n"
            )
            ts = get_last_timestamp(csv_path)
            assert ts == 1718014400

    def test_handles_single_data_row(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test.csv"
            csv_path.write_text(
                "timestamp,open,high,low,close,volume\n"
                "1718000000,3500.0,3600.0,3400.0,3550.0,100.5\n"
            )
            ts = get_last_timestamp(csv_path)
            assert ts == 1718000000


class TestLoadExistingData:
    def test_returns_none_for_missing_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data = load_existing_data("ETH_USDT", "4h", tmpdir)
            assert data is None

    def test_returns_none_for_empty_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "ETH_USDT" / "4h.csv"
            csv_path.parent.mkdir(parents=True)
            csv_path.write_text("timestamp,open,high,low,close,volume\n")
            data = load_existing_data("ETH_USDT", "4h", tmpdir)
            assert data is None

    def test_returns_existing_timestamp(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "ETH_USDT" / "4h.csv"
            csv_path.parent.mkdir(parents=True)
            csv_path.write_text(
                "timestamp,open,high,low,close,volume\n"
                "1718000000,3500.0,3600.0,3400.0,3550.0,100.5\n"
                "1718014400,3550.0,3650.0,3450.0,3600.0,200.0\n"
            )
            data = load_existing_data("ETH_USDT", "4h", tmpdir)
            assert data == 1718014400


class TestRunCandles:
    def _make_response(self, candles):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = candles
        return resp

    def _make_empty_response(self):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = []
        return resp

    def test_full_run_from_scratch(self):
        candles = _make_whitebit_candles(5)
        responses = [self._make_response(candles), self._make_empty_response()]

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("src.collect.requests.get", side_effect=responses):
                with patch("src.collect.time.time", return_value=float(1718100000)):
                    count = run_candles("ETH_USDT", "4h", tmpdir)

            assert count == 5

            csv_path = Path(tmpdir) / "ETH_USDT" / "4h.csv"
            assert csv_path.exists()

            with open(csv_path) as f:
                lines = f.readlines()
            assert len(lines) == 6  # header + 5

    def test_idempotent_no_duplicates(self):
        """Running twice with same data should not duplicate rows."""
        candles = _make_whitebit_candles(5)
        last_candle_ts = candles[-1][0]  # 1718057600
        next_start = last_candle_ts + 14400  # 1718072000

        responses = [self._make_response(candles), self._make_empty_response()]

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("src.collect.requests.get", side_effect=responses):
                with patch("src.collect.time.time", return_value=float(next_start + 100)):
                    count1 = run_candles("ETH_USDT", "4h", tmpdir)

            # Second run: last_ts is in file, start_ts >= end_ts => returns 0
            with patch("src.collect.time.time", return_value=float(next_start)):
                count2 = run_candles("ETH_USDT", "4h", tmpdir)

            assert count1 == 5
            assert count2 == 0  # no new data

            csv_path = Path(tmpdir) / "ETH_USDT" / "4h.csv"
            with open(csv_path) as f:
                lines = f.readlines()
            assert len(lines) == 6  # header + 5 (not 10)

    def test_appends_only_new_data(self):
        """Running with new data should only append new candles."""
        first_batch = _make_whitebit_candles(3, start_time=1718000000)
        second_batch = _make_whitebit_candles(5, start_time=1718000000 + 3 * 4 * 3600)

        with tempfile.TemporaryDirectory() as tmpdir:
            responses_1 = [self._make_response(first_batch), self._make_empty_response()]
            with patch("src.collect.requests.get", side_effect=responses_1):
                with patch("src.collect.time.time", return_value=float(1718100000)):
                    count1 = run_candles("ETH_USDT", "4h", tmpdir)

            responses_2 = [self._make_response(second_batch), self._make_empty_response()]
            with patch("src.collect.requests.get", side_effect=responses_2):
                with patch("src.collect.time.time", return_value=float(1718200000)):
                    count2 = run_candles("ETH_USDT", "4h", tmpdir)

            assert count1 == 3
            assert count2 == 5

            csv_path = Path(tmpdir) / "ETH_USDT" / "4h.csv"
            with open(csv_path) as f:
                lines = f.readlines()
            assert len(lines) == 9  # header + 3 + 5

    def test_creates_directory_structure(self):
        candles = _make_whitebit_candles(2)
        responses = [self._make_response(candles), self._make_empty_response()]

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("src.collect.requests.get", side_effect=responses):
                with patch("src.collect.time.time", return_value=float(1718100000)):
                    run_candles("BTC_USDT", "1h", tmpdir)

            csv_path = Path(tmpdir) / "BTC_USDT" / "1h.csv"
            assert csv_path.exists()


class TestArgparseCLI:
    def test_candles_subcommand(self):
        from src.collect import build_parser

        parser = build_parser()
        args = parser.parse_args(["candles", "--pair", "ETH_USDT", "--interval", "4h"])
        assert args.command == "candles"
        assert args.pair == "ETH_USDT"
        assert args.interval == "4h"

    def test_candles_with_output_dir(self):
        from src.collect import build_parser

        parser = build_parser()
        args = parser.parse_args(
            ["candles", "--pair", "BTC_USDT", "--interval", "1h", "--output-dir", "/tmp/data"]
        )
        assert args.output_dir == "/tmp/data"

    def test_missing_pair_raises(self):
        from src.collect import build_parser

        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["candles", "--interval", "4h"])

    def test_missing_interval_raises(self):
        from src.collect import build_parser

        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["candles", "--pair", "ETH_USDT"])
