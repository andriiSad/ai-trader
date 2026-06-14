import csv
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from src.collect import (
    fetch_funding_rates,
    get_last_funding_timestamp,
    run_funding,
    save_funding_rates,
)

FUNDING_API_RESPONSE = [
    {
        "fundingTime": "1718035200",
        "fundingRate": "0.0001",
        "market": "BTC_PERP",
        "settlementPrice": "70000.0",
        "rateCalculatedTime": "1718006400",
    },
    {
        "fundingTime": "1718064000",
        "fundingRate": "-0.0002",
        "market": "BTC_PERP",
        "settlementPrice": "70500.0",
        "rateCalculatedTime": "1718035200",
    },
    {
        "fundingTime": "1718092800",
        "fundingRate": "0.00015",
        "market": "BTC_PERP",
        "settlementPrice": "70200.0",
        "rateCalculatedTime": "1718064000",
    },
]


def _make_funding_response(count: int, start_time: int = 1718035200) -> list[dict]:
    records = []
    for i in range(count):
        ts = start_time + i * 28800  # 8h intervals
        records.append(
            {
                "fundingTime": str(ts),
                "fundingRate": f"{0.0001 * (i + 1):.7f}",
                "market": "BTC_PERP",
                "settlementPrice": f"{70000.0 + i * 100:.1f}",
                "rateCalculatedTime": str(ts - 28800),
            }
        )
    return records


class TestFetchFundingRates:
    def test_returns_list_of_records(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = FUNDING_API_RESPONSE

        with patch("src.collect.requests.get", return_value=mock_response):
            records = fetch_funding_rates("BTC_USDT", limit=100, offset=0)

        assert len(records) == 3
        assert records[0]["fundingRate"] == "0.0001"

    def test_handles_empty_response(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = []

        with patch("src.collect.requests.get", return_value=mock_response):
            records = fetch_funding_rates("BTC_USDT", limit=100, offset=0)

        assert records == []

    def test_handles_api_error(self):
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.raise_for_status.side_effect = Exception("Rate limited")

        with patch("src.collect.requests.get", return_value=mock_response):
            with pytest.raises(Exception, match="Rate limited"):
                fetch_funding_rates("BTC_USDT", limit=100, offset=0)

    def test_converts_pair_to_perp_format(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = []

        with patch("src.collect.requests.get", return_value=mock_response) as mock_get:
            fetch_funding_rates("BTC_USDT", limit=100, offset=0)

        call_args = mock_get.call_args
        assert "BTC_PERP" in call_args[0][0]

    def test_passes_date_params(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = []

        with patch("src.collect.requests.get", return_value=mock_response) as mock_get:
            fetch_funding_rates("BTC_USDT", limit=50, offset=10, start_date=1000, end_date=2000)

        params = mock_get.call_args[1]["params"]
        assert params["startDate"] == 1000
        assert params["endDate"] == 2000
        assert params["limit"] == 50
        assert params["offset"] == 10


class TestSaveFundingRates:
    def test_creates_csv_with_header(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            save_funding_rates(FUNDING_API_RESPONSE, "BTC_USDT", tmpdir)

            csv_path = Path(tmpdir) / "BTC_USDT" / "funding.csv"
            assert csv_path.exists()

            with open(csv_path) as f:
                reader = csv.reader(f)
                header = next(reader)
            assert header == ["timestamp", "funding_rate"]

    def test_writes_correct_data(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            save_funding_rates(FUNDING_API_RESPONSE, "BTC_USDT", tmpdir)

            csv_path = Path(tmpdir) / "BTC_USDT" / "funding.csv"
            with open(csv_path) as f:
                reader = csv.reader(f)
                next(reader)  # skip header
                rows = list(reader)

            assert len(rows) == 3
            assert rows[0][0] == "1718035200"
            assert rows[0][1] == "0.0001"
            assert rows[1][0] == "1718064000"
            assert rows[1][1] == "-0.0002"

    def test_writes_multiple_records(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            records = _make_funding_response(10)
            save_funding_rates(records, "BTC_USDT", tmpdir)

            csv_path = Path(tmpdir) / "BTC_USDT" / "funding.csv"
            with open(csv_path) as f:
                lines = f.readlines()
            assert len(lines) == 11  # header + 10 rows


class TestGetLastFundingTimestamp:
    def test_returns_none_for_missing_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ts = get_last_funding_timestamp(Path(tmpdir) / "nonexistent.csv")
            assert ts is None

    def test_returns_none_for_empty_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "empty.csv"
            csv_path.write_text("timestamp,funding_rate\n")
            ts = get_last_funding_timestamp(csv_path)
            assert ts is None

    def test_returns_last_timestamp(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test.csv"
            csv_path.write_text("timestamp,funding_rate\n1718035200,0.0001\n1718064000,-0.0002\n")
            ts = get_last_funding_timestamp(csv_path)
            assert ts == 1718064000


class TestRunFunding:
    def _make_response(self, records):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = records
        return resp

    def _make_empty_response(self):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = []
        return resp

    def test_full_run_from_scratch(self):
        records = _make_funding_response(5)
        responses = [self._make_response(records), self._make_empty_response()]

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("src.collect.requests.get", side_effect=responses):
                count = run_funding("BTC_USDT", tmpdir)

            assert count == 5

            csv_path = Path(tmpdir) / "BTC_USDT" / "funding.csv"
            assert csv_path.exists()

            with open(csv_path) as f:
                lines = f.readlines()
            assert len(lines) == 6  # header + 5

    def test_idempotent_no_duplicates(self):
        records = _make_funding_response(5)
        responses = [self._make_response(records), self._make_empty_response()]

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("src.collect.requests.get", side_effect=responses):
                count1 = run_funding("BTC_USDT", tmpdir)

            # Second run: data already exists, should append only new
            responses2 = [self._make_empty_response()]
            with patch("src.collect.requests.get", side_effect=responses2):
                count2 = run_funding("BTC_USDT", tmpdir)

            assert count1 == 5
            assert count2 == 0

            csv_path = Path(tmpdir) / "BTC_USDT" / "funding.csv"
            with open(csv_path) as f:
                lines = f.readlines()
            assert len(lines) == 6  # header + 5 (not 10)

    def test_creates_directory_structure(self):
        records = _make_funding_response(2)
        responses = [self._make_response(records), self._make_empty_response()]

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("src.collect.requests.get", side_effect=responses):
                run_funding("BTC_USDT", tmpdir)

            csv_path = Path(tmpdir) / "BTC_USDT" / "funding.csv"
            assert csv_path.exists()


class TestFundingArgparseCLI:
    def test_funding_subcommand(self):
        from src.collect import build_parser

        parser = build_parser()
        args = parser.parse_args(["funding", "--pair", "BTC_USDT"])
        assert args.command == "funding"
        assert args.pair == "BTC_USDT"

    def test_funding_with_output_dir(self):
        from src.collect import build_parser

        parser = build_parser()
        args = parser.parse_args(["funding", "--pair", "BTC_USDT", "--output-dir", "/tmp/data"])
        assert args.output_dir == "/tmp/data"

    def test_funding_missing_pair_raises(self):
        from src.collect import build_parser

        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["funding"])
