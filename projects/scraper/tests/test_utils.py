import argparse
from pathlib import Path

import pytest
from scraper import (
    CSV_HEADER,
    INTERVAL_MAP,
    append_candles,
    build_config,
    ensure_csv,
    get_csv_path,
    get_last_timestamp,
    load_config,
    parse_interval,
    reorder_candle,
    update_or_append,
)

# --- parse_interval ---


class TestParseInterval:
    def test_known_intervals(self):
        assert parse_interval("1m") == 60
        assert parse_interval("5m") == 300
        assert parse_interval("15m") == 900
        assert parse_interval("1h") == 3600
        assert parse_interval("1d") == 86400

    def test_raw_int(self):
        assert parse_interval("60") == 60
        assert parse_interval("300") == 300

    def test_all_interval_map_keys(self):
        for key, val in INTERVAL_MAP.items():
            assert parse_interval(key) == val

    def test_invalid_interval(self):
        with pytest.raises(argparse.ArgumentTypeError):
            parse_interval("abc")

    def test_zero_interval(self):
        with pytest.raises(argparse.ArgumentTypeError):
            parse_interval("0")

    def test_negative_interval(self):
        with pytest.raises(argparse.ArgumentTypeError):
            parse_interval("-1")


# --- get_csv_path ---


class TestGetCsvPath:
    def test_basic(self):
        p = get_csv_path("data", "BTC_USDT", "1m")
        assert p == Path("data") / "BTC_USDT" / "1m.csv"

    def test_nested_dir(self):
        p = get_csv_path("/tmp/collected", "ETH_USDT", "1h")
        assert p == Path("/tmp/collected") / "ETH_USDT" / "1h.csv"


# --- ensure_csv ---


class TestEnsureCsv:
    def test_creates_file_with_header(self, tmp_path):
        f = tmp_path / "BTC_USDT" / "1m.csv"
        assert not f.exists()
        ensure_csv(f)
        assert f.exists()
        with open(f) as fh:
            header = fh.readline().strip()
            assert header == ",".join(CSV_HEADER)

    def test_does_not_overwrite_existing(self, tmp_path):
        f = tmp_path / "BTC_USDT" / "1m.csv"
        f.parent.mkdir(parents=True)
        f.write_text("timestamp,open,high,low,close,volume,deal\n123,1,2,3,4,5,6\n")
        ensure_csv(f)
        content = f.read_text()
        assert "123" in content  # original data preserved

    def test_creates_parent_dirs(self, tmp_path):
        f = tmp_path / "a" / "b" / "c" / "test.csv"
        ensure_csv(f)
        assert f.exists()


# --- get_last_timestamp ---


class TestGetLastTimestamp:
    def test_empty_file(self, tmp_path):
        f = tmp_path / "test.csv"
        f.write_text("timestamp,open,high,low,close,volume,deal\n")
        assert get_last_timestamp(f) is None

    def test_single_row(self, tmp_path):
        f = tmp_path / "test.csv"
        f.write_text("timestamp,open,high,low,close,volume,deal\n1580860800,1,2,3,4,5,6\n")
        assert get_last_timestamp(f) == 1580860800

    def test_multiple_rows(self, tmp_path):
        f = tmp_path / "test.csv"
        f.write_text(
            "timestamp,open,high,low,close,volume,deal\n"
            "1000,1,2,3,4,5,6\n"
            "2000,1,2,3,4,5,6\n"
            "3000,1,2,3,4,5,6\n"
        )
        assert get_last_timestamp(f) == 3000

    def test_missing_file(self, tmp_path):
        f = tmp_path / "nonexistent.csv"
        assert get_last_timestamp(f) is None

    def test_corrupt_last_line(self, tmp_path):
        f = tmp_path / "test.csv"
        f.write_text("timestamp,open,high,low,close,volume,deal\n1000,1,2,3,4,5,6\ngarbage\n")
        assert get_last_timestamp(f) is None


# --- reorder_candle ---


class TestReorderCandle:
    def test_reorder(self):
        whitebit = [
            1580860800,
            "0.020543",
            "0.020553",
            "0.020614",
            "0.02054",
            "7342.597",
            "151.09",
            "ETH_BTC",
        ]
        ohlcv = reorder_candle(whitebit)
        assert ohlcv == [
            1580860800,
            "0.020543",
            "0.020614",
            "0.02054",
            "0.020553",
            "7342.597",
            "151.09",
        ]

    def test_market_dropped(self):
        whitebit = [100, "o", "c", "h", "l", "v", "d", "BTC_USDT"]
        ohlcv = reorder_candle(whitebit)
        assert len(ohlcv) == 7
        assert "BTC_USDT" not in ohlcv

    def test_ohlcv_order(self):
        whitebit = [100, "open", "close", "high", "low", "vol", "deal", "X"]
        ohlcv = reorder_candle(whitebit)
        assert ohlcv[0] == 100  # timestamp
        assert ohlcv[1] == "open"  # open
        assert ohlcv[2] == "high"  # high (was index 3)
        assert ohlcv[3] == "low"  # low (was index 4)
        assert ohlcv[4] == "close"  # close (was index 2)
        assert ohlcv[5] == "vol"  # volume
        assert ohlcv[6] == "deal"  # deal


# --- append_candles ---


class TestAppendCandles:
    def test_append_to_new_file(self, tmp_path):
        f = tmp_path / "test.csv"
        ensure_csv(f)
        candles = [
            [1000, "100", "100", "105", "95", "10", "1000", "BTC_USDT"],
            [2000, "105", "105", "110", "100", "15", "1500", "BTC_USDT"],
        ]
        append_candles(f, candles)
        lines = f.read_text().strip().split("\n")
        assert len(lines) == 3  # header + 2 rows
        assert lines[1] == "1000,100,105,95,100,10,1000"
        assert lines[2] == "2000,105,110,100,105,15,1500"

    def test_append_preserves_existing(self, tmp_path):
        f = tmp_path / "test.csv"
        ensure_csv(f)
        append_candles(f, [[1000, "1", "1", "1", "1", "1", "1", "X"]])
        append_candles(f, [[2000, "2", "2", "2", "2", "2", "2", "X"]])
        lines = f.read_text().strip().split("\n")
        assert len(lines) == 3


# --- update_or_append ---


class TestUpdateOrAppend:
    def test_first_append(self, tmp_path):
        f = tmp_path / "test.csv"
        ensure_csv(f)
        ts_map = {}
        candle = [1000, "100", "100", "105", "95", "10", "1000", "BTC_USDT"]
        update_or_append(f, candle, ts_map)
        assert ts_map[str(f)] == 1000
        lines = f.read_text().strip().split("\n")
        assert len(lines) == 2

    def test_same_timestamp_overwrites(self, tmp_path):
        f = tmp_path / "test.csv"
        ensure_csv(f)
        ts_map = {}
        candle1 = [1000, "100", "100", "105", "95", "10", "1000", "BTC_USDT"]
        update_or_append(f, candle1, ts_map)
        candle2 = [1000, "101", "101", "106", "96", "12", "1200", "BTC_USDT"]
        update_or_append(f, candle2, ts_map)
        lines = f.read_text().strip().split("\n")
        assert len(lines) == 2  # still 1 data row
        assert lines[1] == "1000,101,106,96,101,12,1200"

    def test_new_timestamp_appends(self, tmp_path):
        f = tmp_path / "test.csv"
        ensure_csv(f)
        ts_map = {}
        update_or_append(f, [1000, "1", "1", "1", "1", "1", "1", "X"], ts_map)
        update_or_append(f, [2000, "2", "2", "2", "2", "2", "2", "X"], ts_map)
        lines = f.read_text().strip().split("\n")
        assert len(lines) == 3

    def test_overwrite_then_append(self, tmp_path):
        f = tmp_path / "test.csv"
        ensure_csv(f)
        ts_map = {}
        update_or_append(f, [1000, "1", "1", "1", "1", "1", "1", "X"], ts_map)
        update_or_append(f, [1000, "2", "2", "2", "2", "2", "2", "X"], ts_map)
        update_or_append(f, [2000, "3", "3", "3", "3", "3", "3", "X"], ts_map)
        lines = f.read_text().strip().split("\n")
        assert len(lines) == 3  # header + 2 data rows
        assert "2000" in lines[2]


# --- build_config / load_config ---


class TestConfig:
    def test_load_config(self, tmp_path):
        cfg = tmp_path / "cfg.yaml"
        cfg.write_text("pairs:\n  - BTC_USDT\nintervals:\n  - 1m\noutput_dir: data\n")
        result = load_config(cfg)
        assert result["pairs"] == ["BTC_USDT"]
        assert result["intervals"] == ["1m"]

    def test_build_config_cli_overrides(self, tmp_path):
        cfg = tmp_path / "cfg.yaml"
        cfg.write_text("pairs:\n  - BTC_USDT\nintervals:\n  - 1m\noutput_dir: data\n")

        class Args:
            config = str(cfg)
            pairs = ["ETH_USDT", "SOL_USDT"]
            intervals = ["5m"]
            start_date = "2026-01-01"
            output_dir = "/tmp/out"

        result = build_config(Args())
        assert result["pairs"] == ["ETH_USDT", "SOL_USDT"]
        assert result["intervals"] == ["5m"]
        assert result["start_date"] == "2026-01-01"
        assert result["output_dir"] == "/tmp/out"

    def test_build_config_no_overrides(self, tmp_path):
        cfg = tmp_path / "cfg.yaml"
        cfg.write_text("pairs:\n  - BTC_USDT\nintervals:\n  - 1m\noutput_dir: data\n")

        class Args:
            config = str(cfg)
            pairs = None
            intervals = None
            start_date = None
            output_dir = None

        result = build_config(Args())
        assert result["pairs"] == ["BTC_USDT"]
        assert result["intervals"] == ["1m"]
