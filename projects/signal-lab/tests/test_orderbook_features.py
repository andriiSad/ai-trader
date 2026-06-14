import tempfile
from pathlib import Path

import pandas as pd
import pytest
from src.features.orderbook import _aggregate_snapshots, generate


def _write_orderbook_snapshot(tmpdir: str, pair: str, timestamp: int, bids: list, asks: list):
    ob_dir = Path(tmpdir) / pair / "orderbook"
    ob_dir.mkdir(parents=True, exist_ok=True)
    csv_path = ob_dir / f"{timestamp}.csv"
    with open(csv_path, "w") as f:
        f.write("timestamp,side,price,quantity\n")
        for price, qty in bids:
            f.write(f"{timestamp},bid,{price},{qty}\n")
        for price, qty in asks:
            f.write(f"{timestamp},ask,{price},{qty}\n")


def _make_candle_timestamps(n: int = 5, interval_sec: int = 14400, start: int = 1718000000):
    return pd.Series([start + i * interval_sec for i in range(n)])


class TestGenerateOrderbook:
    def test_output_columns(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_orderbook_snapshot(
                tmpdir, "BTC_USDT", 1718000000,
                bids=[["60000.00", "1.5"], ["59999.00", "2.0"]],
                asks=[["60001.00", "0.5"], ["60002.00", "1.0"]],
            )
            _write_orderbook_snapshot(
                tmpdir, "BTC_USDT", 1718014400,
                bids=[["60005.00", "1.2"], ["60004.00", "1.8"]],
                asks=[["60006.00", "0.6"], ["60007.00", "0.9"]],
            )

            df = pd.DataFrame({"timestamp": _make_candle_timestamps(2)})
            result = generate(df, data_dir=tmpdir, pair="BTC_USDT")

            expected_cols = {
                "timestamp", "ob_bid_ask_ratio", "ob_total_bid",
                "ob_total_ask", "ob_imbalance", "ob_mid_price_deviation",
            }
            assert set(result.columns) == expected_cols

    def test_no_nans_in_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            for ts in [1718000000, 1718014400, 1718028800]:
                _write_orderbook_snapshot(
                    tmpdir, "BTC_USDT", ts,
                    bids=[["60000.00", "1.5"], ["59999.00", "2.0"]],
                    asks=[["60001.00", "0.5"], ["60002.00", "1.0"]],
                )

            df = pd.DataFrame({"timestamp": _make_candle_timestamps(3)})
            result = generate(df, data_dir=tmpdir, pair="BTC_USDT")

            assert not result.isnull().any().any()

    def test_imbalance_range(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_orderbook_snapshot(
                tmpdir, "BTC_USDT", 1718000000,
                bids=[["60000.00", "3.0"]],
                asks=[["60001.00", "1.0"]],
            )

            df = pd.DataFrame({"timestamp": _make_candle_timestamps(1)})
            result = generate(df, data_dir=tmpdir, pair="BTC_USDT")

            assert -1 <= result["ob_imbalance"].iloc[0] <= 1

    def test_bid_ask_ratio_correct(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_orderbook_snapshot(
                tmpdir, "BTC_USDT", 1718000000,
                bids=[["60000.00", "4.0"]],
                asks=[["60001.00", "2.0"]],
            )

            df = pd.DataFrame({"timestamp": _make_candle_timestamps(1)})
            result = generate(df, data_dir=tmpdir, pair="BTC_USDT")

            assert result["ob_bid_ask_ratio"].iloc[0] == 2.0

    def test_missing_orderbook_dir_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            df = pd.DataFrame({"timestamp": _make_candle_timestamps(1)})
            with pytest.raises(FileNotFoundError):
                generate(df, data_dir=tmpdir, pair="BTC_USDT")

    def test_uses_latest_snapshot_per_candle(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_orderbook_snapshot(
                tmpdir, "BTC_USDT", 1718000000,
                bids=[["60000.00", "1.0"]],
                asks=[["60001.00", "1.0"]],
            )
            _write_orderbook_snapshot(
                tmpdir, "BTC_USDT", 1718007200,
                bids=[["60000.00", "5.0"]],
                asks=[["60001.00", "1.0"]],
            )

            df = pd.DataFrame({"timestamp": _make_candle_timestamps(1, start=1718014400)})
            result = generate(df, data_dir=tmpdir, pair="BTC_USDT")

            assert result["ob_bid_ask_ratio"].iloc[0] == 5.0


class TestAggregateSnapshots:
    def test_empty_snapshots_returns_nan(self):
        snapshots = pd.DataFrame(columns=["timestamp", "side", "price", "quantity"])
        timestamps = pd.Series([1718000000, 1718014400])

        result = _aggregate_snapshots(snapshots, timestamps)

        assert len(result) == 2
        assert result["ob_bid_ask_ratio"].isna().all()

    def test_single_snapshot_applied_to_all_candles(self):
        snapshots = pd.DataFrame({
            "timestamp": [1718000000] * 4,
            "side": ["bid", "bid", "ask", "ask"],
            "price": ["60000", "59999", "60001", "60002"],
            "quantity": ["1.0", "2.0", "0.5", "1.0"],
        })
        timestamps = pd.Series([1718000000, 1718014400])

        result = _aggregate_snapshots(snapshots, timestamps)

        assert len(result) == 2
        assert result["ob_total_bid"].iloc[0] == 3.0
        assert result["ob_total_ask"].iloc[0] == 1.5
        assert result["ob_total_bid"].iloc[1] == 3.0
