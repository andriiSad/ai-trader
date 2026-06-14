import pandas as pd
import pytest
from src.data_loader import load_all_candles, load_candles, load_predictions


def _write_predictions(path, rows=None):
    if rows is None:
        rows = [
            {
                "pair": "BTC_USDT",
                "timestamp": "2025-01-01T00:00:00",
                "probability": 0.8,
                "fold_id": 0,
                "fold_test_start": "2025-01-01",
                "fold_test_end": "2025-02-01",
            },
            {
                "pair": "BTC_USDT",
                "timestamp": "2025-01-01T00:05:00",
                "probability": 0.6,
                "fold_id": 0,
                "fold_test_start": "2025-01-01",
                "fold_test_end": "2025-02-01",
            },
            {
                "pair": "ETH_USDT",
                "timestamp": "2025-01-01T00:00:00",
                "probability": 0.3,
                "fold_id": 0,
                "fold_test_start": "2025-01-01",
                "fold_test_end": "2025-02-01",
            },
        ]
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return path


def _write_candles(path, rows=None):
    if rows is None:
        rows = [
            {
                "timestamp": "2025-01-01T00:00:00",
                "open": 100,
                "high": 105,
                "low": 95,
                "close": 102,
                "volume": 1000,
            },
            {
                "timestamp": "2025-01-01T00:05:00",
                "open": 102,
                "high": 108,
                "low": 100,
                "close": 106,
                "volume": 1200,
            },
        ]
    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def test_load_predictions_basic(tmp_path):
    csv_path = tmp_path / "predictions.csv"
    _write_predictions(csv_path)
    df = load_predictions(str(csv_path))
    assert "timestamp" in df.columns
    assert "probability" in df.columns
    assert "pair" in df.columns
    assert len(df) == 3


def test_load_predictions_filter_by_pair(tmp_path):
    csv_path = tmp_path / "predictions.csv"
    _write_predictions(csv_path)
    df = load_predictions(str(csv_path), pairs=["BTC_USDT"])
    assert len(df) == 2
    assert all(df["pair"] == "BTC_USDT")


def test_load_predictions_missing_file():
    with pytest.raises((FileNotFoundError, Exception)):
        load_predictions("/nonexistent/predictions.csv")


def test_load_predictions_missing_timestamp_column(tmp_path):
    csv_path = tmp_path / "bad.csv"
    pd.DataFrame({"foo": [1, 2]}).to_csv(csv_path, index=False)
    with pytest.raises(ValueError, match="timestamp"):
        load_predictions(str(csv_path))


def test_load_candles_basic(tmp_path):
    pair_dir = tmp_path / "BTC_USDT"
    candle_path = pair_dir / "5m.csv"
    _write_candles(candle_path)
    df = load_candles(str(tmp_path), "BTC_USDT", "5m")
    assert "close" in df.columns
    assert "timestamp" in df.columns
    assert len(df) == 2


def test_load_candles_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_candles(str(tmp_path), "NONEXISTENT", "5m")


def test_load_all_candles(tmp_path):
    for pair in ["BTC_USDT", "ETH_USDT"]:
        pair_dir = tmp_path / pair
        candle_path = pair_dir / "5m.csv"
        _write_candles(candle_path)

    result = load_all_candles(str(tmp_path), ["BTC_USDT", "ETH_USDT"], "5m")
    assert "BTC_USDT" in result
    assert "ETH_USDT" in result
    assert len(result["BTC_USDT"]) == 2
    assert len(result["ETH_USDT"]) == 2
