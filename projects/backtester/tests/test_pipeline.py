import json

import pandas as pd
from src.pipeline import run_backtest


def _create_synthetic_data(tmp_path):
    """Create synthetic predictions + candles for pipeline test."""
    timestamps = pd.date_range("2025-01-01", periods=30, freq="5min")

    for pair in ["BTC_USDT", "ETH_USDT"]:
        base_price = 100.0 if pair == "BTC_USDT" else 50.0
        candle_rows = []
        for i, ts in enumerate(timestamps):
            price = base_price + i * 0.5
            candle_rows.append(
                {
                    "timestamp": ts.isoformat(),
                    "open": price,
                    "high": price + 1,
                    "low": price - 1,
                    "close": price,
                    "volume": 1000.0,
                }
            )
        pair_dir = tmp_path / "candles" / pair
        pair_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(candle_rows).to_csv(pair_dir / "5m.csv", index=False)

    pred_rows = []
    for pair in ["BTC_USDT", "ETH_USDT"]:
        fold_mid = 15
        for i, ts in enumerate(timestamps):
            prob = 0.8 if i < fold_mid else 0.3
            fold_id = 0 if i < fold_mid else 1
            fold_start = timestamps[0] if fold_id == 0 else timestamps[fold_mid]
            fold_end = (
                timestamps[fold_mid] if fold_id == 0 else timestamps[-1] + pd.Timedelta(minutes=5)
            )
            pred_rows.append(
                {
                    "pair": pair,
                    "timestamp": ts.isoformat(),
                    "probability": prob,
                    "fold_id": fold_id,
                    "fold_test_start": fold_start.isoformat(),
                    "fold_test_end": fold_end.isoformat(),
                }
            )
    pred_path = tmp_path / "predictions.csv"
    pd.DataFrame(pred_rows).to_csv(pred_path, index=False)

    results_path = tmp_path / "results" / "backtest_results.json"
    return {
        "pairs": ["BTC_USDT", "ETH_USDT"],
        "interval": "5m",
        "predictions_path": str(pred_path),
        "candles_dir": str(tmp_path / "candles"),
        "initial_capital": 10000,
        "strategy": {"threshold": 0.55, "long_only": True},
        "costs": {"fee_pct": 0.1, "slippage_pct": 0.05},
        "position": {"risk_per_trade": 0.01, "max_duration_candles": 12},
        "output": {
            "results_path": str(results_path),
            "tear_sheet_path": str(tmp_path / "tear_sheet.html"),
        },
    }


def test_pipeline_end_to_end(tmp_path):
    config = _create_synthetic_data(tmp_path)
    result = run_backtest(config)

    assert "overall" in result
    assert "per_pair" in result
    assert "config" in result
    assert "BTC_USDT" in result["overall"]
    assert "ETH_USDT" in result["overall"]


def test_pipeline_output_schema(tmp_path):
    config = _create_synthetic_data(tmp_path)
    result = run_backtest(config)

    assert isinstance(result["overall"], dict)
    assert isinstance(result["per_pair"], dict)
    assert isinstance(result["config"], dict)

    for pair in config["pairs"]:
        pair_result = result["per_pair"][pair]
        assert "per_fold" in pair_result
        assert "metrics" in pair_result
        assert "benchmark" in pair_result
        assert "trade_count" in pair_result

        metrics = pair_result["metrics"]
        assert "sharpe" in metrics
        assert "max_drawdown" in metrics
        assert "win_rate" in metrics
        assert "profit_factor" in metrics
        assert "total_return" in metrics
        assert "num_trades" in metrics

        for fold in pair_result["per_fold"]:
            assert "fold_id" in fold
            assert "fold_start" in fold
            assert "fold_end" in fold
            assert "sharpe" in fold

        benchmark = pair_result["benchmark"]
        assert "sharpe" in benchmark
        assert "total_return" in benchmark


def test_pipeline_saves_results_json(tmp_path):
    config = _create_synthetic_data(tmp_path)
    run_backtest(config)

    results_path = config["output"]["results_path"]
    with open(results_path) as f:
        saved = json.load(f)

    assert "overall" in saved
    assert "per_pair" in saved
    assert "config" in saved
    assert saved["config"]["pairs"] == config["pairs"]


def test_pipeline_config_snapshot(tmp_path):
    config = _create_synthetic_data(tmp_path)
    result = run_backtest(config)

    assert result["config"]["strategy"]["threshold"] == 0.55
    assert result["config"]["costs"]["fee_pct"] == 0.1
    assert result["config"]["pairs"] == ["BTC_USDT", "ETH_USDT"]
