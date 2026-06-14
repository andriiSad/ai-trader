"""Tests for benchmark module."""

import pandas as pd
from src.benchmark import buy_and_hold_returns, compute_benchmark_metrics


class TestBuyAndHold:
    def test_buy_and_hold_basic(self):
        candles = pd.DataFrame({"close": [100.0, 150.0, 200.0]})
        equity = buy_and_hold_returns(candles, 10000)
        assert equity.iloc[0] == 10000
        assert equity.iloc[-1] == 20000

    def test_buy_and_hold_constant_price(self):
        candles = pd.DataFrame({"close": [50.0, 50.0, 50.0]})
        equity = buy_and_hold_returns(candles, 10000)
        assert all(equity == 10000)


class TestBenchmarkMetrics:
    def test_benchmark_metrics_returns_dict(self):
        candles = pd.DataFrame({"close": [100.0, 110.0, 105.0, 115.0]})
        result = compute_benchmark_metrics(candles, 10000)
        expected_keys = {
            "sharpe",
            "max_drawdown",
            "max_drawdown_duration",
            "win_rate",
            "profit_factor",
            "total_return",
            "num_trades",
            "avg_trade_duration",
            "expectancy",
        }
        assert set(result.keys()) == expected_keys
        assert result["num_trades"] == 0
