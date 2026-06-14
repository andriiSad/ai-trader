"""Tests for metrics module."""

import numpy as np
import pandas as pd
from src.metrics import (
    calculate_expectancy,
    calculate_max_drawdown,
    calculate_profit_factor,
    calculate_sharpe,
    calculate_total_return,
    calculate_win_rate,
    compute_all_metrics,
)


class TestSharpe:
    def test_sharpe_positive_returns(self):
        returns = pd.Series([0.01] * 100)
        sharpe = calculate_sharpe(returns, periods_per_year=252)
        assert sharpe > 10

    def test_sharpe_zero_mean(self):
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0, 0.01, 1000))
        sharpe = calculate_sharpe(returns, periods_per_year=252)
        assert abs(sharpe) < 1.0


class TestMaxDrawdown:
    def test_max_drawdown_no_drawdown(self):
        equity = pd.Series([100, 110, 120, 130, 140])
        dd, duration = calculate_max_drawdown(equity)
        assert dd == 0.0
        assert duration == 0

    def test_max_drawdown_known(self):
        equity = pd.Series([100, 110, 100, 80, 90])
        dd, duration = calculate_max_drawdown(equity)
        assert abs(dd - 0.2727) < 0.01


class TestWinRate:
    def test_win_rate_all_winners(self):
        trade_log = [{"pnl": 10.0}] * 10
        assert calculate_win_rate(trade_log) == 1.0

    def test_win_rate_all_losers(self):
        trade_log = [{"pnl": -10.0}] * 10
        assert calculate_win_rate(trade_log) == 0.0

    def test_win_rate_empty(self):
        assert calculate_win_rate([]) == 0.0


class TestProfitFactor:
    def test_profit_factor_all_winners(self):
        trade_log = [{"pnl": 10.0}] * 5
        assert calculate_profit_factor(trade_log) == float("inf")

    def test_profit_factor_equal(self):
        trade_log = [{"pnl": 100.0}, {"pnl": -100.0}]
        assert calculate_profit_factor(trade_log) == 1.0


class TestTotalReturn:
    def test_total_return_basic(self):
        assert calculate_total_return(10000, 12000) == 0.2


class TestExpectancy:
    def test_expectancy_positive(self):
        trade_log = [{"pnl": 50.0}] * 7 + [{"pnl": -20.0}] * 3
        assert calculate_expectancy(trade_log) > 0

    def test_expectancy_empty(self):
        assert calculate_expectancy([]) == 0.0


class TestComputeAll:
    def test_compute_all_returns_dict_with_keys(self):
        equity = pd.Series([10000, 10100, 10200, 10150, 10300])
        trade_log = [
            {"pnl": 50.0, "bars_held": 3},
            {"pnl": -20.0, "bars_held": 2},
        ]
        result = compute_all_metrics(equity, trade_log, 10000)
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
        assert result["num_trades"] == 2
        assert result["total_return"] == 0.03
