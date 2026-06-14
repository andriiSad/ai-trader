import pandas as pd
import plotly.graph_objects as go
from src.charts import (
    build_drawdown_chart,
    build_equity_curve_chart,
    build_fold_metrics_table,
    build_pair_metrics_table,
    build_trade_markers_chart,
)


class TestEquityCurveChart:
    def test_returns_figure(self):
        equity = pd.Series([10000, 10100, 10200, 10150, 10300])
        fig = build_equity_curve_chart(equity)
        assert isinstance(fig, go.Figure)

    def test_with_benchmark(self):
        equity = pd.Series([10000, 10100, 10200])
        benchmark = pd.Series([10000, 10050, 10100])
        fig = build_equity_curve_chart(equity, benchmark_curve=benchmark)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 2

    def test_with_fold_boundaries(self):
        equity = pd.Series([10000, 10100, 10200, 10150, 10300])
        fig = build_equity_curve_chart(equity, fold_boundaries=[2, 4])
        assert isinstance(fig, go.Figure)


class TestDrawdownChart:
    def test_returns_figure(self):
        equity = pd.Series([10000, 10500, 10200, 10800])
        fig = build_drawdown_chart(equity)
        assert isinstance(fig, go.Figure)

    def test_single_trace(self):
        equity = pd.Series([10000, 10500, 10200])
        fig = build_drawdown_chart(equity)
        assert len(fig.data) == 1


class TestTradeMarkersChart:
    def test_returns_figure(self):
        candles = pd.DataFrame(
            {
                "timestamp": ["2025-01-01", "2025-01-02"],
                "open": [100, 102],
                "high": [105, 108],
                "low": [95, 100],
                "close": [102, 106],
            }
        )
        trades = [
            {
                "entry_time": "2025-01-01",
                "entry_price": 100,
                "exit_time": "2025-01-02",
                "exit_price": 106,
            }
        ]
        fig = build_trade_markers_chart(candles, trades)
        assert isinstance(fig, go.Figure)

    def test_no_trades(self):
        candles = pd.DataFrame(
            {
                "timestamp": ["2025-01-01", "2025-01-02"],
                "open": [100, 102],
                "high": [105, 108],
                "low": [95, 100],
                "close": [102, 106],
            }
        )
        fig = build_trade_markers_chart(candles, [])
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1  # only candlestick, no markers


class TestFoldMetricsTable:
    def test_returns_dataframe(self):
        folds = [
            {"fold_id": 0, "sharpe": 1.5, "win_rate": 0.6},
            {"fold_id": 1, "sharpe": 0.8, "win_rate": 0.55},
        ]
        df = build_fold_metrics_table(folds)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "fold_id" in df.columns


class TestPairMetricsTable:
    def test_returns_dataframe(self):
        pairs = {
            "BTC_USDT": {"sharpe": 1.5, "total_return": 0.1},
            "ETH_USDT": {"sharpe": 0.8, "total_return": 0.05},
        }
        df = build_pair_metrics_table(pairs)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "pair" in df.columns

    def test_pair_column_values(self):
        pairs = {"BTC_USDT": {"sharpe": 1.5}, "ETH_USDT": {"sharpe": 0.8}}
        df = build_pair_metrics_table(pairs)
        assert set(df["pair"].tolist()) == {"BTC_USDT", "ETH_USDT"}
