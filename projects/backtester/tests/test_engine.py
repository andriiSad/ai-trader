"""Tests for backtest engine."""

from datetime import datetime

import pandas as pd
from src.engine import BacktestEngine, BacktestResult
from src.position import PositionManager
from src.strategy import ThresholdStrategy


def _make_candles(prices: list[float], start: datetime = datetime(2025, 1, 1)) -> pd.DataFrame:
    rows = []
    for i, price in enumerate(prices):
        ts = pd.Timestamp(start) + pd.Timedelta(hours=i)
        rows.append(
            {
                "timestamp": ts,
                "open": price,
                "high": price,
                "low": price,
                "close": price,
                "volume": 1000.0,
            }
        )
    return pd.DataFrame(rows)


def _make_predictions(
    timestamps: list,
    probabilities: list[float],
    fold_id: int | None = None,
    fold_start=None,
    fold_end=None,
) -> pd.DataFrame:
    data = {"timestamp": timestamps, "probability": probabilities}
    if fold_id is not None:
        data["fold_id"] = [fold_id] * len(timestamps)
    if fold_start is not None:
        data["fold_test_start"] = [fold_start] * len(timestamps)
    if fold_end is not None:
        data["fold_test_end"] = [fold_end] * len(timestamps)
    return pd.DataFrame(data)


def _build_engine(
    threshold=0.55,
    long_only=True,
    risk_per_trade=0.01,
    max_duration=12,
    fee_pct=0.0,
    slippage_pct=0.0,
    initial_capital=10000.0,
) -> BacktestEngine:
    strategy = ThresholdStrategy(threshold=threshold, long_only=long_only)
    pm = PositionManager(
        risk_per_trade=risk_per_trade,
        max_duration_candles=max_duration,
        fee_pct=fee_pct,
        slippage_pct=slippage_pct,
    )
    return BacktestEngine(
        strategy=strategy,
        position_manager=pm,
        initial_capital=initial_capital,
        fee_pct=fee_pct,
        slippage_pct=slippage_pct,
    )


def test_engine_basic_long_profitable():
    prices = [100.0, 102.0, 104.0, 106.0, 108.0]
    candles = _make_candles(prices)
    preds = _make_predictions(candles["timestamp"], [0.8] * 5)

    engine = _build_engine()
    result = engine.run(preds, candles)

    assert isinstance(result, BacktestResult)
    assert result.final_value > result.initial_capital
    assert len(result.trade_log) >= 1
    assert result.trade_log[0]["pnl"] > 0


def test_engine_basic_long_losing():
    prices = [100.0, 98.0, 96.0, 94.0, 92.0]
    candles = _make_candles(prices)
    preds = _make_predictions(candles["timestamp"], [0.8] * 5)

    engine = _build_engine()
    result = engine.run(preds, candles)

    assert result.final_value < result.initial_capital
    assert len(result.trade_log) >= 1
    assert result.trade_log[0]["pnl"] < 0


def test_engine_flat_signal():
    prices = [100.0] * 10
    candles = _make_candles(prices)
    preds = _make_predictions(candles["timestamp"], [0.5] * 10)

    engine = _build_engine()
    result = engine.run(preds, candles)

    assert len(result.trade_log) == 0
    assert result.final_value == result.initial_capital


def test_engine_force_exit():
    prices = [100.0 + i for i in range(20)]
    candles = _make_candles(prices)
    preds = _make_predictions(candles["timestamp"], [0.9] * 20)

    max_dur = 5
    engine = _build_engine(max_duration=max_dur)
    result = engine.run(preds, candles)

    assert len(result.trade_log) >= 1
    for trade in result.trade_log:
        assert trade["bars_held"] <= max_dur


def test_engine_costs_deducted():
    prices = [100.0, 110.0]
    candles = _make_candles(prices)
    preds = _make_predictions(candles["timestamp"], [0.8, 0.8])

    engine_no_cost = _build_engine(fee_pct=0.0, slippage_pct=0.0)
    result_no_cost = engine_no_cost.run(preds, candles)

    engine_with_cost = _build_engine(fee_pct=0.1, slippage_pct=0.05)
    result_with_cost = engine_with_cost.run(preds, candles)

    if result_no_cost.trade_log and result_with_cost.trade_log:
        assert result_with_cost.trade_log[0]["costs"] > 0
        assert result_with_cost.trade_log[0]["pnl"] < result_no_cost.trade_log[0]["pnl"]
        assert result_with_cost.final_value < result_no_cost.final_value


def test_engine_equity_curve_length():
    prices = [100.0, 101.0, 102.0, 103.0, 104.0]
    candles = _make_candles(prices)
    preds = _make_predictions(candles["timestamp"], [0.7] * 5)

    engine = _build_engine()
    result = engine.run(preds, candles)

    assert len(result.equity_curve) == len(candles)


def test_engine_walk_forward():
    start1 = pd.Timestamp("2025-01-01")
    start2 = pd.Timestamp("2025-01-02")

    candles1 = _make_candles([100.0, 102.0, 104.0], start=start1.to_pydatetime())
    candles2 = _make_candles([200.0, 198.0, 196.0], start=start2.to_pydatetime())
    candles = pd.concat([candles1, candles2], ignore_index=True)

    preds1 = _make_predictions(
        candles1["timestamp"],
        [0.8, 0.8, 0.8],
        fold_id=0,
        fold_start=start1,
        fold_end=start1 + pd.Timedelta(hours=3),
    )
    preds2 = _make_predictions(
        candles2["timestamp"],
        [0.8, 0.8, 0.8],
        fold_id=1,
        fold_start=start2,
        fold_end=start2 + pd.Timedelta(hours=3),
    )
    preds = pd.concat([preds1, preds2], ignore_index=True)

    engine = _build_engine()
    results = engine.run_walk_forward(preds, candles)

    assert len(results) == 2
    assert results[0].fold_id == 0
    assert results[1].fold_id == 1
    for r in results:
        assert isinstance(r, BacktestResult)
        assert len(r.equity_curve) > 0


def test_engine_no_lookahead():
    prices = [100.0, 90.0, 110.0, 95.0, 120.0]
    candles = _make_candles(prices)
    probs = [0.3, 0.3, 0.9, 0.3, 0.9]
    preds = _make_predictions(candles["timestamp"], probs)

    engine = _build_engine(threshold=0.55, max_duration=100)
    result = engine.run(preds, candles)

    for trade in result.trade_log:
        entry_idx = list(candles["timestamp"]).index(pd.Timestamp(trade["entry_time"]))
        entry_prob = probs[entry_idx]
        assert entry_prob > 0.55
