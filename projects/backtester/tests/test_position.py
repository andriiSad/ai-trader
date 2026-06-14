"""Tests for position sizing and management."""

from datetime import datetime

from src.position import (
    Position,
    PositionManager,
    calculate_position_size,
    should_force_exit,
)


def test_position_size_basic():
    assert calculate_position_size(10000, 0.01, 100) == 1.0


def test_position_size_fractional():
    result = calculate_position_size(10000, 0.01, 333)
    assert abs(result - 0.3003003003003003) < 1e-10


def test_force_exit_not_yet():
    assert should_force_exit(5, 12) is False


def test_force_exit_at_limit():
    assert should_force_exit(12, 12) is True


def test_force_exit_past_limit():
    assert should_force_exit(15, 12) is True


def test_open_position():
    pm = PositionManager(risk_per_trade=0.01, max_duration_candles=12)
    ts = datetime(2025, 1, 1)
    pos = pm.open_position(price=100.0, direction=1, portfolio_value=10000, timestamp=ts)
    assert isinstance(pos, Position)
    assert pos.entry_price == 100.0
    assert pos.entry_time == ts
    assert pos.direction == 1
    assert pos.quantity == 1.0
    assert pos.bars_held == 0


def test_update_increments_bars():
    pm = PositionManager()
    ts = datetime(2025, 1, 1)
    pos = pm.open_position(price=100.0, direction=1, portfolio_value=10000, timestamp=ts)
    assert pos.bars_held == 0
    pm.update(pos)
    assert pos.bars_held == 1


def test_should_exit_signal_flip():
    pm = PositionManager(max_duration_candles=12)
    ts = datetime(2025, 1, 1)
    pos = pm.open_position(price=100.0, direction=1, portfolio_value=10000, timestamp=ts)
    assert pm.should_exit(pos, current_signal=-1) is True


def test_should_exit_same_signal():
    pm = PositionManager(max_duration_candles=12)
    ts = datetime(2025, 1, 1)
    pos = pm.open_position(price=100.0, direction=1, portfolio_value=10000, timestamp=ts)
    assert pm.should_exit(pos, current_signal=1) is False


def test_should_exit_force_exit():
    pm = PositionManager(max_duration_candles=12)
    ts = datetime(2025, 1, 1)
    pos = pm.open_position(price=100.0, direction=1, portfolio_value=10000, timestamp=ts)
    pos.bars_held = 12
    assert pm.should_exit(pos, current_signal=1) is True


def test_close_position_long_profit():
    pm = PositionManager(fee_pct=0.0, slippage_pct=0.0)
    ts = datetime(2025, 1, 1)
    pos = pm.open_position(price=100.0, direction=1, portfolio_value=10000, timestamp=ts)
    record = pm.close_position(pos, exit_price=110.0, timestamp=datetime(2025, 1, 2))
    assert record["pnl"] > 0
    assert record["pnl"] == 10.0
    assert record["costs"] == 0.0


def test_close_position_long_loss():
    pm = PositionManager(fee_pct=0.0, slippage_pct=0.0)
    ts = datetime(2025, 1, 1)
    pos = pm.open_position(price=100.0, direction=1, portfolio_value=10000, timestamp=ts)
    record = pm.close_position(pos, exit_price=90.0, timestamp=datetime(2025, 1, 2))
    assert record["pnl"] < 0
    assert record["pnl"] == -10.0


def test_close_position_short_profit():
    pm = PositionManager(fee_pct=0.0, slippage_pct=0.0)
    ts = datetime(2025, 1, 1)
    pos = pm.open_position(price=100.0, direction=-1, portfolio_value=10000, timestamp=ts)
    record = pm.close_position(pos, exit_price=90.0, timestamp=datetime(2025, 1, 2))
    assert record["pnl"] > 0
    assert record["pnl"] == 10.0
