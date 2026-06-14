"""Tests for cost computation functions."""

from src.costs import compute_fee, compute_round_trip_cost, compute_slippage


def test_fee_basic():
    assert compute_fee(1000, 1.0) == 10.0


def test_fee_zero():
    assert compute_fee(1000, 0.0) == 0.0


def test_slippage_buy():
    slipped = compute_slippage(100.0, 0.1, 1)
    assert slipped > 100.0
    assert slipped == 100.0 * 1.001


def test_slippage_sell():
    slipped = compute_slippage(100.0, 0.1, -1)
    assert slipped < 100.0
    assert slipped == 100.0 * 0.999


def test_slippage_zero():
    assert compute_slippage(100.0, 0.0, 1) == 100.0
    assert compute_slippage(100.0, 0.0, -1) == 100.0


def test_round_trip_cost():
    result = compute_round_trip_cost(
        entry_price=100.0,
        exit_price=110.0,
        quantity=1.0,
        fee_pct=0.1,
        slippage_pct=0.05,
        direction=1,
    )
    slipped_entry = 100.0 * 1.0005
    slipped_exit = 110.0 * 0.9995
    entry_fee = slipped_entry * 1.0 * 0.001
    exit_fee = slipped_exit * 1.0 * 0.001
    gross_pnl = (slipped_exit - slipped_entry) * 1.0
    expected = gross_pnl - entry_fee - exit_fee
    assert abs(result - expected) < 1e-10
