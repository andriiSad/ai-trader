"""Tests for ThresholdStrategy."""

from src.strategy import ThresholdStrategy


def test_long_only_above_threshold():
    s = ThresholdStrategy(threshold=0.55, long_only=True)
    assert s.signal(0.6) == 1


def test_long_only_below_threshold():
    s = ThresholdStrategy(threshold=0.55, long_only=True)
    assert s.signal(0.5) == 0


def test_long_only_at_threshold():
    s = ThresholdStrategy(threshold=0.55, long_only=True)
    assert s.signal(0.55) == 0


def test_two_way_high():
    s = ThresholdStrategy(threshold=0.55, long_only=False)
    assert s.signal(0.7) == 1


def test_two_way_low():
    s = ThresholdStrategy(threshold=0.55, long_only=False)
    assert s.signal(0.3) == -1


def test_two_way_middle():
    s = ThresholdStrategy(threshold=0.55, long_only=False)
    assert s.signal(0.5) == 0


def test_edge_threshold_zero():
    s = ThresholdStrategy(threshold=0.0, long_only=True)
    assert s.signal(0.01) == 1
    assert s.signal(0.5) == 1
    assert s.signal(0.0) == 0


def test_edge_threshold_one():
    s = ThresholdStrategy(threshold=1.0, long_only=True)
    assert s.signal(0.99) == 0
    assert s.signal(0.5) == 0
    assert s.signal(1.0) == 0


def test_edge_probability_exactly_half():
    s = ThresholdStrategy(threshold=0.55, long_only=True)
    assert s.signal(0.5) == 0
    s2 = ThresholdStrategy(threshold=0.55, long_only=False)
    assert s2.signal(0.5) == 0
