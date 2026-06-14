"""Tests for evaluation metrics."""

import numpy as np
import pandas as pd
import pytest
from src.metrics import compute_metrics, compute_pair_metrics


def test_perfect_predictions():
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([0.9, 0.8, 0.1, 0.2])
    m = compute_metrics(y_true, y_pred)
    assert m["accuracy"] == 1.0
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["f1"] == 1.0
    assert m["confusion_matrix"] == [[2, 0], [0, 2]]
    assert m["sample_count"] == 4


def test_all_wrong():
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([0.1, 0.2, 0.9, 0.8])
    m = compute_metrics(y_true, y_pred)
    assert m["accuracy"] == 0.0
    assert m["precision"] == 0.0
    assert m["recall"] == 0.0
    assert m["f1"] == 0.0


def test_confusion_matrix_values():
    y_true = np.array([1, 0, 1, 0, 1])
    y_pred = np.array([0.9, 0.6, 0.4, 0.1, 0.8])
    m = compute_metrics(y_true, y_pred, threshold=0.5)
    # pred: [1, 1, 0, 0, 1]
    # true: [1, 0, 1, 0, 1]
    # tp=2 (idx 0,4), tn=1 (idx 3), fp=1 (idx 1), fn=1 (idx 2)
    assert m["confusion_matrix"] == [[1, 1], [1, 2]]
    assert m["accuracy"] == pytest.approx(3 / 5)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 3)


def test_sample_count():
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    y_pred = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6])
    m = compute_metrics(y_true, y_pred)
    assert m["sample_count"] == 8


def test_threshold_impact():
    y_true = np.array([1, 0])
    y_pred = np.array([0.6, 0.6])
    m_low = compute_metrics(y_true, y_pred, threshold=0.5)
    m_high = compute_metrics(y_true, y_pred, threshold=0.7)
    # threshold=0.5: pred=[1,1] → tp=1, fp=1, fn=0, tn=0
    assert m_low["accuracy"] == pytest.approx(0.5)
    # threshold=0.7: pred=[0,0] → tp=0, fp=0, fn=1, tn=1
    assert m_high["accuracy"] == pytest.approx(0.5)


def test_empty_inputs():
    y_true = np.array([])
    y_pred = np.array([])
    m = compute_metrics(y_true, y_pred)
    assert m["accuracy"] == 0.0
    assert m["sample_count"] == 0


def test_per_pair_metrics():
    df = pd.DataFrame(
        {
            "pair": ["BTC_USDT"] * 4 + ["ETH_USDT"] * 4,
            "label": [1, 1, 0, 0, 1, 0, 1, 0],
        }
    )
    predictions = np.array([0.9, 0.8, 0.1, 0.2, 0.9, 0.8, 0.1, 0.2])
    result = compute_pair_metrics(df, predictions, ["BTC_USDT", "ETH_USDT"])

    assert "BTC_USDT" in result
    assert "ETH_USDT" in result
    assert result["BTC_USDT"]["accuracy"] == 1.0
    # ETH: true=[1,0,1,0], pred=[1,1,0,0] → tp=1, tn=1, fp=1, fn=1
    assert result["ETH_USDT"]["accuracy"] == pytest.approx(0.5)


def test_per_pair_missing_pair():
    df = pd.DataFrame({"pair": ["BTC_USDT"] * 3, "label": [1, 0, 1]})
    predictions = np.array([0.9, 0.1, 0.8])
    result = compute_pair_metrics(df, predictions, ["BTC_USDT", "XRP_USDT"])
    assert "BTC_USDT" in result
    assert "XRP_USDT" not in result
