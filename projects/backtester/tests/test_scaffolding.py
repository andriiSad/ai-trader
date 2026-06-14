"""Tests for backtester CLI scaffolding."""

import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

PROJECT_DIR = Path(__file__).parent.parent


def test_config_loads_correctly():
    config_path = PROJECT_DIR / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    assert config["pairs"] == ["BTC_USDT", "ETH_USDT"]
    assert config["interval"] == "5m"
    assert config["strategy"]["threshold"] == 0.55
    assert config["strategy"]["long_only"] is True
    assert config["costs"]["fee_pct"] == 0.1
    assert config["costs"]["slippage_pct"] == 0.05
    assert config["position"]["risk_per_trade"] == 0.01
    assert config["position"]["max_duration_candles"] == 12
    assert "predictions_path" in config
    assert "candles_dir" in config
    assert "results_path" in config["output"]
    assert "tear_sheet_path" in config["output"]


def test_run_command_exits_zero(tmp_path):
    timestamps = pd.date_range("2025-01-01", periods=30, freq="5min")
    for pair in ["BTC_USDT", "ETH_USDT"]:
        base_price = 100.0
        rows = [
            {
                "timestamp": ts.isoformat(),
                "open": base_price + i * 0.5,
                "high": base_price + i * 0.5 + 1,
                "low": base_price + i * 0.5 - 1,
                "close": base_price + i * 0.5,
                "volume": 1000.0,
            }
            for i, ts in enumerate(timestamps)
        ]
        pair_dir = tmp_path / "candles" / pair
        pair_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(pair_dir / "5m.csv", index=False)

    pred_rows = []
    for pair in ["BTC_USDT", "ETH_USDT"]:
        for i, ts in enumerate(timestamps):
            pred_rows.append(
                {
                    "pair": pair,
                    "timestamp": ts.isoformat(),
                    "probability": 0.8 if i < 15 else 0.3,
                    "fold_id": 0 if i < 15 else 1,
                    "fold_test_start": timestamps[0].isoformat()
                    if i < 15
                    else timestamps[15].isoformat(),
                    "fold_test_end": timestamps[15].isoformat()
                    if i < 15
                    else timestamps[-1].isoformat(),
                }
            )
    pred_path = tmp_path / "predictions.csv"
    pd.DataFrame(pred_rows).to_csv(pred_path, index=False)

    config = {
        "pairs": ["BTC_USDT", "ETH_USDT"],
        "interval": "5m",
        "predictions_path": str(pred_path),
        "candles_dir": str(tmp_path / "candles"),
        "strategy": {"threshold": 0.55, "long_only": True},
        "costs": {"fee_pct": 0.1, "slippage_pct": 0.05},
        "position": {"risk_per_trade": 0.01, "max_duration_candles": 12},
        "output": {
            "results_path": str(tmp_path / "results.json"),
            "tear_sheet_path": str(tmp_path / "tear.html"),
        },
    }
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    result = subprocess.run(
        [sys.executable, str(PROJECT_DIR / "backtester.py"), "run", "--config", str(config_path)],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_DIR),
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert "Backtest complete" in result.stdout


def test_dashboard_command_exits_zero():
    result = subprocess.run(
        [sys.executable, str(PROJECT_DIR / "backtester.py"), "dashboard"],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_DIR),
    )
    assert result.returncode == 0
    assert "streamlit run" in result.stdout


def test_config_flag_override(tmp_path):
    timestamps = pd.date_range("2025-01-01", periods=10, freq="5min")
    for pair in ["SOL_USDT"]:
        rows = [
            {
                "timestamp": ts.isoformat(),
                "open": 50.0,
                "high": 51.0,
                "low": 49.0,
                "close": 50.0,
                "volume": 500.0,
            }
            for ts in timestamps
        ]
        pair_dir = tmp_path / "candles" / pair
        pair_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(pair_dir / "1h.csv", index=False)

    pred_rows = [
        {
            "pair": "SOL_USDT",
            "timestamp": ts.isoformat(),
            "probability": 0.7,
            "fold_id": 0,
            "fold_test_start": timestamps[0].isoformat(),
            "fold_test_end": timestamps[-1].isoformat(),
        }
        for ts in timestamps
    ]
    pred_path = tmp_path / "predictions.csv"
    pd.DataFrame(pred_rows).to_csv(pred_path, index=False)

    config = {
        "pairs": ["SOL_USDT"],
        "interval": "1h",
        "predictions_path": str(pred_path),
        "candles_dir": str(tmp_path / "candles"),
        "strategy": {"threshold": 0.55, "long_only": True},
        "costs": {"fee_pct": 0.1, "slippage_pct": 0.05},
        "position": {"risk_per_trade": 0.01, "max_duration_candles": 12},
        "output": {
            "results_path": str(tmp_path / "results.json"),
            "tear_sheet_path": str(tmp_path / "tear.html"),
        },
    }
    custom_config = tmp_path / "custom.yaml"
    with open(custom_config, "w") as f:
        yaml.dump(config, f)

    result = subprocess.run(
        [sys.executable, str(PROJECT_DIR / "backtester.py"), "run", "--config", str(custom_config)],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_DIR),
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"


def test_invalid_command_fails():
    result = subprocess.run(
        [sys.executable, str(PROJECT_DIR / "backtester.py"), "invalid"],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_DIR),
    )
    assert result.returncode != 0
