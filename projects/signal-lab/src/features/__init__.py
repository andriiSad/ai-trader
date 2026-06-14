from __future__ import annotations

import importlib

import pandas as pd

from src.features.ohlcv import (
    FEATURE_FUNCS,
    atr_14,
    bb_pct_b,
    ema_cross_9_21,
    macd_hist,
    obv,
    return_lag_1,
    return_lag_5,
    rsi_14,
    sma_cross_5_20,
    volume_ratio,
)
from src.features.ohlcv import (
    generate as _ohlcv_generate,
)


def compose(*feature_dfs: pd.DataFrame) -> pd.DataFrame:
    if len(feature_dfs) == 0:
        raise ValueError("At least one feature DataFrame is required")
    if len(feature_dfs) == 1:
        return feature_dfs[0]
    result = feature_dfs[0]
    for df in feature_dfs[1:]:
        result = result.merge(df, on="timestamp", how="inner")
    result = result.replace([float("inf"), float("-inf")], float("nan"))
    result = result.dropna()
    return result.reset_index(drop=True)


def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    return _ohlcv_generate(df)


def load_module(name: str):
    return importlib.import_module(f"src.features.{name}")


def generate_from_modules(
    df: pd.DataFrame,
    module_names: list[str],
    **kwargs,
) -> pd.DataFrame:
    feature_dfs = []
    for name in module_names:
        mod = load_module(name)
        feature_dfs.append(mod.generate(df, **kwargs))
    return compose(*feature_dfs)


__all__ = [
    "FEATURE_FUNCS",
    "atr_14",
    "bb_pct_b",
    "compose",
    "ema_cross_9_21",
    "generate_features",
    "generate_from_modules",
    "load_module",
    "macd_hist",
    "obv",
    "return_lag_1",
    "return_lag_5",
    "rsi_14",
    "sma_cross_5_20",
    "volume_ratio",
]
