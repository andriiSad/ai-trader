from __future__ import annotations

import numpy as np
import pandas as pd


class NetflowAdapter:
    def fetch(self, timestamps: pd.Series) -> pd.DataFrame:
        raise NotImplementedError


class MockNetflowAdapter(NetflowAdapter):
    def __init__(self, seed: int = 42) -> None:
        self._seed = seed

    def fetch(self, timestamps: pd.Series) -> pd.DataFrame:
        rng = np.random.default_rng(self._seed)
        n = len(timestamps)
        netflow = rng.standard_normal(n) * 1000
        spikes = rng.random(n) < 0.05
        netflow[spikes] *= 10
        return pd.DataFrame({"timestamp": timestamps.values, "netflow": netflow})


def generate(df: pd.DataFrame, seed: int = 42, **kwargs) -> pd.DataFrame:
    adapter = MockNetflowAdapter(seed=seed)
    raw = adapter.fetch(df["timestamp"])
    result = raw.copy()
    result["netflow_raw"] = result["netflow"]
    result["netflow_cum_3"] = result["netflow_raw"].rolling(window=3).sum()
    rolling_mean = result["netflow_raw"].rolling(window=20).mean()
    rolling_std = result["netflow_raw"].rolling(window=20).std()
    result["netflow_zscore_20"] = (result["netflow_raw"] - rolling_mean) / rolling_std
    result = result.drop(columns=["netflow"])
    result = result.replace([np.inf, -np.inf], np.nan)
    return result.reset_index(drop=True)
