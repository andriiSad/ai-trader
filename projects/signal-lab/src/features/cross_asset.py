from __future__ import annotations

from pathlib import Path

import pandas as pd


def _load_eth_data(data_dir: str, interval: str) -> pd.DataFrame:
    path = Path(data_dir) / "ETH_USDT" / f"{interval}.csv"
    if not path.exists():
        raise FileNotFoundError(f"ETH data not found at {path}")
    return pd.read_csv(path, parse_dates=["timestamp"])


def generate(
    df: pd.DataFrame,
    data_dir: str = "data",
    interval: str = "4h",
    **kwargs,
) -> pd.DataFrame:
    eth_df = _load_eth_data(data_dir, interval)
    merged = df[["timestamp", "close"]].merge(
        eth_df[["timestamp", "close"]],
        on="timestamp",
        how="inner",
        suffixes=("_btc", "_eth"),
    )
    btc_returns = merged["close_btc"].pct_change()
    eth_returns = merged["close_eth"].pct_change()

    corr_14 = btc_returns.rolling(window=14).corr(eth_returns)
    corr_28 = btc_returns.rolling(window=28).corr(eth_returns)
    ratio = merged["close_btc"] / merged["close_eth"]

    result = merged[["timestamp"]].copy()
    result["btc_eth_corr_14"] = corr_14
    result["btc_eth_corr_28"] = corr_28
    result["btc_eth_ratio"] = ratio

    result = result.iloc[28:].reset_index(drop=True)
    return result
