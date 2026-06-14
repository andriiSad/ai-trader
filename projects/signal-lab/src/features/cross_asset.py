from __future__ import annotations

from pathlib import Path

import pandas as pd


def _load_eth_data(data_dir: str, interval: str) -> pd.DataFrame:
    path = Path(data_dir) / "ETH_USDT" / f"{interval}.csv"
    if not path.exists():
        raise FileNotFoundError(f"ETH data not found at {path}")
    return pd.read_csv(path)


def _normalize_timestamps(source_ts: pd.Series, target_ts: pd.Series) -> pd.Series:
    target_is_datetime = hasattr(getattr(target_ts, "dtype", None), "tz") or str(getattr(target_ts, "dtype", "")).startswith("datetime")
    if target_is_datetime:
        return pd.to_datetime(source_ts, errors="coerce")
    source_numeric = pd.to_numeric(source_ts, errors="coerce")
    target_numeric = pd.to_numeric(target_ts, errors="coerce")
    source_max = source_numeric.max()
    target_max = target_numeric.max()
    if pd.isna(source_max) or pd.isna(target_max):
        return source_ts
    if target_max > 1e12 and source_max < 1e12:
        return source_numeric * 1000
    if source_max > 1e12 and target_max < 1e12:
        return source_numeric // 1000
    return source_ts


def generate(
    df: pd.DataFrame,
    data_dir: str = "data",
    interval: str = "4h",
    **kwargs,
) -> pd.DataFrame:
    eth_df = _load_eth_data(data_dir, interval)
    eth_df["timestamp"] = _normalize_timestamps(eth_df["timestamp"], df["timestamp"])

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

    return result.reset_index(drop=True)
