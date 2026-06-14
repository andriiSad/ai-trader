from __future__ import annotations

import numpy as np
import pandas as pd


def generate(
    df: pd.DataFrame, data_dir: str = "data", pair: str = "BTC_USDT", **kwargs
) -> pd.DataFrame:
    """Generate funding rate features from candle data.

    Loads funding rate data from CSV, forward-fills to candle timestamps,
    and computes derived features.

    Args:
        df: OHLCV DataFrame with 'timestamp' column.
        data_dir: Root data directory containing funding CSVs.
        pair: Trading pair name (default "BTC_USDT").

    Returns:
        DataFrame with columns: timestamp, funding_rate_raw,
        funding_rate_cum_3, funding_rate_roc.
    """
    from pathlib import Path

    funding_path = Path(data_dir) / pair / "funding.csv"
    if not funding_path.exists():
        return (
            df[["timestamp"]]
            .iloc[0:0]
            .copy()
            .assign(
                funding_rate_raw=pd.Series(dtype=float),
                funding_rate_cum_3=pd.Series(dtype=float),
                funding_rate_roc=pd.Series(dtype=float),
            )
        )

    funding_df = pd.read_csv(funding_path)
    if funding_df.empty:
        return (
            df[["timestamp"]]
            .iloc[0:0]
            .copy()
            .assign(
                funding_rate_raw=pd.Series(dtype=float),
                funding_rate_cum_3=pd.Series(dtype=float),
                funding_rate_roc=pd.Series(dtype=float),
            )
        )

    funding_df["timestamp"] = funding_df["timestamp"].astype(int)
    funding_df["funding_rate"] = funding_df["funding_rate"].astype(float)

    candle_ts = df[["timestamp"]].copy()
    candle_ts["timestamp"] = candle_ts["timestamp"].astype(int)

    candle_max = candle_ts["timestamp"].max()
    funding_max = funding_df["timestamp"].max()
    if candle_max > 1e12 and funding_max < 1e12:
        funding_df["timestamp"] = funding_df["timestamp"] * 1000
    elif candle_max < 1e12 and funding_max > 1e12:
        funding_df["timestamp"] = funding_df["timestamp"] // 1000

    merged = candle_ts.merge(
        funding_df[["timestamp", "funding_rate"]],
        on="timestamp",
        how="left",
    )

    merged["funding_rate"] = merged["funding_rate"].ffill()

    if merged.empty:
        return (
            df[["timestamp"]]
            .iloc[0:0]
            .copy()
            .assign(
                funding_rate_raw=pd.Series(dtype=float),
                funding_rate_cum_3=pd.Series(dtype=float),
                funding_rate_roc=pd.Series(dtype=float),
            )
        )

    result = merged[["timestamp"]].copy()
    result["funding_rate_raw"] = merged["funding_rate"].values
    result["funding_rate_cum_3"] = (
        merged["funding_rate"].rolling(window=3, min_periods=3).sum().values
    )

    prev = merged["funding_rate"].shift(1)
    result["funding_rate_roc"] = (merged["funding_rate"] - prev) / prev.abs()

    result = result.replace([np.inf, -np.inf], np.nan)
    return result.reset_index(drop=True)
