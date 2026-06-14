from __future__ import annotations

import pandas as pd


def generate_labels(df: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    future_close = df["close"].shift(-horizon)
    label = (future_close > df["close"]).astype(int)
    result = df[["timestamp"]].copy()
    result["label"] = label
    result = result.iloc[: len(result) - horizon]
    return result.reset_index(drop=True)
