from __future__ import annotations

import numpy as np
import pandas as pd


def fit_scaler(feature_df: pd.DataFrame) -> dict:
    numeric = feature_df.drop(columns=["timestamp"], errors="ignore")
    std = numeric.std()
    std = std.replace(0, 1)
    return {"mean": numeric.mean(), "std": std}


def transform(feature_df: pd.DataFrame, scaler: dict) -> pd.DataFrame:
    timestamp = feature_df["timestamp"] if "timestamp" in feature_df.columns else None
    numeric = feature_df.drop(columns=["timestamp"], errors="ignore")
    scaled = (numeric - scaler["mean"]) / scaler["std"]
    scaled = scaled.replace([np.inf, -np.inf], np.nan)
    scaled = scaled.dropna()
    if timestamp is not None:
        scaled["timestamp"] = timestamp.loc[scaled.index].values
    return scaled.reset_index(drop=True)
