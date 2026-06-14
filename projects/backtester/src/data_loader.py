from pathlib import Path

import pandas as pd


def load_predictions(path: str, pairs: list[str] = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamp" not in df.columns:
        raise ValueError("Predictions CSV missing 'timestamp' column")
    if pairs:
        df = df[df["pair"].isin(pairs)]
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def load_candles(data_dir: str, pair: str, interval: str) -> pd.DataFrame:
    path = Path(data_dir) / pair / f"{interval}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Candle file not found: {path}")
    df = pd.read_csv(path)
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def load_all_candles(data_dir: str, pairs: list[str], interval: str) -> dict[str, pd.DataFrame]:
    return {pair: load_candles(data_dir, pair, interval) for pair in pairs}
