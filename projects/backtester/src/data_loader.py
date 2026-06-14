from pathlib import Path

import pandas as pd


def load_predictions(path: str, pairs: list[str] = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamp" not in df.columns:
        raise ValueError("Predictions CSV missing 'timestamp' column")
    for col in ["timestamp", "fold_test_start", "fold_test_end"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col]).dt.strftime("%Y-%m-%d %H:%M:%S")
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
        if df["timestamp"].dtype in ("int64", "float64"):
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s").dt.strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        else:
            df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S")
        df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def load_all_candles(
    data_dir: str, pairs: list[str], interval: str
) -> dict[str, pd.DataFrame]:
    return {pair: load_candles(data_dir, pair, interval) for pair in pairs}
