"""Load candle CSV files into pandas DataFrames."""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

CANDLE_DTYPES = {
    "timestamp": "int64",
    "open": "float64",
    "high": "float64",
    "low": "float64",
    "close": "float64",
    "volume": "float64",
    "deal": "float64",
}

CANDLE_COLUMNS = list(CANDLE_DTYPES.keys())


def load_candles(path: str, pair: str) -> pd.DataFrame:
    """Read a 5m candle CSV and return a DataFrame with proper dtypes.

    Parameters
    ----------
    path : str
        Path to the CSV file (e.g. ``data/BTC_USDT/5m.csv``).
    pair : str
        Trading pair name (e.g. ``BTC_USDT``). Used only for logging.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns matching CANDLE_COLUMNS. Returns an empty
        DataFrame with the correct schema if the file is missing or unreadable.
    """
    csv_path = Path(path)
    if not csv_path.exists():
        logger.warning("File not found for %s: %s", pair, csv_path)
        return pd.DataFrame(columns=CANDLE_COLUMNS)

    try:
        df = pd.read_csv(csv_path, dtype=CANDLE_DTYPES)
    except Exception:
        logger.warning("Failed to read CSV for %s: %s", pair, csv_path, exc_info=True)
        return pd.DataFrame(columns=CANDLE_COLUMNS)

    missing_cols = set(CANDLE_COLUMNS) - set(df.columns)
    if missing_cols:
        logger.warning("Missing columns %s in %s", missing_cols, csv_path)
        for col in missing_cols:
            df[col] = 0.0

    df = df[CANDLE_COLUMNS]
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def load_all_pairs(
    data_dir: str, pairs: list[str], interval: str = "5m"
) -> dict[str, pd.DataFrame]:
    """Load candle CSVs for multiple trading pairs.

    Parameters
    ----------
    data_dir : str
        Root data directory (e.g. ``data``).
    pairs : list[str]
        List of pair names (e.g. ``["BTC_USDT", "ETH_USDT"]``).
    interval : str
        Candle interval, used to locate the CSV filename.

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping of pair name → DataFrame.
    """
    result = {}
    for pair in pairs:
        csv_path = Path(data_dir) / pair / f"{interval}.csv"
        result[pair] = load_candles(str(csv_path), pair)
    return result
