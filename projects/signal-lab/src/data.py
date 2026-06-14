from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta

import pandas as pd
from binance.client import Client

COLUMNS = ["timestamp", "open", "high", "low", "close", "volume", "deal"]
MAX_LIMIT = 1000


def download_candles(
    pair: str,
    interval: str,
    start_date: str | None,
    output_dir: str,
) -> pd.DataFrame:
    """Download candles from Binance REST API and save as CSV.

    Args:
        pair: Trading pair (e.g. "BTC_USDT" or "BTCUSDT").
        interval: Candle interval (e.g. "4h").
        start_date: Start date string (ISO format) or None for 2+ years ago.
        output_dir: Directory to save CSV files.

    Returns:
        DataFrame with candle data.
    """
    symbol = pair.replace("_", "")
    client = Client()

    if start_date:
        start_ts = int(datetime.fromisoformat(start_date).timestamp() * 1000)
    else:
        two_years_ago = datetime.now(UTC) - timedelta(days=730)
        start_ts = int(two_years_ago.timestamp() * 1000)

    all_candles: list[list] = []
    current_start = start_ts

    while True:
        klines = client.get_klines(
            symbol=symbol,
            interval=interval,
            startTime=current_start,
            limit=MAX_LIMIT,
        )
        if not klines:
            break

        all_candles.extend(klines)

        if len(klines) < MAX_LIMIT:
            break

        last_open_time = klines[-1][0]
        current_start = last_open_time + 1

    rows = []
    for k in all_candles:
        rows.append(
            {
                "timestamp": k[0],
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5]),
                "deal": int(k[8]),
            }
        )

    df = pd.DataFrame(rows, columns=COLUMNS)

    pair_dir = os.path.join(output_dir, pair)
    os.makedirs(pair_dir, exist_ok=True)
    csv_path = os.path.join(pair_dir, f"{interval}.csv")
    df.to_csv(csv_path, index=False)

    return df


def load_candles(
    pair: str,
    interval: str,
    data_dir: str,
) -> pd.DataFrame:
    """Load candles from a CSV file.

    Args:
        pair: Trading pair (e.g. "BTC_USDT").
        interval: Candle interval (e.g. "4h").
        data_dir: Directory containing CSV files.

    Returns:
        DataFrame with candle data and proper dtypes.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
    """
    csv_path = os.path.join(data_dir, pair, f"{interval}.csv")
    df = pd.read_csv(csv_path)
    df["timestamp"] = df["timestamp"].astype(int)
    df["deal"] = df["deal"].astype(int)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    return df
