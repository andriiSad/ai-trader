"""Buy-and-hold benchmark for comparison."""

import pandas as pd

from src.metrics import compute_all_metrics


def buy_and_hold_returns(candles_df: pd.DataFrame, initial_capital: float) -> pd.Series:
    """Equity curve from buying at first candle and holding."""
    first_price = candles_df["close"].iloc[0]
    quantity = initial_capital / first_price
    equity = quantity * candles_df["close"]
    return equity


def compute_benchmark_metrics(candles_df: pd.DataFrame, initial_capital: float) -> dict:
    """Compute same metrics as strategy for buy-and-hold."""
    equity = buy_and_hold_returns(candles_df, initial_capital)
    return compute_all_metrics(equity, [], initial_capital)
