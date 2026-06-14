"""Label generation: binary UP/DOWN labels from forward returns."""

import pandas as pd


def generate_labels(df: pd.DataFrame, horizon: int = 5, threshold: float = 0.0) -> pd.Series:
    """Generate binary UP/DOWN labels based on forward return.

    For each row, compute the return over the next ``horizon`` bars::

        return = (close[t + horizon] - close[t]) / close[t]

    If return > threshold: label = 1 (UP)
    Else: label = 0 (DOWN)

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a ``close`` column.
    horizon : int
        Number of bars to look ahead (default 5 = 25 minutes for 5m candles).
    threshold : float
        Minimum return to qualify as UP (default 0.0 = any positive move).

    Returns
    -------
    pd.Series
        Binary labels aligned with input index. Last ``horizon`` rows are NaN.
    """
    future_close = df["close"].shift(-horizon)
    forward_return = (future_close - df["close"]) / df["close"]
    labels = (forward_return > threshold).astype(float)
    labels.iloc[-horizon:] = float("nan")
    return labels
