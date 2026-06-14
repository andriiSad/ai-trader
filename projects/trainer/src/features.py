"""Feature engineering: technical indicators + lagged values from OHLCV data."""

import numpy as np
import pandas as pd


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_bollinger_bands(
    close: pd.Series, period: int = 20, std_dev: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    middle = close.rolling(window=period).mean()
    rolling_std = close.rolling(window=period).std()
    upper = middle + std_dev * rolling_std
    lower = middle - std_dev * rolling_std
    return upper, middle, lower


def compute_sma(close: pd.Series, period: int) -> pd.Series:
    return close.rolling(window=period).mean()


def compute_ema(close: pd.Series, period: int) -> pd.Series:
    return close.ewm(span=period, adjust=False).mean()


def compute_lagged_returns(
    close: pd.Series, periods: list[int] | None = None
) -> dict[str, pd.Series]:
    if periods is None:
        periods = [1, 3, 5]
    result = {}
    for p in periods:
        result[f"return_lag_{p}"] = close.pct_change(periods=p)
    return result


def compute_lagged_volume(
    volume: pd.Series, periods: list[int] | None = None
) -> dict[str, pd.Series]:
    if periods is None:
        periods = [1, 3, 5]
    result = {}
    for p in periods:
        result[f"volume_change_lag_{p}"] = volume.pct_change(periods=p)
    return result


def compute_range_features(df: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "high_low_range_pct": (df["high"] - df["low"]) / df["close"],
        "close_open_range_pct": (df["close"] - df["open"]) / df["close"],
    }


def compute_rolling_volatility(close: pd.Series, period: int) -> pd.Series:
    log_returns = np.log(close / close.shift(1))
    return log_returns.rolling(window=period).std()


def compute_williams_r(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    highest = high.rolling(window=period).max()
    lowest = low.rolling(window=period).min()
    return (highest - close) / (highest - lowest) * -100


def compute_cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
    typical_price = (high + low + close) / 3
    sma_tp = typical_price.rolling(window=period).mean()
    mad = typical_price.rolling(window=period).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
    )
    return (typical_price - sma_tp) / (0.015 * mad)


def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Master function: takes OHLCV DataFrame, returns feature matrix.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: open, high, low, close, volume.
        May contain: timestamp, deal, pair.

    Returns
    -------
    pd.DataFrame
        Feature matrix with metadata columns preserved, NaN warmup rows dropped.
    """
    close = df["close"]
    volume = df["volume"]
    high = df["high"]
    low = df["low"]

    feature_parts: list[pd.DataFrame] = []

    for period in [7, 14, 21]:
        feature_parts.append(compute_rsi(close, period).to_frame(f"rsi_{period}"))

    macd_line, macd_signal, macd_hist = compute_macd(close)
    feature_parts.append(macd_line.to_frame("macd_line"))
    feature_parts.append(macd_signal.to_frame("macd_signal"))
    feature_parts.append(macd_hist.to_frame("macd_hist"))

    bb_upper, bb_middle, bb_lower = compute_bollinger_bands(close)
    feature_parts.append(bb_upper.to_frame("bb_upper"))
    feature_parts.append(bb_middle.to_frame("bb_middle"))
    feature_parts.append(bb_lower.to_frame("bb_lower"))
    bb_bandwidth = (bb_upper - bb_lower) / bb_middle
    feature_parts.append(bb_bandwidth.to_frame("bb_bandwidth"))
    bb_pct_b = (close - bb_lower) / (bb_upper - bb_lower)
    feature_parts.append(bb_pct_b.to_frame("bb_pct_b"))

    sma_periods = [5, 10, 20, 50, 100]
    for period in sma_periods:
        feature_parts.append(compute_sma(close, period).to_frame(f"sma_{period}"))

    ema_periods = [5, 9, 12, 21, 26, 50]
    for period in ema_periods:
        feature_parts.append(compute_ema(close, period).to_frame(f"ema_{period}"))

    for period in sma_periods:
        sma_val = compute_sma(close, period)
        feature_parts.append((close / sma_val).to_frame(f"close_to_sma_{period}"))

    for period in ema_periods:
        ema_val = compute_ema(close, period)
        feature_parts.append((close / ema_val).to_frame(f"close_to_ema_{period}"))

    if len(sma_periods) >= 2:
        for i in range(len(sma_periods) - 1):
            fast = compute_sma(close, sma_periods[i])
            slow = compute_sma(close, sma_periods[i + 1])
            feature_parts.append(
                (fast / slow).to_frame(f"sma_cross_{sma_periods[i]}_{sma_periods[i + 1]}")
            )

    if len(ema_periods) >= 2:
        fast_ema = compute_ema(close, ema_periods[0])
        slow_ema = compute_ema(close, ema_periods[-1])
        feature_parts.append(
            (fast_ema / slow_ema).to_frame(f"ema_cross_{ema_periods[0]}_{ema_periods[-1]}")
        )

    lagged_returns = compute_lagged_returns(close, [1, 3, 5, 10, 20])
    for name, series in lagged_returns.items():
        feature_parts.append(series.to_frame(name))

    lagged_volume = compute_lagged_volume(volume, [1, 3, 5, 10, 20])
    for name, series in lagged_volume.items():
        feature_parts.append(series.to_frame(name))

    range_feats = compute_range_features(df)
    for name, series in range_feats.items():
        feature_parts.append(series.to_frame(name))

    for period in [10, 20]:
        feature_parts.append(compute_sma(volume, period).to_frame(f"volume_sma_{period}"))
        vol_sma = compute_sma(volume, period)
        feature_parts.append((volume / vol_sma).to_frame(f"volume_to_sma_{period}"))

    for period in [5, 10, 20]:
        feature_parts.append(
            compute_rolling_volatility(close, period).to_frame(f"volatility_{period}")
        )

    roc_periods = [5, 10]
    for period in roc_periods:
        feature_parts.append(close.pct_change(periods=period).to_frame(f"roc_{period}"))

    feature_parts.append(compute_williams_r(high, low, close, 14).to_frame("williams_r_14"))
    feature_parts.append(compute_cci(high, low, close, 20).to_frame("cci_20"))

    highest_20 = high.rolling(window=20).max()
    lowest_20 = low.rolling(window=20).min()
    feature_parts.append(
        ((close - lowest_20) / (highest_20 - lowest_20)).to_frame("close_position_20")
    )

    highest_50 = high.rolling(window=50).max()
    lowest_50 = low.rolling(window=50).min()
    feature_parts.append(
        ((close - lowest_50) / (highest_50 - lowest_50)).to_frame("close_position_50")
    )

    features_df = pd.concat(feature_parts, axis=1)

    meta_cols = []
    if "timestamp" in df.columns:
        meta_cols.append("timestamp")
    if "pair" in df.columns:
        meta_cols.append("pair")

    result = pd.concat([df[meta_cols], features_df], axis=1) if meta_cols else features_df.copy()

    result = result.replace([np.inf, -np.inf], np.nan)
    result = result.dropna()
    return result
