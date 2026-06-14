from __future__ import annotations

import numpy as np
import pandas as pd


def rsi_14(df: pd.DataFrame) -> pd.Series:
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def macd_hist(df: pd.DataFrame) -> pd.Series:
    ema_fast = df["close"].ewm(span=12, adjust=False).mean()
    ema_slow = df["close"].ewm(span=26, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    return macd_line - signal_line


def bb_pct_b(df: pd.DataFrame) -> pd.Series:
    sma = df["close"].rolling(window=20).mean()
    std = df["close"].rolling(window=20).std()
    upper = sma + 2.0 * std
    lower = sma - 2.0 * std
    return (df["close"] - lower) / (upper - lower)


def atr_14(df: pd.DataFrame) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()


def obv(df: pd.DataFrame) -> pd.Series:
    sign = np.sign(df["close"].diff())
    sign.iloc[0] = 0
    return (sign * df["volume"]).cumsum()


def sma_cross_5_20(df: pd.DataFrame) -> pd.Series:
    sma_5 = df["close"].rolling(window=5).mean()
    sma_20 = df["close"].rolling(window=20).mean()
    return sma_5 / sma_20


def ema_cross_9_21(df: pd.DataFrame) -> pd.Series:
    ema_9 = df["close"].ewm(span=9, adjust=False).mean()
    ema_21 = df["close"].ewm(span=21, adjust=False).mean()
    return ema_9 / ema_21


def volume_ratio(df: pd.DataFrame) -> pd.Series:
    sma_vol = df["volume"].rolling(window=20).mean()
    return df["volume"] / sma_vol


def return_lag_1(df: pd.DataFrame) -> pd.Series:
    return df["close"].pct_change(1)


def return_lag_5(df: pd.DataFrame) -> pd.Series:
    return df["close"].pct_change(5)


FEATURE_FUNCS = {
    "rsi_14": rsi_14,
    "macd_hist": macd_hist,
    "bb_pct_b": bb_pct_b,
    "atr_14": atr_14,
    "obv": obv,
    "sma_cross_5_20": sma_cross_5_20,
    "ema_cross_9_21": ema_cross_9_21,
    "volume_ratio": volume_ratio,
    "return_lag_1": return_lag_1,
    "return_lag_5": return_lag_5,
}


def generate(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    result = df[["timestamp"]].copy()
    for name, func in FEATURE_FUNCS.items():
        result[name] = func(df)
    result = result.replace([np.inf, -np.inf], np.nan)
    result = result.dropna()
    return result.reset_index(drop=True)
