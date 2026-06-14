"""Walk-forward split engine for time-series cross-validation."""

from __future__ import annotations

import re

import pandas as pd


def _normalize_offset(alias: str) -> str:
    """Convert legacy pandas offset aliases (M, Y) to current (ME, YE)."""
    return re.sub(r"(\d+)M$", r"\1ME", re.sub(r"(\d+)Y$", r"\1YE", alias))


def walk_forward_splits(
    df: pd.DataFrame,
    train_window: str = "3M",
    test_window: str = "1M",
    step: str = "1M",
) -> list[dict]:
    """Generate walk-forward train/test splits.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a ``timestamp`` column (datetime or unix seconds).
    train_window : str
        Training window size (pandas offset alias, e.g. ``"3M"`` = 3 months).
    test_window : str
        Test window size.
    step : str
        How far to slide forward each fold.

    Returns
    -------
    list[dict]
        Each dict contains:
        - fold_id: int
        - train_start: pd.Timestamp
        - train_end: pd.Timestamp
        - test_start: pd.Timestamp
        - test_end: pd.Timestamp
        - train_mask: pd.Series[bool]
        - test_mask: pd.Series[bool]
    """
    ts = df["timestamp"]
    if not pd.api.types.is_datetime64_any_dtype(ts):
        ts = pd.to_datetime(ts, unit="s")

    data_start = ts.min()
    data_end = ts.max()

    train_offset = pd.tseries.frequencies.to_offset(_normalize_offset(train_window))
    test_offset = pd.tseries.frequencies.to_offset(_normalize_offset(test_window))
    step_offset = pd.tseries.frequencies.to_offset(_normalize_offset(step))

    folds: list[dict] = []
    fold_id = 0
    train_start = data_start

    while True:
        train_end = train_start + train_offset
        test_start = train_end
        test_end = test_start + test_offset

        if test_end > data_end + pd.Timedelta(milliseconds=1):
            break

        train_mask = (ts >= train_start) & (ts < train_end)
        test_mask = (ts >= test_start) & (ts < test_end)

        if train_mask.sum() == 0 or test_mask.sum() == 0:
            train_start += step_offset
            fold_id += 1
            continue

        folds.append(
            {
                "fold_id": fold_id,
                "train_start": train_start,
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
                "train_mask": train_mask,
                "test_mask": test_mask,
            }
        )

        train_start += step_offset
        fold_id += 1

    return folds
