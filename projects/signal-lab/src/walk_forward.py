from __future__ import annotations

from collections.abc import Iterator

import pandas as pd


def split(
    df: pd.DataFrame,
    train_months: int = 6,
    test_months: int = 2,
) -> Iterator[tuple[pd.DataFrame, pd.DataFrame]]:
    ts = df["timestamp"]
    if ts.dtype == "int64" or ts.dtype == "float64":
        dates = pd.to_datetime(ts, unit="ms", utc=True)
    else:
        dates = pd.to_datetime(ts, utc=True)

    start = dates.min()
    end = dates.max()

    train_start = start
    while True:
        train_end = train_start + pd.DateOffset(months=train_months)
        test_end = train_end + pd.DateOffset(months=test_months)

        if test_end > end + pd.Timedelta(days=1):
            break

        train_mask = (dates >= train_start) & (dates < train_end)
        test_mask = (dates >= train_end) & (dates < test_end)

        train_df = df.loc[train_mask].reset_index(drop=True)
        test_df = df.loc[test_mask].reset_index(drop=True)

        if len(train_df) == 0 or len(test_df) == 0:
            train_start = train_end + pd.DateOffset(months=test_months)
            continue

        yield train_df, test_df

        train_start = train_start + pd.DateOffset(months=test_months)
