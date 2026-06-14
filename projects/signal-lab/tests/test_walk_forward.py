import numpy as np
import pandas as pd
from src.walk_forward import split


def _make_df(years: int = 2) -> pd.DataFrame:
    start = pd.Timestamp("2022-01-01", tz="UTC")
    end = start + pd.DateOffset(years=years)
    dates = pd.date_range(start, end, freq="4h", inclusive="left")
    rng = np.random.default_rng(42)
    close = 100 + np.cumsum(rng.normal(0, 1, len(dates)))
    return pd.DataFrame(
        {
            "timestamp": dates.astype(np.int64).values // 10**3,
            "open": close + rng.normal(0, 0.5, len(dates)),
            "high": close + abs(rng.normal(0, 1, len(dates))),
            "low": close - abs(rng.normal(0, 1, len(dates))),
            "close": close,
            "volume": rng.uniform(100, 1000, len(dates)),
            "deal": rng.integers(0, 100, len(dates)),
        }
    )


def test_fold_count():
    df = _make_df(2)
    folds = list(split(df, train_months=6, test_months=2))
    assert len(folds) >= 5, f"Expected at least 5 folds, got {len(folds)}"


def test_no_overlap_between_test_folds():
    df = _make_df(2)
    folds = list(split(df, train_months=6, test_months=2))
    test_timestamps = []
    for _, test_df in folds:
        test_timestamps.append(set(test_df["timestamp"].values))
    for i in range(len(test_timestamps)):
        for j in range(i + 1, len(test_timestamps)):
            overlap = test_timestamps[i] & test_timestamps[j]
            assert len(overlap) == 0, f"Overlap between fold {i} and {j}: {len(overlap)} rows"


def test_train_test_sizes_reasonable():
    df = _make_df(2)
    folds = list(split(df, train_months=6, test_months=2))
    for i, (train_df, test_df) in enumerate(folds):
        assert len(train_df) > 100, f"Fold {i} train too small: {len(train_df)}"
        assert len(test_df) > 100, f"Fold {i} test too small: {len(test_df)}"
        assert len(train_df) > len(test_df), f"Fold {i} train should be larger than test"


def test_generator_yields_tuples():
    df = _make_df(2)
    for item in split(df, train_months=6, test_months=2):
        assert isinstance(item, tuple)
        assert len(item) == 2
        train_df, test_df = item
        assert isinstance(train_df, pd.DataFrame)
        assert isinstance(test_df, pd.DataFrame)
        break
