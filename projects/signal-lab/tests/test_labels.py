import numpy as np
import pandas as pd
from src.labels import generate_labels


def _make_df(closes: list[float]) -> pd.DataFrame:
    n = len(closes)
    return pd.DataFrame(
        {
            "timestamp": np.arange(n),
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": [100.0] * n,
            "deal": list(range(n)),
        }
    )


class TestGenerateLabels:
    def test_up_label(self):
        df = _make_df([100, 110, 120])
        labels = generate_labels(df, horizon=1)
        assert labels["label"].iloc[0] == 1  # 110 > 100
        assert labels["label"].iloc[1] == 1  # 120 > 110

    def test_down_label(self):
        df = _make_df([100, 90, 80])
        labels = generate_labels(df, horizon=1)
        assert labels["label"].iloc[0] == 0  # 90 < 100
        assert labels["label"].iloc[1] == 0  # 80 < 90

    def test_mixed_labels(self):
        df = _make_df([100, 110, 95, 120])
        labels = generate_labels(df, horizon=1)
        assert list(labels["label"]) == [1, 0, 1]

    def test_binary_values(self):
        df = _make_df([100, 110, 95, 120, 105])
        labels = generate_labels(df, horizon=1)
        assert set(labels["label"].unique()).issubset({0, 1})

    def test_horizon_2(self):
        df = _make_df([100, 90, 120, 80, 150])
        labels = generate_labels(df, horizon=2)
        assert labels["label"].iloc[0] == 1  # 120 > 100
        assert labels["label"].iloc[1] == 0  # 80 < 90
        assert labels["label"].iloc[2] == 1  # 150 > 120

    def test_output_has_timestamp_and_label(self):
        df = _make_df([100, 110, 120])
        labels = generate_labels(df, horizon=1)
        assert list(labels.columns) == ["timestamp", "label"]

    def test_length_matches_input_minus_horizon(self):
        df = _make_df([100, 110, 120, 130, 140])
        labels = generate_labels(df, horizon=1)
        assert len(labels) == 4  # 5 - 1

    def test_horizon_3_length(self):
        df = _make_df([100, 110, 120, 130, 140, 150])
        labels = generate_labels(df, horizon=3)
        assert len(labels) == 3  # 6 - 3

    def test_equal_close_gives_down(self):
        df = _make_df([100, 100, 100])
        labels = generate_labels(df, horizon=1)
        assert labels["label"].iloc[0] == 0  # 100 not > 100
