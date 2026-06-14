import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _make_candle_ts(n: int = 20, start: int = 1718035200, interval: int = 14400) -> list[int]:
    return [start + i * interval for i in range(n)]


def _make_funding_csv(tmpdir: str, records: list[tuple[int, str]]) -> None:
    csv_path = Path(tmpdir) / "BTC_USDT" / "funding.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w") as f:
        f.write("timestamp,funding_rate\n")
        for ts, rate in records:
            f.write(f"{ts},{rate}\n")


def _make_ohlcv_with_timestamps(timestamps: list[int]) -> pd.DataFrame:
    n = len(timestamps)
    rng = np.random.default_rng(42)
    close = 70000 + np.cumsum(rng.standard_normal(n) * 100)
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": close - 50,
            "high": close + 200,
            "low": close - 200,
            "close": close,
            "volume": rng.uniform(100, 1000, n),
        }
    )


class TestFundingGenerate:
    def test_output_columns(self):
        candle_ts = _make_candle_ts(20)
        # Funding every 8h (28800s), candles every 4h (14400s)
        funding_ts = candle_ts[::2]  # every other candle = 8h
        funding_records = [(ts, "0.0001") for ts in funding_ts]

        with tempfile.TemporaryDirectory() as tmpdir:
            _make_funding_csv(tmpdir, funding_records)
            df = _make_ohlcv_with_timestamps(candle_ts)

            from src.features.funding import generate

            result = generate(df, data_dir=tmpdir)

        expected_cols = ["timestamp", "funding_rate_raw", "funding_rate_cum_3", "funding_rate_roc"]
        assert list(result.columns) == expected_cols

    def test_forward_fill_8h_to_4h(self):
        candle_ts = _make_candle_ts(10, start=1718035200)
        # Funding at t0, t2, t4, t6, t8 (every 8h)
        funding_ts = [candle_ts[i] for i in range(0, 10, 2)]
        funding_records = [(ts, f"{0.0001 * (i + 1):.6f}") for i, ts in enumerate(funding_ts)]

        with tempfile.TemporaryDirectory() as tmpdir:
            _make_funding_csv(tmpdir, funding_records)
            df = _make_ohlcv_with_timestamps(candle_ts)

            from src.features.funding import generate

            result = generate(df, data_dir=tmpdir)

        # Candle at index 1 (4h after first funding) should have same funding_rate_raw as index 0
        assert result.iloc[0]["funding_rate_raw"] == result.iloc[1]["funding_rate_raw"]
        assert result.iloc[2]["funding_rate_raw"] == result.iloc[3]["funding_rate_raw"]

    def test_drops_rows_before_first_funding(self):
        candle_ts = _make_candle_ts(20, start=1718035200)
        # Funding starts at candle index 4 (skip first 4 candles)
        funding_ts = candle_ts[4::2]
        funding_records = [(ts, "0.0001") for ts in funding_ts]

        with tempfile.TemporaryDirectory() as tmpdir:
            _make_funding_csv(tmpdir, funding_records)
            df = _make_ohlcv_with_timestamps(candle_ts)

            from src.features.funding import generate

            result = generate(df, data_dir=tmpdir)

        # Rows before first funding have NaN (compose will drop them)
        assert pd.isna(result["funding_rate_raw"].iloc[0])
        # Rows after first funding have values
        assert result["funding_rate_raw"].iloc[4:].notna().all()

    def test_no_nans_in_output(self):
        candle_ts = _make_candle_ts(20)
        funding_ts = candle_ts[::2]
        funding_records = [(ts, f"{0.0001 * (i + 1):.6f}") for i, ts in enumerate(funding_ts)]

        with tempfile.TemporaryDirectory() as tmpdir:
            _make_funding_csv(tmpdir, funding_records)
            df = _make_ohlcv_with_timestamps(candle_ts)

            from src.features.funding import generate

            result = generate(df, data_dir=tmpdir)

        # No NaN after first funding row (compose handles dropping warmup)
        assert not result["funding_rate_raw"].iloc[1:].isna().any()

    def test_cum_3_is_rolling_sum(self):
        candle_ts = _make_candle_ts(12)
        funding_ts = candle_ts[::2]
        rates = ["0.0001", "0.0002", "-0.0001", "0.0003", "0.0001", "0.0002"]
        funding_records = list(zip(funding_ts, rates, strict=True))

        with tempfile.TemporaryDirectory() as tmpdir:
            _make_funding_csv(tmpdir, funding_records)
            df = _make_ohlcv_with_timestamps(candle_ts)

            from src.features.funding import generate

            result = generate(df, data_dir=tmpdir)

        cum_values = result["funding_rate_cum_3"].dropna().values

        # Rolling sum of 3 forward-filled values
        assert cum_values[0] == pytest.approx(0.0001 + 0.0001 + 0.0002, abs=1e-8)

    def test_roc_is_rate_of_change(self):
        candle_ts = _make_candle_ts(12)
        funding_ts = candle_ts[::2]
        rates = ["0.0001", "0.0002", "0.0003", "0.0001", "0.0004", "0.0002"]
        funding_records = list(zip(funding_ts, rates, strict=True))

        with tempfile.TemporaryDirectory() as tmpdir:
            _make_funding_csv(tmpdir, funding_records)
            df = _make_ohlcv_with_timestamps(candle_ts)

            from src.features.funding import generate

            result = generate(df, data_dir=tmpdir)

        roc = result["funding_rate_roc"].dropna()
        # At candle 1 (ffill, same as candle 0): ROC = 0.0
        assert roc.iloc[0] == pytest.approx(0.0, abs=1e-8)
        # At candle 2 (second funding 0.0002 vs ffill 0.0001): ROC = 1.0
        assert roc.iloc[1] == pytest.approx(1.0, abs=1e-8)
        # At candle 4 (third funding 0.0003 vs ffill 0.0002): ROC = 0.5
        assert roc.iloc[3] == pytest.approx(0.5, abs=1e-8)

    def test_missing_funding_file_returns_empty(self):
        candle_ts = _make_candle_ts(10)
        df = _make_ohlcv_with_timestamps(candle_ts)

        with tempfile.TemporaryDirectory() as tmpdir:
            from src.features.funding import generate

            result = generate(df, data_dir=tmpdir)

        assert len(result) == 0
        assert "timestamp" in result.columns

    def test_timestamp_preserved(self):
        candle_ts = _make_candle_ts(10)
        funding_ts = candle_ts[::2]
        funding_records = [(ts, "0.0001") for ts in funding_ts]

        with tempfile.TemporaryDirectory() as tmpdir:
            _make_funding_csv(tmpdir, funding_records)
            df = _make_ohlcv_with_timestamps(candle_ts)

            from src.features.funding import generate

            result = generate(df, data_dir=tmpdir)

        assert "timestamp" in result.columns
        assert result["timestamp"].is_unique

    def test_compose_with_ohlcv(self):
        candle_ts = _make_candle_ts(50)
        funding_ts = candle_ts[::2]
        funding_records = [(ts, f"{0.0001 * (i % 5 + 1):.6f}") for i, ts in enumerate(funding_ts)]

        with tempfile.TemporaryDirectory() as tmpdir:
            _make_funding_csv(tmpdir, funding_records)
            df = _make_ohlcv_with_timestamps(candle_ts)

            from src.features import compose
            from src.features.funding import generate as funding_generate
            from src.features.ohlcv import generate as ohlcv_generate

            ohlcv_feat = ohlcv_generate(df)
            funding_feat = funding_generate(df, data_dir=tmpdir)
            result = compose(ohlcv_feat, funding_feat)

        assert "rsi_14" in result.columns
        assert "funding_rate_raw" in result.columns
        assert "timestamp" in result.columns
        assert not result.isna().any().any()
