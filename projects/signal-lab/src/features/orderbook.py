from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _load_orderbook_snapshots(data_dir: str, pair: str) -> pd.DataFrame:
    orderbook_dir = Path(data_dir) / pair / "orderbook"
    if not orderbook_dir.exists():
        raise FileNotFoundError(f"Order book data not found at {orderbook_dir}")

    frames = []
    for csv_file in sorted(orderbook_dir.glob("*.csv")):
        try:
            df = pd.read_csv(csv_file)
            if not df.empty:
                frames.append(df)
        except Exception:
            continue

    if not frames:
        return pd.DataFrame(columns=["timestamp", "side", "price", "quantity"])

    return pd.concat(frames, ignore_index=True)


def _aggregate_snapshots(snapshots: pd.DataFrame, candle_timestamps: pd.Series) -> pd.DataFrame:
    if snapshots.empty:
        result = pd.DataFrame({"timestamp": candle_timestamps})
        for col in [
            "ob_bid_ask_ratio",
            "ob_total_bid",
            "ob_total_ask",
            "ob_imbalance",
            "ob_mid_price_deviation",
        ]:
            result[col] = np.nan
        return result

    snapshots = snapshots.copy()
    snapshots["price"] = pd.to_numeric(snapshots["price"], errors="coerce")
    snapshots["quantity"] = pd.to_numeric(snapshots["quantity"], errors="coerce")
    snapshots["timestamp"] = pd.to_numeric(snapshots["timestamp"], errors="coerce")
    snapshots = snapshots.dropna(subset=["price", "quantity", "timestamp"])

    candle_ts_sorted = candle_timestamps.sort_values().reset_index(drop=True)
    snapshot_ts = snapshots["timestamp"].unique()
    snapshot_ts_sorted = np.sort(snapshot_ts)

    results = []
    for cts in candle_ts_sorted:
        mask = snapshot_ts_sorted <= cts
        if not mask.any():
            results.append(
                {
                    "timestamp": cts,
                    "ob_bid_ask_ratio": np.nan,
                    "ob_total_bid": np.nan,
                    "ob_total_ask": np.nan,
                    "ob_imbalance": np.nan,
                    "ob_mid_price_deviation": np.nan,
                }
            )
            continue

        relevant_ts = snapshot_ts_sorted[mask][-1]
        snap = snapshots[snapshots["timestamp"] == relevant_ts]

        bids = snap[snap["side"] == "bid"]
        asks = snap[snap["side"] == "ask"]

        total_bid = bids["quantity"].sum()
        total_ask = asks["quantity"].sum()

        bid_ask_ratio = total_bid / total_ask if total_ask > 0 else np.nan
        imbalance = (
            (total_bid - total_ask) / (total_bid + total_ask)
            if (total_bid + total_ask) > 0
            else np.nan
        )

        best_bid = bids["price"].max() if not bids.empty else np.nan
        best_ask = asks["price"].min() if not asks.empty else np.nan

        if not np.isnan(best_bid) and not np.isnan(best_ask) and best_ask > 0:
            actual_mid = (best_bid + best_ask) / 2
            weighted_bid = (
                (bids["price"] * bids["quantity"]).sum() / total_bid if total_bid > 0 else np.nan
            )
            weighted_ask = (
                (asks["price"] * asks["quantity"]).sum() / total_ask if total_ask > 0 else np.nan
            )
            weighted_mid = (
                (weighted_bid + weighted_ask) / 2
                if not np.isnan(weighted_bid) and not np.isnan(weighted_ask)
                else np.nan
            )
            mid_price_deviation = (
                (weighted_mid - actual_mid) / actual_mid if actual_mid > 0 else np.nan
            )
        else:
            mid_price_deviation = np.nan

        results.append(
            {
                "timestamp": cts,
                "ob_bid_ask_ratio": bid_ask_ratio,
                "ob_total_bid": total_bid,
                "ob_total_ask": total_ask,
                "ob_imbalance": imbalance,
                "ob_mid_price_deviation": mid_price_deviation,
            }
        )

    return pd.DataFrame(results)


def generate(
    df: pd.DataFrame,
    data_dir: str = "data",
    pair: str = "BTC_USDT",
    **kwargs,
) -> pd.DataFrame:
    snapshots = _load_orderbook_snapshots(data_dir, pair)
    result = _aggregate_snapshots(snapshots, df["timestamp"])
    return result
