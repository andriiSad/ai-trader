"""Plotly chart generators for data exploration."""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add basic derived features used by multiple charts."""
    df = df.copy()
    df["return_1"] = df["close"].pct_change()
    df["return_5"] = df["close"].pct_change(5)
    df["volume_change"] = df["volume"].pct_change()
    df["range"] = df["high"] - df["low"]
    return df


def candlestick_chart(df: pd.DataFrame, pair: str) -> go.Figure:
    """OHLC candlestick chart with volume bar subplot."""
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.7, 0.3],
        subplot_titles=[f"{pair} Candlestick", "Volume"],
    )

    fig.add_trace(
        go.Candlestick(
            x=df["timestamp"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="OHLC",
        ),
        row=1,
        col=1,
    )

    colors = np.where(df["close"] >= df["open"], "#26a69a", "#ef5350")
    fig.add_trace(
        go.Bar(x=df["timestamp"], y=df["volume"], marker_color=colors, name="Volume"),
        row=2,
        col=1,
    )

    fig.update_layout(
        xaxis_rangeslider_visible=False,
        height=700,
        title=f"{pair} – 5m Candlestick",
        showlegend=False,
    )
    return fig


def volume_profile(df: pd.DataFrame, pair: str) -> go.Figure:
    """Horizontal histogram of volume by price level."""
    price_bins = np.linspace(df["low"].min(), df["high"].max(), 80)
    bin_volume = np.zeros(len(price_bins) - 1)

    for _, row in df.iterrows():
        for i in range(len(price_bins) - 1):
            if row["high"] >= price_bins[i] and row["low"] <= price_bins[i + 1]:
                bin_volume[i] += row["volume"]

    bin_centers = (price_bins[:-1] + price_bins[1:]) / 2

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=bin_centers,
            x=bin_volume,
            orientation="h",
            marker_color="#5c6bc0",
            name="Volume",
        )
    )
    fig.update_layout(
        title=f"{pair} – Volume Profile",
        xaxis_title="Volume",
        yaxis_title="Price",
        height=700,
    )
    return fig


def return_distribution(df: pd.DataFrame, pair: str, periods: list[int] | None = None) -> go.Figure:
    """Histogram of N-bar returns."""
    if periods is None:
        periods = [1, 5]

    fig = make_subplots(
        rows=1, cols=len(periods), subplot_titles=[f"{p}-bar returns" for p in periods]
    )

    for idx, p in enumerate(periods, start=1):
        returns = df["close"].pct_change(p).dropna()
        fig.add_trace(
            go.Histogram(x=returns, nbinsx=100, name=f"{p}-bar", opacity=0.75),
            row=1,
            col=idx,
        )

    fig.update_layout(
        title=f"{pair} – Return Distributions",
        height=400,
        showlegend=False,
    )
    return fig


def correlation_heatmap(df: pd.DataFrame, pair: str) -> go.Figure:
    """Heatmap of feature correlations (returns, volume_change, range)."""
    df = _add_derived_features(df)
    cols = ["return_1", "return_5", "volume_change", "range"]
    corr = df[cols].corr()

    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=cols,
            y=cols,
            colorscale="RdBu_r",
            zmin=-1,
            zmax=1,
            text=np.round(corr.values, 2),
            texttemplate="%{text}",
        )
    )
    fig.update_layout(title=f"{pair} – Feature Correlations", height=500)
    return fig


def feature_distributions(df: pd.DataFrame, pair: str) -> go.Figure:
    """Box plots of basic features."""
    df = _add_derived_features(df)
    features = ["return_1", "volume_change", "range"]

    fig = go.Figure()
    for feat in features:
        series = df[feat].dropna()
        fig.add_trace(go.Box(y=series, name=feat))

    fig.update_layout(
        title=f"{pair} – Feature Distributions",
        height=400,
        showlegend=False,
    )
    return fig


def build_dashboard(data_dir: str, pairs: list[str], output_path: str) -> None:
    """Combine all charts into a single self-contained HTML file."""
    from src.data_loader import load_all_pairs

    data = load_all_pairs(data_dir, pairs)

    html_parts: list[str] = []

    for pair, df in data.items():
        if df.empty:
            html_parts.append(f"<h2>{pair} – No data available</h2>")
            continue

        html_parts.append(f"<h1>{pair}</h1>")

        figs = [
            candlestick_chart(df, pair),
            volume_profile(df, pair),
            return_distribution(df, pair),
            correlation_heatmap(df, pair),
            feature_distributions(df, pair),
        ]
        for fig in figs:
            html_parts.append(fig.to_html(full_html=False, include_plotlyjs=False))

    plotly_cdn = '<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>'
    full_html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Data Exploration Dashboard</title>
{plotly_cdn}
<style>body{{font-family:sans-serif;margin:20px;}}</style>
</head><body>
<h1 style="border-bottom:2px solid #333;padding-bottom:8px;">Data Exploration Dashboard</h1>
{"".join(html_parts)}
</body></html>"""

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(full_html, encoding="utf-8")
