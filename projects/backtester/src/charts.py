import pandas as pd
import plotly.graph_objects as go


def build_equity_curve_chart(
    equity_curve: pd.Series,
    benchmark_curve: pd.Series = None,
    fold_boundaries: list = None,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=equity_curve.index,
            y=equity_curve.values,
            mode="lines",
            name="Strategy",
            line=dict(color="blue"),
        )
    )
    if benchmark_curve is not None:
        fig.add_trace(
            go.Scatter(
                x=benchmark_curve.index,
                y=benchmark_curve.values,
                mode="lines",
                name="Buy & Hold",
                line=dict(color="gray", dash="dash"),
            )
        )
    if fold_boundaries:
        for boundary in fold_boundaries:
            fig.add_vline(x=boundary, line_dash="dot", line_color="red", opacity=0.5)
    fig.update_layout(title="Equity Curve", xaxis_title="Time", yaxis_title="Portfolio Value")
    return fig


def build_drawdown_chart(equity_curve: pd.Series) -> go.Figure:
    peak = equity_curve.expanding(min_periods=1).max()
    drawdown = (equity_curve - peak) / peak * 100
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=drawdown.index,
            y=drawdown.values,
            fill="tozeroy",
            fillcolor="rgba(255,0,0,0.3)",
            line=dict(color="red"),
            name="Drawdown",
        )
    )
    fig.update_layout(title="Drawdown", xaxis_title="Time", yaxis_title="Drawdown %")
    return fig


def build_trade_markers_chart(candles_df: pd.DataFrame, trade_log: list[dict]) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=candles_df["timestamp"],
            open=candles_df["open"],
            high=candles_df["high"],
            low=candles_df["low"],
            close=candles_df["close"],
            name="Price",
        )
    )
    if trade_log:
        entry_times = [t["entry_time"] for t in trade_log]
        entry_prices = [t["entry_price"] for t in trade_log]
        exit_times = [t["exit_time"] for t in trade_log]
        exit_prices = [t["exit_price"] for t in trade_log]
        fig.add_trace(
            go.Scatter(
                x=entry_times,
                y=entry_prices,
                mode="markers",
                marker=dict(symbol="triangle-up", size=10, color="green"),
                name="Entry",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=exit_times,
                y=exit_prices,
                mode="markers",
                marker=dict(symbol="triangle-down", size=10, color="red"),
                name="Exit",
            )
        )
    fig.update_layout(title="Trade Markers", xaxis_title="Time", yaxis_title="Price")
    return fig


def build_fold_metrics_table(per_fold_metrics: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(per_fold_metrics)


def build_pair_metrics_table(per_pair_metrics: dict) -> pd.DataFrame:
    rows = []
    for pair, metrics in per_pair_metrics.items():
        row = {"pair": pair}
        row.update(metrics)
        rows.append(row)
    return pd.DataFrame(rows)
