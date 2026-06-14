"""Performance metrics for backtesting."""

import numpy as np
import pandas as pd


def calculate_sharpe(returns: pd.Series, periods_per_year: int = 72576) -> float:
    """Annualized Sharpe ratio. periods_per_year = 252 * 24 * 12 for 5-min candles."""
    if returns.std() == 0:
        return 0.0
    return (returns.mean() / returns.std()) * np.sqrt(periods_per_year)


def calculate_max_drawdown(equity_curve: pd.Series) -> tuple[float, int]:
    """Returns (max_drawdown_pct, max_drawdown_duration_bars)."""
    peak = equity_curve.expanding(min_periods=1).max()
    drawdown = (equity_curve - peak) / peak
    max_dd = drawdown.min()

    dd_duration = 0
    max_dd_duration = 0
    for i in range(len(drawdown)):
        if drawdown.iloc[i] < 0:
            dd_duration += 1
            max_dd_duration = max(max_dd_duration, dd_duration)
        else:
            dd_duration = 0

    return abs(max_dd), max_dd_duration


def calculate_win_rate(trade_log: list[dict]) -> float:
    """Percentage of profitable trades."""
    if not trade_log:
        return 0.0
    wins = sum(1 for t in trade_log if t["pnl"] > 0)
    return wins / len(trade_log)


def calculate_profit_factor(trade_log: list[dict]) -> float:
    """Gross profits / gross losses."""
    if not trade_log:
        return 0.0
    gross_profit = sum(t["pnl"] for t in trade_log if t["pnl"] > 0)
    gross_loss = abs(sum(t["pnl"] for t in trade_log if t["pnl"] < 0))
    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else 0.0
    return gross_profit / gross_loss


def calculate_total_return(initial: float, final: float) -> float:
    """Total return as a fraction."""
    return (final - initial) / initial


def calculate_expectancy(trade_log: list[dict]) -> float:
    """Expected value per trade: avg_win * win_rate - avg_loss * loss_rate."""
    if not trade_log:
        return 0.0
    wins = [t["pnl"] for t in trade_log if t["pnl"] > 0]
    losses = [t["pnl"] for t in trade_log if t["pnl"] < 0]
    win_rate = len(wins) / len(trade_log)
    loss_rate = len(losses) / len(trade_log)
    avg_win = np.mean(wins) if wins else 0.0
    avg_loss = abs(np.mean(losses)) if losses else 0.0
    return avg_win * win_rate - avg_loss * loss_rate


def compute_all_metrics(
    equity_curve: pd.Series, trade_log: list[dict], initial_capital: float
) -> dict:
    """Compute all metrics and return as dict."""
    returns = equity_curve.pct_change().dropna()
    max_dd, max_dd_duration = calculate_max_drawdown(equity_curve)

    return {
        "sharpe": calculate_sharpe(returns),
        "max_drawdown": max_dd,
        "max_drawdown_duration": max_dd_duration,
        "win_rate": calculate_win_rate(trade_log),
        "profit_factor": calculate_profit_factor(trade_log),
        "total_return": calculate_total_return(initial_capital, equity_curve.iloc[-1]),
        "num_trades": len(trade_log),
        "avg_trade_duration": (np.mean([t["bars_held"] for t in trade_log]) if trade_log else 0.0),
        "expectancy": calculate_expectancy(trade_log),
    }
