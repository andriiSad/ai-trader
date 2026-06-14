"""Backtest simulation engine."""

from dataclasses import dataclass

import pandas as pd

from src.position import Position, PositionManager
from src.strategy import ThresholdStrategy


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    trade_log: list[dict]
    initial_capital: float
    final_value: float
    fold_id: int | None
    fold_start: str | None
    fold_end: str | None


class BacktestEngine:
    def __init__(
        self,
        strategy: ThresholdStrategy,
        position_manager: PositionManager,
        initial_capital: float,
        fee_pct: float,
        slippage_pct: float,
    ):
        self.strategy = strategy
        self.position_manager = position_manager
        self.initial_capital = initial_capital
        self.fee_pct = fee_pct
        self.slippage_pct = slippage_pct

    def run(
        self,
        predictions_df: pd.DataFrame,
        candles_df: pd.DataFrame,
        fold_id: int | None = None,
        fold_start: str | None = None,
        fold_end: str | None = None,
    ) -> BacktestResult:
        merged = pd.merge(candles_df, predictions_df[["timestamp", "probability"]], on="timestamp")
        merged = merged.sort_values("timestamp").reset_index(drop=True)

        portfolio_value = self.initial_capital
        equity_values: list[float] = []
        trade_log: list[dict] = []
        current_position: Position | None = None

        for _, candle in merged.iterrows():
            if current_position is not None:
                self.position_manager.update(current_position)
                signal = self.strategy.signal(candle["probability"])
                if self.position_manager.should_exit(current_position, signal):
                    trade = self.position_manager.close_position(
                        current_position, candle["close"], candle["timestamp"]
                    )
                    portfolio_value += trade["pnl"]
                    trade_log.append(trade)
                    current_position = None

            if current_position is None:
                signal = self.strategy.signal(candle["probability"])
                if signal != 0:
                    current_position = self.position_manager.open_position(
                        candle["close"], signal, portfolio_value, candle["timestamp"]
                    )

            unrealized = 0.0
            if current_position is not None:
                unrealized = (
                    current_position.direction
                    * (candle["close"] - current_position.entry_price)
                    * current_position.quantity
                )
            equity_values.append(portfolio_value + unrealized)

        if current_position is not None:
            last_candle = merged.iloc[-1]
            trade = self.position_manager.close_position(
                current_position, last_candle["close"], last_candle["timestamp"]
            )
            portfolio_value += trade["pnl"]
            trade_log.append(trade)

        equity_curve = pd.Series(equity_values, index=merged["timestamp"].values)

        return BacktestResult(
            equity_curve=equity_curve,
            trade_log=trade_log,
            initial_capital=self.initial_capital,
            final_value=portfolio_value,
            fold_id=fold_id,
            fold_start=fold_start,
            fold_end=fold_end,
        )

    def run_walk_forward(
        self, predictions_df: pd.DataFrame, candles_df: pd.DataFrame
    ) -> list[BacktestResult]:
        results: list[BacktestResult] = []
        for fold_id in sorted(predictions_df["fold_id"].unique()):
            fold_preds = predictions_df[predictions_df["fold_id"] == fold_id]
            fold_start = fold_preds["fold_test_start"].iloc[0]
            fold_end = fold_preds["fold_test_end"].iloc[0]

            fold_candles = candles_df[
                (candles_df["timestamp"] >= fold_start) & (candles_df["timestamp"] < fold_end)
            ]

            result = self.run(fold_preds, fold_candles, int(fold_id), fold_start, fold_end)
            results.append(result)

        return results
