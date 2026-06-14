"""Position sizing and management."""

from dataclasses import dataclass
from datetime import datetime

from src.costs import compute_fee, compute_slippage


@dataclass
class Position:
    entry_price: float
    entry_time: datetime
    direction: int  # 1 for long, -1 for short
    quantity: float
    bars_held: int = 0


class PositionManager:
    def __init__(
        self,
        risk_per_trade: float = 0.01,
        max_duration_candles: int = 12,
        fee_pct: float = 0.1,
        slippage_pct: float = 0.05,
    ):
        self.risk_per_trade = risk_per_trade
        self.max_duration_candles = max_duration_candles
        self.fee_pct = fee_pct
        self.slippage_pct = slippage_pct

    def open_position(
        self, price: float, direction: int, portfolio_value: float, timestamp: datetime
    ) -> Position:
        quantity = calculate_position_size(portfolio_value, self.risk_per_trade, price)
        return Position(
            entry_price=price,
            entry_time=timestamp,
            direction=direction,
            quantity=quantity,
            bars_held=0,
        )

    def update(self, position: Position) -> Position:
        position.bars_held += 1
        return position

    def should_exit(self, position: Position, current_signal: int) -> bool:
        if current_signal != position.direction:
            return True
        return should_force_exit(position.bars_held, self.max_duration_candles)

    def close_position(self, position: Position, exit_price: float, timestamp: datetime) -> dict:
        slipped_entry = compute_slippage(
            position.entry_price, self.slippage_pct, position.direction
        )
        slipped_exit = compute_slippage(exit_price, self.slippage_pct, -position.direction)

        entry_value = slipped_entry * position.quantity
        exit_value = slipped_exit * position.quantity

        entry_fee = compute_fee(entry_value, self.fee_pct)
        exit_fee = compute_fee(exit_value, self.fee_pct)
        total_costs = entry_fee + exit_fee

        gross_pnl = position.direction * (slipped_exit - slipped_entry) * position.quantity
        pnl = gross_pnl - total_costs

        return {
            "entry_price": position.entry_price,
            "exit_price": exit_price,
            "direction": position.direction,
            "quantity": position.quantity,
            "pnl": pnl,
            "entry_time": position.entry_time,
            "exit_time": timestamp,
            "bars_held": position.bars_held,
            "costs": total_costs,
        }


def calculate_position_size(portfolio_value: float, risk_per_trade: float, price: float) -> float:
    risk_amount = portfolio_value * risk_per_trade
    return risk_amount / price


def should_force_exit(bars_held: int, max_duration: int) -> bool:
    return bars_held >= max_duration
