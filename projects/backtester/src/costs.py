"""Pure functions for trade cost computation."""


def compute_fee(trade_value: float, fee_pct: float) -> float:
    return trade_value * fee_pct / 100


def compute_slippage(price: float, slippage_pct: float, direction: int) -> float:
    return price * (1 + direction * slippage_pct / 100)


def compute_round_trip_cost(
    entry_price: float,
    exit_price: float,
    quantity: float,
    fee_pct: float,
    slippage_pct: float,
    direction: int,
) -> float:
    slipped_entry = compute_slippage(entry_price, slippage_pct, direction)
    slipped_exit = compute_slippage(exit_price, slippage_pct, -direction)

    entry_value = slipped_entry * quantity
    exit_value = slipped_exit * quantity

    entry_fee = compute_fee(entry_value, fee_pct)
    exit_fee = compute_fee(exit_value, fee_pct)

    cost_basis = direction * (slipped_exit - slipped_entry) * quantity
    total_fees = entry_fee + exit_fee

    return cost_basis - total_fees
