from config import (
    STARTING_CAPITAL,
    MAX_POSITION_PERCENT,
    STOP_LOSS_PERCENT,
    TAKE_PROFIT_PERCENT,
    MAX_DAILY_LOSS_PERCENT,
    MAX_TRADES_PER_DAY,
    MIN_STRATEGY_SCORE,
)


def calculate_position_size(capital):
    """
    Calculate the maximum amount allowed for one trade.
    """
    return capital * MAX_POSITION_PERCENT


def calculate_stop_loss(entry_price):
    """
    Calculate stop-loss price.
    """
    return entry_price * (1 - STOP_LOSS_PERCENT)


def calculate_take_profit(entry_price):
    """
    Calculate take-profit price.
    """
    return entry_price * (1 + TAKE_PROFIT_PERCENT)


def risk_check(
    capital,
    daily_loss,
    trades_today,
    strategy_score,
    entry_price,
    daily_starting_capital=None,
):
    """
    Determine whether a trade is allowed.
    """

    # Strategy must pass first
    if strategy_score < MIN_STRATEGY_SCORE:
        return False, "Strategy score is below minimum."

    # Daily loss protection
    # Operational callers pass the capital at the start of the current day.
    # Keep the legacy fallback for existing callers and compatibility tests.
    loss_basis = (
        capital
        if daily_starting_capital is None
        else daily_starting_capital
    )
    max_daily_loss = loss_basis * MAX_DAILY_LOSS_PERCENT

    if daily_loss >= max_daily_loss:
        return False, "Daily loss limit reached."

    # Maximum trades per day
    if trades_today >= MAX_TRADES_PER_DAY:
        return False, "Maximum daily trades reached."

    # Entry price must be valid
    if entry_price <= 0:
        return False, "Invalid entry price."

    return True, "Risk checks passed."


def get_trade_plan(capital, entry_price):
    """
    Create the planned trade parameters.
    """

    position_size = calculate_position_size(capital)

    stop_loss = calculate_stop_loss(entry_price)

    take_profit = calculate_take_profit(entry_price)

    return {
        "position_size": position_size,
        "entry_price": entry_price,
        "stop_loss": stop_loss,
        "take_profit": take_profit
    }