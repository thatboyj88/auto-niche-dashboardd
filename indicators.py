def calculate_ema(prices, period):
    """
    Calculate Exponential Moving Average.
    """
    if len(prices) < period:
        return None

    multiplier = 2 / (period + 1)

    ema = sum(prices[:period]) / period

    for price in prices[period:]:
        ema = (price - ema) * multiplier + ema

    return ema


def calculate_rsi(prices, period=14):
    """
    Calculate Relative Strength Index using
    the most recent price changes.
    """

    if len(prices) < period + 1:
        return None

    recent_prices = prices[-(period + 1):]

    gains = []
    losses = []

    for i in range(1, len(recent_prices)):

        change = (
            recent_prices[i] -
            recent_prices[i - 1]
        )

        if change > 0:
            gains.append(change)
            losses.append(0)
        else:
            gains.append(0)
            losses.append(abs(change))

    average_gain = sum(gains) / period
    average_loss = sum(losses) / period

    if average_loss == 0:
        return 100

    rs = average_gain / average_loss

    return 100 - (
        100 / (1 + rs)
    )


def calculate_average_volume(volumes, period=20):
    """
    Calculate average trading volume.
    """
    if len(volumes) < period:
        return None

    return sum(volumes[-period:]) / period