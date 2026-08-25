from config import (
    MIN_STRATEGY_SCORE,
    RSI_MIN,
    RSI_MAX,
    VOLUME_MULTIPLIER
)


def calculate_strategy_score(
    ema_9,
    ema_21,
    ema_50,
    ema_200,
    rsi,
    current_price,
    average_volume,
    current_volume
):

    score = 0
    reasons = []
    conditions = {}

    # Long-term trend
    if (
        ema_50 is not None
        and ema_200 is not None
        and ema_50 > ema_200
    ):
        score += 25
        conditions["long_term_trend"] = True
        reasons.append("Long-term trend bullish")
    else:
        conditions["long_term_trend"] = False
        reasons.append("Long-term trend not bullish")

    # Short-term momentum
    if ema_9 > ema_21:
        score += 20
        conditions["short_term_momentum"] = True
        reasons.append("Short-term momentum bullish")
    else:
        conditions["short_term_momentum"] = False
        reasons.append("Short-term momentum weak")

    # RSI
    if RSI_MIN <= rsi <= RSI_MAX:
        score += 15
        conditions["rsi"] = True
        reasons.append("RSI acceptable")
    else:
        conditions["rsi"] = False
        reasons.append("RSI outside preferred range")

    # Volume
    if current_volume >= average_volume * VOLUME_MULTIPLIER:
        score += 20
        conditions["volume"] = True
        reasons.append("Volume confirmed")
    else:
        conditions["volume"] = False
        reasons.append("Volume confirmation weak")

    # Price above EMA21
    if current_price > ema_21:
        score += 20
        conditions["price_above_ema21"] = True
        reasons.append("Price above EMA21")
    else:
        conditions["price_above_ema21"] = False
        reasons.append("Price below EMA21")

    if score >= MIN_STRATEGY_SCORE:
        decision = "BUY CANDIDATE"
    else:
        decision = "NO TRADE"

    return score, decision, reasons, conditions