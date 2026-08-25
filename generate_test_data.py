import random


def generate_candles(count=1000):

    random.seed(42)

    candles = []
    price = 100.0

    for i in range(count):

        # Create alternating market regimes
        if i < 150:
            trend = 0.05

        elif i < 300:
            trend = -0.04

        elif i < 450:
            trend = 0.03

        elif i < 600:
            trend = -0.05

        elif i < 750:
            trend = 0.02

        else:
            trend = 0.04

        # Much larger random movement
        noise = random.uniform(-0.30, 0.30)

        change = trend + noise

        price += change

        if price < 20:
            price = 20

        open_price = price + random.uniform(-0.15, 0.15)

        high = max(
            open_price,
            price
        ) + random.uniform(0.05, 0.25)

        low = min(
            open_price,
            price
        ) - random.uniform(0.05, 0.25)

        volume = random.randint(80, 120)

        # Occasional volume spikes
        if random.random() < 0.10:
            volume *= random.uniform(1.5, 2.5)

        candles.append({
            "timestamp": i,
            "open": open_price,
            "high": high,
            "low": low,
            "close": price,
            "volume": volume
        })

    return candles