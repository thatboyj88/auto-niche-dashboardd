class HistoricalDataLoader:
    """
    Converts external historical market data into the
    candle format used by our trading bot.
    """

    def __init__(self):
        self.candles = []

    def load_candles(self, raw_data):
        """
        Accepts a list of candle dictionaries.

        Expected format:

        {
            "timestamp": ...,
            "open": ...,
            "high": ...,
            "low": ...,
            "close": ...,
            "volume": ...
        }
        """

        self.candles = []

        for candle in raw_data:

            required_fields = [
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume"
            ]

            if not all(
                field in candle
                for field in required_fields
            ):
                continue

            self.candles.append({
                "timestamp": candle["timestamp"],
                "open": float(candle["open"]),
                "high": float(candle["high"]),
                "low": float(candle["low"]),
                "close": float(candle["close"]),
                "volume": float(candle["volume"])
            })

        return self.candles

    def count(self):
        return len(self.candles)

    def get_closes(self):
        return [
            candle["close"]
            for candle in self.candles
        ]

    def get_volumes(self):
        return [
            candle["volume"]
            for candle in self.candles
        ]