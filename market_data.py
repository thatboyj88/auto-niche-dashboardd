class HistoricalMarketData:

    def __init__(self, candles=None):
        self.candles = candles or []

    def add_candle(
        self,
        timestamp,
        open_price,
        high,
        low,
        close,
        volume
    ):
        candle = {
            "timestamp": timestamp,
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume
        }

        self.candles.append(candle)

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

    def count(self):
        return len(self.candles)