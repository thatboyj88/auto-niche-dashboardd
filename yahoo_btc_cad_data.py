import json
import math
from datetime import datetime, timezone
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from historical_data_loader import HistoricalDataLoader


class YahooFinanceMarketDataError(RuntimeError):
    """Raised when public Yahoo Finance BTC/CAD data cannot be loaded."""


class YahooBTCADMarketData:
    """
    Loads public, aggregated BTC/CAD daily OHLCV candles from Yahoo Finance.

    This is a market-data-only loader. It never accesses an exchange account,
    private API, wallet, or order endpoint. The returned values are not
    Kraken-specific candles and must be labeled as Yahoo Finance data.
    """

    CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/BTC-CAD"
    SYMBOL = "BTC-CAD"
    CURRENCY = "CAD"
    INTERVAL = "1d"
    # The previous three-year study did not contain a completed sideways
    # period. Yahoo's public ten-year daily window includes an additional,
    # independent historical sample without changing strategy inputs.
    RANGE = "10y"
    # This completed 365-candle interval is retained as a supplemental
    # research sample. It is fetched only when the rolling window no longer
    # contains a completed Sideways period.
    ANCHORED_SAMPLE_START = "2019-08-20"
    ANCHORED_SAMPLE_END = "2020-08-18"
    ANCHORED_SAMPLE_PERIOD2 = "2020-08-19"
    ANCHORED_SAMPLE_CANDLES = 365
    ROLLING_SOURCE_LABEL = "Rolling Yahoo Finance 10-year window"
    ANCHORED_SOURCE_LABEL = (
        "Anchored Yahoo Finance BTC/CAD sample "
        "(2019-08-20 to 2020-08-18)"
    )
    SOURCE_LABEL = "Yahoo Finance public aggregated BTC/CAD daily data"

    def __init__(self, data_range=RANGE, timeout=20):
        if not isinstance(data_range, str) or not data_range:
            raise ValueError("data_range must be a non-empty string")

        if not isinstance(timeout, (int, float)) or timeout <= 0:
            raise ValueError("timeout must be positive")

        self.data_range = data_range
        self.timeout = timeout
        self.candles = []
        self.rolling_candles = []
        self.anchored_candles = []
        self.last_error = None
        self.last_anchored_error = None

    def load(self):
        """Fetch, validate, and store completed daily BTC/CAD candles."""
        self.candles = []
        self.rolling_candles = []
        self.anchored_candles = []
        self.last_error = None
        self.last_anchored_error = None

        try:
            payload = self._request_json()
            self.candles = self._parse_payload(payload)

            if not self.candles:
                raise YahooFinanceMarketDataError(
                    "Yahoo Finance returned no completed BTC/CAD candles"
                )

            self.rolling_candles = self.get_candles()
            return self.get_candles()
        except YahooFinanceMarketDataError as error:
            self.last_error = str(error)
            self.candles = []
            self.rolling_candles = []
            return []

    def fetch(self):
        """Alias for load() for callers that prefer fetch terminology."""
        return self.load()

    def load_anchored_sample(self):
        """
        Fetch the fixed completed sample used to preserve Sideways coverage.

        This sample is deliberately separate from the rolling window. Callers
        decide whether it is needed after classifying the rolling periods.
        """
        self.anchored_candles = []
        self.last_anchored_error = None

        try:
            payload = self._request_anchored_json()
            candles = self._parse_payload(payload)
            if len(candles) < self.ANCHORED_SAMPLE_CANDLES:
                raise YahooFinanceMarketDataError(
                    "Yahoo Finance anchored BTC/CAD sample returned fewer "
                    f"than {self.ANCHORED_SAMPLE_CANDLES} completed candles"
                )

            self.anchored_candles = candles[:self.ANCHORED_SAMPLE_CANDLES]
            return self.get_anchored_candles()
        except YahooFinanceMarketDataError as error:
            self.last_anchored_error = str(error)
            self.anchored_candles = []
            return []

    def _request_json(self):
        return self._request_payload(
            {
                "range": self.data_range,
                "interval": self.INTERVAL,
            }
        )

    def _request_anchored_json(self):
        start_timestamp = int(
            datetime.strptime(
                self.ANCHORED_SAMPLE_START,
                "%Y-%m-%d",
            ).replace(tzinfo=timezone.utc).timestamp()
        )
        end_timestamp = int(
            datetime.strptime(
                self.ANCHORED_SAMPLE_PERIOD2,
                "%Y-%m-%d",
            ).replace(tzinfo=timezone.utc).timestamp()
        )
        return self._request_payload(
            {
                "period1": start_timestamp,
                "period2": end_timestamp,
                "interval": self.INTERVAL,
            }
        )

    def _request_payload(self, query):
        url = (
            f"{self.CHART_URL}?"
            f"{urlencode(query)}"
        )
        request = Request(
            url,
            headers={
                "Accept": "application/json",
                "User-Agent": "paper-trading-historical-research/1.0",
            },
        )

        try:
            with urlopen(request, timeout=self.timeout) as response:
                raw_body = response.read()
        except HTTPError as error:
            raise YahooFinanceMarketDataError(
                f"Yahoo Finance HTTP error {error.code}"
            ) from error
        except (URLError, TimeoutError, OSError) as error:
            raise YahooFinanceMarketDataError(
                f"Yahoo Finance connection failed: {error}"
            ) from error

        try:
            payload = json.loads(raw_body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise YahooFinanceMarketDataError(
                "Yahoo Finance returned invalid JSON"
            ) from error

        if not isinstance(payload, dict):
            raise YahooFinanceMarketDataError(
                "Yahoo Finance returned an invalid response object"
            )

        return payload

    @classmethod
    def _parse_payload(cls, payload, now=None):
        chart = payload.get("chart")
        if not isinstance(chart, dict):
            raise YahooFinanceMarketDataError(
                "Yahoo Finance response did not contain a chart object"
            )

        if chart.get("error"):
            raise YahooFinanceMarketDataError(
                f"Yahoo Finance error: {chart['error']}"
            )

        result = chart.get("result")
        if not isinstance(result, list) or not result:
            raise YahooFinanceMarketDataError(
                "Yahoo Finance response did not contain a result set"
            )

        series = result[0]
        if not isinstance(series, dict):
            raise YahooFinanceMarketDataError(
                "Yahoo Finance returned an invalid BTC/CAD series"
            )

        metadata = series.get("meta", {})
        if not isinstance(metadata, dict):
            raise YahooFinanceMarketDataError(
                "Yahoo Finance BTC/CAD data is missing series metadata"
            )

        symbol = str(metadata.get("symbol", "")).upper()
        currency = str(metadata.get("currency", "")).upper()
        if symbol != cls.SYMBOL or currency != cls.CURRENCY:
            raise YahooFinanceMarketDataError(
                "Yahoo Finance did not return the BTC-CAD CAD series"
            )

        timestamps = series.get("timestamp")
        indicators = series.get("indicators")
        if not isinstance(timestamps, list) or not isinstance(indicators, dict):
            raise YahooFinanceMarketDataError(
                "Yahoo Finance BTC/CAD data is missing timestamps or indicators"
            )

        quotes = indicators.get("quote")
        if not isinstance(quotes, list) or not quotes:
            raise YahooFinanceMarketDataError(
                "Yahoo Finance BTC/CAD data is missing OHLCV quotes"
            )

        quote = quotes[0]
        if not isinstance(quote, dict):
            raise YahooFinanceMarketDataError(
                "Yahoo Finance returned invalid OHLCV quote data"
            )

        now = now or datetime.now(timezone.utc)
        current_date = now.astimezone(timezone.utc).date()
        raw_candles = []

        for index, timestamp in enumerate(timestamps):
            try:
                values = {
                    "timestamp": int(timestamp),
                    "open": quote["open"][index],
                    "high": quote["high"][index],
                    "low": quote["low"][index],
                    "close": quote["close"][index],
                    "volume": quote["volume"][index],
                }
                numeric_values = [
                    float(values[field])
                    for field in ("open", "high", "low", "close", "volume")
                ]
            except (KeyError, IndexError, TypeError, ValueError):
                continue

            if not all(math.isfinite(value) for value in numeric_values):
                continue

            if min(numeric_values[:4]) <= 0 or numeric_values[4] < 0:
                continue

            try:
                candle_date = datetime.fromtimestamp(
                    values["timestamp"],
                    tz=timezone.utc,
                ).date()
            except (OverflowError, OSError, ValueError):
                continue

            if candle_date >= current_date:
                continue

            raw_candles.append(values)

        raw_candles.sort(key=lambda candle: candle["timestamp"])
        unique_candles = {
            candle["timestamp"]: candle
            for candle in raw_candles
        }
        normalized = HistoricalDataLoader().load_candles(
            [
                unique_candles[timestamp]
                for timestamp in sorted(unique_candles)
            ]
        )

        return normalized

    def count(self):
        return len(self.candles)

    def get_candles(self):
        return list(self.candles)

    def get_rolling_candles(self):
        return list(self.rolling_candles)

    def get_anchored_candles(self):
        return list(self.anchored_candles)
