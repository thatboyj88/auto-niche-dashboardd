import json
import math
from datetime import datetime, timezone
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from market_data_health import UNAVAILABLE, inspect_candles


class KrakenMarketDataError(RuntimeError):
    """Raised when Kraken public market data cannot be loaded."""


class KrakenMarketData:
    """
    Loads public Kraken OHLC candles without authentication.

    This class intentionally uses only Kraken's public AssetPairs and OHLC
    endpoints. It does not access account, wallet, or order functionality.
    """

    API_BASE_URL = "https://api.kraken.com/0/public"

    def __init__(self, interval=60, timeout=20):
        if not isinstance(interval, int) or interval <= 0:
            raise ValueError("interval must be a positive integer")

        if not isinstance(timeout, (int, float)) or timeout <= 0:
            raise ValueError("timeout must be positive")

        self.interval = interval
        self.timeout = timeout
        self.candles = []
        self.pair_identifier = None
        self.pair_name = None
        self.last_error = None
        self.health = inspect_candles(
            [],
            interval_minutes=self.interval,
            provider_available=False,
        )

    def load(self):
        """Fetch and store the latest committed OHLC candles."""
        self.candles = []
        self.pair_identifier = None
        self.pair_name = None
        self.last_error = None
        self.health = inspect_candles(
            [],
            interval_minutes=self.interval,
            provider_available=False,
        )

        try:
            pair_identifier, pair_name = self._resolve_btc_cad_pair()
            response = self._request_json(
                "/OHLC",
                {
                    "pair": pair_identifier,
                    "interval": self.interval
                }
            )

            result = response.get("result")
            if not isinstance(result, dict):
                raise KrakenMarketDataError(
                    "Kraken OHLC response did not contain a result object"
                )

            rows = self._find_ohlc_rows(result)
            parsed_candles = self._parse_rows(rows)

            if parsed_candles:
                latest_timestamp = max(
                    candle["timestamp"]
                    for candle in parsed_candles
                )

                parsed_candles = [
                    candle
                    for candle in parsed_candles
                    if candle["timestamp"] != latest_timestamp
                ]

            parsed_candles.sort(
                key=lambda candle: candle["timestamp"]
            )

            self.pair_identifier = pair_identifier
            self.pair_name = pair_name
            self.candles = parsed_candles
            self.health = inspect_candles(
                self.candles,
                interval_minutes=self.interval,
            )
            return self.get_candles()

        except KrakenMarketDataError as error:
            self.last_error = str(error)
            self.candles = []
            self.health = inspect_candles(
                [],
                interval_minutes=self.interval,
                provider_available=False,
                provider_error=self.last_error,
            )
            return []

    def fetch(self):
        """Alias for load() for callers that prefer fetch terminology."""
        return self.load()

    def _resolve_btc_cad_pair(self):
        response = self._request_json("/AssetPairs")
        result = response.get("result")

        if not isinstance(result, dict):
            raise KrakenMarketDataError(
                "Kraken AssetPairs response did not contain a result object"
            )

        candidates = []

        for identifier, details in result.items():
            if not isinstance(details, dict):
                continue

            base = self._canonical_asset(details.get("base"))
            quote = self._canonical_asset(details.get("quote"))

            wsname = str(details.get("wsname", "")).upper()
            if "/" in wsname:
                ws_base, ws_quote = wsname.split("/", 1)
                base = base or self._canonical_asset(ws_base)
                quote = quote or self._canonical_asset(ws_quote)

            if base != "XBT" or quote != "CAD":
                continue

            display_name = (
                details.get("wsname")
                or details.get("altname")
                or identifier
            )
            status = str(details.get("status", "")).lower()
            exact_display_match = (
                str(display_name).upper() == "XBT/CAD"
            )

            candidates.append({
                "identifier": identifier,
                "display_name": display_name,
                "online": status == "online",
                "exact_display_match": exact_display_match
            })

        if not candidates:
            raise KrakenMarketDataError(
                "Kraken did not report an XBT/CAD market pair"
            )

        candidates.sort(
            key=lambda candidate: (
                not candidate["online"],
                not candidate["exact_display_match"],
                candidate["identifier"]
            )
        )

        selected = candidates[0]
        return (
            selected["identifier"],
            selected["display_name"]
        )

    def _request_json(self, endpoint, params=None):
        url = f"{self.API_BASE_URL}{endpoint}"

        if params:
            url = f"{url}?{urlencode(params)}"

        request = Request(
            url,
            headers={
                "Accept": "application/json",
                "User-Agent": "paper-trading-market-data/1.0"
            }
        )

        try:
            with urlopen(request, timeout=self.timeout) as response:
                raw_body = response.read()
        except HTTPError as error:
            raise KrakenMarketDataError(
                f"Kraken HTTP error {error.code}"
            ) from error
        except (URLError, TimeoutError, OSError) as error:
            raise KrakenMarketDataError(
                f"Kraken connection failed: {error}"
            ) from error

        try:
            payload = json.loads(raw_body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise KrakenMarketDataError(
                "Kraken returned invalid JSON"
            ) from error

        if not isinstance(payload, dict):
            raise KrakenMarketDataError(
                "Kraken returned an invalid response object"
            )

        api_errors = payload.get("error", [])
        if api_errors:
            if isinstance(api_errors, list):
                message = "; ".join(str(item) for item in api_errors)
            else:
                message = str(api_errors)

            raise KrakenMarketDataError(
                f"Kraken API error: {message}"
            )

        return payload

    @staticmethod
    def _find_ohlc_rows(result):
        for key, value in result.items():
            if key != "last" and isinstance(value, list):
                return value

        raise KrakenMarketDataError(
            "Kraken OHLC response did not contain candle rows"
        )

    @classmethod
    def _parse_rows(cls, rows):
        candles = []

        for row in rows:
            if not isinstance(row, (list, tuple)) or len(row) < 7:
                continue

            try:
                values = [
                    float(row[index])
                    for index in (0, 1, 2, 3, 4, 6)
                ]
            except (TypeError, ValueError, IndexError):
                continue

            if not all(math.isfinite(value) for value in values):
                continue

            timestamp, open_price, high, low, close, volume = values

            if timestamp < 0 or min(
                open_price,
                high,
                low,
                close,
                volume
            ) < 0:
                continue

            candles.append({
                "timestamp": int(timestamp),
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume
            })

        return candles

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

    def get_candles(self):
        return list(self.candles)

    @staticmethod
    def _canonical_asset(value):
        if value is None:
            return None

        symbol = str(value).upper().replace("/", "")

        if symbol in {"XXBT", "XBT", "BTC"}:
            return "XBT"

        if symbol in {"ZCAD", "CAD"}:
            return "CAD"

        return symbol or None


def _format_timestamp(timestamp):
    return datetime.fromtimestamp(
        timestamp,
        tz=timezone.utc
    ).isoformat()


if __name__ == "__main__":
    print("--------------------------------")
    print("KRAKEN LIVE DATA TEST")
    print("--------------------------------")

    loader = KrakenMarketData(interval=60)
    candles = loader.load()

    if not candles:
        print(f"ERROR: {loader.last_error or 'No candles loaded'}")
        print("--------------------------------")
        print("REAL MONEY TRADING: DISABLED")
        print("================================")
        raise SystemExit(1)

    first_candle = candles[0]
    last_candle = candles[-1]

    print(f"Pair: {loader.pair_name}")
    print(f"Kraken Pair Identifier: {loader.pair_identifier}")
    print(f"Interval: {loader.interval} minutes")
    print(f"Candles Loaded: {loader.count()}")
    print(
        "First Timestamp: "
        f"{_format_timestamp(first_candle['timestamp'])}"
    )
    print(
        "Last Timestamp: "
        f"{_format_timestamp(last_candle['timestamp'])}"
    )
    print(f"First Close: ${first_candle['close']:.2f}")
    print(f"Last Close: ${last_candle['close']:.2f}")
    print("--------------------------------")
    print("REAL MONEY TRADING: DISABLED")
    print("================================")