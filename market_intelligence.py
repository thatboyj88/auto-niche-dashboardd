"""Read-only, provider-neutral market intelligence for dashboard consumers.

The observation runner intentionally does not use this service.  Its existing
Kraken loader and cadence remain the source of genuine paper evidence.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from xml.etree import ElementTree

from kraken_live_data import KrakenMarketData
from market_data_health import DEGRADED, HEALTHY, UNAVAILABLE, inspect_candles


@dataclass(frozen=True)
class MarketSnapshot:
    provider: str
    pair: str | None
    price: float | None
    bid: float | None
    ask: float | None
    volume: float | None
    candles: tuple[dict, ...]
    observed_timestamp: int | None
    received_timestamp: float
    freshness_age_seconds: float | None
    latency_ms: float | None
    health: dict
    error: str | None = None
    rate_limited: bool = False
    credentials_required: bool = False

    @property
    def pair_name(self):
        return self.pair

    @property
    def last_error(self):
        return self.error

    @property
    def data_range(self):
        return "live"

    def count(self):
        return len(self.candles)


@dataclass(frozen=True)
class NewsEvent:
    title: str
    source: str
    url: str
    observed_at: str
    freshness_age_seconds: float | None


@dataclass(frozen=True)
class NewsSnapshot:
    source: str
    status: str
    items: tuple[NewsEvent, ...]
    fetched_at: str
    error: str | None = None


def fetch_public_news_events(
    *,
    feed_url: str = "https://www.coindesk.com/arc/outboundfeeds/rss/",
    source: str = "CoinDesk RSS",
    request_text: Callable | None = None,
    max_age_seconds: float = 48 * 60 * 60,
    timeout: float = 10,
) -> NewsSnapshot:
    """Read attributed public RSS context; never supplies synthetic headlines."""
    fetched_at = datetime.now(timezone.utc)
    try:
        if request_text is not None:
            payload = request_text(feed_url)
        else:
            request = Request(
                feed_url,
                headers={
                    "Accept": "application/rss+xml, application/xml, text/xml",
                    "User-Agent": "paper-trading-market-data/1.0",
                },
            )
            with urlopen(request, timeout=timeout) as response:
                payload = response.read()
        if isinstance(payload, bytes):
            payload = payload.decode("utf-8")
        root = ElementTree.fromstring(payload)
        items = []
        for item in root.findall(".//item"):
            title = (item.findtext("title") or "").strip()
            url = (item.findtext("link") or "").strip()
            published = (item.findtext("pubDate") or "").strip()
            if not title or not url or not published:
                continue
            observed = parsedate_to_datetime(published)
            if observed.tzinfo is None:
                observed = observed.replace(tzinfo=timezone.utc)
            age = max(0.0, (fetched_at - observed).total_seconds())
            if age > max_age_seconds:
                continue
            items.append(
                NewsEvent(
                    title=title,
                    source=source,
                    url=url,
                    observed_at=observed.isoformat(),
                    freshness_age_seconds=round(age, 1),
                )
            )
        if not items:
            return NewsSnapshot(
                source=source,
                status=UNAVAILABLE,
                items=(),
                fetched_at=fetched_at.isoformat(),
                error="News feed contained no fresh, well-formed items.",
            )
        return NewsSnapshot(
            source=source,
            status=HEALTHY,
            items=tuple(items[:10]),
            fetched_at=fetched_at.isoformat(),
        )
    except Exception as error:
        return NewsSnapshot(
            source=source,
            status=UNAVAILABLE,
            items=(),
            fetched_at=fetched_at.isoformat(),
            error=f"News feed unavailable: {error}",
        )


class MarketProvider:
    """Small extension seam for additional legitimate public providers."""

    name = "unknown"

    def fetch(self) -> MarketSnapshot:
        raise NotImplementedError


@dataclass(frozen=True)
class MarketComparison:
    primary: MarketSnapshot
    secondary: MarketSnapshot
    status: str
    price_difference_percent: float | None
    message: str

    @property
    def agreement(self):
        return self.status == HEALTHY


class KrakenPublicMarketProvider(MarketProvider):
    name = "Kraken public API"

    def __init__(
        self,
        *,
        loader_factory: Callable[[], KrakenMarketData] = KrakenMarketData,
        interval: int = 60,
        max_attempts: int = 2,
    ):
        self.loader_factory = loader_factory
        self.interval = interval
        self.max_attempts = max(1, int(max_attempts))

    def fetch(self) -> MarketSnapshot:
        received = time.time()
        started = time.perf_counter()
        last_error = None
        loader = None
        candles = []
        for attempt in range(self.max_attempts):
            loader = self.loader_factory()
            candles = loader.load()
            if candles:
                break
            last_error = getattr(loader, "last_error", None) or "No candles returned"
            if attempt + 1 < self.max_attempts:
                time.sleep(0.05 * (2**attempt))

        latency_ms = round((time.perf_counter() - started) * 1000, 1)
        if loader is None or not candles:
            return self._unavailable(
                received,
                latency_ms,
                last_error or "Kraken market data unavailable",
            )

        health = inspect_candles(
            candles,
            interval_minutes=self.interval,
            provider_available=True,
            now_timestamp=received,
        )
        if health["status"] != HEALTHY:
            return self._snapshot(
                loader,
                candles,
                received,
                latency_ms,
                health,
                health["issues"][0] if health["issues"] else "Invalid market data",
            )

        ticker = self._load_ticker(loader)
        if ticker["error"]:
            return self._snapshot(
                loader,
                candles,
                received,
                latency_ms,
                health,
                ticker["error"],
            )

        return self._snapshot(
            loader,
            candles,
            received,
            latency_ms,
            health,
            None,
            ticker=ticker,
        )

    @staticmethod
    def _load_ticker(loader):
        try:
            response = loader._request_json(
                "/Ticker",
                {"pair": loader.pair_identifier or "XXBTZCAD"},
            )
            result = response.get("result")
            if not isinstance(result, dict):
                raise ValueError("ticker result is not an object")
            row = next(iter(result.values()), None)
            if not isinstance(row, dict):
                raise ValueError("ticker pair is missing")

            def number(name, index, required=False):
                values = row.get(name)
                value = values[index] if isinstance(values, (list, tuple)) and len(values) > index else None
                if value is None and required:
                    raise ValueError(f"ticker field {name} is missing")
                if value is None:
                    return None
                value = float(value)
                if value <= 0:
                    raise ValueError(f"ticker field {name} is not positive")
                return value

            price = number("c", 0, required=True)
            bid = number("b", 0)
            ask = number("a", 0)
            volume = number("v", 1)
            if bid is not None and ask is not None and bid > ask:
                raise ValueError("ticker bid exceeds ask")
            return {"price": price, "bid": bid, "ask": ask, "volume": volume, "error": None}
        except Exception as error:
            return {
                "price": None,
                "bid": None,
                "ask": None,
                "volume": None,
                "error": f"Kraken ticker rejected: {error}",
            }

    def _snapshot(self, loader, candles, received, latency_ms, health, error, ticker=None):
        latest = candles[-1] if candles else {}
        age = health.get("data_age_seconds")
        if error:
            health = dict(health)
            health["status"] = DEGRADED
            health["provider_error"] = error
        return MarketSnapshot(
            provider=self.name,
            pair=getattr(loader, "pair_name", None) or "XBT/CAD",
            price=ticker["price"] if ticker else None,
            bid=ticker["bid"] if ticker else None,
            ask=ticker["ask"] if ticker else None,
            volume=ticker["volume"] if ticker else None,
            candles=tuple(candles),
            observed_timestamp=latest.get("timestamp"),
            received_timestamp=received,
            freshness_age_seconds=age,
            latency_ms=latency_ms,
            health=health,
            error=error,
            rate_limited=bool(error and ("429" in error or "rate" in error.lower())),
        )

    def _unavailable(self, received, latency_ms, error):
        return MarketSnapshot(
            provider=self.name,
            pair=None,
            price=None,
            bid=None,
            ask=None,
            volume=None,
            candles=(),
            observed_timestamp=None,
            received_timestamp=received,
            freshness_age_seconds=None,
            latency_ms=latency_ms,
            health=inspect_candles(
                [],
                interval_minutes=self.interval,
                provider_available=False,
                provider_error=error,
            ),
            error=error,
        )


class CoinGeckoPublicMarketProvider(MarketProvider):
    """Loads public CoinGecko BTC/CAD spot and OHLC data.

    CoinGecko's public OHLC response does not include candle volume.  Those
    candles remain visible as incomplete comparison data and are never marked
    healthy or used as trading evidence.
    """

    name = "CoinGecko public API"
    API_BASE_URL = "https://api.coingecko.com/api/v3"

    def __init__(
        self,
        *,
        request_json: Callable | None = None,
        interval: int = 60,
        max_attempts: int = 2,
        timeout: float = 20,
    ):
        self.interval = interval
        self.max_attempts = max(1, int(max_attempts))
        self.timeout = timeout
        self._request_json_impl = request_json

    def fetch(self) -> MarketSnapshot:
        received = time.time()
        started = time.perf_counter()
        last_error = None
        for attempt in range(self.max_attempts):
            try:
                ticker = self._request_json(
                    "/simple/price",
                    {
                        "ids": "bitcoin",
                        "vs_currencies": "cad",
                        "include_24hr_vol": "true",
                    },
                )
                rows = self._request_json(
                    "/coins/bitcoin/ohlc",
                    {"vs_currency": "cad", "days": "1"},
                )
                candles = self._parse_candles(rows)
                if not candles:
                    raise ValueError("CoinGecko returned no usable candles")
                health = inspect_candles(
                    candles,
                    interval_minutes=self.interval,
                    provider_available=True,
                    now_timestamp=received,
                    allow_missing_volume=True,
                )
                if health["status"] != HEALTHY:
                    return self._snapshot(
                        ticker, candles, received,
                        round((time.perf_counter() - started) * 1000, 1),
                        health, health["issues"][0],
                    )
                return self._snapshot(
                    ticker, candles, received,
                    round((time.perf_counter() - started) * 1000, 1),
                    health, None,
                )
            except Exception as error:
                last_error = str(error)
                if attempt + 1 < self.max_attempts:
                    time.sleep(0.05 * (2**attempt))

        return self._unavailable(
            received,
            round((time.perf_counter() - started) * 1000, 1),
            f"CoinGecko provider unavailable: {last_error or 'unknown error'}",
        )

    def _request_json(self, endpoint, params=None):
        if self._request_json_impl is not None:
            return self._request_json_impl(endpoint, params)
        url = f"{self.API_BASE_URL}{endpoint}"
        if params:
            url = f"{url}?{urlencode(params)}"
        request = Request(
            url,
            headers={
                "Accept": "application/json",
                "User-Agent": "paper-trading-market-data/1.0",
            },
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                payload = response.read().decode("utf-8")
        except HTTPError as error:
            raise RuntimeError(f"CoinGecko HTTP error {error.code}") from error
        except (URLError, TimeoutError, OSError) as error:
            raise RuntimeError(f"CoinGecko connection failed: {error}") from error
        import json
        try:
            value = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RuntimeError("CoinGecko returned invalid JSON") from error
        if isinstance(value, dict) and value.get("message"):
            raise RuntimeError(f"CoinGecko API error: {value['message']}")
        return value

    @staticmethod
    def _parse_candles(rows):
        candles = []
        if not isinstance(rows, list):
            return candles
        for row in rows:
            if not isinstance(row, (list, tuple)) or len(row) < 5:
                continue
            try:
                timestamp, open_price, high, low, close = (
                    float(row[index]) for index in range(5)
                )
            except (TypeError, ValueError):
                continue
            if not all(
                value > 0 and value != float("inf") and value != float("-inf")
                for value in (timestamp, low, high, open_price, close)
            ):
                continue
            if timestamp > 10_000_000_000:
                timestamp /= 1000
            candles.append({
                "timestamp": int(timestamp),
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
                "volume": None,
            })
        return sorted(candles, key=lambda candle: candle["timestamp"])

    def _snapshot(self, ticker, candles, received, latency_ms, health, error):
        try:
            quote = ticker["bitcoin"]
            price = float(quote["cad"])
            bid = ask = None
            volume = (
                float(quote["cad_24h_vol"])
                if quote.get("cad_24h_vol") is not None else None
            )
            if price <= 0:
                raise ValueError("ticker quote is not positive")
        except (AttributeError, KeyError, TypeError, ValueError):
            price = bid = ask = volume = None
            error = error or "CoinGecko ticker rejected: missing or invalid quote"
            health = dict(health)
            health["status"] = DEGRADED
            health["provider_error"] = error
        latest = candles[-1] if candles else {}
        return MarketSnapshot(
            provider=self.name,
            pair="BTC/CAD",
            price=price,
            bid=bid,
            ask=ask,
            volume=volume,
            candles=tuple(candles),
            observed_timestamp=latest.get("timestamp"),
            received_timestamp=received,
            freshness_age_seconds=health.get("data_age_seconds"),
            latency_ms=latency_ms,
            health=health,
            error=error,
            rate_limited=bool(error and ("429" in error or "rate" in error.lower())),
        )

    def _unavailable(self, received, latency_ms, error):
        return MarketSnapshot(
            provider=self.name,
            pair=None,
            price=None,
            bid=None,
            ask=None,
            volume=None,
            candles=(),
            observed_timestamp=None,
            received_timestamp=received,
            freshness_age_seconds=None,
            latency_ms=latency_ms,
            health=inspect_candles(
                [],
                interval_minutes=self.interval,
                provider_available=False,
                provider_error=error,
            ),
            error=error,
            rate_limited=bool("429" in error or "rate" in error.lower()),
        )


class CoinbasePublicMarketProvider(MarketProvider):
    """Loads the public Coinbase Exchange BTC-CAD market and OHLCV candles.

    Coinbase is comparison-only.  Product verification, ticker quote, and
    candles must all succeed before this provider can be healthy.
    """

    name = "Coinbase Exchange public API"
    API_BASE_URL = "https://api.exchange.coinbase.com"

    def __init__(
        self,
        *,
        request_json: Callable | None = None,
        interval: int = 60,
        max_attempts: int = 2,
        timeout: float = 20,
        product_id: str = "BTC-CAD",
    ):
        self.interval = interval
        self.max_attempts = max(1, int(max_attempts))
        self.timeout = timeout
        self.product_id = product_id
        self._request_json_impl = request_json

    def fetch(self) -> MarketSnapshot:
        received = time.time()
        started = time.perf_counter()
        last_error = None
        for attempt in range(self.max_attempts):
            try:
                product = self._request_json(f"/products/{self.product_id}")
                self._validate_product(product)
                ticker = self._request_json(f"/products/{self.product_id}/ticker")
                rows = self._request_json(
                    f"/products/{self.product_id}/candles",
                    {"granularity": str(self.interval * 60)},
                )
                candles = self._parse_candles(rows)
                if not candles:
                    raise ValueError("Coinbase returned no usable OHLCV candles")
                health = inspect_candles(
                    candles,
                    interval_minutes=self.interval,
                    provider_available=True,
                    now_timestamp=received,
                )
                if health["status"] != HEALTHY:
                    return self._snapshot(
                        ticker, candles, received,
                        round((time.perf_counter() - started) * 1000, 1),
                        health, health["issues"][0] if health["issues"] else "Invalid market data",
                    )
                return self._snapshot(
                    ticker, candles, received,
                    round((time.perf_counter() - started) * 1000, 1),
                    health, None,
                )
            except Exception as error:
                last_error = str(error)
                if attempt + 1 < self.max_attempts:
                    time.sleep(0.05 * (2**attempt))
        return self._unavailable(
            received,
            round((time.perf_counter() - started) * 1000, 1),
            f"Coinbase provider unavailable: {last_error or 'unknown error'}",
        )

    def _request_json(self, endpoint, params=None):
        if self._request_json_impl is not None:
            return self._request_json_impl(endpoint, params)
        url = f"{self.API_BASE_URL}{endpoint}"
        if params:
            url = f"{url}?{urlencode(params)}"
        request = Request(
            url,
            headers={
                "Accept": "application/json",
                "User-Agent": "paper-trading-market-data/1.0",
            },
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                payload = response.read().decode("utf-8")
        except HTTPError as error:
            raise RuntimeError(f"Coinbase HTTP error {error.code}") from error
        except (URLError, TimeoutError, OSError) as error:
            raise RuntimeError(f"Coinbase connection failed: {error}") from error
        import json
        try:
            return json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RuntimeError("Coinbase returned invalid JSON") from error

    def _validate_product(self, product):
        if not isinstance(product, dict):
            raise ValueError("Coinbase product response is not an object")
        if product.get("id") != self.product_id:
            raise ValueError("Coinbase BTC-CAD product is missing")
        if product.get("status") not in {None, "online"}:
            raise ValueError(f"Coinbase BTC-CAD product is {product['status']}")
        if product.get("trading_disabled") is True:
            raise ValueError("Coinbase BTC-CAD trading is disabled")

    @staticmethod
    def _parse_candles(rows):
        candles = []
        if not isinstance(rows, list):
            return candles
        for row in rows:
            if not isinstance(row, (list, tuple)) or len(row) < 6:
                continue
            try:
                timestamp, low, high, open_price, close, volume = (
                    float(row[index]) for index in range(6)
                )
            except (TypeError, ValueError):
                continue
            values = (timestamp, low, high, open_price, close, volume)
            if not all(value > 0 and value != float("inf") and value != float("-inf")
                       for value in values):
                continue
            candles.append({
                "timestamp": int(timestamp),
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            })
        return sorted(candles, key=lambda candle: candle["timestamp"])

    def _snapshot(self, ticker, candles, received, latency_ms, health, error):
        try:
            if not isinstance(ticker, dict):
                raise ValueError("ticker response is not an object")
            price = float(ticker["price"])
            bid = float(ticker["bid"])
            ask = float(ticker["ask"])
            volume = float(ticker["volume"])
            if min(price, bid, ask, volume) <= 0 or bid > ask:
                raise ValueError("ticker quote is invalid")
        except (KeyError, TypeError, ValueError):
            price = bid = ask = volume = None
            error = error or "Coinbase ticker rejected: missing or invalid quote"
            health = dict(health)
            health["status"] = DEGRADED
            health["provider_error"] = error
        latest = candles[-1] if candles else {}
        return MarketSnapshot(
            provider=self.name,
            pair=self.product_id,
            price=price,
            bid=bid,
            ask=ask,
            volume=volume,
            candles=tuple(candles),
            observed_timestamp=latest.get("timestamp"),
            received_timestamp=received,
            freshness_age_seconds=health.get("data_age_seconds"),
            latency_ms=latency_ms,
            health=health,
            error=error,
            rate_limited=bool(error and ("429" in error or "rate" in error.lower())),
        )

    def _unavailable(self, received, latency_ms, error):
        return MarketSnapshot(
            provider=self.name,
            pair=None,
            price=None,
            bid=None,
            ask=None,
            volume=None,
            candles=(),
            observed_timestamp=None,
            received_timestamp=received,
            freshness_age_seconds=None,
            latency_ms=latency_ms,
            health=inspect_candles(
                [],
                interval_minutes=self.interval,
                provider_available=False,
                provider_error=error,
            ),
            error=error,
            rate_limited=bool("429" in error or "rate" in error.lower()),
        )


def compare_market_snapshots(
    primary: MarketSnapshot,
    secondary: MarketSnapshot,
    *,
    max_difference_percent: float = 2.0,
):
    primary_health = getattr(primary, "health", {}) or {}
    secondary_health = getattr(secondary, "health", {}) or {}
    if primary_health.get("status") != HEALTHY:
        return MarketComparison(
            primary, secondary, UNAVAILABLE, None,
            "Primary provider is not healthy; no comparison is selected.",
        )
    if secondary.price is None:
        return MarketComparison(
            primary, secondary, UNAVAILABLE, None,
            "Comparison provider is unavailable or unhealthy.",
        )
    if primary.price is None or primary.price <= 0:
        return MarketComparison(
            primary, secondary, UNAVAILABLE, None,
            "Primary provider has no usable price.",
        )
    difference = abs(primary.price - secondary.price) / primary.price * 100
    if secondary_health.get("status") != HEALTHY:
        return MarketComparison(
            primary, secondary, DEGRADED, difference,
            "Comparison price is available, but its market data is incomplete or unhealthy; "
            "no blended or automatic failover value is used.",
        )
    status = HEALTHY if difference <= max_difference_percent else DEGRADED
    message = (
        "Providers agree within the comparison threshold."
        if status == HEALTHY
        else "Provider prices disagree; no blended or automatic failover value is used."
    )
    return MarketComparison(primary, secondary, status, difference, message)


class MarketDataService:
    """Single-flight short-lived cache shared by dashboard reruns."""

    def __init__(self, provider: MarketProvider | None = None, ttl_seconds=45):
        self.provider = provider or KrakenPublicMarketProvider()
        self.ttl_seconds = max(1.0, float(ttl_seconds))
        self._lock = threading.Lock()
        self._cached: MarketSnapshot | None = None
        self._cached_at = 0.0

    def get(self, *, force_refresh=False) -> MarketSnapshot:
        now = time.time()
        with self._lock:
            if not force_refresh and self._cached and now - self._cached_at < self.ttl_seconds:
                return self._cached
            try:
                snapshot = self.provider.fetch()
            except Exception as error:
                if isinstance(self.provider, KrakenPublicMarketProvider):
                    snapshot = self.provider._unavailable(
                        time.time(),
                        None,
                        f"Kraken provider failed safely: {error}",
                    )
                else:
                    snapshot = MarketSnapshot(
                        provider=self.provider.name,
                        pair=None,
                        price=None,
                        bid=None,
                        ask=None,
                        volume=None,
                        candles=(),
                        observed_timestamp=None,
                        received_timestamp=time.time(),
                        freshness_age_seconds=None,
                        latency_ms=None,
                        health=inspect_candles(
                            [],
                            provider_available=False,
                            provider_error=str(error),
                        ),
                        error=str(error),
                    )
            self._cached = snapshot
            self._cached_at = time.time()
            return snapshot

    def clear(self):
        with self._lock:
            self._cached = None
            self._cached_at = 0.0
