import time
import unittest

from market_intelligence import (
    CoinbasePublicMarketProvider,
    CoinGeckoPublicMarketProvider,
    DEGRADED,
    HEALTHY,
    UNAVAILABLE,
    KrakenPublicMarketProvider,
    MarketDataService,
    compare_market_snapshots,
    fetch_public_news_events,
)


def candles():
    now = int(time.time())
    return [
        {
            "timestamp": now - 120,
            "open": 100,
            "high": 105,
            "low": 99,
            "close": 103,
            "volume": 10,
        },
        {
            "timestamp": now - 60,
            "open": 103,
            "high": 108,
            "low": 102,
            "close": 107,
            "volume": 12,
        },
    ]


class FakeLoader:
    calls = 0
    ticker = {
        "error": [],
        "result": {
            "XXBTZCAD": {
                "c": ["107.50", "1"],
                "b": ["107.25", "1", "1"],
                "a": ["107.75", "1", "1"],
                "v": ["20", "24"],
            }
        },
    }

    def __init__(self):
        self.pair_identifier = "XXBTZCAD"
        self.pair_name = "XBT/CAD"
        self.last_error = None

    def load(self):
        type(self).calls += 1
        return candles()

    def _request_json(self, endpoint, params=None):
        self.endpoint = endpoint
        return self.ticker


class MarketIntelligenceTests(unittest.TestCase):
    def setUp(self):
        FakeLoader.calls = 0

    def test_provider_normalizes_genuine_kraken_snapshot(self):
        snapshot = KrakenPublicMarketProvider(
            loader_factory=FakeLoader,
            max_attempts=1,
        ).fetch()

        self.assertEqual(snapshot.health["status"], HEALTHY)
        self.assertEqual(snapshot.provider, "Kraken public API")
        self.assertEqual(snapshot.pair, "XBT/CAD")
        self.assertEqual(snapshot.price, 107.5)
        self.assertEqual(snapshot.bid, 107.25)
        self.assertEqual(snapshot.ask, 107.75)
        self.assertEqual(snapshot.volume, 24.0)
        self.assertEqual(len(snapshot.candles), 2)
        self.assertIsNotNone(snapshot.observed_timestamp)
        self.assertIsNotNone(snapshot.received_timestamp)
        self.assertIsNotNone(snapshot.freshness_age_seconds)

    def test_malformed_ticker_fails_closed_without_fabricating_quote(self):
        class Malformed(FakeLoader):
            ticker = {"result": {"XXBTZCAD": {"c": ["not-a-number"]}}}

        snapshot = KrakenPublicMarketProvider(
            loader_factory=Malformed,
            max_attempts=1,
        ).fetch()

        self.assertEqual(snapshot.health["status"], DEGRADED)
        self.assertIsNone(snapshot.price)
        self.assertIsNone(snapshot.bid)
        self.assertIn("ticker rejected", snapshot.error)

    def test_duplicate_and_out_of_order_candles_are_not_healthy(self):
        class BadCandles(FakeLoader):
            def load(self):
                values = candles()
                return [values[1], values[1]]

        snapshot = KrakenPublicMarketProvider(
            loader_factory=BadCandles,
            max_attempts=1,
        ).fetch()

        self.assertEqual(snapshot.health["status"], DEGRADED)
        self.assertIn("duplicate timestamps", snapshot.health["issues"])

    def test_service_cache_prevents_duplicate_dashboard_polls(self):
        service = MarketDataService(
            KrakenPublicMarketProvider(loader_factory=FakeLoader, max_attempts=1),
            ttl_seconds=60,
        )
        first = service.get()
        second = service.get()

        self.assertIs(first, second)
        self.assertEqual(FakeLoader.calls, 1)

    def test_provider_outage_retries_then_reports_unavailable(self):
        class Outage(FakeLoader):
            def load(self):
                type(self).calls += 1
                return []

        snapshot = KrakenPublicMarketProvider(
            loader_factory=Outage,
            max_attempts=2,
        ).fetch()

        self.assertEqual(snapshot.health["status"], UNAVAILABLE)
        self.assertEqual(Outage.calls, 2)
        self.assertIsNone(snapshot.price)
        self.assertIsNotNone(snapshot.error)

    def test_coingecko_provider_normalizes_public_snapshot(self):
        now = int(time.time() * 1000)
        responses = {
            "/simple/price": {
                "bitcoin": {
                    "cad": "108.00",
                    "cad_24h_vol": "12.5",
                },
            },
            "/coins/bitcoin/ohlc": [
                [now - 120000, 107, 108, 106, 107.5],
                [now - 60000, 107.5, 109, 107, 108],
            ],
        }

        def request_json(endpoint, params=None):
            return responses[endpoint]

        snapshot = CoinGeckoPublicMarketProvider(
            request_json=request_json,
            max_attempts=1,
        ).fetch()

        self.assertEqual(snapshot.health["status"], DEGRADED)
        self.assertEqual(snapshot.provider, "CoinGecko public API")
        self.assertEqual(snapshot.pair, "BTC/CAD")
        self.assertEqual(snapshot.price, 108.0)
        self.assertIsNone(snapshot.bid)
        self.assertIsNone(snapshot.ask)
        self.assertEqual(snapshot.volume, 12.5)
        self.assertEqual(snapshot.candles[-1]["close"], 108.0)

    def test_comparison_never_blends_disagreeing_provider_prices(self):
        primary = KrakenPublicMarketProvider(
            loader_factory=FakeLoader,
            max_attempts=1,
        ).fetch()
        secondary = CoinGeckoPublicMarketProvider(
            request_json=lambda endpoint, params=None: (
                {
                    "bitcoin": {
                        "cad": 120,
                        "cad_24h_vol": 2,
                    }
                }
                if endpoint == "/simple/price"
                else [
                    [int(time.time() * 1000) - 120000, 107, 108, 106, 107.5],
                    [int(time.time() * 1000) - 60000, 107.5, 109, 107, 108],
                ]
            ),
            max_attempts=1,
        ).fetch()

        comparison = compare_market_snapshots(primary, secondary)

        self.assertEqual(comparison.status, DEGRADED)
        self.assertFalse(comparison.agreement)
        self.assertGreater(comparison.price_difference_percent, 2)
        self.assertIn("no blended", comparison.message)

    def test_comparison_fails_closed_when_secondary_is_unavailable(self):
        primary = KrakenPublicMarketProvider(
            loader_factory=FakeLoader,
            max_attempts=1,
        ).fetch()
        secondary = CoinGeckoPublicMarketProvider(
            request_json=lambda endpoint, params=None: (_ for _ in ()).throw(
                RuntimeError("CoinGecko HTTP error 429")
            ),
            max_attempts=1,
        ).fetch()

        comparison = compare_market_snapshots(primary, secondary)

        self.assertEqual(comparison.status, UNAVAILABLE)
        self.assertIsNone(comparison.price_difference_percent)
        self.assertTrue(secondary.rate_limited)

    def test_stale_primary_blocks_comparison_even_with_fresh_secondary(self):
        class Stale(FakeLoader):
            def load(self):
                rows = candles()
                return [
                    {**row, "timestamp": row["timestamp"] - 100_000}
                    for row in rows
                ]

        primary = KrakenPublicMarketProvider(
            loader_factory=Stale, max_attempts=1
        ).fetch()
        secondary = CoinbasePublicMarketProvider(
            request_json=lambda endpoint, params=None: {
                "/products/BTC-CAD": {"id": "BTC-CAD", "status": "online"},
                "/products/BTC-CAD/ticker": {
                    "price": "107", "bid": "106.9", "ask": "107.1", "volume": "20"
                },
                "/products/BTC-CAD/candles": [
                    [int(time.time()) - 60, 106, 108, 107, 107, 10],
                    [int(time.time()) - 120, 105, 107, 106, 106, 9],
                ],
            }[endpoint],
            max_attempts=1,
        ).fetch()
        comparison = compare_market_snapshots(primary, secondary)
        self.assertEqual(primary.health["status"], DEGRADED)
        self.assertEqual(comparison.status, UNAVAILABLE)
        self.assertIsNone(comparison.price_difference_percent)
        self.assertIn("Primary provider", comparison.message)

    def test_unhealthy_secondary_is_not_selected_or_used_as_failover(self):
        primary = KrakenPublicMarketProvider(
            loader_factory=FakeLoader, max_attempts=1
        ).fetch()

        class BadSecondary(FakeLoader):
            def load(self):
                rows = candles()
                rows[-1]["volume"] = 0
                return rows

        secondary = KrakenPublicMarketProvider(
            loader_factory=BadSecondary, max_attempts=1
        ).fetch()
        comparison = compare_market_snapshots(primary, secondary)
        self.assertEqual(secondary.health["status"], DEGRADED)
        self.assertEqual(comparison.status, UNAVAILABLE)
        self.assertIsNone(comparison.price_difference_percent)
        self.assertIn("unavailable", comparison.message.lower())

    def test_malformed_secondary_payload_is_unavailable(self):
        secondary = CoinbasePublicMarketProvider(
            request_json=lambda endpoint, params=None: (
                {"id": "BTC-CAD", "status": "online"}
                if endpoint == "/products/BTC-CAD"
                else {"price": "not-a-number"}
                if endpoint == "/products/BTC-CAD/ticker"
                else [["malformed"]]
            ),
            max_attempts=1,
        ).fetch()
        self.assertEqual(secondary.health["status"], UNAVAILABLE)
        self.assertIsNone(secondary.price)
        self.assertEqual(secondary.candles, ())

    def test_market_service_converts_unexpected_provider_failure_to_unavailable(self):
        class Broken:
            name = "Broken comparison provider"

            def fetch(self):
                raise RuntimeError("unexpected provider failure")

        snapshot = MarketDataService(Broken(), ttl_seconds=60).get()
        self.assertEqual(snapshot.health["status"], UNAVAILABLE)
        self.assertIsNone(snapshot.price)
        self.assertIn("unexpected provider failure", snapshot.error)

    def test_coinbase_provider_normalizes_verified_public_ohlcv(self):
        now = int(time.time())
        responses = {
            "/products/BTC-CAD": {
                "id": "BTC-CAD",
                "status": "online",
                "trading_disabled": False,
            },
            "/products/BTC-CAD/ticker": {
                "price": "108.00",
                "bid": "107.90",
                "ask": "108.10",
                "volume": "24.5",
            },
            "/products/BTC-CAD/candles": [
                [now - 60, 107, 109, 108, 108.5, 12.0],
                [now - 120, 106, 108, 107, 107.5, 11.0],
            ],
        }

        snapshot = CoinbasePublicMarketProvider(
            request_json=lambda endpoint, params=None: responses[endpoint],
            max_attempts=1,
        ).fetch()

        self.assertEqual(snapshot.health["status"], HEALTHY)
        self.assertEqual(snapshot.provider, "Coinbase Exchange public API")
        self.assertEqual(snapshot.pair, "BTC-CAD")
        self.assertEqual(snapshot.price, 108.0)
        self.assertEqual(snapshot.bid, 107.9)
        self.assertEqual(snapshot.ask, 108.1)
        self.assertEqual(snapshot.volume, 24.5)
        self.assertEqual(snapshot.candles[-1]["volume"], 12.0)
        self.assertEqual(snapshot.observed_timestamp, now - 60)

    def test_coinbase_provider_retries_rate_limit_and_fails_closed(self):
        calls = []

        def rate_limited(endpoint, params=None):
            calls.append(endpoint)
            raise RuntimeError("Coinbase HTTP error 429")

        snapshot = CoinbasePublicMarketProvider(
            request_json=rate_limited,
            max_attempts=2,
        ).fetch()

        self.assertEqual(snapshot.health["status"], UNAVAILABLE)
        self.assertTrue(snapshot.rate_limited)
        self.assertEqual(len(calls), 2)
        self.assertIsNone(snapshot.price)

    def test_coinbase_product_verification_rejects_non_live_market(self):
        responses = {
            "/products/BTC-CAD": {"id": "BTC-CAD", "status": "offline"},
        }
        snapshot = CoinbasePublicMarketProvider(
            request_json=lambda endpoint, params=None: responses[endpoint],
            max_attempts=1,
        ).fetch()
        self.assertEqual(snapshot.health["status"], UNAVAILABLE)
        self.assertIn("product is offline", snapshot.error)

    def test_complete_coinbase_comparison_is_healthy_without_blending(self):
        now = int(time.time())
        rows = [
            [now - 60, 107, 109, 108, 108, 12],
            [now - 120, 106, 108, 107, 107, 11],
        ]
        responses = {
            "/products/BTC-CAD": {"id": "BTC-CAD", "status": "online"},
            "/products/BTC-CAD/ticker": {
                "price": "108", "bid": "107.9", "ask": "108.1", "volume": "24"
            },
            "/products/BTC-CAD/candles": rows,
        }
        secondary = CoinbasePublicMarketProvider(
            request_json=lambda endpoint, params=None: responses[endpoint],
            max_attempts=1,
        ).fetch()
        primary = KrakenPublicMarketProvider(
            loader_factory=FakeLoader, max_attempts=1
        ).fetch()
        comparison = compare_market_snapshots(primary, secondary)
        self.assertEqual(comparison.status, HEALTHY)
        self.assertTrue(comparison.agreement)
        self.assertIn("agree", comparison.message)

    def test_public_news_keeps_attribution_and_freshness(self):
        now = time.strftime("%a, %d %b %Y %H:%M:%S +0000", time.gmtime())
        xml = (
            "<rss><channel><item>"
            "<title>Bitcoin market update</title>"
            "<link>https://news.example/item</link>"
            f"<pubDate>{now}</pubDate>"
            "</item></channel></rss>"
        )
        snapshot = fetch_public_news_events(
            feed_url="https://news.example/rss",
            source="Example RSS",
            request_text=lambda _url: xml,
        )
        self.assertEqual(snapshot.status, HEALTHY)
        self.assertEqual(snapshot.items[0].source, "Example RSS")
        self.assertEqual(snapshot.items[0].url, "https://news.example/item")
        self.assertIsNotNone(snapshot.items[0].freshness_age_seconds)

    def test_public_news_rejects_stale_and_malformed_items(self):
        xml = (
            "<rss><channel>"
            "<item><title>Old</title><link>https://news.example/old</link>"
            "<pubDate>Tue, 01 Jan 2019 00:00:00 +0000</pubDate></item>"
            "<item><title>Missing date</title><link>https://news.example/missing</link></item>"
            "</channel></rss>"
        )
        snapshot = fetch_public_news_events(
            request_text=lambda _url: xml,
            max_age_seconds=60,
        )
        self.assertEqual(snapshot.status, UNAVAILABLE)
        self.assertEqual(snapshot.items, ())
        self.assertIn("no fresh", snapshot.error.lower())

    def test_public_news_failure_is_explicitly_unavailable(self):
        snapshot = fetch_public_news_events(
            request_text=lambda _url: (_ for _ in ()).throw(RuntimeError("offline"))
        )
        self.assertEqual(snapshot.status, UNAVAILABLE)
        self.assertEqual(snapshot.items, ())
        self.assertIn("unavailable", snapshot.error.lower())


if __name__ == "__main__":
    unittest.main()