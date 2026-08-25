import unittest

from market_data_health import (
    DEGRADED,
    HEALTHY,
    UNAVAILABLE,
    inspect_candles,
)


def candle(timestamp, price=100.0):
    return {
        "timestamp": timestamp,
        "open": price,
        "high": price + 1,
        "low": price - 1,
        "close": price + 0.5,
        "volume": 10,
    }


class MarketDataHealthTests(unittest.TestCase):
    def test_healthy_candles_report_fresh_data(self):
        result = inspect_candles(
            [candle(1000), candle(4600)],
            interval_minutes=60,
            now_timestamp=4700,
        )

        self.assertEqual(result["status"], HEALTHY)
        self.assertEqual(result["issues"], [])
        self.assertEqual(result["last_known_good_timestamp"], 4600)
        self.assertEqual(result["data_age_seconds"], 100.0)

    def test_missing_candles_and_stale_data_are_degraded(self):
        result = inspect_candles(
            [candle(1000), candle(8200)],
            interval_minutes=60,
            now_timestamp=16000,
        )

        self.assertEqual(result["status"], DEGRADED)
        self.assertIn("excessive timestamp gap", result["issues"])
        self.assertIn("stale market data", result["issues"])
        self.assertIsNone(result["last_known_good_timestamp"])

    def test_malformed_ohlc_and_duplicate_timestamps_are_degraded(self):
        invalid = candle(4600)
        invalid["high"] = invalid["low"] - 1

        result = inspect_candles(
            [candle(1000), invalid, candle(4600)],
            interval_minutes=60,
            now_timestamp=4700,
        )

        self.assertEqual(result["status"], DEGRADED)
        self.assertIn("duplicate timestamps", result["issues"])
        self.assertTrue(
            any("has an invalid high" in issue for issue in result["issues"])
        )
        self.assertTrue(
            any("has high below low" in issue for issue in result["issues"])
        )

    def test_unavailable_provider_is_explicit(self):
        result = inspect_candles(
            [],
            interval_minutes=60,
            provider_available=False,
            provider_error="timeout",
        )

        self.assertEqual(result["status"], UNAVAILABLE)
        self.assertEqual(result["provider_error"], "timeout")
        self.assertIn("provider unavailable", result["issues"])