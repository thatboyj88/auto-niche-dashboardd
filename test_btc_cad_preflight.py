import json
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from btc_cad_preflight import (
    REQUESTED_CANDLE_COUNT,
    _notification_text,
    main,
    run_preflight,
    send_slack_notification,
)


def make_candle(timestamp, close):
    return {
        "timestamp": timestamp,
        "open": close,
        "high": close * 1.01,
        "low": close * 0.99,
        "close": close,
        "volume": 1000.0,
    }


def make_requested_period(starting_price=100.0, ending_price=120.0):
    start_timestamp = int(datetime(
        2019,
        8,
        20,
        tzinfo=timezone.utc,
    ).timestamp())
    return [
        make_candle(
            start_timestamp + (index * 86400),
            starting_price + (
                (ending_price - starting_price) *
                (index / (REQUESTED_CANDLE_COUNT - 1))
            ),
        )
        for index in range(REQUESTED_CANDLE_COUNT)
    ]


class FakeYahooMarketData:
    last_anchored_error = None

    def __init__(self, candles):
        self.candles = candles

    def load_anchored_sample(self):
        return self.candles


class BTCADPreflightTests(unittest.TestCase):
    def test_successful_validation_calculates_return_and_regime(self):
        result = run_preflight(
            FakeYahooMarketData(make_requested_period(100.0, 120.0))
        )

        self.assertTrue(result["ok"])
        self.assertEqual(result["candle_count"], 365)
        self.assertAlmostEqual(result["market_return"], 20.0)
        self.assertEqual(result["regime"], "Bull")

    def test_failure_handling_rejects_non_positive_ohlcv_without_notifying(self):
        candles = make_requested_period()
        candles[100]["volume"] = 0

        result = run_preflight(FakeYahooMarketData(candles))

        self.assertFalse(result["ok"])
        self.assertIn("non-positive", result["failure"])

    def test_failure_handling_rejects_duplicate_timestamp(self):
        candles = make_requested_period()
        candles[40]["timestamp"] = candles[39]["timestamp"]

        result = run_preflight(FakeYahooMarketData(candles))

        self.assertFalse(result["ok"])
        self.assertIn("chronological and unique", result["failure"])

    def test_cli_entry_point_fails_for_malformed_ohlc(self):
        candles = make_requested_period()
        candles[80]["volume"] = 0
        notify = unittest.mock.Mock()

        exit_code = main(
            [],
            market_data=FakeYahooMarketData(candles),
            notify=notify,
        )

        self.assertEqual(exit_code, 1)
        self.assertFalse(notify.call_args.args[0]["ok"])

    def test_validation_workflow_runs_the_strict_preflight(self):
        replit_config = Path(".replit").read_text(encoding="utf-8")

        self.assertIn(
            'args = "uv run python btc_cad_preflight.py"',
            replit_config,
        )

    def test_main_notifies_for_success_without_real_slack_delivery(self):
        notify = unittest.mock.Mock()

        exit_code = main(
            [],
            market_data=FakeYahooMarketData(make_requested_period()),
            notify=notify,
        )

        self.assertEqual(exit_code, 0)
        self.assertTrue(notify.call_args.args[0]["ok"])

    def test_main_returns_failure_and_notifies_without_real_slack_delivery(self):
        candles = make_requested_period()
        candles[-1]["close"] = 0
        notify = unittest.mock.Mock()

        exit_code = main(
            [],
            market_data=FakeYahooMarketData(candles),
            notify=notify,
        )

        self.assertEqual(exit_code, 1)
        self.assertFalse(notify.call_args.args[0]["ok"])

    def test_slack_notification_uses_mocked_webhook(self):
        result = run_preflight(
            FakeYahooMarketData(make_requested_period(100.0, 80.0))
        )

        class FakeResponse:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def getcode(self):
                return self.status

        captured = {}

        def fake_urlopen(request, timeout):
            captured["request"] = request
            captured["timeout"] = timeout
            return FakeResponse()

        with patch("btc_cad_preflight.urlopen", fake_urlopen):
            send_slack_notification(
                result,
                "https://hooks.slack.com/services/test/webhook",
            )

        self.assertEqual(captured["request"].method, "POST")
        self.assertEqual(captured["timeout"], 10)
        text = json.loads(captured["request"].data)["text"]
        self.assertIn("PASS:", text)
        self.assertIn("Bear", text)

    def test_failure_notification_contains_safety_boundary(self):
        message = _notification_text({
            "ok": False,
            "failure": "Yahoo Finance returned no candles",
            "candle_count": 0,
        })

        self.assertIn("FAIL:", message)
        self.assertIn("No trades", message)
        self.assertIn("Kraken private APIs", message)


if __name__ == "__main__":
    unittest.main()