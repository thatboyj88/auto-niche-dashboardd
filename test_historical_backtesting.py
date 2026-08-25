import json
import unittest
from datetime import datetime, timezone
from unittest.mock import patch

from check_yahoo_btc_cad_data import (
    _notification_message,
    main,
    send_maintainer_notification,
    validate_yahoo_btc_cad_sources,
)
from dashboard import load_historical_btc_cad_data
from generate_test_data import generate_candles
from multi_period_backtest import (
    BEAR_RETURN_PERCENT,
    BULL_RETURN_PERCENT,
    PERIOD_CANDLES,
    MultiPeriodBacktester,
)
from strategy_backtest import StrategyBacktester
from yahoo_btc_cad_data import YahooBTCADMarketData


def make_candle(timestamp, close, volume=1000.0):
    return {
        "timestamp": timestamp,
        "open": close,
        "high": close * 1.01,
        "low": close * 0.99,
        "close": close,
        "volume": volume,
    }


def make_period(start_timestamp, opening_price, closing_price):
    candles = []
    for index in range(PERIOD_CANDLES):
        progress = index / (PERIOD_CANDLES - 1)
        close = (
            opening_price +
            ((closing_price - opening_price) * progress)
        )
        candles.append(
            make_candle(
                start_timestamp + (index * 86400),
                close,
            )
        )
    return candles


class YahooBTCADMarketDataTests(unittest.TestCase):
    def test_extended_history_is_the_default(self):
        loader = YahooBTCADMarketData()

        self.assertEqual(loader.data_range, "10y")

    def test_parser_normalizes_sorts_and_excludes_incomplete_candles(self):
        first = 1704067200
        second = first + 86400
        current = 1735689600
        payload = {
            "chart": {
                "result": [
                    {
                        "meta": {
                            "symbol": "BTC-CAD",
                            "currency": "CAD",
                        },
                        "timestamp": [second, first, current],
                        "indicators": {
                            "quote": [
                                {
                                    "open": [42000, 41000, 43000],
                                    "high": [42500, 41500, 43500],
                                    "low": [41500, 40500, 42500],
                                    "close": [42100, 41100, None],
                                    "volume": [200, 100, 300],
                                }
                            ]
                        },
                    }
                ],
                "error": None,
            }
        }

        candles = YahooBTCADMarketData._parse_payload(
            payload,
            now=datetime(2025, 1, 1, tzinfo=timezone.utc),
        )

        self.assertEqual([candle["timestamp"] for candle in candles], [
            first,
            second,
        ])
        self.assertEqual(candles[0]["close"], 41100.0)
        self.assertEqual(candles[1]["volume"], 200.0)

    def test_parser_rejects_non_btc_cad_series(self):
        payload = {
            "chart": {
                "result": [
                    {
                        "meta": {
                            "symbol": "BTC-USD",
                            "currency": "USD",
                        }
                    }
                ],
                "error": None,
            }
        }

        with self.assertRaisesRegex(
            RuntimeError,
            "BTC-CAD CAD series",
        ):
            YahooBTCADMarketData._parse_payload(payload)

    def test_anchored_sample_uses_a_fixed_completed_window(self):
        loader = YahooBTCADMarketData()
        captured = {}
        start_timestamp = int(datetime(
            2019,
            8,
            20,
            tzinfo=timezone.utc,
        ).timestamp())
        end_timestamp = int(datetime(
            2020,
            8,
            19,
            tzinfo=timezone.utc,
        ).timestamp())

        def request_payload(query):
            captured.update(query)
            return {
                "chart": {
                    "result": [{
                        "meta": {
                            "symbol": "BTC-CAD",
                            "currency": "CAD",
                        },
                        "timestamp": [
                            start_timestamp + (index * 86400)
                            for index in range(365)
                        ],
                        "indicators": {
                            "quote": [{
                                "open": [100] * 365,
                                "high": [101] * 365,
                                "low": [99] * 365,
                                "close": [100] * 365,
                                "volume": [1000] * 365,
                            }]
                        },
                    }],
                    "error": None,
                }
            }

        loader._request_payload = request_payload

        candles = loader.load_anchored_sample()

        self.assertEqual(len(candles), 365)
        self.assertEqual(captured, {
            "period1": start_timestamp,
            "period2": end_timestamp,
            "interval": "1d",
        })
        self.assertEqual(candles[0]["timestamp"], start_timestamp)
        self.assertEqual(candles[-1]["timestamp"], end_timestamp - 86400)
        self.assertEqual(loader.get_anchored_candles(), candles)

    def test_anchored_sample_fails_explicitly_when_incomplete(self):
        loader = YahooBTCADMarketData()
        loader._request_anchored_json = lambda: {
            "chart": {
                "result": [{
                    "meta": {
                        "symbol": "BTC-CAD",
                        "currency": "CAD",
                    },
                    "timestamp": [1704067200],
                    "indicators": {
                        "quote": [{
                            "open": [100],
                            "high": [101],
                            "low": [99],
                            "close": [100],
                            "volume": [1000],
                        }]
                    },
                }],
                "error": None,
            }
        }

        self.assertEqual(loader.load_anchored_sample(), [])
        self.assertIn("fewer than 365", loader.last_anchored_error)

    def test_out_of_range_timestamp_becomes_a_load_error(self):
        payload = {
            "chart": {
                "result": [
                    {
                        "meta": {
                            "symbol": "BTC-CAD",
                            "currency": "CAD",
                        },
                        "timestamp": [10 ** 100],
                        "indicators": {
                            "quote": [
                                {
                                    "open": [42000],
                                    "high": [42500],
                                    "low": [41500],
                                    "close": [42100],
                                    "volume": [200],
                                }
                            ]
                        },
                    }
                ],
                "error": None,
            }
        }
        loader = YahooBTCADMarketData()
        loader._request_json = lambda: payload

        self.assertEqual(loader.load(), [])
        self.assertIn(
            "no completed BTC/CAD candles",
            loader.last_error,
        )


class YahooBTCADDataPreflightTests(unittest.TestCase):
    def test_preflight_confirms_both_normalized_sources(self):
        rolling = [
            make_candle(1704067200 + (index * 86400), 100)
            for index in range(365)
        ]
        anchored_start = int(datetime(
            2019,
            8,
            20,
            tzinfo=timezone.utc,
        ).timestamp())
        anchored = [
            make_candle(anchored_start + (index * 86400), 100)
            for index in range(365)
        ]

        class FakeMarketData:
            last_error = None
            last_anchored_error = None

            def load(self):
                return rolling

            def load_anchored_sample(self):
                return anchored

        result = validate_yahoo_btc_cad_sources(FakeMarketData())

        self.assertTrue(result["ok"])
        self.assertEqual(result["rolling_candle_count"], 365)
        self.assertEqual(result["anchored_candle_count"], 365)
        self.assertEqual(result["failures"], [])

    def test_preflight_reports_missing_anchored_evidence(self):
        rolling = [
            make_candle(1704067200 + (index * 86400), 100)
            for index in range(365)
        ]

        class FakeMarketData:
            last_error = None
            last_anchored_error = "Yahoo Finance HTTP error 404"

            def load(self):
                return rolling

            def load_anchored_sample(self):
                return []

        result = validate_yahoo_btc_cad_sources(FakeMarketData())

        self.assertFalse(result["ok"])
        self.assertEqual(result["rolling_candle_count"], 365)
        self.assertEqual(result["anchored_candle_count"], 0)
        self.assertIn("Anchored Yahoo Finance BTC/CAD sample", result["failures"][0])
        self.assertIn("HTTP error 404", result["failures"][0])

    def test_preflight_rejects_a_shifted_anchored_date_range(self):
        rolling = [
            make_candle(1704067200 + (index * 86400), 100)
            for index in range(365)
        ]
        shifted_anchored_start = int(datetime(
            2019,
            8,
            21,
            tzinfo=timezone.utc,
        ).timestamp())
        shifted_anchored = [
            make_candle(shifted_anchored_start + (index * 86400), 100)
            for index in range(365)
        ]

        class FakeMarketData:
            last_error = None
            last_anchored_error = None

            def load(self):
                return rolling

            def load_anchored_sample(self):
                return shifted_anchored

        result = validate_yahoo_btc_cad_sources(FakeMarketData())

        self.assertFalse(result["ok"])
        self.assertIn(
            "does not cover the expected daily range",
            result["failures"][0],
        )

    def test_notification_names_each_failed_source(self):
        result = {
            "rolling_failure": "rolling unavailable",
            "anchored_failure": "anchored unavailable",
            "failures": [
                "Rolling Yahoo Finance BTC/CAD source unavailable",
                "Anchored Yahoo Finance BTC/CAD sample unavailable",
            ],
        }

        message = _notification_message(result)

        self.assertIn("rolling source", message)
        self.assertIn("anchored 2019-08-20 through 2020-08-18 sample", message)

    def test_notification_posts_only_to_slack_webhook(self):
        result = {
            "rolling_failure": "rolling unavailable",
            "anchored_failure": None,
            "failures": ["Rolling Yahoo Finance BTC/CAD source unavailable"],
        }

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

        with patch("check_yahoo_btc_cad_data.urlopen", fake_urlopen):
            self.assertTrue(
                send_maintainer_notification(
                    result,
                    "https://hooks.slack.com/services/test/webhook",
                )
            )

        self.assertEqual(captured["request"].method, "POST")
        self.assertEqual(captured["timeout"], 10)
        self.assertIn(
            "rolling source",
            json.loads(captured["request"].data)["text"],
        )

    def test_notification_rejects_non_slack_endpoint(self):
        result = {"failures": ["Rolling source unavailable"]}

        with self.assertRaises(ValueError):
            send_maintainer_notification(
                result,
                "https://query1.finance.yahoo.com/trading",
            )

    def test_notification_delivery_test_is_clearly_labelled(self):
        message = _notification_message({"test_notification": True})

        self.assertTrue(message.startswith("TEST:"))
        self.assertIn("no evidence source was checked", message)

    def test_failed_preflight_invokes_maintainer_notification(self):
        result = {
            "ok": False,
            "rolling_candle_count": 0,
            "anchored_candle_count": 365,
            "rolling_failure": "rolling unavailable",
            "anchored_failure": None,
            "failures": ["Rolling Yahoo Finance BTC/CAD source unavailable"],
        }

        with (
            patch(
                "check_yahoo_btc_cad_data.validate_yahoo_btc_cad_sources",
                return_value=result,
            ),
            patch(
                "check_yahoo_btc_cad_data.send_maintainer_notification",
                return_value=True,
            ) as notify,
        ):
            self.assertEqual(main([]), 1)

        notify.assert_called_once_with(result)

    def test_unexpected_preflight_error_still_invokes_notification(self):
        with (
            patch(
                "check_yahoo_btc_cad_data.validate_yahoo_btc_cad_sources",
                side_effect=RuntimeError("unexpected response shape"),
            ),
            patch(
                "check_yahoo_btc_cad_data.send_maintainer_notification",
                return_value=True,
            ) as notify,
        ):
            self.assertEqual(main([]), 1)

        self.assertIn(
            "unexpected response shape",
            notify.call_args.args[0]["failures"][0],
        )
        self.assertTrue(notify.call_args.args[0]["rolling_failure"])
        self.assertTrue(notify.call_args.args[0]["anchored_failure"])

    def test_cli_notification_test_uses_the_configured_notifier(self):
        with patch(
            "check_yahoo_btc_cad_data.send_maintainer_notification",
            return_value=True,
        ) as notify:
            self.assertEqual(main(["--test-notification"]), 0)

        notify.assert_called_once_with({"test_notification": True})


class MultiPeriodBacktesterTests(unittest.TestCase):
    def test_regime_boundaries_are_predeclared(self):
        starting_timestamp = 1704067200
        bull_period = make_period(
            starting_timestamp,
            100,
            100 * (1 + (BULL_RETURN_PERCENT / 100)),
        )
        bear_period = make_period(
            starting_timestamp,
            100,
            100 * (1 + (BEAR_RETURN_PERCENT / 100)),
        )
        sideways_period = make_period(
            starting_timestamp,
            100,
            108,
        )

        self.assertEqual(
            MultiPeriodBacktester.classify_regime(bull_period)[0],
            "Bull",
        )
        self.assertEqual(
            MultiPeriodBacktester.classify_regime(bear_period)[0],
            "Bear",
        )
        self.assertEqual(
            MultiPeriodBacktester.classify_regime(sideways_period)[0],
            "Sideways",
        )

    def test_periods_are_isolated_and_aggregate_their_results(self):
        starting_timestamp = 1704067200
        candles = (
            make_period(starting_timestamp, 100, 130) +
            make_period(
                starting_timestamp + (PERIOD_CANDLES * 86400),
                130,
                100,
            ) +
            make_period(
                starting_timestamp + (2 * PERIOD_CANDLES * 86400),
                100,
                105,
            ) +
            [make_candle(
                starting_timestamp + (3 * PERIOD_CANDLES * 86400),
                105,
            )]
        )

        results = MultiPeriodBacktester().run(
            candles,
            notifier=lambda _result: None,
        )
        periods = results["periods"]
        aggregate = results["aggregate"]

        self.assertEqual(len(periods), 3)
        self.assertEqual(results["unused_candles"], 1)
        self.assertEqual(
            [period["regime"] for period in periods],
            ["Bull", "Bear", "Sideways"],
        )
        self.assertEqual(
            [
                period["period"]
                for period in results["regime_summary"]["Sideways"]
            ],
            ["Period C"],
        )
        self.assertTrue(all(
            period["starting_capital"] == 25.00
            for period in periods
        ))
        self.assertTrue(all(
            period["candle_count"] == PERIOD_CANDLES
            for period in periods
        ))
        self.assertTrue(all(
            set(period["condition_counts"]) == {
                "long_term_trend",
                "short_term_momentum",
                "rsi",
                "volume",
                "price_above_ema21",
            }
            for period in periods
        ))
        self.assertEqual(
            aggregate["total_return"],
            sum(period["return_percent"] for period in periods),
        )
        self.assertEqual(
            aggregate["best_period"]["period"],
            max(
                periods,
                key=lambda period: period["return_percent"],
            )["period"],
        )
        self.assertEqual(
            aggregate["worst_period"]["period"],
            min(
                periods,
                key=lambda period: period["return_percent"],
            )["period"],
        )

    def test_sources_remain_separate_independent_periods(self):
        starting_timestamp = 1704067200
        rolling = make_period(starting_timestamp, 100, 130)
        anchored = make_period(
            starting_timestamp + (PERIOD_CANDLES * 86400),
            100,
            105,
        )

        results = MultiPeriodBacktester().run_sources(
            [
                {
                    "candles": rolling,
                    "label": "Rolling Yahoo Finance 10-year window",
                    "kind": "rolling",
                },
                {
                    "candles": anchored,
                    "label": "Anchored Yahoo Finance BTC/CAD sample",
                    "kind": "anchored",
                },
            ],
            notifier=lambda _result: None,
        )

        self.assertEqual(len(results["periods"]), 2)
        self.assertEqual(
            [period["source_kind"] for period in results["periods"]],
            ["rolling", "anchored"],
        )
        self.assertTrue(all(
            period["starting_capital"] == 25.00
            for period in results["periods"]
        ))
        self.assertEqual(
            [source["period_count"] for source in results["sources"]],
            [1, 1],
        )


class HistoricalSourceSelectionTests(unittest.TestCase):
    def test_dashboard_historical_render_does_not_send_slack_notifications(self):
        rolling = make_period(1704067200, 100, 105)

        with (
            patch("dashboard.YahooBTCADMarketData") as market_data_class,
            patch(
                "btc_cad_preflight.send_slack_notification",
            ) as notify,
        ):
            market_data = market_data_class.return_value
            market_data.load.return_value = rolling
            market_data.data_range = "10y"
            market_data.ROLLING_SOURCE_LABEL = (
                "Rolling Yahoo Finance 10-year window"
            )

            load_historical_btc_cad_data()

        notify.assert_not_called()

    def test_anchored_sample_is_loaded_when_rolling_data_lacks_sideways(self):
        rolling = make_period(1704067200, 100, 130)
        anchored = make_period(
            1608508800,
            100,
            105,
        )

        with patch("dashboard.YahooBTCADMarketData") as market_data_class:
            market_data = market_data_class.return_value
            market_data.load.return_value = rolling
            market_data.load_anchored_sample.return_value = anchored
            market_data.data_range = "10y"
            market_data.ROLLING_SOURCE_LABEL = (
                "Rolling Yahoo Finance 10-year window"
            )
            market_data.ANCHORED_SOURCE_LABEL = (
                "Anchored Yahoo Finance BTC/CAD sample"
            )

            _, sources = load_historical_btc_cad_data()

        market_data.load_anchored_sample.assert_called_once_with()
        self.assertEqual(
            [source["kind"] for source in sources],
            ["rolling", "anchored"],
        )

    def test_anchored_sample_is_not_loaded_when_rolling_data_has_sideways(self):
        rolling = make_period(1704067200, 100, 105)

        with patch("dashboard.YahooBTCADMarketData") as market_data_class:
            market_data = market_data_class.return_value
            market_data.load.return_value = rolling
            market_data.data_range = "10y"
            market_data.ROLLING_SOURCE_LABEL = (
                "Rolling Yahoo Finance 10-year window"
            )

            _, sources = load_historical_btc_cad_data()

        market_data.load_anchored_sample.assert_not_called()
        self.assertEqual(
            [source["kind"] for source in sources],
            ["rolling"],
        )


class SyntheticBaselineTests(unittest.TestCase):
    def test_synthetic_strategy_baseline_uses_calendar_day_risk_buckets(self):
        backtester = StrategyBacktester(25.00)
        backtester.run(generate_candles(1000))
        results = backtester.results()

        self.assertAlmostEqual(results["ending_capital"], 25.3263, places=4)
        self.assertAlmostEqual(results["profit"], 0.3263, places=4)
        self.assertEqual(results["trades"], 3)
        self.assertAlmostEqual(results["win_rate"], 66.67, places=2)
        self.assertAlmostEqual(results["max_drawdown"], 1.17, places=2)
        self.assertTrue(results["trades_history"])
        self.assertIn(
            "gross_profit_loss_before_costs",
            results["trades_history"][0],
        )
        self.assertIn(
            "market_entry_price",
            results["trades_history"][0],
        )
        self.assertIn(
            "market_exit_price",
            results["trades_history"][0],
        )


if __name__ == "__main__":
    unittest.main()
