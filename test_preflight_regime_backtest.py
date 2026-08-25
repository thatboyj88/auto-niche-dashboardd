import unittest
from datetime import datetime, timezone
from unittest.mock import Mock, patch

from btc_cad_preflight import BTCADPreflightError
from multi_period_backtest import MultiPeriodBacktester, PERIOD_CANDLES
from strategy_backtest import StrategyBacktester


def make_candle(timestamp, close):
    return {
        "timestamp": timestamp,
        "open": close,
        "high": close * 1.01,
        "low": close * 0.99,
        "close": close,
        "volume": 1000.0,
    }


def make_period(starting_price=100.0, ending_price=120.0):
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
                (index / (PERIOD_CANDLES - 1))
            ),
        )
        for index in range(PERIOD_CANDLES)
    ]


class PreflightGatedRegimeBacktestTests(unittest.TestCase):
    def test_valid_period_notifies_then_preserves_strategy_results(self):
        candles = make_period()
        notifier = Mock()

        expected_backtester = StrategyBacktester(starting_capital=25.00)
        expected_backtester.run(candles)
        expected = expected_backtester.results()

        result = MultiPeriodBacktester().run(
            candles,
            notifier=notifier,
        )
        period = result["periods"][0]

        notifier.assert_called_once()
        self.assertTrue(notifier.call_args.args[0]["ok"])
        self.assertEqual(period["regime"], "Bull")
        self.assertAlmostEqual(period["market_return"], 20.0)
        self.assertEqual(period["preflight"]["candle_count"], 365)
        self.assertEqual(period["ending_capital"], expected["ending_capital"])
        self.assertEqual(period["profit"], expected["profit"])
        self.assertEqual(period["trades"], expected["trades"])
        self.assertEqual(period["evaluations"], expected["evaluations"])
        self.assertEqual(
            period["evaluation_history"],
            expected["evaluation_history"],
        )
        self.assertEqual(
            period["condition_counts"],
            expected["condition_counts"],
        )

    def test_invalid_period_notifies_and_never_constructs_backtester(self):
        candles = make_period()
        candles[50]["volume"] = 0
        notifier = Mock()

        with patch(
            "multi_period_backtest.StrategyBacktester",
        ) as strategy_backtester:
            with self.assertRaisesRegex(
                BTCADPreflightError,
                "preflight failed",
            ):
                MultiPeriodBacktester().run(
                    candles,
                    notifier=notifier,
                )

        notifier.assert_called_once()
        self.assertFalse(notifier.call_args.args[0]["ok"])
        strategy_backtester.assert_not_called()

    def test_invalid_first_timestamp_notifies_before_backtesting(self):
        candles = make_period()
        candles[0]["timestamp"] = "not-a-timestamp"
        notifier = Mock()

        with patch(
            "multi_period_backtest.StrategyBacktester",
        ) as strategy_backtester:
            with self.assertRaisesRegex(
                BTCADPreflightError,
                "preflight failed",
            ):
                MultiPeriodBacktester().run(
                    candles,
                    notifier=notifier,
                )

        notifier.assert_called_once()
        self.assertFalse(notifier.call_args.args[0]["ok"])
        strategy_backtester.assert_not_called()

    def test_missing_first_timestamp_notifies_for_source_runner(self):
        candles = make_period()
        del candles[0]["timestamp"]
        notifier = Mock()

        with patch(
            "multi_period_backtest.StrategyBacktester",
        ) as strategy_backtester:
            with self.assertRaisesRegex(
                BTCADPreflightError,
                "preflight failed",
            ):
                MultiPeriodBacktester().run_sources(
                    [{
                        "candles": candles,
                        "label": "Yahoo Finance public BTC/CAD",
                        "kind": "rolling",
                    }],
                    notifier=notifier,
                )

        notifier.assert_called_once()
        self.assertFalse(notifier.call_args.args[0]["ok"])
        strategy_backtester.assert_not_called()

    def test_sources_gate_each_independent_period_with_mocked_notifications(self):
        first_period = make_period(100.0, 120.0)
        second_period = [
            {
                **candle,
                "timestamp": candle["timestamp"] + (PERIOD_CANDLES * 86400),
            }
            for candle in make_period(120.0, 90.0)
        ]
        notifier = Mock()

        results = MultiPeriodBacktester().run_sources(
            [{
                "candles": first_period + second_period,
                "label": "Yahoo Finance public BTC/CAD",
                "kind": "rolling",
            }],
            notifier=notifier,
        )

        self.assertEqual(len(results["periods"]), 2)
        self.assertEqual(notifier.call_count, 2)
        self.assertTrue(all(
            call.args[0]["ok"]
            for call in notifier.call_args_list
        ))


if __name__ == "__main__":
    unittest.main()