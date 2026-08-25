import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import patch

from exit_capture_study import (
    BREAK_EVEN_MOVE_PERCENT,
    ExitCaptureStudy,
    _print_group,
    print_report,
    run_exit_capture_study,
    _wilson_interval,
)
from score_effectiveness_study import SCORE_STUDY_PERIODS


# Included by offline_study_report_check.py; network/live-data tests remain
# intentionally unmarked and outside that runner.
REPORT_REGRESSION_MODULE = True


def candle(timestamp, close, high=None, low=None):
    return {
        "timestamp": timestamp,
        "open": close,
        "high": high if high is not None else close,
        "low": low if low is not None else close,
        "close": close,
        "volume": 100.0,
    }


def trade(entry=0, exit_candle=3, reason="STOP LOSS"):
    return {
        "trade_number": 1,
        "entry_candle": entry,
        "exit_candle": exit_candle,
        "entry_timestamp": 1_700_000_000,
        "exit_timestamp": 1_700_259_200,
        "entry_price": 100.0,
        "exit_price": 98.0,
        "market_entry_price": 100.0,
        "net_profit_loss": -0.30,
        "gross_profit_loss_before_costs": -0.20,
        "reason": reason,
    }


class ExitCaptureStudyTests(unittest.TestCase):
    def setUp(self):
        self.study = ExitCaptureStudy()

    def test_targets_are_classified_before_at_and_after_exit(self):
        candles = [candle(index, 100.0) for index in range(8)]
        candles[1]["high"] = 101.1
        candles[3]["high"] = 102.1
        candles[5]["high"] = 104.1

        result = self.study.analyze_trade(trade(), candles)

        self.assertEqual(
            result["targets"]["break_even"]["phase"],
            "before_exit",
        )
        self.assertEqual(
            result["targets"]["two_percent"]["phase"],
            "at_exit",
        )
        self.assertEqual(
            result["targets"]["four_percent"]["phase"],
            "after_exit",
        )
        self.assertEqual(
            result["targets"]["four_percent"]["candles_after_exit"],
            2,
        )

    def test_unreached_target_is_never_and_post_exit_mfe_is_zero(self):
        candles = [candle(index, 100.0) for index in range(4)]

        result = self.study.analyze_trade(trade(exit_candle=3), candles)

        self.assertEqual(
            result["targets"]["break_even"]["phase"],
            "never",
        )
        self.assertEqual(result["mfe_after_exit_percent"], 0.0)
        self.assertFalse(result["next_candle"]["available"])

    def test_pre_and_post_exit_mfe_and_next_candle_are_separate(self):
        candles = [candle(index, 100.0) for index in range(6)]
        candles[1]["high"] = 101.5
        candles[4]["high"] = 103.0
        candles[4]["close"] = 102.0
        candles[4]["low"] = 99.5

        result = self.study.analyze_trade(trade(exit_candle=3), candles)

        self.assertAlmostEqual(result["mfe_before_exit_percent"], 1.5)
        self.assertAlmostEqual(result["mfe_after_exit_percent"], 3.0)
        self.assertAlmostEqual(result["next_candle"]["high_percent"], 3.0)
        self.assertAlmostEqual(result["next_candle"]["close_percent"], 2.0)

    def test_reason_summary_groups_stop_and_take_profit_paths(self):
        candles = [candle(index, 100.0) for index in range(6)]
        stop = self.study.analyze_trade(
            trade(reason="STOP LOSS"),
            candles,
        )
        target = self.study.analyze_trade(
            trade(reason="TAKE PROFIT"),
            candles,
        )

        summary = self.study.summarize_trades([stop, target])

        self.assertEqual(
            summary["by_exit_reason"]["STOP LOSS"]["trades"],
            1,
        )
        self.assertEqual(
            summary["by_exit_reason"]["TAKE PROFIT"]["trades"],
            1,
        )

    def test_cost_threshold_and_post_exit_reach_are_reported_separately(self):
        candles = [candle(index, 100.0) for index in range(8)]
        candles[5]["high"] = 102.0
        stop = self.study.analyze_trade(
            trade(reason="STOP LOSS"),
            candles,
        )
        target = self.study.analyze_trade(
            trade(reason="TAKE PROFIT"),
            candles,
        )

        summary = self.study.summarize_trades([stop, target])

        self.assertEqual(
            summary["cost_model"]["required_move_percent"],
            BREAK_EVEN_MOVE_PERCENT,
        )
        self.assertEqual(
            summary["target_capture"]["break_even"]["after_exit_reached"],
            2,
        )
        self.assertEqual(
            summary["by_exit_reason"]["STOP LOSS"]["targets_after_exit"][
                "break_even"
            ]["after_exit_reached"],
            1,
        )
        self.assertEqual(
            summary["by_exit_reason"]["TAKE PROFIT"]["targets_after_exit"][
                "break_even"
            ]["after_exit_reached"],
            1,
        )
        confidence_interval = summary["target_capture"]["break_even"][
            "after_exit_reached_confidence_interval_percent"
        ]
        self.assertEqual(confidence_interval["method"], "Wilson score interval")
        self.assertEqual(confidence_interval["confidence_level"], 0.95)
        self.assertAlmostEqual(confidence_interval["lower"], 34.24, places=2)
        self.assertAlmostEqual(confidence_interval["upper"], 100.0, places=2)
        self.assertEqual(
            summary["by_exit_reason"]["STOP LOSS"]["targets_after_exit"][
                "break_even"
            ]["after_exit_reached_confidence_interval_percent"],
            summary["by_exit_reason"]["TAKE PROFIT"]["targets_after_exit"][
                "break_even"
            ]["after_exit_reached_confidence_interval_percent"],
        )

    def test_group_keeps_period_trade_paths_independent(self):
        candles = [candle(index, 100.0) for index in range(6)]
        first = {
            "period": "Period A",
            "start_date": "2020-01-01",
            "end_date": "2020-01-06",
            "regime": "Bull",
            "trades_history": [trade()],
        }
        second_trade = trade(reason="TAKE PROFIT")
        second_trade["trade_number"] = 2
        second = {
            "period": "Period B",
            "start_date": "2020-01-07",
            "end_date": "2020-01-12",
            "regime": "Bear",
            "trades_history": [second_trade],
        }

        result = self.study.analyze_group(
            [first, second],
            [candles, candles],
        )

        self.assertEqual(result["period_count"], 2)
        self.assertEqual(result["trade_count"], 2)
        self.assertEqual(len(result["periods"][0]["trades"]), 1)
        self.assertEqual(len(result["periods"][1]["trades"]), 1)
        self.assertEqual(
            result["summary"]["by_exit_reason"]["STOP LOSS"]["trades"],
            1,
        )
        self.assertEqual(
            result["summary"]["by_exit_reason"]["TAKE PROFIT"]["trades"],
            1,
        )

    def test_period_summaries_include_intervals_and_exit_reasons(self):
        candles = [candle(index, 100.0) for index in range(6)]
        candles[4]["high"] = 102.0
        first = {
            "period": "Period A",
            "start_date": "2020-01-01",
            "end_date": "2020-01-06",
            "regime": "Bull",
            "trades_history": [trade(reason="STOP LOSS")],
        }
        second_trade = trade(reason="TAKE PROFIT")
        second_trade["trade_number"] = 2
        second = {
            "period": "Period B",
            "start_date": "2020-01-07",
            "end_date": "2020-01-12",
            "regime": "Bear",
            "trades_history": [second_trade],
        }

        result = self.study.analyze_group(
            [first, second],
            [candles, candles],
        )

        first_summary = result["periods"][0]["summary"]
        second_summary = result["periods"][1]["summary"]
        first_interval = first_summary["target_capture"]["break_even"][
            "after_exit_reached_confidence_interval_percent"
        ]
        second_interval = second_summary["target_capture"]["break_even"][
            "after_exit_reached_confidence_interval_percent"
        ]
        self.assertEqual(first_interval["method"], "Wilson score interval")
        self.assertEqual(first_interval["confidence_level"], 0.95)
        self.assertEqual(first_interval, second_interval)
        self.assertEqual(
            first_summary["by_exit_reason"]["STOP LOSS"]["trades"],
            1,
        )
        self.assertEqual(
            second_summary["by_exit_reason"]["TAKE PROFIT"]["trades"],
            1,
        )
        self.assertEqual(
            first_summary["by_exit_reason"]["TAKE PROFIT"]["trades"],
            0,
        )
        self.assertEqual(
            second_summary["by_exit_reason"]["STOP LOSS"]["trades"],
            0,
        )

    def test_empty_period_has_zero_interval_and_empty_exit_reasons(self):
        period = {
            "period": "Empty period",
            "start_date": "2020-01-01",
            "end_date": "2020-01-06",
            "regime": "Sideways",
            "trades_history": [],
        }

        result = self.study.analyze_group([period], [[]])
        summary = result["periods"][0]["summary"]
        interval = summary["target_capture"]["break_even"][
            "after_exit_reached_confidence_interval_percent"
        ]

        self.assertEqual(summary["target_capture"]["break_even"]["signals"], 0)
        self.assertEqual(interval["method"], "Wilson score interval")
        self.assertEqual(interval["confidence_level"], 0.95)
        self.assertEqual(interval["lower"], 0.0)
        self.assertEqual(interval["upper"], 0.0)
        self.assertEqual(
            summary["by_exit_reason"]["STOP LOSS"]["trades"],
            0,
        )
        self.assertEqual(
            summary["by_exit_reason"]["TAKE PROFIT"]["trades"],
            0,
        )

    def test_print_group_keeps_period_intervals_and_available_reasons_visible(self):
        candles = [candle(index, 100.0) for index in range(6)]
        first = {
            "period": "Period A",
            "start_date": "2020-01-01",
            "end_date": "2020-01-06",
            "regime": "Bull",
            "trades_history": [trade(reason="STOP LOSS")],
        }
        second_trade = trade(reason="TAKE PROFIT")
        second_trade["trade_number"] = 2
        second = {
            "period": "Period B",
            "start_date": "2020-01-07",
            "end_date": "2020-01-12",
            "regime": "Bear",
            "trades_history": [second_trade],
        }
        empty = {
            "period": "Empty period",
            "start_date": "2020-01-13",
            "end_date": "2020-01-18",
            "regime": "Sideways",
            "trades_history": [],
        }
        result = self.study.analyze_group(
            [first, second, empty],
            [candles, candles, []],
        )

        output = StringIO()
        with redirect_stdout(output):
            _print_group("Research", result)
        report = output.getvalue()

        self.assertIn("Period A (2020-01-01 to 2020-01-06, regime=Bull)", report)
        self.assertIn("Period B (2020-01-07 to 2020-01-12, regime=Bear)", report)
        self.assertIn(
            "Empty period (2020-01-13 to 2020-01-18, regime=Sideways)",
            report,
        )
        self.assertGreaterEqual(report.count("95% Wilson CI="), 9)

        period_a = report.split("  Period A ", 1)[1].split("  Period B ", 1)[0]
        period_b = report.split("  Period B ", 1)[1].split("  Empty period ", 1)[0]
        empty_period = report.split("  Empty period ", 1)[1]
        self.assertIn("STOP LOSS: trades=1", period_a)
        self.assertNotIn("TAKE PROFIT: trades=", period_a)
        self.assertIn("TAKE PROFIT: trades=1", period_b)
        self.assertNotIn("STOP LOSS: trades=", period_b)
        self.assertNotIn("STOP LOSS:", empty_period)
        self.assertNotIn("TAKE PROFIT:", empty_period)

    def test_print_report_keeps_research_validation_and_interpretation_sections(self):
        candles = [candle(index, 100.0) for index in range(6)]
        research_period = {
            "period": "Research period",
            "start_date": "2020-01-01",
            "end_date": "2020-01-06",
            "regime": "Bull",
            "trades_history": [trade(reason="STOP LOSS")],
        }
        validation_period = {
            "period": "Validation period",
            "start_date": "2020-01-07",
            "end_date": "2020-01-12",
            "regime": "Bear",
            "trades_history": [trade(reason="TAKE PROFIT")],
        }
        results = {
            "split": {
                "research_start": "2020-01-01",
                "research_end": "2020-01-06",
                "research_periods": 1,
                "research_candles": 6,
                "validation_start": "2020-01-07",
                "validation_end": "2020-01-12",
                "validation_periods": 1,
                "validation_candles": 6,
            },
            "research": self.study.analyze_group(
                [research_period],
                [candles],
            ),
            "validation": self.study.analyze_group(
                [validation_period],
                [candles],
            ),
        }

        output = StringIO()
        with redirect_stdout(output):
            print_report(results)
        report = output.getvalue()

        self.assertIn("=== Research ===", report)
        self.assertIn("=== Untouched validation ===", report)
        self.assertIn("=== Interpretation boundary ===", report)

    def test_complete_report_path_smoke_checks_real_shaped_results(self):
        candles = [candle(index, 100.0) for index in range(8)]
        candles[1]["high"] = 101.5
        candles[3]["high"] = 102.5
        candles[5]["high"] = 104.5

        selected = [
            {**period, "candles": candles}
            for period in SCORE_STUDY_PERIODS
        ]

        def fake_period_group(periods, notifier):
            del notifier
            return [
                {
                    **period,
                    "trades_history": [
                        trade(
                            exit_candle=3,
                            reason=(
                                "STOP LOSS"
                                if index % 2 == 0
                                else "TAKE PROFIT"
                            ),
                        )
                    ],
                }
                for index, period in enumerate(periods)
            ]

        class FakeMarketData:
            last_error = None

            def __init__(self, *_args, **_kwargs):
                pass

            def load(self):
                return candles

        output = StringIO()
        with (
            patch("exit_capture_study.YahooBTCADMarketData", FakeMarketData),
            patch("exit_capture_study.select_score_study_periods",
                  return_value=selected),
            patch("exit_capture_study._run_period_group",
                  side_effect=fake_period_group),
            redirect_stdout(output),
        ):
            results = run_exit_capture_study(notifier=lambda *_args: None)
            print_report(results)

        report = output.getvalue()
        self.assertEqual(results["research"]["period_count"], 8)
        self.assertEqual(results["validation"]["period_count"], 2)
        self.assertIn("BTC/CAD EXIT-CAPTURE STUDY — ANALYSIS ONLY", report)
        self.assertIn("=== Research ===", report)
        self.assertIn("=== Untouched validation ===", report)
        self.assertIn(
            f"break_even ({BREAK_EVEN_MOVE_PERCENT:.3f}%",
            report,
        )
        self.assertIn("STOP LOSS: trades=", report)
        self.assertIn("TAKE PROFIT: trades=", report)
        self.assertIn("=== Interpretation boundary ===", report)
        self.assertLess(
            report.index("=== Untouched validation ==="),
            report.index("=== Interpretation boundary ==="),
        )

    def test_break_even_threshold_uses_existing_cost_formula(self):
        self.assertGreater(BREAK_EVEN_MOVE_PERCENT, 1.0)
        self.assertLess(BREAK_EVEN_MOVE_PERCENT, 1.1)

    def test_wilson_interval_handles_empty_and_rejects_invalid_counts(self):
        self.assertEqual(_wilson_interval(0, 0), {"lower": 0.0, "upper": 0.0})
        with self.assertRaises(ValueError):
            _wilson_interval(2, 1)


if __name__ == "__main__":
    unittest.main()