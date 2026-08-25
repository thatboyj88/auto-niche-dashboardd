import unittest

from exit_economics_study import (
    ExitEconomicsStudy,
    _wilson_interval,
)


def candle(timestamp, close, high=None, low=None):
    return {
        "timestamp": timestamp,
        "open": close,
        "high": high if high is not None else close,
        "low": low if low is not None else close,
        "close": close,
        "volume": 100.0,
    }


def trade(exit_candle=1, reason="STOP LOSS"):
    return {
        "trade_number": 1,
        "entry_candle": 0,
        "exit_candle": exit_candle,
        "entry_timestamp": 1_700_000_000,
        "exit_timestamp": 1_700_086_400,
        "entry_price": 100.0,
        "exit_price": 98.0,
        "market_entry_price": 100.0,
        "position_size": 1.0,
        "net_profit_loss": -2.0,
        "gross_profit_loss_before_costs": -2.0,
        "reason": reason,
    }


class ExitEconomicsStudyTests(unittest.TestCase):
    def setUp(self):
        self.study = ExitEconomicsStudy()

    def test_hypothetical_exit_applies_same_fees_and_slippage(self):
        candles = [candle(index, 100.0) for index in range(5)]
        candles[3]["high"] = 102.1

        result = self.study.analyze_trade(trade(exit_candle=1), candles)
        hypothetical = result["hypothetical_exits"]["two_percent"]

        self.assertIsNotNone(hypothetical)
        self.assertEqual(hypothetical["candles_after_original_exit"], 2)
        self.assertGreater(hypothetical["fees"], 0.0)
        self.assertGreater(hypothetical["slippage"], 0.0)
        self.assertGreater(hypothetical["net_profit_loss"], -2.0)
        self.assertAlmostEqual(
            hypothetical["net_improvement_vs_original"],
            hypothetical["net_profit_loss"] + 2.0,
        )

    def test_non_post_exit_target_has_no_hypothetical_exit(self):
        candles = [candle(index, 100.0) for index in range(4)]
        candles[1]["high"] = 102.1

        result = self.study.analyze_trade(trade(exit_candle=1), candles)

        self.assertIsNone(result["hypothetical_exits"]["two_percent"])

    def test_summary_reports_costs_improvement_and_positive_interval(self):
        candles = [candle(index, 100.0) for index in range(5)]
        candles[3]["high"] = 102.1
        analyzed = self.study.analyze_trade(trade(), candles)

        summary = self.study._group_summary([analyzed])
        target = summary["targets"]["two_percent"]

        self.assertEqual(target["post_exit_opportunities"], 1)
        self.assertEqual(
            target["post_exit_opportunity_interval"]["confidence_level"],
            0.95,
        )
        self.assertGreater(target["hypothetical_fees"], 0.0)
        self.assertGreater(target["hypothetical_slippage"], 0.0)
        self.assertEqual(target["net_improvement"]["count"], 1)
        self.assertEqual(target["profitable_improvement_count"], 1)
        self.assertEqual(
            target["profitable_improvement_interval"]["confidence_level"],
            0.95,
        )

    def test_exit_reason_groups_are_kept_separate(self):
        candles = [candle(index, 100.0) for index in range(5)]
        candles[3]["high"] = 104.1
        stop = self.study.analyze_trade(
            trade(reason="STOP LOSS"),
            candles,
        )
        take_profit = self.study.analyze_trade(
            trade(reason="TAKE PROFIT"),
            candles,
        )

        summary = self.study._group_summary([stop, take_profit])

        self.assertEqual(
            summary["by_exit_reason"]["STOP LOSS"]["trades"],
            1,
        )
        self.assertEqual(
            summary["by_exit_reason"]["TAKE PROFIT"]["trades"],
            1,
        )

    def test_wilson_interval_handles_empty_and_full_samples(self):
        empty = _wilson_interval(0, 0)
        full = _wilson_interval(10, 10)

        self.assertIsNone(empty["lower_percent"])
        self.assertAlmostEqual(full["upper_percent"], 100.0)
        self.assertGreaterEqual(full["lower_percent"], 0.0)


if __name__ == "__main__":
    unittest.main()