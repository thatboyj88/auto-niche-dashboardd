import unittest

from pre_stop_market_state_study import (
    CONTINUED_LOSS,
    PRE_STOP_OFFSETS,
    RECOVERED_ENTRY,
    PreStopMarketStateStudy,
)


def candle(timestamp, close, high=None, low=None, volume=100.0):
    return {
        "timestamp": timestamp,
        "open": close,
        "high": high if high is not None else close,
        "low": low if low is not None else close,
        "close": close,
        "volume": volume,
    }


def evaluation(index, score=85, rsi=55.0):
    return {
        "candle": index,
        "strategy_score": score,
        "rsi": rsi,
        "ema21": 99.0,
        "ema50": 98.0,
        "ema200": 95.0,
        "short_term_momentum": True,
        "long_term_trend": True,
    }


def trade(exit_candle=5):
    return {
        "trade_number": 1,
        "entry_candle": 1,
        "exit_candle": exit_candle,
        "entry_timestamp": 1_700_000_000,
        "exit_timestamp": 1_700_345_600,
        "market_entry_price": 100.0,
        "market_exit_price": 98.0,
        "reason": "STOP LOSS",
    }


class PreStopMarketStateStudyTests(unittest.TestCase):
    def setUp(self):
        self.study = PreStopMarketStateStudy()
        self.candles = [candle(index, 100.0) for index in range(12)]
        self.evaluations = {
            index: evaluation(index, rsi=50 + index)
            for index in range(12)
        }

    def test_captures_all_pre_stop_offsets_and_changes(self):
        result = self.study.analyze_trade(
            trade(),
            self.candles,
            self.evaluations,
            "Period A",
            "Bull",
        )

        self.assertEqual(set(result["states"]), set(PRE_STOP_OFFSETS))
        self.assertEqual(result["states"][-1]["rsi_change_1"], 1.0)
        self.assertEqual(result["states"][0]["rsi_change_3"], 3.0)
        self.assertIn("price_vs_ema21_percent", result["states"][0])
        self.assertIn("candle_range_percent", result["states"][0])

    def test_recovery_classification_and_targets(self):
        self.candles[6]["high"] = 100.0
        self.candles[7]["high"] = 102.1
        self.candles[8]["high"] = 104.1

        result = self.study.analyze_trade(
            trade(),
            self.candles,
            self.evaluations,
            "Period B",
            "Bull",
        )

        self.assertEqual(result["classification"], RECOVERED_ENTRY)
        self.assertTrue(result["reached_two_percent"])
        self.assertTrue(result["reached_four_percent"])

    def test_continued_loss_is_separate(self):
        result = self.study.analyze_trade(
            trade(),
            [candle(index, 95.0) for index in range(12)],
            self.evaluations,
            "Period C",
            "Bear",
        )

        self.assertEqual(result["classification"], CONTINUED_LOSS)
        self.assertFalse(result["reached_two_percent"])
        self.assertFalse(result["reached_four_percent"])

    def test_missing_pre_stop_context_is_explicit(self):
        result = self.study.analyze_trade(
            trade(exit_candle=2),
            self.candles,
            self.evaluations,
            "Period D",
            "Sideways",
        )

        self.assertIsNone(result["states"][-3])
        self.assertIsNotNone(result["states"][0])

    def test_summary_reports_group_differences_and_empty_groups(self):
        recovered = self.study.analyze_trade(
            trade(),
            self.candles,
            self.evaluations,
            "Period E",
            "Bull",
        )
        continued = self.study.analyze_trade(
            trade(),
            [candle(index, 95.0) for index in range(12)],
            self.evaluations,
            "Period F",
            "Bear",
        )

        summary = self.study.summarize_trades([recovered, continued])
        comparison = summary["comparisons"][0]["rsi"]

        self.assertEqual(summary["stop_loss_count"], 2)
        self.assertEqual(summary["recovered_entry_count"], 1)
        self.assertEqual(comparison["recovering"]["count"], 1)
        self.assertEqual(comparison["continued_loss"]["count"], 1)
        self.assertEqual(
            comparison["difference_recovering_minus_continued"],
            0.0,
        )
        self.assertIn("pattern_assessment", summary)


if __name__ == "__main__":
    unittest.main()