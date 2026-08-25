import unittest

from stop_loss_recovery_study import StopLossRecoveryStudy


def candle(timestamp, close, high=None, low=None):
    return {
        "timestamp": timestamp,
        "open": close,
        "high": high if high is not None else close,
        "low": low if low is not None else close,
        "close": close,
        "volume": 100.0,
    }


def evaluation(score=85, rsi=55):
    return {
        "strategy_score": score,
        "rsi": rsi,
        "long_term_trend": True,
        "short_term_momentum": True,
        "rsi_condition": True,
        "volume": False,
        "price_above_ema21": True,
    }


def trade(exit_candle=1):
    return {
        "trade_number": 1,
        "entry_candle": 0,
        "exit_candle": exit_candle,
        "entry_timestamp": 1_700_000_000,
        "exit_timestamp": 1_700_086_400,
        "entry_price": 100.0,
        "market_entry_price": 100.0,
        "market_exit_price": 98.0,
        "reason": "STOP LOSS",
    }


class StopLossRecoveryStudyTests(unittest.TestCase):
    def setUp(self):
        self.study = StopLossRecoveryStudy()

    def test_recovery_and_target_detection(self):
        candles = [candle(index, 100.0) for index in range(8)]
        candles[2]["high"] = 99.0
        candles[3]["high"] = 100.0
        candles[4]["high"] = 102.1
        candles[5]["high"] = 104.1

        result = self.study.analyze_trade(
            trade(),
            candles,
            evaluation(),
            "Period A",
            "Bull",
        )

        self.assertTrue(result["recovered_entry"])
        self.assertTrue(result["reached_two_percent"])
        self.assertTrue(result["reached_four_percent"])
        self.assertEqual(
            result["recovery_timing"]["recover_50_percent"][
                "candles_after_exit"
            ],
            1,
        )
        self.assertEqual(
            result["recovery_timing"]["recover_100_percent"][
                "candles_after_exit"
            ],
            2,
        )

    def test_continued_loser_has_no_recovery(self):
        candles = [candle(index, 95.0) for index in range(5)]

        result = self.study.analyze_trade(
            trade(),
            candles,
            evaluation(),
            "Period B",
            "Bear",
        )

        self.assertTrue(result["continued_loser"])
        self.assertFalse(result["recovered_entry"])
        self.assertFalse(result["reached_two_percent"])
        self.assertFalse(result["reached_four_percent"])

    def test_horizon_boundaries_and_period_end_are_explicit(self):
        candles = [candle(index, 100.0) for index in range(4)]
        candles[2]["high"] = 99.5
        candles[3]["high"] = 99.5

        result = self.study.analyze_trade(
            trade(),
            candles,
            evaluation(),
            "Period C",
            "Sideways",
        )

        self.assertEqual(
            result["horizon_recovery"][5]["observed_candles"],
            2,
        )
        self.assertTrue(
            result["horizon_recovery"][5]["period_end_limited"]
        )
        self.assertFalse(result["recovered_entry"])

    def test_empty_stop_loss_period_summarizes_cleanly(self):
        summary = self.study.summarize_trades([])

        self.assertEqual(summary["stop_loss_count"], 0)
        self.assertEqual(summary["recovered_entry_percent"], 0.0)
        self.assertEqual(summary["groups"]["continued_losers"]["count"], 0)
        self.assertEqual(summary["recovery_timing"]["recover_50_percent"]["count"], 0)

    def test_group_summary_preserves_entry_pattern_fields(self):
        candles = [candle(index, 100.0) for index in range(5)]
        candles[3]["high"] = 100.0
        analyzed = self.study.analyze_trade(
            trade(),
            candles,
            evaluation(score=92, rsi=68),
            "Period D",
            "Bull",
        )

        summary = self.study.summarize_trades([analyzed])
        group = summary["groups"]["recovering_trades"]

        self.assertEqual(group["count"], 1)
        self.assertAlmostEqual(group["average_entry_score"], 92.0)
        self.assertAlmostEqual(group["average_entry_rsi"], 68.0)
        self.assertEqual(group["regimes"]["Bull"], 1)
        self.assertEqual(group["conditions"]["volume"], 0.0)


if __name__ == "__main__":
    unittest.main()