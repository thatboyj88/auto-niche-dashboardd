import unittest

from trade_path_exit_timing_study import TradePathExitTimingStudy


def candle(timestamp, close, high=None, low=None):
    return {
        "timestamp": timestamp,
        "open": close,
        "high": high if high is not None else close,
        "low": low if low is not None else close,
        "close": close,
        "volume": 100.0,
    }


def trade(reason="STOP LOSS", exit_candle=2, net=-0.3):
    return {
        "trade_number": 1,
        "entry_candle": 0,
        "exit_candle": exit_candle,
        "entry_timestamp": 1_700_000_000,
        "exit_timestamp": 1_700_172_800,
        "market_entry_price": 100.0,
        "market_exit_price": 98.0 if reason == "STOP LOSS" else 104.0,
        "position_size": 1.0,
        "strategy_score": 85,
        "rsi_at_entry": 55.0,
        "net_profit_loss": net,
        "reason": reason,
    }


class TradePathExitTimingStudyTests(unittest.TestCase):
    def setUp(self):
        self.study = TradePathExitTimingStudy()

    def test_path_metrics_and_early_movement_are_bounded(self):
        candles = [candle(index, 100.0) for index in range(7)]
        candles[1]["high"] = 101.0
        candles[3]["low"] = 97.0
        candles[5]["high"] = 104.0

        result = self.study.analyze_trade(
            trade(),
            candles,
            "Period A",
            "Bull",
        )

        self.assertAlmostEqual(result["mfe_percent"], 4.0)
        self.assertAlmostEqual(result["mae_percent"], -3.0)
        self.assertEqual(result["early_movement_percent"][20], None)
        self.assertIn(result["exit_timing"], ("before_strongest", "near_strongest", "after_strongest"))

    def test_exit_classification_and_stop_recovery(self):
        candles = [candle(index, 100.0) for index in range(8)]
        candles[4]["high"] = 102.1
        candles[5]["high"] = 104.1

        result = self.study.analyze_trade(
            trade(),
            candles,
            "Period B",
            "Bear",
        )

        self.assertTrue(result["stop_before_target"])
        self.assertTrue(result["post_exit_stop_first"]["recovered_entry"])
        self.assertTrue(result["post_exit_stop_first"]["reached_two_percent"])
        self.assertTrue(result["post_exit_stop_first"]["reached_four_percent"])

    def test_target_first_aftermath_and_counterfactual_costs(self):
        candles = [candle(index, 100.0) for index in range(6)]
        candles[1]["high"] = 104.1
        candles[4]["low"] = 99.0
        target_trade = trade("TAKE PROFIT", exit_candle=2, net=0.2)
        target_trade["market_exit_price"] = 104.0

        result = self.study.analyze_trade(
            target_trade,
            candles,
            "Period C",
            "Bull",
        )

        self.assertTrue(result["target_before_stop"])
        self.assertIn("four_percent", result["hypothetical_exits"])
        self.assertTrue(
            result["post_exit_stop_first"]["fell_below_entry"]
        )
        self.assertTrue(
            result["post_exit_stop_first"][
                "reached_two_percent_beyond_target"
            ] is False
        )
        self.assertGreater(
            result["hypothetical_exits"]["four_percent"]["fees"],
            0,
        )

    def test_summary_keeps_empty_groups_valid(self):
        summary = self.study.summarize_trades([])

        self.assertEqual(summary["trade_count"], 0)
        self.assertEqual(summary["exit_timing"]["before_strongest"], 0)
        self.assertEqual(summary["groups"]["winners"]["count"], 0)
        self.assertEqual(summary["counterfactual"]["two_percent"]["count"], 0)


if __name__ == "__main__":
    unittest.main()