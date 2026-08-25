import unittest

from trade_sequencing_clustering_study import (
    SEQUENCE_WINDOWS,
    TradeSequencingClusteringStudy,
    _summarize,
)


def evaluation(candle, decision="NO TRADE"):
    return {"candle": candle, "decision": decision}


def trade(number, entry, exit_candle, net, reason="TAKE PROFIT"):
    return {
        "trade_number": number,
        "entry_candle": entry,
        "exit_candle": exit_candle,
        "gross_profit_loss_before_costs": net + 0.25,
        "net_profit_loss": net,
        "fees": 0.20,
        "estimated_slippage": 0.05,
        "market_entry_price": 100.0,
        "market_exit_price": 104.0 if net > 0 else 98.0,
        "position_size": 0.4,
        "strategy_score": 85,
        "rsi_at_entry": 62,
        "reason": reason,
    }


class TradeSequencingClusteringStudyTests(unittest.TestCase):
    def test_sequence_metrics_and_groups_are_assigned(self):
        candles = [
            {
                "open": 100.0,
                "high": 104.0,
                "low": 98.0,
                "close": 101.0,
                "volume": 1.0,
                "timestamp": index,
            }
            for index in range(40)
        ]
        result = {
            "period": "Test",
            "regime": "Bull",
            "trades_history": [
                trade(1, 10, 12, 1.0),
                trade(2, 15, 16, -1.0, reason="STOP LOSS"),
                trade(3, 19, 20, 1.0),
            ],
            "evaluation_history": [
                evaluation(10, "BUY"),
                evaluation(14, "BUY"),
                evaluation(18, "BUY"),
            ],
        }

        items = TradeSequencingClusteringStudy().analyze_period(
            result, candles
        )

        self.assertEqual(len(items), 3)
        self.assertEqual(items[1]["candles_since_previous_buy_signal"], 1)
        self.assertEqual(items[1]["candles_since_previous_completed_trade"], 3)
        self.assertTrue(items[1]["clustered"])
        self.assertTrue(items[2]["immediately_after_loss"])
        self.assertEqual(
            set(items[0]["signals_previous"]),
            set(SEQUENCE_WINDOWS),
        )

    def test_empty_summary_is_safe(self):
        summary = _summarize([])

        self.assertEqual(summary["trade_count"], 0)
        self.assertEqual(summary["net_profit_loss"], 0.0)
        self.assertEqual(summary["exit_reasons"]["STOP LOSS"], 0)


if __name__ == "__main__":
    unittest.main()