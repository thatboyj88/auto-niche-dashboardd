import unittest

from historical_validation import (
    measure_decision_consistency,
    summarize_genuine_paper_records,
    summarize_historical_results,
)
from generate_test_data import generate_candles


class HistoricalValidationSummaryTests(unittest.TestCase):
    def test_decision_consistency_replays_cost_and_boundary_cases(self):
        result = measure_decision_consistency(
            [generate_candles(365), generate_candles(365)]
        )

        self.assertEqual(result["case_count"], 18)
        self.assertEqual(result["repeatable_cases"], 18)
        self.assertEqual(result["non_repeatable_cases"], 0)
        self.assertEqual(result["repeatability_percent"], 100.0)
        self.assertIn("PASS", result["conclusion"])

    def test_paper_summary_is_read_only_and_cost_aware(self):
        records = [
            {
                "dataset": "PAPER_OPERATIONAL",
                "record_type": "SIGNAL",
                "payload": {"strategy_score": 80},
            },
            {
                "dataset": "PAPER_OPERATIONAL",
                "record_type": "TRADE",
                "payload": {
                    "profit_loss": 1.0,
                    "fees": 0.2,
                    "slippage": 0.1,
                },
            },
            {
                "dataset": "PAPER_OPERATIONAL",
                "record_type": "TRADE",
                "payload": {
                    "profit_loss": -0.5,
                    "fees": 0.2,
                    "slippage": 0.1,
                },
            },
            {
                "dataset": "HISTORICAL",
                "record_type": "TRADE",
                "payload": {"profit_loss": 100},
            },
        ]

        result = summarize_genuine_paper_records(records)

        self.assertEqual(result["records"], 3)
        self.assertEqual(result["trades"], 2)
        self.assertEqual(result["wins"], 1)
        self.assertEqual(result["losses"], 1)
        self.assertEqual(result["profit"], 0.5)
        self.assertEqual(result["fees"], 0.4)
        self.assertEqual(result["slippage"], 0.2)
        self.assertEqual(result["strategy_score_distribution"], {80.0: 1})
        self.assertIsNone(result["max_drawdown"])
        self.assertIsNone(result["market_condition_performance"])

    def test_historical_summary_aggregates_periods_and_regimes(self):
        result = summarize_historical_results(
            {
                "source_candles": 730,
                "periods": [
                    {
                        "trades": 2,
                        "wins": 1,
                        "profit": 1.5,
                        "evaluation_history": [
                            {"strategy_score": 60},
                            {"strategy_score": 80},
                        ],
                        "trades_history": [{"strategy_score": 80}],
                    },
                    {
                        "trades": 1,
                        "wins": 1,
                        "profit": -0.2,
                        "evaluation_history": [{"strategy_score": 100}],
                        "trades_history": [{"strategy_score": 100}],
                    },
                ],
                "aggregate": {
                    "total_profit": 1.3,
                    "total_fees": 0.3,
                    "total_slippage": 0.1,
                    "worst_drawdown": 4.0,
                },
                "regime_summary": {
                    "Bull": [
                        {"trades": 2, "profit": 1.5},
                    ],
                    "Bear": [
                        {"trades": 1, "profit": -0.2},
                    ],
                    "Sideways": [],
                },
            }
        )

        self.assertEqual(result["records"], 730)
        self.assertEqual(result["trades"], 3)
        self.assertAlmostEqual(result["win_rate"], 66.6666667)
        self.assertEqual(
            result["strategy_score_distribution"],
            {60: 1, 80: 1, 100: 1},
        )
        self.assertEqual(result["market_condition_performance"]["Bear"]["profit"], -0.2)


if __name__ == "__main__":
    unittest.main()