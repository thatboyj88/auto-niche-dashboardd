import unittest

from execution_economics_study import (
    SCENARIOS,
    OUTCOME_INCONCLUSIVE_SMALL_SAMPLE,
    _trade_diagnostics,
)


class ExecutionEconomicsStudyTests(unittest.TestCase):
    def test_five_scenarios_are_predeclared(self):
        self.assertEqual([item[0] for item in SCENARIOS], ["C0", "C1", "C2", "C3", "C4"])
        self.assertEqual(SCENARIOS[0][1:3], (1.0, 1.0))
        self.assertEqual(SCENARIOS[-1][1:3], (0.0, 0.0))

    def test_trade_diagnostics_calculate_cost_hurdle_and_path(self):
        trade = {
            "entry_candle": 0,
            "exit_candle": 2,
            "market_entry_price": 100.0,
            "position_size": 1.0,
            "gross_profit_loss_before_costs": 2.0,
            "fees": 0.8,
            "estimated_slippage": 0.2,
            "net_profit_loss": 1.0,
            "reason": "TAKE PROFIT",
        }
        result = _trade_diagnostics(
            trade,
            [{"close": 100}, {"close": 103}, {"close": 98}],
            0.004,
            0.001,
        )
        self.assertEqual(result["required_break_even_percent"], 1.0)
        self.assertAlmostEqual(result["mfe_percent"], 3.0)
        self.assertAlmostEqual(result["mae_percent"], -2.0)
        self.assertTrue(result["mfe_cleared_cost_hurdle"])

    def test_no_promotion_classification_exists(self):
        self.assertEqual(OUTCOME_INCONCLUSIVE_SMALL_SAMPLE, "INCONCLUSIVE_SMALL_SAMPLE")


if __name__ == "__main__":
    unittest.main()