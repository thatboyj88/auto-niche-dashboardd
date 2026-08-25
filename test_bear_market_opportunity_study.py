import unittest

from bear_market_opportunity_study import (
    FORWARD_HORIZONS,
    BearMarketOpportunityStudy,
)


class BearMarketOpportunityStudyTests(unittest.TestCase):
    def test_exit_simulation_applies_costs(self):
        path = [{"close": 104.0, "high": 104.0, "low": 100.0}]
        result = BearMarketOpportunityStudy._simulate_exit(path, 100.0)

        self.assertEqual(result["reason"], "TAKE PROFIT")
        self.assertLess(result["net_profit_loss"], 0.4)
        self.assertGreater(result["fees"], 0.0)
        self.assertGreater(result["slippage"], 0.0)

    def test_horizons_are_fixed(self):
        self.assertEqual(FORWARD_HORIZONS, (1, 3, 5, 10, 20))

    def test_empty_group_summary_is_safe(self):
        result = BearMarketOpportunityStudy().analyze_group([])

        self.assertEqual(result["all"]["opportunities"], 0)
        self.assertEqual(
            result["all"]["profitable_at_horizon"][1],
            0,
        )


if __name__ == "__main__":
    unittest.main()