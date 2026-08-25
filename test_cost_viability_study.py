import unittest

from cost_viability_study import COST_MODELS, CostViabilityStudy


def trade(gross, fees=0.20, slippage=0.05, reason="TAKE PROFIT"):
    return {
        "trade_number": 1,
        "period": "Period A",
        "regime": "Bull",
        "reason": reason,
        "entry_candle": 1,
        "market_entry_price": 100.0,
        "position_size": 1.0,
        "gross_profit_loss_before_costs": gross,
        "fees": fees,
        "estimated_slippage": slippage,
    }


class CostViabilityStudyTests(unittest.TestCase):
    def setUp(self):
        self.study = CostViabilityStudy()

    def test_required_move_and_current_net_use_same_cost_model(self):
        analyzed = self.study.analyze_trade(
            trade(gross=1.0),
            "Period A",
            "Bull",
        )

        self.assertAlmostEqual(analyzed["required_move_percent"], 0.25)
        self.assertAlmostEqual(analyzed["current_net_profit_loss"], 0.75)

    def test_cost_factors_reprice_without_changing_gross(self):
        analyzed = self.study.analyze_trade(trade(gross=1.0), "A", "Bull")

        current = self.study.reprice_trade(analyzed, 1.0)
        half = self.study.reprice_trade(analyzed, 0.5)
        zero = self.study.reprice_trade(analyzed, 0.0)

        self.assertAlmostEqual(current["net_profit_loss"], 0.75)
        self.assertAlmostEqual(half["net_profit_loss"], 0.875)
        self.assertAlmostEqual(zero["net_profit_loss"], 1.0)

    def test_profitability_counts_and_concentration_are_reported(self):
        trades = [
            self.study.analyze_trade(trade(1.0), "A", "Bull"),
            self.study.analyze_trade(trade(0.1), "A", "Bull"),
            self.study.analyze_trade(trade(-0.2, reason="STOP LOSS"), "A", "Bull"),
            self.study.analyze_trade(trade(0.4), "A", "Bull"),
        ]
        summary = self.study.analyze_trades(trades)

        self.assertEqual(summary["trade_count"], 4)
        self.assertEqual(summary["positive_gross_trade_count"], 3)
        self.assertIn("current", summary["cost_models"])
        self.assertIn("zero", summary["cost_models"])
        self.assertGreater(
            summary["gross_profit_concentration"]["5"][
                "share_of_positive_gross_percent"
            ],
            0,
        )

    def test_all_requested_cost_models_exist(self):
        labels = [label for _, _, label in COST_MODELS]
        self.assertEqual(
            labels,
            ["current model", "25% lower", "50% lower", "75% lower", "zero costs"],
        )

    def test_empty_group_is_safe(self):
        summary = self.study.analyze_trades([])

        self.assertEqual(summary["trade_count"], 0)
        self.assertEqual(summary["gross_profit_loss"], 0)
        self.assertEqual(summary["cost_models"]["current"]["profitable_trade_count"], 0)
        self.assertEqual(summary["gross_profit_concentration"]["5"]["top_trade_count"], 0)


if __name__ == "__main__":
    unittest.main()