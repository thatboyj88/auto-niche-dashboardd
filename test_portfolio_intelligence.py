import unittest
from datetime import datetime, timezone

from portfolio_intelligence import analyze_portfolio


class PortfolioIntelligenceTests(unittest.TestCase):
    def test_normal_state_has_allocation_sizing_and_provenance(self):
        analysis = analyze_portfolio(
            cash=1000,
            positions=[
                {
                    "symbol": "BTC/CAD",
                    "quantity": 0.01,
                    "price": 30000,
                    "expected_return": 0.12,
                    "volatility": 0.30,
                },
                {"symbol": "ETH/CAD", "quantity": 0.1, "price": 2000},
            ],
            target_allocations={"BTC/CAD": 0.40, "ETH/CAD": 0.20},
        )
        self.assertEqual(analysis.status, "OK")
        self.assertAlmostEqual(analysis.total_value, 1500)
        self.assertAlmostEqual(analysis.current_allocation["BTC/CAD"], 0.2)
        self.assertAlmostEqual(analysis.position_sizing["max_position_value"], 400)
        self.assertEqual(analysis.risk_adjusted_allocation["status"], "AVAILABLE")
        self.assertIn("BTC/CAD", analysis.risk_adjusted_allocation["ranking"])

    def test_empty_state_is_explicit_and_recommends_cash(self):
        analysis = analyze_portfolio(cash=100, positions=[])
        self.assertEqual(analysis.status, "OK")
        self.assertEqual(analysis.current_allocation, {"CASH": 1.0})
        self.assertEqual(analysis.diversification["status"], "LIMITED")
        self.assertEqual(analysis.risk_adjusted_allocation["status"], "UNAVAILABLE")
        self.assertTrue(any("cash" in item.lower() for item in analysis.recommendations))

    def test_invalid_inputs_fail_closed(self):
        analysis = analyze_portfolio(cash=100, positions=[{"symbol": "BTC", "quantity": 1, "price": None}])
        self.assertEqual(analysis.status, "UNAVAILABLE")
        self.assertIsNone(analysis.total_value)
        self.assertTrue(analysis.issues)

    def test_stale_valuation_fails_closed(self):
        analysis = analyze_portfolio(
            cash=100,
            positions=[],
            observed_at="2026-08-25T10:00:00+00:00",
            now=datetime(2026, 8, 25, 10, 16, tzinfo=timezone.utc),
            max_age_seconds=900,
        )
        self.assertEqual(analysis.status, "UNAVAILABLE")
        self.assertIn("stale", analysis.issues[0])

    def test_concentration_breach_has_explicit_reason_and_veto(self):
        analysis = analyze_portfolio(
            cash=100,
            positions=[{"symbol": "BTC", "quantity": 1, "price": 900}],
            max_position_percent=0.40,
        )
        self.assertEqual(analysis.concentration["status"], "BREACH")
        self.assertTrue(analysis.concentration["breaches"])
        self.assertTrue(any("VETO" in item for item in analysis.recommendations))

    def test_target_drift_recommendation_is_deterministic(self):
        analysis = analyze_portfolio(
            cash=50,
            positions=[{"symbol": "BTC", "quantity": 1, "price": 50}],
            target_allocations={"BTC": 0.8},
        )
        self.assertTrue(any("material allocation drift" in item for item in analysis.recommendations))

    def test_default_proposal_caps_concentration_into_cash(self):
        analysis = analyze_portfolio(
            cash=100,
            positions=[{"symbol": "BTC", "quantity": 1, "price": 900}],
        )
        self.assertAlmostEqual(analysis.proposed_allocation["BTC"], 0.40)
        self.assertAlmostEqual(analysis.proposed_allocation["CASH"], 0.60)

    def test_target_weights_over_one_fail_closed(self):
        analysis = analyze_portfolio(
            cash=100,
            positions=[],
            target_allocations={"BTC": 0.8, "ETH": 0.3},
        )
        self.assertEqual(analysis.status, "UNAVAILABLE")
        self.assertIn("exceed 100%", analysis.issues[0])


if __name__ == "__main__":
    unittest.main()