import unittest
from unittest.mock import patch

from generate_test_data import generate_candles
from strategy import calculate_strategy_score
from strategy_backtest import StrategyBacktester
from strategy_candidate_study import (
    MAX_VALIDATION_COST_SHARE_PERCENT,
    _candidate_classification,
    _compare_groups,
    evaluate_promotion_gates,
    patched_candidate_score,
    rsi_floor_score,
)


class StrategyCandidateStudyTests(unittest.TestCase):
    def test_candidate_keeps_score_and_conditions_but_blocks_low_rsi_buy(self):
        args = (110, 100, 95, 90, 55, 120, 100, 300)
        original = calculate_strategy_score(*args)
        candidate = rsi_floor_score(*args)

        self.assertEqual(candidate[0], original[0])
        self.assertEqual(candidate[3], original[3])
        self.assertEqual(candidate[1], "NO TRADE")

    def test_candidate_preserves_qualifying_rsi_decision(self):
        args = (110, 100, 95, 90, 65, 120, 100, 300)
        original = calculate_strategy_score(*args)
        candidate = rsi_floor_score(*args)

        self.assertEqual(candidate[0], original[0])
        self.assertEqual(candidate[1], original[1])

    def test_control_backtest_is_unchanged_after_candidate_context(self):
        candles = generate_candles(1000)
        baseline = StrategyBacktester(25.00)
        baseline.run(candles)
        baseline_results = baseline.results()

        with patched_candidate_score("candidate_a"):
            candidate = StrategyBacktester(25.00)
            candidate.run(candles)
        after = StrategyBacktester(25.00)
        after.run(candles)

        self.assertEqual(
            after.results(),
            baseline_results,
        )

    def test_group_comparison_reports_net_delta(self):
        control = {
            "performance": {
                "net_profit_loss": 1.0,
                "net_return_percent": 4.0,
                "completed_trades": 4,
                "cost_share_of_abs_gross_percent": 60.0,
            },
        }
        candidate = {
            "performance": {
                "net_profit_loss": 1.5,
                "net_return_percent": 6.0,
                "completed_trades": 3,
                "cost_share_of_abs_gross_percent": 40.0,
            },
        }

        comparison = _compare_groups(control, candidate)

        self.assertAlmostEqual(comparison["net_profit_delta"], 0.5)
        self.assertEqual(comparison["classification"], "improvement")

    def test_promotion_requires_validation_sample_and_cost_control(self):
        candidate = {
            "name": "Candidate A",
            "validation": {
                "performance": {
                    "buy_signals": 60,
                    "cost_share_of_abs_gross_percent": (
                        MAX_VALIDATION_COST_SHARE_PERCENT
                    ),
                },
            },
        }
        comparison = {
            "net_profit_delta": 1.0,
        }

        classification = _candidate_classification(
            candidate,
            comparison,
            comparison,
        )

        self.assertEqual(classification["classification"], "PROMISING")

    def test_promotion_requires_every_lifecycle_stage_and_gate(self):
        candidate = {
            "name": "Candidate A",
            "research": {"periods": [{"period": "A"}], "source": "fixed study"},
            "validation": {"performance": {
                "buy_signals": 25,
                "cost_share_of_abs_gross_percent": 10.0,
            }},
            "research_period_results": [{"period": "A"}],
            "validation_period_results": [{"period": "B"}],
        }
        evidence = {
            "research": {"status": "PASS"},
            "candidate": {"status": "PASS"},
            "backtest": {"status": "PASS"},
            "stress_test": {"status": "PASS"},
            "paper_test": {"status": "PASS"},
            "validate": {"status": "PASS"},
            "data_quality": "PASS",
            "freshness": "FRESH",
            "robustness": "PASS",
            "risk": "PASS",
            "paper_observation": "PASS",
        }
        result = evaluate_promotion_gates(candidate, evidence)
        self.assertEqual(result["status"], "PROMOTED")
        self.assertFalse(result["blocked_reasons"])

    def test_skipped_and_stale_evidence_blocks_promotion_with_reasons(self):
        candidate = {
            "name": "Candidate A",
            "research": {"periods": [{"period": "A"}]},
            "validation": {"performance": {"buy_signals": 2}},
            "research_period_results": [{"period": "A"}],
            "validation_period_results": [{"period": "B"}],
        }
        result = evaluate_promotion_gates(candidate, {
            "data_quality": "PASS",
            "freshness": "STALE",
            "robustness": "PASS",
            "risk": "PASS",
            "paper_observation": "PASS",
            "stress_test": {"status": "STALE", "reason": "stress report expired"},
        })
        self.assertEqual(result["status"], "BLOCKED")
        self.assertTrue(any("20" in reason for reason in result["blocked_reasons"]))
        self.assertIn("stress report expired", result["blocked_reasons"])


if __name__ == "__main__":
    unittest.main()