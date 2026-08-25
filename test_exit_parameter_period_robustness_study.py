import unittest

from exit_parameter_period_robustness_study import (
    ADDITIONAL_PERIODS,
    CONCENTRATION_THRESHOLD_PERCENT,
    MIN_PERIODS_FOR_BREADTH,
    MIN_TRADES_PER_PERIOD,
    VARIANTS,
    _breadth,
    _overall_outcome,
    _promotion_gate,
)
from config import EXIT_PROMOTION_MIN_UNTOUCHED_PERIODS


class ExitParameterPeriodRobustnessTests(unittest.TestCase):
    def test_only_control_and_two_6_percent_candidates_are_run(self):
        self.assertEqual(
            VARIANTS,
            (
                ("control", (2.0, 4.0)),
                ("stop_2.0_target_6.0", (2.0, 6.0)),
                ("stop_1.5_target_6.0", (1.5, 6.0)),
            ),
        )

    def test_additional_periods_are_pinned_and_regime_labeled(self):
        self.assertEqual(
            ADDITIONAL_PERIODS,
            (
                {
                    "period": "Supplemental Period K",
                    "start_date": "2014-09-17",
                    "end_date": "2015-09-16",
                    "regime": "Bear",
                },
                {
                    "period": "Supplemental Period L",
                    "start_date": "2015-09-17",
                    "end_date": "2016-09-15",
                    "regime": "Bull",
                },
                {
                    "period": "Supplemental Period M",
                    "start_date": "2016-09-16",
                    "end_date": "2017-09-15",
                    "regime": "Bull",
                },
            ),
        )

    def test_additional_periods_are_disjoint_and_cover_three_periods(self):
        self.assertGreaterEqual(
            len(ADDITIONAL_PERIODS),
            EXIT_PROMOTION_MIN_UNTOUCHED_PERIODS,
        )
        for previous, current in zip(ADDITIONAL_PERIODS, ADDITIONAL_PERIODS[1:]):
            self.assertLess(previous["end_date"], current["start_date"])
        self.assertEqual(
            [period["end_date"] for period in ADDITIONAL_PERIODS],
            ["2015-09-16", "2016-09-15", "2017-09-15"],
        )

    def test_breadth_flags_sparse_and_concentrated_results(self):
        periods = [
            {
                "insufficient_evidence": False,
                "net_delta_vs_control": 1.0,
                "contribution_to_total_improvement_percent": 70.0,
            },
            {
                "insufficient_evidence": False,
                "net_delta_vs_control": -0.2,
                "contribution_to_total_improvement_percent": -20.0,
            },
            {
                "insufficient_evidence": True,
                "net_delta_vs_control": 0.1,
                "contribution_to_total_improvement_percent": 50.0,
            },
        ]
        result = _breadth(periods, periods, 1.0)

        self.assertEqual(result["eligible_periods"], 2)
        self.assertEqual(result["positive_periods"], 1)
        self.assertTrue(result["concentrated_in_one_or_few_periods"])
        self.assertTrue(result["insufficient_evidence"])
        self.assertEqual(MIN_TRADES_PER_PERIOD, 3)
        self.assertEqual(
            MIN_PERIODS_FOR_BREADTH,
            EXIT_PROMOTION_MIN_UNTOUCHED_PERIODS,
        )
        self.assertEqual(CONCENTRATION_THRESHOLD_PERCENT, 60.0)

    def test_breadth_marks_too_few_eligible_periods(self):
        periods = [{
            "insufficient_evidence": False,
            "net_delta_vs_control": 0.1,
            "contribution_to_total_improvement_percent": 100.0,
        }]

        self.assertTrue(_breadth(periods, periods, 0.1)["insufficient_evidence"])

    def test_outcome_is_yellow_when_uplift_is_positive_but_unproven(self):
        breadth = {
            "stop_2.0_target_6.0": {
                "positive_period_share_percent": 100.0,
                "concentrated_in_one_or_few_periods": True,
                "insufficient_evidence": False,
            },
            "stop_1.5_target_6.0": {
                "positive_period_share_percent": 100.0,
                "concentrated_in_one_or_few_periods": True,
                "insufficient_evidence": False,
            },
        }
        analysis = {
            "breadth": breadth,
            "aggregate": {
                "control": {"net": 0.0},
                "stop_2.0_target_6.0": {"net": 1.0},
                "stop_1.5_target_6.0": {"net": 0.5},
            },
        }
        validation = {
            "breadth": {
                label: {"insufficient_evidence": True}
                for label in breadth
            }
        }
        outcome = _overall_outcome(analysis, validation, analysis)
        self.assertEqual(outcome["label"], "INTERESTING BUT UNPROVEN")
        self.assertEqual(outcome["color"], "yellow")

    def test_promotion_gate_blocks_concentrated_or_sparse_candidate(self):
        def analysis(breadth, net_delta=1.0, cost_share=1.0, drawdown_delta=0.0):
            return {
                "breadth": {"candidate": breadth},
                "aggregate": {
                    "control": {
                        "net": 0.0,
                        "cost_share": 1.0,
                        "maximum_drawdown": 1.0,
                    },
                    "candidate": {
                        "net": net_delta,
                        "net_delta_vs_control": net_delta,
                        "cost_share": cost_share,
                        "drawdown_delta_vs_control": drawdown_delta,
                    },
                },
            }

        breadth = {
            "eligible_periods": 2,
            "positive_period_share_percent": 100.0,
            "concentrated_in_one_or_few_periods": True,
            "insufficient_evidence": True,
        }
        decision = _promotion_gate(
            "candidate",
            analysis(breadth),
            analysis(breadth),
            analysis(breadth),
        )

        self.assertEqual(decision["status"], "RESEARCH_ONLY")
        self.assertTrue(decision["research_only"])
        self.assertTrue(any("eligible periods" in reason for reason in decision["reasons"]))
        self.assertTrue(any("concentration" in reason for reason in decision["reasons"]))

    def test_promotion_gate_allows_only_broad_positive_low_risk_candidate(self):
        def analysis():
            breadth = {
                "eligible_periods": 3,
                "positive_period_share_percent": 100.0,
                "concentrated_in_one_or_few_periods": False,
                "insufficient_evidence": False,
            }
            return {
                "breadth": {"candidate": breadth},
                "aggregate": {
                    "control": {"net": 0.0, "cost_share": 1.0},
                    "candidate": {
                        "net": 1.0,
                        "net_delta_vs_control": 1.0,
                        "cost_share": 1.0,
                        "drawdown_delta_vs_control": 0.0,
                    },
                },
            }

        decision = _promotion_gate(
            "candidate", analysis(), analysis(), analysis()
        )

        self.assertEqual(decision["status"], "PROMOTION_ELIGIBLE")
        self.assertFalse(decision["research_only"])


if __name__ == "__main__":
    unittest.main()