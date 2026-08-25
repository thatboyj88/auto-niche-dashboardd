import io
import unittest
from contextlib import redirect_stdout

from out_of_sample_validation import (
    MAX_MEANINGFUL_COST_SHARE_PERCENT,
    MIN_CALIBRATION_SIGNALS,
    OutOfSampleValidationStudy,
    print_report,
    _split_periods,
)
from score_effectiveness_study import SCORE_STUDY_PERIODS


# Included by offline_study_report_check.py; network/live-data tests remain
# intentionally unmarked and outside that runner.
REPORT_REGRESSION_MODULE = True


HYPOTHESIS_SURVIVAL_NAMES = (
    "rsi_60_vs_50_59",
    "score_85_89_vs_80_84",
    "break_even_reach",
    "net_after_costs",
    "regime_consistency",
)


def render(report, result):
    output = io.StringIO()
    with redirect_stdout(output):
        report(result)
    return output.getvalue()


def report_fixture():
    split = {
        "research_start": "2020-01-01",
        "research_end": "2020-01-02",
        "research_periods": 1,
        "research_candles": 2,
        "validation_start": "2020-01-03",
        "validation_end": "2020-01-04",
        "validation_periods": 1,
        "validation_candles": 2,
    }
    group = {
        "performance": {
            "periods": 1,
            "candles": 2,
            "gross_profit_loss": 1.0,
            "fees": 0.1,
            "slippage": 0.1,
            "net_profit_loss": 0.8,
            "cost_share_of_abs_gross_percent": 20.0,
        },
        "periods": [{
            "start_date": "2020-01-01",
            "end_date": "2020-01-02",
        }],
        "signal_count": 1,
        "completed_entry_count": 1,
        "early_movement": {},
        "rsi_bands": {},
        "score_bands": {},
        "condition_combinations": {},
        "cost_break_even": {"overall": {"horizons": {}}},
        "regime_performance": {},
    }
    group_hypotheses = {
        "rsi_60_vs_50_59": {
            "lower": {"horizons": {"5": {"average": 0.1}}},
            "higher": {"horizons": {"5": {"average": 0.2}}},
        },
        "score_85_89_vs_80_84": {
            "lower": {"horizons": {"5": {"average": 0.1}}},
            "higher": {"horizons": {"5": {"average": 0.2}}},
        },
    }
    return {
        "split": split,
        "research": group,
        "validation": group,
        "research_hypotheses": group_hypotheses,
        "validation_hypotheses": group_hypotheses,
        "hypotheses": {
            name: {
                "research": "supported",
                "validation": (
                    "supported" if name == "net_after_costs"
                    else "not supported"
                ),
                "survived": name == "net_after_costs",
            }
            for name in HYPOTHESIS_SURVIVAL_NAMES
        },
    }


def computed_hypothesis_group(
    return_values,
    reached_break_even_percent,
    signal_count=40,
):
    signals = [
        {
            "entry_rsi": 55 if index < 20 else 60,
            "score": 82 if index < 20 else 87,
            "regime": "Bull" if index < 20 else "Bear",
            "forward_returns": {
                horizon: return_values[0 if index < 20 else 1]
                for horizon in (3, 5, 10, 20)
            },
        }
        for index in range(signal_count)
    ]
    return {
        "signals": signals,
        "cost_break_even": {
            "overall": {
                "horizons": {
                    "5": {
                        "signals": len(signals),
                        "reached_break_even_percent": (
                            reached_break_even_percent
                        ),
                    },
                },
            },
        },
        "regime_performance": {
            "Bull": {"buy_signals": 20},
            "Bear": {"buy_signals": 20},
        },
        "performance": {
            "gross_profit_loss": 10.0 if return_values[1] > 0 else -10.0,
            "net_profit_loss": 9.0 if return_values[1] > 0 else -11.0,
            "fees": 0.5,
            "slippage": 0.5,
            "cost_share_of_abs_gross_percent": 10.0,
        },
    }


class OutOfSampleValidationTests(unittest.TestCase):
    def test_report_has_validation_sections_and_final_interpretation_boundary(self):
        report = render(print_report, report_fixture())

        self.assertIn(
            "BTC/CAD OUT-OF-SAMPLE VALIDATION — ANALYSIS ONLY",
            report,
        )
        self.assertIn("=== Research / calibration ===", report)
        self.assertIn("=== Out-of-sample validation ===", report)
        self.assertIn("=== Hypothesis survival ===", report)
        self.assertIn(
            "Observed relationships are not proven predictive relationships.",
            report,
        )
        self.assertLess(
            report.index("=== Hypothesis survival ==="),
            report.index("Observed relationships are not proven predictive relationships."),
        )

    def test_report_renders_every_hypothesis_survival_line(self):
        report = render(print_report, report_fixture())
        survival_section = report.split("=== Hypothesis survival ===", 1)[1].split(
            "Observed relationships are not proven predictive relationships.",
            1,
        )[0]
        lines = [
            line.strip()
            for line in survival_section.splitlines()
            if line.strip()
        ]

        self.assertEqual(
            {line.split(":", 1)[0] for line in lines},
            set(HYPOTHESIS_SURVIVAL_NAMES),
        )
        self.assertEqual(len(lines), len(HYPOTHESIS_SURVIVAL_NAMES))
        for line in lines:
            self.assertRegex(line, r"research=[^,]+")
            self.assertRegex(line, r"validation=[^,]+")
            self.assertRegex(line, r"survived=(YES|NO)")

    def test_split_is_chronological_and_uses_newest_two_periods(self):
        research, validation = _split_periods(SCORE_STUDY_PERIODS)

        self.assertEqual(len(research), 8)
        self.assertEqual(len(validation), 2)
        self.assertEqual(research[0]["start_date"], "2016-08-20")
        self.assertEqual(research[-1]["end_date"], "2024-08-17")
        self.assertEqual(validation[0]["start_date"], "2024-08-18")
        self.assertEqual(validation[-1]["end_date"], "2026-08-17")

    def test_comparison_requires_minimum_sample_in_both_groups(self):
        study = OutOfSampleValidationStudy()
        sparse = {
            "signals": MIN_CALIBRATION_SIGNALS - 1,
            "insufficient_evidence": True,
            "horizons": {
                "5": {"average": 5.0},
            },
        }
        established = {
            "signals": MIN_CALIBRATION_SIGNALS,
            "insufficient_evidence": False,
            "horizons": {
                "5": {"average": 1.0},
            },
        }

        comparison = study._compare_groups(established, sparse)

        self.assertEqual(comparison["status"], "insufficient evidence")

    def test_regime_consistency_requires_two_established_regimes(self):
        result = OutOfSampleValidationStudy._regime_consistency([
            {
                "regime": "Bull",
                "signals": MIN_CALIBRATION_SIGNALS,
                "average_5_candle_return": 1.0,
                "positive_5_candle_percent": 55.0,
            },
        ])

        self.assertEqual(result["status"], "insufficient evidence")

    def test_regime_consistency_rejects_mixed_signs(self):
        result = OutOfSampleValidationStudy._regime_consistency([
            {
                "regime": "Bull",
                "signals": MIN_CALIBRATION_SIGNALS,
                "average_5_candle_return": 1.0,
                "positive_5_candle_percent": 55.0,
            },
            {
                "regime": "Bear",
                "signals": MIN_CALIBRATION_SIGNALS,
                "average_5_candle_return": -1.0,
                "positive_5_candle_percent": 45.0,
            },
        ])

        self.assertEqual(result["status"], "not supported")

    def test_cost_heavy_net_gain_is_not_meaningful_after_costs(self):
        result = OutOfSampleValidationStudy._net_after_costs_hypothesis({
            "gross_profit_loss": 1.0,
            "net_profit_loss": 0.01,
            "fees": 0.8,
            "slippage": 0.19,
            "cost_share_of_abs_gross_percent": (
                MAX_MEANINGFUL_COST_SHARE_PERCENT
            ),
        })

        self.assertEqual(result["status"], "not supported")

    def test_computed_hypotheses_keep_status_and_survival_contract(self):
        study = OutOfSampleValidationStudy()
        research = study.hypothesis_results(
            computed_hypothesis_group((1.0, 2.0), 100.0)
        )
        validation = study.hypothesis_results(
            computed_hypothesis_group((1.0, -1.0), 0.0)
        )

        comparison = study._compare_hypotheses(research, validation)

        self.assertEqual(set(comparison), set(HYPOTHESIS_SURVIVAL_NAMES))
        for name in HYPOTHESIS_SURVIVAL_NAMES:
            result = comparison[name]
            self.assertIn(result["research"], {"supported", "not supported"})
            self.assertIn(
                result["validation"],
                {"supported", "not supported"},
            )
            self.assertEqual(
                result["survived"],
                (
                    result["research"] == "supported"
                    and result["validation"] == "supported"
                ),
            )
        self.assertTrue(all(
            comparison[name]["research"] == "supported"
            for name in HYPOTHESIS_SURVIVAL_NAMES
        ))
        self.assertTrue(all(
            comparison[name]["validation"] == "not supported"
            for name in HYPOTHESIS_SURVIVAL_NAMES
        ))
        self.assertTrue(all(
            not comparison[name]["survived"]
            for name in HYPOTHESIS_SURVIVAL_NAMES
        ))

    def test_insufficient_evidence_remains_explicit_in_rendered_report(self):
        study = OutOfSampleValidationStudy()
        research = study.hypothesis_results(
            computed_hypothesis_group((1.0, 2.0), 100.0)
        )
        validation = study.hypothesis_results(
            computed_hypothesis_group(
                (1.0, 2.0),
                100.0,
                signal_count=MIN_CALIBRATION_SIGNALS,
            )
        )
        comparison = study._compare_hypotheses(research, validation)

        sparse_name = "rsi_60_vs_50_59"
        self.assertEqual(research[sparse_name]["status"], "supported")
        self.assertEqual(
            validation[sparse_name]["status"],
            "insufficient evidence",
        )
        self.assertEqual(comparison[sparse_name]["research"], "supported")
        self.assertEqual(
            comparison[sparse_name]["validation"],
            "insufficient evidence",
        )
        self.assertFalse(comparison[sparse_name]["survived"])

        results = report_fixture()
        results.update({
            "research_hypotheses": research,
            "validation_hypotheses": validation,
            "hypotheses": comparison,
        })
        report = render(print_report, results)

        self.assertIn(
            "rsi_60_vs_50_59: research=supported, "
            "validation=insufficient evidence, survived=NO",
            report,
        )

    def test_empty_validation_group_renders_insufficient_evidence(self):
        study = OutOfSampleValidationStudy()
        results = report_fixture()
        results["validation"] = study.analyze_group([], [])
        results["validation_hypotheses"] = study.hypothesis_results(
            results["validation"]
        )
        results["hypotheses"] = {
            name: {
                **result,
                "validation": "insufficient evidence",
                "survived": False,
            }
            for name, result in results["hypotheses"].items()
        }
        results["split"].update({
            "validation_start": None,
            "validation_end": None,
            "validation_periods": 0,
            "validation_candles": 0,
        })

        report = render(print_report, results)

        self.assertIn(
            "Insufficient evidence: no study-period results or BUY signals",
            report,
        )
        self.assertIn("Validation: unavailable (0 periods, 0 candles)", report)
        for name in HYPOTHESIS_SURVIVAL_NAMES:
            self.assertIn(
                f"{name}: research=supported, "
                "validation=insufficient evidence, survived=NO",
                report,
            )

    def test_empty_research_group_renders_insufficient_evidence(self):
        study = OutOfSampleValidationStudy()
        results = report_fixture()
        results["research"] = study.analyze_group([], [])
        results["research_hypotheses"] = study.hypothesis_results(
            results["research"]
        )
        results["hypotheses"] = {
            name: {
                **result,
                "research": "insufficient evidence",
                "survived": False,
            }
            for name, result in results["hypotheses"].items()
        }
        results["split"].update({
            "research_start": None,
            "research_end": None,
            "research_periods": 0,
            "research_candles": 0,
        })

        report = render(print_report, results)

        self.assertIn(
            "Insufficient evidence: no study-period results or BUY signals",
            report,
        )
        self.assertIn("Research: unavailable (0 periods, 0 candles)", report)
        for name in HYPOTHESIS_SURVIVAL_NAMES:
            self.assertEqual(
                results["research_hypotheses"][name]["status"],
                "insufficient evidence",
            )
            self.assertIn(
                f"{name}: research=insufficient evidence, "
                f"validation={results['hypotheses'][name]['validation']}, "
                "survived=NO",
                report,
            )

if __name__ == "__main__":
    unittest.main()
