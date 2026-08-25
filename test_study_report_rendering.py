import io
import unittest
from contextlib import redirect_stdout

from cost_viability_study import print_report as print_cost_report
from counterfactual_exit_study import print_report as print_counterfactual_report
from exit_economics_study import print_report as print_exit_economics_report
from exit_parameter_period_robustness_study import print_report as print_exit_period_report
from pre_stop_market_state_study import print_report as print_pre_stop_report
from rsi_candidate_robustness_study import print_report as print_rsi_report
from score_effectiveness_study import print_report as print_score_report
from stop_loss_recovery_study import print_report as print_recovery_report
from strategy_calibration_study import print_report as print_calibration_report
from strategy_candidate_study import print_report as print_candidate_report
from strategy_diagnostic_study import print_report as print_diagnostic_report
from trade_economics_study import print_report as print_trade_economics_report
from trade_filter_candidate_study import print_report as print_filter_report
from trade_path_exit_timing_study import print_report as print_path_report


# Included by offline_study_report_check.py; network/live-data tests remain
# intentionally unmarked and outside that runner.
REPORT_REGRESSION_MODULE = True


class MissingValue:
    def __getitem__(self, key):
        return self

    def items(self):
        return ()

    def values(self):
        return ()

    def __iter__(self):
        return iter(())

    def __bool__(self):
        return False

    def __format__(self, spec):
        return format(0.0, spec)

    def __str__(self):
        return "0"


class RenderFixture(dict):
    """Deterministic empty result shape for exercising report formatting.

    Reports receive a mapping with the same nested lookup behavior as their
    production result. Empty iterables keep the fixture small while still
    running every report section and its formatting statements.
    """

    def __missing__(self, key):
        return MissingValue()


def fixture():
    return RenderFixture()


def split():
    return {
        "research_start": "2020-01-01",
        "research_end": "2020-01-02",
        "research_periods": 1,
        "research_candles": 2,
        "validation_start": "2020-01-03",
        "validation_end": "2020-01-04",
        "validation_periods": 1,
        "validation_candles": 2,
    }


def render(report, result):
    output = io.StringIO()
    with redirect_stdout(output):
        report(result)
    return output.getvalue()


class StudyReportRenderingTests(unittest.TestCase):
    def test_cost_viability_report_has_boundary(self):
        result = {"split": split(), "research": fixture(), "validation": fixture()}
        report = render(print_cost_report, result)
        self.assertIn("BTC/CAD COST VIABILITY & GROSS EDGE STUDY", report)
        self.assertIn("=== Research ===", report)
        self.assertIn("=== Untouched validation ===", report)
        self.assertIn("=== Interpretation boundary ===", report)

    def test_counterfactual_report_has_scenario_section(self):
        report = render(print_counterfactual_report, {
            "Period A": {
                "start_date": "2020-01-01",
                "end_date": "2020-01-02",
                "candles": 2,
                "market_return": 1.0,
                "scenarios": {},
                "losing_trade_diagnostics": [],
            }
        })
        self.assertIn("COUNTERFACTUAL EXIT STUDY — ANALYSIS ONLY", report)
        self.assertIn("Scenario results:", report)
        self.assertIn("Original losing-trade diagnostics:", report)

    def test_exit_economics_report_has_boundary(self):
        result = {"split": split(), "research": fixture(), "validation": fixture()}
        report = render(print_exit_economics_report, result)
        self.assertIn("BTC/CAD EXIT ECONOMICS — STEP 13 ANALYSIS ONLY", report)
        self.assertIn("=== Research ===", report)
        self.assertIn("=== Untouched validation ===", report)
        self.assertIn("=== Interpretation boundary ===", report)

    def test_exit_parameter_period_report_has_boundary(self):
        result = {
            "note": "Deterministic report fixture.",
            "research": fixture(),
            "validation": fixture(),
            "additional": fixture(),
            "outcome": {
                "label": "INTERESTING BUT UNPROVEN",
                "color": "yellow",
                "reason": "Fixture outcome.",
            },
        }
        report = render(print_exit_period_report, result)
        self.assertIn("BTC/CAD 6% TARGET PERIOD ROBUSTNESS STUDY — ANALYSIS ONLY", report)
        self.assertIn("=== Research per-period comparison ===", report)
        self.assertIn("=== Untouched validation regime comparison ===", report)
        self.assertIn("=== Additional untouched history aggregate comparison ===", report)
        self.assertIn("=== Overall outcome ===", report)
        self.assertIn("INTERESTING BUT UNPROVEN (yellow)", report)
        self.assertIn("=== Interpretation boundary ===", report)

    def test_pre_stop_report_has_boundary(self):
        result = {"research": {"summary": fixture()}, "validation": {"summary": fixture()}}
        report = render(print_pre_stop_report, result)
        self.assertIn("BTC/CAD PRE-STOP MARKET-STATE STUDY", report)
        self.assertIn("=== Research ===", report)
        self.assertIn("=== Untouched validation ===", report)
        self.assertIn("=== Interpretation boundary ===", report)

    def test_rsi_report_has_threshold_and_boundary(self):
        result = {"research": fixture(), "validation": fixture()}
        report = render(print_rsi_report, result)
        self.assertIn("BTC/CAD RSI CANDIDATE ROBUSTNESS STUDY", report)
        self.assertIn("=== Research ===", report)
        self.assertIn("=== Threshold stability ===", report)
        self.assertIn("=== Interpretation boundary ===", report)

    def test_score_report_has_each_regime_section(self):
        result = {
            "periods": [],
            "by_regime": {
                "Bull": {
                    "valid_evaluations": 0,
                    "periods": 0,
                    "insufficient_period_coverage": False,
                    "buckets": {},
                },
                "Bear": {
                    "valid_evaluations": 0,
                    "periods": 0,
                    "insufficient_period_coverage": False,
                    "buckets": {},
                },
                "Sideways": {
                    "valid_evaluations": 0,
                    "periods": 0,
                    "insufficient_period_coverage": False,
                    "buckets": {},
                },
            },
        }
        report = render(print_score_report, result)
        self.assertIn("STRATEGY SCORE EFFECTIVENESS STUDY", report)
        for regime in ("Bull", "Bear", "Sideways"):
            self.assertIn(f"=== {regime} periods: none ===", report)

    def test_stop_loss_recovery_report_has_boundary(self):
        result = {"split": split(), "research": fixture(), "validation": fixture()}
        report = render(print_recovery_report, result)
        self.assertIn("BTC/CAD STOP-LOSS RECOVERY & FALSE-STOP STUDY", report)
        self.assertIn("=== Research ===", report)
        self.assertIn("=== Untouched validation ===", report)
        self.assertIn("=== Interpretation boundary ===", report)

    def test_strategy_calibration_report_has_diagnosis(self):
        result = {
            "signal_count": 0,
            "completed_entry_count": 0,
            "score_bands": {},
            "rsi_bands": {},
            "condition_combinations": {},
            "early_movement": {},
            "cost_break_even": fixture(),
            "diagnosis": fixture(),
        }
        report = render(print_calibration_report, result)
        self.assertIn("BTC/CAD SCORE / ENTRY CALIBRATION STUDY", report)
        self.assertIn("=== Condition combinations ===", report)
        self.assertIn("=== Cost break-even ===", report)
        self.assertIn("=== Final entry-quality diagnosis ===", report)

    def test_strategy_candidate_report_has_control_section(self):
        result = {
            "split": split(),
            "control": {"research": fixture()},
            "candidates": {},
            "candidate_definitions": {},
            "comparisons": {},
        }
        report = render(print_candidate_report, result)
        self.assertIn("BTC/CAD CONTROLLED STRATEGY IMPROVEMENT", report)
        self.assertIn("CONTROL · research", report)

    def test_strategy_diagnostic_report_has_final_diagnosis(self):
        result = {
            "periods": [],
            "by_regime": {},
            "score_effectiveness": {},
            "condition_effectiveness": {},
            "entry_timing": {},
            "exit_behavior": {},
            "mfe_mae": {},
            "cost_sensitivity": {},
            "diagnosis": {
                "primary_findings": [],
                "evidence": [],
                "insufficient_evidence": [],
            },
        }
        report = render(print_diagnostic_report, result)
        self.assertIn("BTC/CAD STRATEGY DIAGNOSTIC STUDY", report)
        self.assertIn("=== Period coverage ===", report)
        self.assertIn("=== Cost sensitivity ===", report)
        self.assertIn("=== Final diagnosis ===", report)

    def test_trade_economics_report_has_final_verdict(self):
        result = {
            "split": split(),
            "research": fixture(),
            "validation": fixture(),
            "research_diagnosis": fixture(),
            "validation_diagnosis": fixture(),
        }
        report = render(print_trade_economics_report, result)
        self.assertIn("BTC/CAD TRADE ECONOMICS & OPPORTUNITY QUALITY", report)
        self.assertIn("=== Research → validation consistency ===", report)
        self.assertIn("=== Final verdict ===", report)

    def test_trade_filter_report_has_boundary(self):
        result = {
            "research_control": fixture(),
            "research": {},
            "validation_control": fixture(),
            "validation": {},
        }
        report = render(print_filter_report, result)
        self.assertIn("BTC/CAD TRADE-FILTER CANDIDATE SCREEN", report)
        self.assertIn("=== Research ===", report)
        self.assertIn("=== Untouched validation ===", report)
        self.assertIn("=== Interpretation boundary ===", report)

    def test_trade_path_report_has_boundary(self):
        result = {
            "research": {"summary": fixture()},
            "validation": {"summary": fixture()},
        }
        report = render(print_path_report, result)
        self.assertIn("BTC/CAD TRADE PATH & EXIT TIMING STUDY", report)
        self.assertIn("=== Research ===", report)
        self.assertIn("=== Validation ===", report)
        self.assertIn("=== Interpretation boundary ===", report)


if __name__ == "__main__":
    unittest.main()