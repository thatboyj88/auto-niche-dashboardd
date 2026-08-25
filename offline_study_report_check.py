"""Run the offline smoke checks for user-visible study reports."""

import ast
from pathlib import Path
import sys
import unittest


REPORT_REGRESSION_MARKER = "REPORT_REGRESSION_MODULE"


def discover_report_regression_modules() -> tuple[str, ...]:
    """Find test modules that declare offline report regression coverage."""
    project_root = Path(__file__).resolve().parent
    modules = []
    for test_file in sorted(project_root.glob("test_*.py")):
        tree = ast.parse(test_file.read_text(encoding="utf-8"), test_file.name)
        if any(
            isinstance(node, (ast.Assign, ast.AnnAssign))
            and (
                isinstance(node, ast.AnnAssign)
                and isinstance(node.target, ast.Name)
                or isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) for target in node.targets)
            )
            and isinstance(node.value, ast.Constant)
            and node.value.value is True
            and (
                isinstance(node, ast.AnnAssign)
                and isinstance(node.target, ast.Name)
                and node.target.id == REPORT_REGRESSION_MARKER
                or isinstance(node, ast.Assign)
                and any(
                    isinstance(target, ast.Name)
                    and target.id == REPORT_REGRESSION_MARKER
                    for target in node.targets
                )
            )
            for node in tree.body
        ):
            modules.append(test_file.stem)
    return tuple(modules)


def validate_report_test_coverage(report_test_modules: tuple[str, ...]) -> None:
    """Fail if marked report regressions drift out of the offline runner."""
    discovered = set(discover_report_regression_modules())
    configured = set(report_test_modules)
    missing = sorted(discovered - configured)
    if missing:
        raise RuntimeError(
            "Offline report smoke coverage is missing marked regression "
            f"module(s): {', '.join(missing)}. "
            "Only deterministic in-memory report tests are marked; "
            "Yahoo/network and live-data tests intentionally remain "
            "outside this runner."
        )


def main() -> int:
    """Run report tests and return a release-friendly exit status."""
    # Keep this list explicit and offline-only: marked report regressions use
    # deterministic, in-memory candles and fixtures. Preflight, historical
    # backtesting, and other live-data tests are intentionally not marked or
    # included because they may exercise Yahoo/network boundaries.
    report_test_modules = (
        "test_cost_viability_study",
        "test_counterfactual_exit_study",
        "test_exit_capture_study",
        "test_exit_economics_study",
        "test_exit_parameter_period_robustness_study",
        "test_out_of_sample_validation",
        "test_pre_stop_market_state_study",
        "test_score_effectiveness_study",
        "test_stop_loss_recovery_study",
        "test_strategy_calibration_study",
        "test_strategy_candidate_study",
        "test_strategy_diagnostic_study",
        "test_study_report_rendering",
        "test_trade_economics_study",
        "test_trade_filter_candidate_study",
        "test_trade_path_exit_timing_study",
    )
    try:
        validate_report_test_coverage(report_test_modules)
    except RuntimeError as error:
        print(error, file=sys.stderr)
        return 1

    suite = unittest.TestSuite(
        unittest.defaultTestLoader.loadTestsFromName(module)
        for module in report_test_modules
    )
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())