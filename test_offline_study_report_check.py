import unittest

from offline_study_report_check import (
    discover_report_regression_modules,
    validate_report_test_coverage,
)


class OfflineStudyReportCheckTests(unittest.TestCase):
    def test_missing_marked_modules_are_named_in_failure(self):
        with self.assertRaises(RuntimeError) as context:
            validate_report_test_coverage(("test_configured_report",))
        message = str(context.exception)
        self.assertIn("missing marked regression module(s):", message)
        for module in discover_report_regression_modules():
            self.assertIn(module, message)


if __name__ == "__main__":
    unittest.main()