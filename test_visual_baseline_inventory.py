import tempfile
import unittest
from pathlib import Path

from visual_baseline_inventory import (
    BASELINE_MANIFEST,
    discover_snapshot_tests,
    validate_inventory,
)


class VisualBaselineInventoryTests(unittest.TestCase):
    def test_current_snapshot_suite_has_a_reviewed_baseline_for_every_case(self):
        errors = validate_inventory()
        self.assertEqual(errors, [])
        self.assertEqual(len(discover_snapshot_tests()), len(BASELINE_MANIFEST))

    def test_missing_baseline_reports_exact_test_and_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            baseline_dir = Path(temp_dir)
            for filename in next(iter(BASELINE_MANIFEST.values())):
                (baseline_dir / filename).touch()
            errors = validate_inventory(
                baseline_dir=baseline_dir,
                test_source=Path("test_dashboard_ai_operations_assistant.py"),
            )
        self.assertTrue(any(error.startswith("MISSING baseline: test=") for error in errors))
        self.assertTrue(any("orbit-summary-firefox-desktop.png" in error for error in errors))

    def test_unexpected_baseline_reports_exact_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            baseline_dir = Path(temp_dir)
            for filenames in BASELINE_MANIFEST.values():
                for filename in filenames:
                    (baseline_dir / filename).touch()
            unexpected = baseline_dir / "unreviewed.png"
            unexpected.touch()
            errors = validate_inventory(baseline_dir=baseline_dir)
        self.assertEqual(errors, [f"UNEXPECTED baseline: file={unexpected}"])

    def test_new_snapshot_test_cannot_be_added_without_manifest_entry(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "browser_tests.py"
            source.write_text(
                "class DashboardAssistantBrowserTests:\n"
                "    def test_new_snapshot(self):\n"
                "        self._run_orbit_summary_snapshot_comparison()\n",
                encoding="utf-8",
            )
            errors = validate_inventory(test_source=source)
        self.assertIn(
            "UNREGISTERED snapshot test: test=test_new_snapshot",
            errors,
        )


if __name__ == "__main__":
    unittest.main()