import unittest
from types import SimpleNamespace
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

import dashboard
from testing_center import (
    BLOCKED,
    FAIL,
    NOT_RUN,
    NOT_CONFIGURED,
    PASS,
    diagnostic_registry,
    run_diagnostics,
    run_pre_live_validation,
)


def render_testing_center_fixture():
    import dashboard
    from types import SimpleNamespace

    dashboard.render_mc_testing_center(
        SimpleNamespace(health={"status": "HEALTHY"}), []
    )


class TestingCenterTests(unittest.TestCase):
    def test_registry_is_explicit_and_read_only(self):
        registry = diagnostic_registry()
        self.assertEqual(len(registry), 8)
        self.assertEqual({item["name"] for item in registry}, {
            "Authentication boundary", "Dashboard routes", "Market provider",
            "Paper mode safety", "State recovery", "Evidence integrity",
            "API health", "Broker readiness",
        })

    def test_complete_safe_snapshot_reports_expected_states(self):
        results = run_diagnostics({
            "authenticated": True,
            "routes": ("OVERVIEW", "SYSTEM"),
            "routes_valid": True,
            "market_health": {"status": "HEALTHY"},
            "paper_trading": True,
            "live_trading": False,
            "observation_status": "RUNNING",
            "evidence_reconciled": True,
            "api_status": "HEALTHY",
        })
        self.assertEqual(
            [result["status"] for result in results],
            [PASS] * 7 + [BLOCKED],
        )
        self.assertTrue(all(result["safety"] == "READ_ONLY" for result in results))

    def test_missing_snapshots_fail_closed_without_fabrication(self):
        results = run_diagnostics()
        statuses = {result["name"]: result["status"] for result in results}
        self.assertEqual(statuses["Authentication boundary"], BLOCKED)
        self.assertEqual(statuses["State recovery"], NOT_RUN)
        self.assertEqual(statuses["Evidence integrity"], NOT_RUN)
        self.assertEqual(statuses["API health"], BLOCKED)
        self.assertEqual(statuses["Broker readiness"], BLOCKED)

    def test_pre_live_report_distinguishes_missing_evidence_from_unsafe_config(self):
        report = run_pre_live_validation({
            "paper_trading": True,
            "live_trading": False,
            "api_contract_valid": True,
        })
        statuses = {item["name"]: item["status"] for item in report["checks"]}
        self.assertEqual(report["status"], BLOCKED)
        self.assertEqual(statuses["Market provider"], NOT_CONFIGURED)
        self.assertEqual(statuses["Dashboard routes"], NOT_CONFIGURED)
        self.assertEqual(statuses["Broker readiness"], BLOCKED)
        self.assertFalse(report["safety"]["mutation_attempted"])
        self.assertFalse(report["safety"]["broker_contacted"])

    def test_failure_fixtures_are_isolated_and_repeatable(self):
        for fixture, expected in (
            ("stale_data", FAIL),
            ("provider_outage", FAIL),
            ("duplicate_execution", PASS),
            ("recovery", PASS),
            ("veto", BLOCKED),
        ):
            first = run_diagnostics(fixture=fixture)
            second = run_diagnostics(fixture=fixture)
            self.assertEqual(first[-1]["status"], expected)
            self.assertEqual(first[-1]["detail"], second[-1]["detail"])
            self.assertIn("isolated", first[-1]["detail"].lower())
            self.assertNotIn("write", first[-1]["detail"].lower())

    def test_paper_risk_stress_suite_isolated_and_paper_only(self):
        first = run_pre_live_validation(
            {
                "paper_trading": True,
                "live_trading": False,
            },
            fixture="risk_stress",
        )
        second = run_pre_live_validation(
            {
                "paper_trading": True,
                "live_trading": False,
            },
            fixture="risk_stress",
        )
        statuses = {item["name"]: item["status"] for item in first["checks"]}
        self.assertEqual(statuses["Stress: daily drawdown guard"], PASS)
        self.assertEqual(statuses["Stress: daily trade cap"], PASS)
        self.assertEqual(statuses["Stress: provider outage rejection"], PASS)
        self.assertEqual(statuses["Stress: temporary state recovery"], PASS)
        self.assertEqual(statuses["Stress: Risk Governor veto"], PASS)
        self.assertEqual(statuses["Paper-only exchange gate"], BLOCKED)
        self.assertEqual(first["status"], BLOCKED)
        self.assertEqual(
            [item["detail"] for item in first["checks"] if item["name"].startswith("Stress:")],
            [item["detail"] for item in second["checks"] if item["name"].startswith("Stress:")],
        )
        self.assertFalse(first["safety"]["mutation_attempted"])
        self.assertFalse(first["safety"]["broker_contacted"])

    def test_dashboard_center_is_repeatable_and_exposes_safe_run_control(self):
        with (
            patch("dashboard._authenticated_user_key", return_value=None),
            patch("dashboard.load_live_observation_status", return_value={
                "available": False,
            }),
        ):
            app = AppTest.from_function(render_testing_center_fixture).run()
            self.assertEqual(app.button[0].label, "Run read-only checks")
            self.assertIn("No run yet", "\n".join(item.value for item in app.caption))
            app.button[0].click().run()
            rendered = "\n".join(item.value for item in app.caption)
            self.assertNotIn("Authenticated session detected", rendered)
            self.assertTrue(any("Broker readiness" in item.value for item in app.markdown))


if __name__ == "__main__":
    unittest.main()