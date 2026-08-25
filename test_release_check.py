import unittest
import json
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.error import URLError
from unittest.mock import patch

from release_check import (
    CheckResult,
    check_published_pwa_assets,
    check_workflow_action_pins,
    check_workflow_lint_container_pin,
    check_workflow_uv_version,
    format_json,
    format_summary,
    main,
    run_check,
)


class ReleaseCheckTests(unittest.TestCase):
    def test_run_check_terminates_timed_out_process_group(self):
        result = run_check(
            "bounded fixture",
            (
                sys.executable,
                "-c",
                "import subprocess, sys, time; "
                "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)']); "
                "time.sleep(30)",
            ),
            timeout=0.1,
        )

        self.assertEqual(result.returncode, 124)
        self.assertIn("exceeded its 0.1-second timeout", result.output)

    @patch("release_check.os.killpg", side_effect=OSError("group cleanup unavailable"))
    def test_run_check_reports_cleanup_failure_and_uses_process_fallback(self, _killpg):
        result = run_check(
            "cleanup fallback fixture",
            (sys.executable, "-c", "import time; time.sleep(30)"),
            timeout=0.1,
        )

        self.assertEqual(result.returncode, 124)
        self.assertIn("cleanup failed while terminating process group", result.output)

    def test_workflow_lint_container_pin_requires_sha256_digest(self):
        pinned_workflow = "      rhysd/actionlint:1.7.7@sha256:" + "a" * 64 + "\n"
        mutable_workflow = "      rhysd/actionlint:1.7.7\n"

        result = check_workflow_lint_container_pin()
        self.assertEqual(result.returncode, 0)

        with NamedTemporaryFile(mode="w", encoding="utf-8") as workflow_file:
            workflow_file.write(pinned_workflow)
            workflow_file.flush()
            result = check_workflow_lint_container_pin(Path(workflow_file.name))
            self.assertEqual(result.returncode, 0)

            workflow_file.seek(0)
            workflow_file.truncate()
            workflow_file.write(mutable_workflow)
            workflow_file.flush()
            result = check_workflow_lint_container_pin(Path(workflow_file.name))
            self.assertEqual(result.returncode, 1)
            self.assertIn("sha256 digest", result.output)

    def test_workflow_action_pins_require_immutable_commit_shas(self):
        pinned_workflow = "      - uses: actions/checkout@" + "a" * 40 + "\n"
        mutable_workflow = "      - uses: actions/checkout@v4\n"

        with NamedTemporaryFile(mode="w", encoding="utf-8") as workflow_file:
            workflow_file.write(pinned_workflow)
            workflow_file.flush()
            result = check_workflow_action_pins(Path(workflow_file.name))
            self.assertEqual(result.returncode, 0)
            self.assertIn("1 GitHub action references", result.output)

            workflow_file.seek(0)
            workflow_file.truncate()
            workflow_file.write(mutable_workflow)
            workflow_file.flush()
            result = check_workflow_action_pins(Path(workflow_file.name))
            self.assertEqual(result.returncode, 1)
            self.assertIn("immutable 40-character commit SHAs", result.output)

    def test_workflow_uv_version_requires_reviewed_exact_version(self):
        pinned_workflow = (
            "      - uses: astral-sh/setup-uv@"
            + "a" * 40
            + "\n"
            "        with:\n"
            "          version: 0.9.24\n"
        )
        mutable_workflow = pinned_workflow.replace("0.9.24", "latest")
        missing_workflow = pinned_workflow.replace("version: 0.9.24", "foo: bar")
        empty_workflow = "name: workflow without setup-uv\n"

        with NamedTemporaryFile(mode="w", encoding="utf-8") as workflow_file:
            workflow_file.write(pinned_workflow)
            workflow_file.flush()
            result = check_workflow_uv_version(Path(workflow_file.name))
            self.assertEqual(result.returncode, 0)
            self.assertIn("reviewed uv 0.9.24", result.output)

            workflow_file.seek(0)
            workflow_file.truncate()
            workflow_file.write(mutable_workflow)
            workflow_file.flush()
            result = check_workflow_uv_version(Path(workflow_file.name))
            self.assertEqual(result.returncode, 1)
            self.assertIn("reviewed uv version 0.9.24", result.output)
            self.assertIn("'latest'", result.output)

            workflow_file.seek(0)
            workflow_file.truncate()
            workflow_file.write(missing_workflow)
            workflow_file.flush()
            result = check_workflow_uv_version(Path(workflow_file.name))
            self.assertEqual(result.returncode, 1)
            self.assertIn("must declare a version", result.output)

            workflow_file.seek(0)
            workflow_file.truncate()
            workflow_file.write(empty_workflow)
            workflow_file.flush()
            result = check_workflow_uv_version(Path(workflow_file.name))
            self.assertEqual(result.returncode, 1)
            self.assertIn("Expected at least one setup-uv step", result.output)

    def test_scheduled_browser_drift_notification_contract(self):
        workflow = (
            Path(__file__).resolve().parent
            / ".github"
            / "workflows"
            / "webkit-browser.yml"
        ).read_text(encoding="utf-8")
        notification_job = workflow.split(
            "  scheduled-browser-drift-notification:\n", maxsplit=1
        )[1]
        notification_lines = notification_job.splitlines()
        next_job_index = next(
            (
                index
                for index, line in enumerate(notification_lines)
                if index > 0
                and line.startswith("  ")
                and not line.startswith("    ")
            ),
            len(notification_lines),
        )
        notification_job = "\n".join(notification_lines[:next_job_index])

        self.assertIn("github.event_name == 'schedule'", notification_job)
        self.assertIn("always()", notification_job)

        needs_block = notification_job.split("    needs:\n", maxsplit=1)[1].split(
            "    runs-on:", maxsplit=1
        )[0]
        self.assertEqual(
            {
                line.strip()[2:]
                for line in needs_block.splitlines()
                if line.strip().startswith("- ")
            },
            {
                "workflow-lint",
                "release-gate",
                "webkit-dashboard",
                "orbit-visual-regression",
                "firefox-orbit-visual-regression",
            },
        )

        self.assertIn(
            "BTC_CAD_PREFLIGHT_SLACK_WEBHOOK_URL: "
            "${{ secrets.BTC_CAD_PREFLIGHT_SLACK_WEBHOOK_URL }}",
            notification_job,
        )
        run_body = notification_job.split("        run: |\n", maxsplit=1)[1]
        self.assertNotIn("secrets.BTC_CAD_PREFLIGHT_SLACK_WEBHOOK_URL", run_body)
        self.assertNotIn("BTC_CAD_PREFLIGHT_SLACK_WEBHOOK_URL", run_body)
        self.assertIn("GITHUB_RUN_URL", notification_job)
        self.assertIn("GITHUB_ARTIFACTS_URL", notification_job)
        self.assertIn('"run_url": os.environ["GITHUB_RUN_URL"]', notification_job)
        self.assertIn(
            '"artifacts_url": os.environ["GITHUB_ARTIFACTS_URL"]', notification_job
        )

    def test_summary_keeps_check_names_statuses_and_details(self):
        summary = format_summary(
            (
                CheckResult("Offline report coverage", 1, "missing report module"),
                CheckResult("BTC/CAD Yahoo live-data preflight", 0, "validated candles"),
            )
        )

        self.assertIn("[FAIL] Offline report coverage (exit 1)", summary)
        self.assertIn("missing report module", summary)
        self.assertIn("[PASS] BTC/CAD Yahoo live-data preflight (exit 0)", summary)
        self.assertIn("validated candles", summary)

    def test_json_result_keeps_checks_separate_with_status_and_details(self):
        rendered = format_json(
            (
                CheckResult("Offline report coverage", 1, "coverage failed"),
                CheckResult("BTC/CAD Yahoo live-data preflight", 0, "network passed"),
            )
        )
        self.assertEqual(
            json.loads(rendered),
            {
                "status": "fail",
                "checks": [
                    {
                        "name": "Offline report coverage",
                        "status": "fail",
                        "exit_code": 1,
                        "details": "coverage failed",
                    },
                    {
                        "name": "BTC/CAD Yahoo live-data preflight",
                        "status": "pass",
                        "exit_code": 0,
                        "details": "network passed",
                    },
                ],
            },
        )

    def test_published_pwa_check_requires_deployment_metadata(self):
        result = check_published_pwa_assets(None)

        self.assertEqual(result.returncode, 1)
        self.assertIn("--published-url", result.output)
        self.assertIn("PUBLISHED_DASHBOARD_URL", result.output)

    @patch("release_check.run_check")
    def test_local_allow_flag_reports_skip_without_failing(self, run_check):
        run_check.side_effect = (
            CheckResult("Full regression suite", 0, "tests passed"),
            CheckResult("Offline report coverage", 0, "coverage passed"),
            CheckResult("BTC/CAD Yahoo live-data preflight", 0, "network passed"),
            CheckResult("API restart and health check", 0, "restart passed"),
        )

        with patch("builtins.print") as print_mock:
            exit_code = main(["--allow-missing-published-url"])

        self.assertEqual(exit_code, 0)
        rendered = print_mock.call_args.args[0]
        self.assertIn("SKIP:", rendered)
        self.assertIn("Hosted asset validation remains required", rendered)

    @patch("release_check.run_check")
    def test_json_flag_emits_only_machine_readable_output(self, run_check):
        run_check.side_effect = (
            CheckResult("Full regression suite", 0, "tests passed"),
            CheckResult("Offline report coverage", 0, "coverage passed"),
            CheckResult("BTC/CAD Yahoo live-data preflight", 0, "network passed"),
            CheckResult("API restart and health check", 0, "restart passed"),
        )

        with patch("builtins.print") as print_mock:
            exit_code = main(["--json", "--published-url", "https://dashboard.example"])

        self.assertEqual(exit_code, 0)
        self.assertEqual(json.loads(print_mock.call_args.args[0])["status"], "pass")

    @patch("release_check.urlopen")
    def test_published_pwa_check_verifies_status_and_content_types(self, urlopen):
        responses = []
        for content_type in (
            "application/manifest+json",
            "image/png",
            "image/png",
            "image/png",
            "image/png",
        ):
            response = unittest.mock.MagicMock()
            response.status = 200
            response.getcode.return_value = 200
            response.headers.get_content_type.return_value = content_type
            response.__enter__.return_value = response
            responses.append(response)
        urlopen.side_effect = responses

        result = check_published_pwa_assets("https://dashboard.example", timeout=3)

        self.assertEqual(result.returncode, 0)
        self.assertEqual(urlopen.call_count, 5)
        self.assertEqual(
            [call.args[0].full_url for call in urlopen.call_args_list],
            [
                "https://dashboard.example/app/static/manifest.json",
                "https://dashboard.example/app/static/kova/kova-pwa-192.png",
                "https://dashboard.example/app/static/kova/kova-pwa-512.png",
                "https://dashboard.example/app/static/kova/apple-touch-icon.png",
                "https://dashboard.example/app/static/kova/favicon.png",
            ],
        )

    @patch("release_check.urlopen")
    def test_published_pwa_check_reports_stale_url_and_bad_content_type(self, urlopen):
        response = unittest.mock.MagicMock()
        response.status = 200
        response.headers.get_content_type.return_value = "text/html"
        response.__enter__.return_value = response
        urlopen.side_effect = [
            response,
            URLError("connection refused"),
            response,
            response,
            response,
        ]

        result = check_published_pwa_assets("https://stale.example")

        self.assertEqual(result.returncode, 1)
        self.assertIn("kova-pwa-192.png", result.output)
        self.assertIn("expected Content-Type", result.output)
        self.assertIn("could not reach published deployment", result.output)

    @patch("release_check.check_published_pwa_assets")
    @patch("release_check.run_check")
    def test_api_restart_failure_fails_release_with_actionable_details(
        self, run_check, pwa_check
    ):
        run_check.side_effect = (
            CheckResult("Full regression suite", 0, "tests passed"),
            CheckResult("Offline report coverage", 0, "coverage passed"),
            CheckResult("BTC/CAD Yahoo live-data preflight", 0, "network passed"),
            CheckResult(
                "API restart and health check",
                1,
                "API did not become healthy within 30000ms",
            ),
        )
        pwa_check.return_value = CheckResult(
            "Published PWA asset smoke check", 0, "hosted assets passed"
        )

        with patch("builtins.print") as print_mock:
            exit_code = main(["--published-url", "https://dashboard.example"])

        self.assertEqual(exit_code, 1)
        rendered = print_mock.call_args.args[0]
        self.assertIn("[FAIL] API restart and health check", rendered)
        self.assertIn("API did not become healthy", rendered)

    @patch("release_check.check_published_pwa_assets")
    @patch("release_check.run_check")
    def test_offline_failure_is_not_hidden_by_live_data_success(
        self, run_check, pwa_check
    ):
        run_check.side_effect = (
            CheckResult("Full regression suite", 0, "tests passed"),
            CheckResult("Offline report coverage", 1, "coverage failed"),
            CheckResult("BTC/CAD Yahoo live-data preflight", 0, "network passed"),
            CheckResult("API restart and health check", 0, "restart passed"),
        )
        pwa_check.return_value = CheckResult(
            "Published PWA asset smoke check", 0, "hosted assets passed"
        )

        with patch("builtins.print") as print_mock:
            exit_code = main(["--published-url", "https://dashboard.example"])

        self.assertEqual(exit_code, 1)
        rendered = print_mock.call_args.args[0]
        self.assertIn("coverage failed", rendered)
        self.assertIn("network passed", rendered)

    @patch("release_check.check_published_pwa_assets")
    @patch("release_check.run_check")
    def test_json_flag_emits_only_machine_readable_output(self, run_check, pwa_check):
        run_check.side_effect = (
            CheckResult("Full regression suite", 0, "tests passed"),
            CheckResult("Offline report coverage", 0, "coverage passed"),
            CheckResult("BTC/CAD Yahoo live-data preflight", 0, "network passed"),
            CheckResult("API restart and health check", 0, "restart passed"),
        )
        pwa_check.return_value = CheckResult(
            "Published PWA asset smoke check", 0, "hosted assets passed"
        )

        with patch("builtins.print") as print_mock:
            exit_code = main(["--json", "--published-url", "https://dashboard.example"])

        self.assertEqual(exit_code, 0)
        self.assertEqual(json.loads(print_mock.call_args.args[0])["status"], "pass")


if __name__ == "__main__":
    unittest.main()
