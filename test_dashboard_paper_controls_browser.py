import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from playwright.sync_api import sync_playwright

from test_dashboard_ai_operations_assistant import (
    _browser_dashboard_wrapper,
    _cleanup_fixture_process,
    _dashboard_wait,
    _find_free_port,
    _navigate_fixture_until_ready,
    _start_bounded_output_reader,
    _fixture_output,
)


def _paper_controls_wrapper():
    """Use the full dashboard fixture with isolated paper-control state."""
    source = _browser_dashboard_wrapper()
    return source.replace(
        'patch("dashboard.get_provider_health", return_value=provider_health),\n',
        'patch("dashboard.get_provider_health", return_value=provider_health),\n'
    ).replace(
        '    patch(\n'
        '        "dashboard._authenticated_user_key",\n'
        '        side_effect=test_authenticated_user_key,\n'
        '    ),',
        '    patch(\n'
        '        "dashboard._authenticated_user_key",\n'
        '        return_value=(\n'
        '            "sub:browser-fixture"\n'
        '            if os.environ.get("DASHBOARD_TEST_AUTH") == "1"\n'
        '            else None\n'
        '        ),\n'
        '    ),',
    )


@unittest.skipIf(
    os.environ.get("RELEASE_CHECK_SKIP_BROWSER") == "1",
    "Browser regressions run in dedicated release workflow jobs.",
)
class PaperControlsBrowserTests(unittest.TestCase):
    def _open_controls(self, authenticated):
        temp_dir = tempfile.TemporaryDirectory()
        project_root = Path(__file__).resolve().parent
        env = {
            **os.environ,
            "DASHBOARD_TEST_PROVIDER_STATE": "HEALTHY",
            "DASHBOARD_TEST_AUTH": "1" if authenticated else "0",
            "OBSERVATION_DATA_DIR": temp_dir.name,
            "OBSERVATION_CONTROLLER_STATE_PATH": str(
                Path(temp_dir.name) / "controller.json"
            ),
            "OBSERVATION_RUNNER_LOCK_PATH": str(Path(temp_dir.name) / "runner.lock"),
            "OBSERVATION_STORE_PATH": str(Path(temp_dir.name) / "observations.jsonl"),
        }
        wrapper_path = Path(temp_dir.name) / "dashboard_paper_controls_fixture.py"
        wrapper_path.write_text(_paper_controls_wrapper(), encoding="utf-8")
        port = _find_free_port()
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                str(wrapper_path),
                "--server.address",
                "127.0.0.1",
                "--server.port",
                str(port),
                "--server.headless",
                "true",
                "--server.enableCORS",
                "false",
                "--browser.gatherUsageStats",
                "false",
            ],
            cwd=project_root,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        output, reader = _start_bounded_output_reader(process)
        playwright = sync_playwright().start()
        browser = playwright.chromium.launch(
            headless=True,
            executable_path=shutil.which("chromium"),
            env={"HOME": temp_dir.name},
        )
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        _navigate_fixture_until_ready(
            lambda: page.goto(
                f"http://127.0.0.1:{port}/?section=SETTINGS",
                wait_until="domcontentloaded",
                timeout=3000,
            ),
            process,
            output,
            reader,
            "paper controls fixture",
            page=page,
        )
        _dashboard_wait(
            lambda: page.wait_for_function(
                "() => document.body.innerText.includes('Pre-live readiness diagnostics')",
                timeout=20000,
            ),
            page, output, reader, "Paper controls readiness diagnostics",
        )
        page.get_by_text("Pre-live readiness diagnostics", exact=True).click()
        _dashboard_wait(
            lambda: page.wait_for_function(
                "() => document.body.innerText.includes('These controls affect the genuine paper observation loop only.')",
                timeout=10000,
            ),
            page, output, reader, "Paper controls panel",
        )
        if authenticated:
            _dashboard_wait(
                lambda: page.get_by_test_id("paper-controls-ready").wait_for(
                    state="visible",
                    timeout=10000,
                ),
                page, output, reader, "Paper controls",
            )
        return temp_dir, process, reader, browser, playwright, page, output

    @staticmethod
    def _close_controls(temp_dir, process, reader, browser, playwright, page):
        try:
            try:
                page.close()
            finally:
                try:
                    browser.close()
                finally:
                    try:
                        playwright.stop()
                    finally:
                        _cleanup_fixture_process(process, reader)
        finally:
            temp_dir.cleanup()

    def test_anonymous_session_cannot_mutate_paper_controller(self):
        resources = self._open_controls(authenticated=False)
        try:
            _, _, _, _, _, page, _ = resources
            self.assertIn("Sign in to unlock paper controls", page.locator("body").inner_text())
            self.assertIn("KOVA voice remains read-only", page.locator("body").inner_text())
            self.assertEqual(page.get_by_role("button", name="START", exact=True).count(), 0)
            self.assertEqual(page.get_by_role("button", name="PAUSE", exact=True).count(), 0)
            self.assertEqual(page.get_by_role("button", name="STOP", exact=True).count(), 0)
            state = Path(resources[0].name) / "controller.json"
            self.assertFalse(state.exists())
        finally:
            self._close_controls(*resources[:5], resources[5])

    def test_authenticated_session_confirms_start_pause_and_stop_paper_only(self):
        resources = self._open_controls(authenticated=True)
        try:
            temp_dir, _, _, _, _, page, _ = resources
            body = page.locator("body").inner_text()
            self.assertIn("AUTHENTICATED RUNNER CONTROLS", body)
            self.assertIn("AVAILABLE", body)
            self.assertIn("Live trading, live options, margin, and undefined-risk options remain disabled.", body)

            page.get_by_text("Confirm START", exact=True).click()
            page.get_by_role("button", name="START", exact=True).click()
            _dashboard_wait(
                lambda: page.wait_for_function(
                    "() => document.body.innerText.includes('Current persisted state: RUNNING')"
                ),
                page, resources[6], resources[2], "Paper START state",
            )

            page.get_by_text("Confirm PAUSE", exact=True).click()
            page.get_by_role("button", name="PAUSE", exact=True).click()
            _dashboard_wait(
                lambda: page.wait_for_function(
                    "() => document.body.innerText.includes('Current persisted state: PAUSED')"
                ),
                page, resources[6], resources[2], "Paper PAUSE state",
            )

            page.get_by_text("Confirm STOP", exact=True).click()
            page.get_by_role("button", name="STOP", exact=True).click()
            _dashboard_wait(
                lambda: page.wait_for_function(
                    "() => document.body.innerText.includes('Current persisted state: STOPPED_MANUAL')"
                ),
                page, resources[6], resources[2], "Paper STOP state",
            )

            state = Path(temp_dir.name) / "controller.json"
            self.assertIn('"status":"STOPPED_MANUAL"', state.read_text(encoding="utf-8"))
            evidence = Path(temp_dir.name) / "observations.jsonl"
            self.assertFalse(evidence.exists() and evidence.read_text(encoding="utf-8").strip())
        finally:
            self._close_controls(*resources[:5], resources[5])


if __name__ == "__main__":
    unittest.main()