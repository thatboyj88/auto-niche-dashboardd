import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from datetime import datetime, timezone

from test_dashboard_ai_operations_assistant import (
    _cleanup_fixture_process,
    _dashboard_wait,
    _find_free_port,
    _fixture_output,
    _navigate_fixture_until_ready,
    _start_bounded_output_reader,
)


def _options_browser_wrapper(
    provider_outage_after_first=False, provider_recovery_after_outage=False
):
    """Return a local-only Streamlit fixture for the options review."""
    outage_mode = "true" if provider_outage_after_first else "false"
    recovery_mode = "true" if provider_recovery_after_outage else "false"
    return """
import os
from datetime import datetime, timezone
from unittest.mock import patch

import streamlit as st
os.environ["OPTIONS_QUOTE_CACHE_TTL_SECONDS"] = "5"
os.environ["OPTIONS_FIXTURE_OUTAGE_AFTER_FIRST"] = "__OUTAGE_MODE__"
os.environ["OPTIONS_FIXTURE_RECOVERY_AFTER_OUTAGE"] = "__RECOVERY_MODE__"
import dashboard

if "fixture_request_count" not in st.session_state:
    st.session_state.fixture_request_count = 0
if "fixture_cache_initialized" not in st.session_state:
    dashboard.fetch_cached_public_option_quote_candidates.clear()
    st.session_state.fixture_cache_initialized = True

def fake_option_provider(symbol):
    st.session_state.fixture_request_count += 1
    snapshot_number = st.session_state.fixture_request_count
    fetched_at = datetime.now(timezone.utc)
    if (
        os.environ["OPTIONS_FIXTURE_OUTAGE_AFTER_FIRST"] == "true"
        and snapshot_number > 1
        and not (
            os.environ["OPTIONS_FIXTURE_RECOVERY_AFTER_OUTAGE"] == "true"
            and snapshot_number > 2
        )
    ):
        return {
            "source": "Local unavailable fixture provider",
            "symbol": symbol.upper(),
            "available": False,
            "expiration": None,
            "candidates": [],
            "error": "Local provider outage during refresh.",
            "fetched_at": fetched_at.isoformat(),
        }
    return {
        "source": f"Local available fixture snapshot {snapshot_number}",
        "symbol": symbol.upper(),
        "available": True,
        "expiration": "2099-01-01T20:00:00+00:00",
        "candidates": [{
            "instrument": f"LOCAL-{symbol.upper()}-100C-SNAPSHOT-{snapshot_number}",
            "strategy": "LONG_CALL",
            "stock_price": 100,
            "quantity": 1,
            "contracts": [{
                "underlying": symbol.upper(),
                "option_type": "CALL",
                "strike": 100 + snapshot_number,
                "expiration": "2099-01-01T20:00:00+00:00",
                "bid": 4 + snapshot_number,
                "ask": 6 + snapshot_number,
                "underlying_price": 100,
                "observed_at": fetched_at.isoformat(),
            }],
        }],
        "error": None,
        "fetched_at": fetched_at.isoformat(),
    }

with patch(
    "dashboard.fetch_public_option_quote_candidates",
    side_effect=fake_option_provider,
):
    dashboard.inject_mission_control_theme()
    dashboard.render_mc_options_review()
    st.button("Rerun fixture")
    st.caption(
        "FIXTURE PROVIDER REQUESTS: "
        f"{st.session_state.fixture_request_count}"
    )
""".replace("__OUTAGE_MODE__", outage_mode).replace(
    "__RECOVERY_MODE__", recovery_mode
)


@unittest.skipUnless(
    shutil.which("chromium")
    and os.environ.get("RELEASE_CHECK_SKIP_BROWSER") != "1",
    "Browser regressions run in dedicated release workflow jobs.",
)
class DashboardOptionsBrowserTests(unittest.TestCase):
    def test_cached_quotes_refresh_after_expiry_and_keep_provider_state_visible(self):
        """Protect cache reuse, automatic expiry, explicit refresh, and UI state."""
        from playwright.sync_api import sync_playwright

        project_root = os.path.dirname(os.path.abspath(__file__))
        with tempfile.TemporaryDirectory() as temp_dir:
            wrapper_path = os.path.join(
                temp_dir, "dashboard_options_browser_fixture.py"
            )
            with open(wrapper_path, "w", encoding="utf-8") as wrapper_file:
                wrapper_file.write(_options_browser_wrapper())

            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(
                    headless=True,
                    executable_path=shutil.which("chromium"),
                    env={**os.environ, "HOME": temp_dir},
                )
                process = None
                page = None
                output_reader = None
                try:
                    port = _find_free_port()
                    process = subprocess.Popen(
                        [
                            sys.executable,
                            "-m",
                            "streamlit",
                            "run",
                            wrapper_path,
                            "--server.address",
                            "127.0.0.1",
                            "--server.port",
                            str(port),
                            "--server.headless",
                            "true",
                            "--browser.gatherUsageStats",
                            "false",
                        ],
                        cwd=project_root,
                        env={**os.environ, "HOME": temp_dir},
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                    )
                    startup_output, output_reader = _start_bounded_output_reader(
                        process
                    )
                    page = browser.new_page(viewport={"width": 1024, "height": 900})
                    _navigate_fixture_until_ready(
                        lambda: page.goto(
                            f"http://127.0.0.1:{port}",
                            wait_until="domcontentloaded",
                            timeout=3000,
                        ),
                        process,
                        startup_output,
                        output_reader,
                        "Options review fixture",
                        page=page,
                    )

                    _dashboard_wait(
                        lambda: page.get_by_text(
                            "DEFINED-RISK OPTIONS", exact=True
                        ).wait_for(timeout=20000),
                        page, startup_output, output_reader,
                        "Options review section",
                    )
                    body = page.locator("body")
                    self.assertIn("FIXTURE PROVIDER REQUESTS: 1", body.inner_text())
                    self.assertIn(
                        "Local available fixture snapshot 1", body.inner_text()
                    )
                    self.assertIn(
                        "LOCAL-SPY-100C-SNAPSHOT-1", body.inner_text()
                    )

                    # A normal Streamlit rerun must reuse the short-lived snapshot.
                    page.get_by_role("button", name="Rerun fixture").click()
                    _dashboard_wait(
                        lambda: page.get_by_text(
                            "FIXTURE PROVIDER REQUESTS: 1", exact=True
                        ).wait_for(timeout=20000),
                        page, startup_output, output_reader,
                        "Options cached rerun",
                    )
                    self.assertIn("FIXTURE PROVIDER REQUESTS: 1", body.inner_text())

                    # The explicit action invalidates the cache exactly once.
                    page.get_by_role("button", name="Refresh quotes").click()
                    _dashboard_wait(
                        lambda: page.get_by_text(
                            "FIXTURE PROVIDER REQUESTS: 2", exact=True
                        ).wait_for(timeout=20000),
                        page, startup_output, output_reader,
                        "Options refresh",
                    )
                    self.assertNotIn(
                        "FIXTURE PROVIDER REQUESTS: 3", body.inner_text()
                    )
                    refreshed_text = body.inner_text()
                    self.assertIn(
                        "Local available fixture snapshot 2", refreshed_text
                    )
                    self.assertIn("LOCAL-SPY-100C-SNAPSHOT-2", refreshed_text)
                    self.assertNotIn(
                        "Local available fixture snapshot 1", refreshed_text
                    )
                    self.assertNotIn("LOCAL-SPY-100C-SNAPSHOT-1", refreshed_text)

                    # The fixture uses a five-second TTL so expiry is deterministic
                    # without waiting for the production sixty-second TTL.
                    time.sleep(5.2)
                    page.get_by_role("button", name="Rerun fixture").click()
                    _dashboard_wait(
                        lambda: page.get_by_text(
                            "FIXTURE PROVIDER REQUESTS: 3", exact=True
                        ).wait_for(timeout=20000),
                        page, startup_output, output_reader,
                        "Options expiry refresh",
                    )
                    self.assertNotIn(
                        "FIXTURE PROVIDER REQUESTS: 4", body.inner_text()
                    )

                    rendered_text = body.inner_text()
                    self.assertIn("SNAPSHOT AGE", rendered_text)
                    self.assertRegex(rendered_text, r"\d+s ago")
                    self.assertIn(
                        "Local available fixture snapshot 3", rendered_text
                    )
                    self.assertIn("LOCAL-SPY-100C-SNAPSHOT-3", rendered_text)
                    self.assertNotIn(
                        "Local available fixture snapshot 2", rendered_text
                    )
                    self.assertNotIn("LOCAL-SPY-100C-SNAPSHOT-2", rendered_text)
                finally:
                    if page is not None:
                        page.close()
                    _cleanup_fixture_process(process, output_reader)
                    browser.close()

    def test_refresh_provider_outage_replaces_prior_option_snapshot(self):
        """A refresh outage must not leave the prior candidate visible."""
        from playwright.sync_api import sync_playwright

        project_root = os.path.dirname(os.path.abspath(__file__))
        with tempfile.TemporaryDirectory() as temp_dir:
            wrapper_path = os.path.join(
                temp_dir, "dashboard_options_browser_fixture.py"
            )
            with open(wrapper_path, "w", encoding="utf-8") as wrapper_file:
                wrapper_file.write(
                    _options_browser_wrapper(provider_outage_after_first=True)
                )

            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(
                    headless=True,
                    executable_path=shutil.which("chromium"),
                    env={**os.environ, "HOME": temp_dir},
                )
                process = None
                page = None
                try:
                    port = _find_free_port()
                    process = subprocess.Popen(
                        [
                            sys.executable,
                            "-m",
                            "streamlit",
                            "run",
                            wrapper_path,
                            "--server.address",
                            "127.0.0.1",
                            "--server.port",
                            str(port),
                            "--server.headless",
                            "true",
                            "--browser.gatherUsageStats",
                            "false",
                        ],
                        cwd=project_root,
                        env={**os.environ, "HOME": temp_dir},
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                    )
                    startup_output, output_reader = _start_bounded_output_reader(
                        process
                    )
                    page = browser.new_page(viewport={"width": 1024, "height": 900})
                    _navigate_fixture_until_ready(
                        lambda: page.goto(
                            f"http://127.0.0.1:{port}",
                            wait_until="domcontentloaded",
                            timeout=3000,
                        ),
                        process,
                        startup_output,
                        output_reader,
                        "Options outage fixture",
                        page=page,
                    )

                    _dashboard_wait(
                        lambda: page.get_by_text(
                            "DEFINED-RISK OPTIONS", exact=True
                        ).wait_for(timeout=20000),
                        page, startup_output, output_reader,
                        "Options outage section",
                    )
                    body = page.locator("body")
                    initial_text = body.inner_text()
                    self.assertIn("FIXTURE PROVIDER REQUESTS: 1", initial_text)
                    self.assertIn("Local available fixture snapshot 1", initial_text)
                    self.assertIn("LOCAL-SPY-100C-SNAPSHOT-1", initial_text)

                    page.get_by_role("button", name="Refresh quotes").click()
                    _dashboard_wait(
                        lambda: page.get_by_text(
                            "FIXTURE PROVIDER REQUESTS: 2", exact=True
                        ).wait_for(timeout=20000),
                        page, startup_output, output_reader,
                        "Options outage refresh",
                    )
                    outage_text = body.inner_text()
                    self.assertIn("UNAVAILABLE", outage_text)
                    self.assertIn(
                        "REJECTED · Public option quote provider · "
                        "Local provider outage during refresh.",
                        outage_text,
                    )
                    self.assertIn(
                        "No quote has been invented or substituted.",
                        outage_text,
                    )
                    self.assertNotIn("LOCAL-SPY-100C-SNAPSHOT-1", outage_text)
                    self.assertNotIn("Local available fixture snapshot 1", outage_text)
                finally:
                    if page is not None:
                        page.close()
                    _cleanup_fixture_process(process, output_reader)
                    browser.close()

    def test_expired_quote_provider_outage_replaces_prior_option_snapshot(self):
        """An outage after automatic cache expiry must clear stale candidates."""
        from playwright.sync_api import sync_playwright

        project_root = os.path.dirname(os.path.abspath(__file__))
        with tempfile.TemporaryDirectory() as temp_dir:
            wrapper_path = os.path.join(
                temp_dir, "dashboard_options_browser_fixture.py"
            )
            with open(wrapper_path, "w", encoding="utf-8") as wrapper_file:
                wrapper_file.write(
                    _options_browser_wrapper(provider_outage_after_first=True)
                )

            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(
                    headless=True,
                    executable_path=shutil.which("chromium"),
                    env={**os.environ, "HOME": temp_dir},
                )
                process = None
                page = None
                try:
                    port = _find_free_port()
                    process = subprocess.Popen(
                        [
                            sys.executable,
                            "-m",
                            "streamlit",
                            "run",
                            wrapper_path,
                            "--server.address",
                            "127.0.0.1",
                            "--server.port",
                            str(port),
                            "--server.headless",
                            "true",
                            "--browser.gatherUsageStats",
                            "false",
                        ],
                        cwd=project_root,
                        env={**os.environ, "HOME": temp_dir},
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                    )
                    startup_output, output_reader = _start_bounded_output_reader(
                        process
                    )
                    page = browser.new_page(viewport={"width": 1024, "height": 900})
                    _navigate_fixture_until_ready(
                        lambda: page.goto(
                            f"http://127.0.0.1:{port}",
                            wait_until="domcontentloaded",
                            timeout=3000,
                        ),
                        process,
                        startup_output,
                        output_reader,
                        "Options expiry outage fixture",
                    )

                    page.get_by_text(
                        "DEFINED-RISK OPTIONS", exact=True
                    ).wait_for(timeout=20000)
                    body = page.locator("body")
                    initial_text = body.inner_text()
                    self.assertIn("FIXTURE PROVIDER REQUESTS: 1", initial_text)
                    self.assertIn("Local available fixture snapshot 1", initial_text)
                    self.assertIn("LOCAL-SPY-100C-SNAPSHOT-1", initial_text)

                    # Let the short-lived quote cache expire, then perform only
                    # the normal rerun path (without clicking Refresh quotes).
                    time.sleep(5.2)
                    page.get_by_role("button", name="Rerun fixture").click()
                    page.get_by_text(
                        "FIXTURE PROVIDER REQUESTS: 2", exact=True
                    ).wait_for(timeout=20000)

                    outage_text = body.inner_text()
                    self.assertIn("UNAVAILABLE", outage_text)
                    self.assertIn(
                        "REJECTED · Public option quote provider · "
                        "Local provider outage during refresh.",
                        outage_text,
                    )
                    self.assertIn(
                        "No quote has been invented or substituted.",
                        outage_text,
                    )
                    self.assertNotIn("LOCAL-SPY-100C-SNAPSHOT-1", outage_text)
                    self.assertNotIn(
                        "Local available fixture snapshot 1", outage_text
                    )
                finally:
                    if page is not None:
                        page.close()
                    _cleanup_fixture_process(process, output_reader)
                    browser.close()

    def test_expired_quote_provider_recovers_on_normal_rerun(self):
        """A later healthy response must replace an expired outage state."""
        from playwright.sync_api import sync_playwright

        project_root = os.path.dirname(os.path.abspath(__file__))
        with tempfile.TemporaryDirectory() as temp_dir:
            wrapper_path = os.path.join(
                temp_dir, "dashboard_options_browser_fixture.py"
            )
            with open(wrapper_path, "w", encoding="utf-8") as wrapper_file:
                wrapper_file.write(
                    _options_browser_wrapper(
                        provider_outage_after_first=True,
                        provider_recovery_after_outage=True,
                    )
                )

            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(
                    headless=True,
                    executable_path=shutil.which("chromium"),
                    env={**os.environ, "HOME": temp_dir},
                )
                process = None
                page = None
                try:
                    port = _find_free_port()
                    process = subprocess.Popen(
                        [
                            sys.executable,
                            "-m",
                            "streamlit",
                            "run",
                            wrapper_path,
                            "--server.address",
                            "127.0.0.1",
                            "--server.port",
                            str(port),
                            "--server.headless",
                            "true",
                            "--browser.gatherUsageStats",
                            "false",
                        ],
                        cwd=project_root,
                        env={**os.environ, "HOME": temp_dir},
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                    )
                    startup_output, output_reader = _start_bounded_output_reader(
                        process
                    )
                    page = browser.new_page(viewport={"width": 1024, "height": 900})
                    _navigate_fixture_until_ready(
                        lambda: page.goto(
                            f"http://127.0.0.1:{port}",
                            wait_until="domcontentloaded",
                            timeout=3000,
                        ),
                        process,
                        startup_output,
                        output_reader,
                        "Options recovery fixture",
                        page=page,
                    )

                    _dashboard_wait(
                        lambda: page.get_by_text(
                            "DEFINED-RISK OPTIONS", exact=True
                        ).wait_for(timeout=20000),
                        page, startup_output, output_reader,
                        "Options recovery section",
                    )
                    body = page.locator("body")
                    initial_text = body.inner_text()
                    self.assertIn("FIXTURE PROVIDER REQUESTS: 1", initial_text)
                    self.assertIn("Local available fixture snapshot 1", initial_text)
                    self.assertIn("LOCAL-SPY-100C-SNAPSHOT-1", initial_text)

                    # The first normal rerun after expiry reaches the provider
                    # outage; no explicit refresh action is used.
                    time.sleep(5.2)
                    page.get_by_role("button", name="Rerun fixture").click()
                    _dashboard_wait(
                        lambda: page.get_by_text(
                            "FIXTURE PROVIDER REQUESTS: 2", exact=True
                        ).wait_for(timeout=20000),
                        page, startup_output, output_reader,
                        "Options recovery outage",
                    )
                    outage_text = body.inner_text()
                    self.assertIn("UNAVAILABLE", outage_text)
                    self.assertIn(
                        "Local provider outage during refresh.", outage_text
                    )
                    self.assertNotIn("LOCAL-SPY-100C-SNAPSHOT-1", outage_text)

                    # Once that outage snapshot expires, a second normal rerun
                    # must request and display the fresh provider response.
                    time.sleep(5.2)
                    page.get_by_role("button", name="Rerun fixture").click()
                    _dashboard_wait(
                        lambda: page.get_by_text(
                            "FIXTURE PROVIDER REQUESTS: 3", exact=True
                        ).wait_for(timeout=20000),
                        page, startup_output, output_reader,
                        "Options recovery refresh",
                    )

                    recovered_text = body.inner_text()
                    self.assertIn("AVAILABLE", recovered_text)
                    self.assertIn(
                        "Local available fixture snapshot 3", recovered_text
                    )
                    self.assertIn(
                        "LOCAL-SPY-100C-SNAPSHOT-3", recovered_text
                    )
                    self.assertNotIn(
                        "Local provider outage during refresh.", recovered_text
                    )
                    self.assertNotIn("UNAVAILABLE", recovered_text)
                    self.assertNotIn(
                        "Local available fixture snapshot 1", recovered_text
                    )
                    self.assertNotIn(
                        "LOCAL-SPY-100C-SNAPSHOT-1", recovered_text
                    )
                finally:
                    if page is not None:
                        page.close()
                    _cleanup_fixture_process(process, output_reader)
                    browser.close()


if __name__ == "__main__":
    unittest.main()
