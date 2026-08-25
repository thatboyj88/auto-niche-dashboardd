import unittest
import json
import os
import inspect
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlencode
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

import dashboard


ORBIT_SNAPSHOT_VIEWPORTS = (
    ("desktop", 1280, 900),
    ("mobile-320", 320, 844),
    ("mobile-390", 390, 844),
    ("mobile-430", 430, 844),
)
ORBIT_SNAPSHOT_BASELINE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "visual_baselines",
    "orbit_summary",
)


def _compare_orbit_snapshot_to_baseline(
    screenshot,
    baseline_path,
    diff_path,
    *,
    max_channel_delta=8,
    max_changed_ratio=0.01,
):
    """Compare a screenshot and save a tightly cropped diff when it changes."""
    from io import BytesIO

    from PIL import Image, ImageChops, ImageEnhance

    with Image.open(BytesIO(screenshot)).convert("RGB") as actual:
        if not os.path.isfile(baseline_path):
            raise AssertionError(
                f"Missing approved Orbit Summary baseline: {baseline_path}. "
                "Run with ORBIT_SNAPSHOT_UPDATE=1 ORBIT_SNAPSHOT_UPDATE_APPROVED=1 "
                "after reviewing the new screenshots."
            )
        with Image.open(baseline_path).convert("RGB") as expected:
            if actual.size != expected.size:
                raise AssertionError(
                    f"Orbit Summary baseline size changed: expected {expected.size}, "
                    f"got {actual.size} ({baseline_path})."
                )

            difference = ImageChops.difference(actual, expected)
            changed = difference.point(
                lambda channel: 255 if channel > max_channel_delta else 0
            ).convert("L")
            changed_pixels = changed.histogram()[255]
            changed_ratio = changed_pixels / (actual.width * actual.height)
            if changed_ratio <= max_changed_ratio:
                return

            bbox = changed.getbbox() or (0, 0, actual.width, actual.height)
            padding = 12
            bbox = (
                max(0, bbox[0] - padding),
                max(0, bbox[1] - padding),
                min(actual.width, bbox[2] + padding),
                min(actual.height, bbox[3] + padding),
            )
            os.makedirs(os.path.dirname(diff_path), exist_ok=True)
            focused_diff = ImageEnhance.Contrast(difference).enhance(4)
            focused_diff.crop(bbox).save(diff_path)
            raise AssertionError(
                f"Orbit Summary visual drift detected in {baseline_path}: "
                f"{changed_pixels:,} pixels ({changed_ratio:.3%}) changed; "
                f"focused diff: {diff_path}"
            )


def render_assistant_health_view():
    import dashboard

    dashboard.inject_mission_control_theme()
    dashboard.render_mc_provider_health()


def render_full_dashboard():
    import dashboard

    dashboard.render_dashboard()


def _find_free_port():
    with socket.socket() as server_socket:
        server_socket.bind(("127.0.0.1", 0))
        return server_socket.getsockname()[1]


def _start_bounded_output_reader(process):
    """Read fixture output without allowing an unbounded log buffer."""
    output = deque(maxlen=200)

    def read_output():
        try:
            for line in iter(process.stdout.readline, ""):
                output.append(line.rstrip()[:2000])
        except (OSError, ValueError):
            # Teardown may close the pipe while the daemon reader is draining it.
            pass
        finally:
            stream = process.stdout
            if stream is not None and not stream.closed:
                stream.close()

    reader = threading.Thread(target=read_output, daemon=True)
    reader.start()
    return output, reader


def _fixture_output(output, reader):
    reader.join(timeout=1)
    captured = "\n".join(output).strip()
    return captured or "(no Streamlit fixture output captured)"


def _dashboard_wait(wait, page, output, reader, description):
    """Wait for a dashboard condition with actionable fixture diagnostics."""
    try:
        return wait()
    except Exception as error:
        try:
            page_text = page.locator("body").inner_text()
        except Exception as page_error:
            page_text = f"(page text unavailable: {page_error})"
        raise AssertionError(
            f"{description} did not become ready:\n"
            f"page text:\n{page_text}\n"
            f"fixture output:\n{_fixture_output(output, reader)}"
        ) from error


def _click_dashboard_link_until_ready(page, label, timeout=10000):
    """Click a dashboard link even if Streamlit replaces it mid-rerender."""
    deadline = time.monotonic() + timeout / 1000
    last_error = None
    while time.monotonic() < deadline:
        try:
            link = page.get_by_role("link", name=label, exact=True)
            link.wait_for(state="visible", timeout=1000)
            link.click(timeout=1000)
            return
        except Exception as error:
            last_error = error
            time.sleep(0.1)
    if last_error is not None:
        raise last_error
    raise AssertionError(f"Dashboard link {label!r} did not become ready")


def _traverse_dashboard_history(page, direction, timeout=10000):
    """Traverse browser history without waiting on a rerender to finish loading."""
    deadline = time.monotonic() + timeout / 1000
    last_error = None
    while time.monotonic() < deadline:
        try:
            if direction == "back":
                page.go_back(wait_until="commit", timeout=1000)
            else:
                page.go_forward(wait_until="commit", timeout=1000)
            return
        except Exception as error:
            last_error = error
            time.sleep(0.1)
    if last_error is not None:
        raise last_error
    raise AssertionError(f"Could not traverse browser history {direction!r}")


def _cleanup_fixture_process(process, output_reader=None, *, timeout=5):
    """Stop a browser fixture and close all resources without hiding test errors."""
    cleanup_errors = []
    if process is not None:
        try:
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=timeout)
        except (OSError, subprocess.SubprocessError) as error:
            cleanup_errors.append(f"process cleanup failed: {error}")
        finally:
            stream = getattr(process, "stdout", None)
            if stream is not None and not stream.closed:
                try:
                    stream.close()
                except OSError as error:
                    cleanup_errors.append(f"fixture output close failed: {error}")

    if output_reader is not None:
        output_reader.join(timeout=1)
        if output_reader.is_alive():
            cleanup_errors.append("fixture output reader did not stop")
    return "; ".join(cleanup_errors)


def _navigate_fixture_until_ready(
    navigate,
    process,
    startup_output,
    output_reader,
    fixture_name,
    timeout=20,
    page=None,
):
    """Retry fixture navigation while reporting every startup failure consistently."""
    deadline = time.monotonic() + timeout
    while True:
        try:
            navigate()
            return
        except Exception as error:
            if process.poll() is not None:
                message = (
                    f"{fixture_name} exited before navigation completed:\n"
                    f"{_fixture_output(startup_output, output_reader)}"
                )
                if page is not None:
                    try:
                        message += f"\npage text:\n{page.locator('body').inner_text()}"
                    except Exception as page_error:
                        message += f"\npage text unavailable: {page_error}"
                raise AssertionError(message) from error
            if time.monotonic() >= deadline:
                message = (
                    f"{fixture_name} did not start:\n"
                    f"{_fixture_output(startup_output, output_reader)}"
                )
                if page is not None:
                    try:
                        message += f"\npage text:\n{page.locator('body').inner_text()}"
                    except Exception as page_error:
                        message += f"\npage text unavailable: {page_error}"
                raise AssertionError(message) from error
            time.sleep(0.25)


def _write_orbit_observation_fixture(data_dir):
    """Create stable paper-observation metadata for visual snapshots only."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    controller = {
        "status": "RUNNING",
        "started_at": "2024-01-01T00:00:00+00:00",
        "last_cycle_at": "2024-01-02T00:57:36+00:00",
        "last_data_health": "HEALTHY",
        "cycles": 100,
        "healthy_cycles": 100,
        "unhealthy_cycles": 0,
    }
    engine = {
        "capital": 15.19,
        "position": 0,
        "genuine_signals": 1,
        "genuine_completed_trades": 1,
        "last_signal": {
            "payload": {
                "entry_eligible": False,
                "strategy_score": 80,
            }
        },
        "last_completed_trade": {
            "trade_number": 1,
            "reason": "completed",
        },
        "persistence_health": {
            "status": "AVAILABLE",
            "error_code": None,
            "last_error": None,
            "operation": "read",
        },
    }
    records = [
        {"dataset": "PAPER_OPERATIONAL", "record_type": "SIGNAL"},
        {"dataset": "PAPER_OPERATIONAL", "record_type": "TRADE"},
    ]
    (data_dir / "observation_controller.json").write_text(
        json.dumps(controller), encoding="utf-8"
    )
    (data_dir / "paper_engine_state.json").write_text(
        json.dumps(engine), encoding="utf-8"
    )
    (data_dir / "observations.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def _browser_dashboard_wrapper():
    """Return a Streamlit entry point with entirely local dashboard fixtures."""
    return """
import os
import json
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path

import dashboard

preferences_path = os.environ.get("KOVA_DASHBOARD_PREFERENCES_PATH")
if preferences_path:
    dashboard.DASHBOARD_PREFERENCES_PATH = Path(preferences_path)
def test_authenticated_user_key():
    auth_path = os.environ.get("KOVA_TEST_AUTH_PATH")
    if not auth_path:
        return None
    try:
        with open(auth_path, encoding="utf-8") as auth_file:
            test_user = json.load(auth_file).get("user")
    except (OSError, json.JSONDecodeError, AttributeError):
        test_user = None
    return {
        "account-a": "sub:browser-account-a",
        "account-b": "sub:browser-account-b",
    }.get(test_user)


state = os.environ["DASHBOARD_TEST_PROVIDER_STATE"]
provider_health = {
    "provider": "Managed provider",
    "availability": state,
    "requests": 7,
    "successes": 4 if state != "UNAVAILABLE" else 0,
    "failures": 3 if state != "HEALTHY" else 0,
    "success_rate_percent": 57.1 if state != "UNAVAILABLE" else 0.0,
    "last_latency_ms": 1200.0,
    "last_outcome": "SUCCESS" if state == "HEALTHY" else "FAILURE",
    "last_failure_category": (
        None if state == "HEALTHY"
        else "provider_outage" if state == "DEGRADED"
        else "timeout"
    ),
    "failure_categories": (
        {} if state == "HEALTHY"
        else {"rate_limit": 2, "provider_outage": 1}
        if state == "DEGRADED"
        else {"timeout": 1}
    ),
}
latest_evaluation = {
    "evaluation_number": 1,
    "candle": 220,
    "timestamp": 1700007200,
    "current_price": 49975.00,
    "strategy_score": 80,
    "decision": "HOLD",
    "rsi": 50.0,
    "ema21": 49800.0,
    "ema50": 49700.0,
    "ema200": 49500.0,
    "long_term_trend": True,
    "short_term_momentum": True,
    "rsi_condition": True,
    "volume": True,
    "price_above_ema21": True,
}
results = {
    "starting_capital": 25.00,
    "ending_capital": 25.00,
    "profit": 0.00,
    "trades": 0,
    "wins": 0,
    "losses": 0,
    "win_rate": 0.0,
    "max_drawdown": 0.0,
    "total_fees": 0.0,
    "total_slippage": 0.0,
    "equity_curve": [25.00, 25.00, 25.00],
    "trades_history": [],
    "evaluations": 1,
    "highest_score": 80,
    "score_80_or_more": 1,
    "evaluation_history": [latest_evaluation],
}
market_data = SimpleNamespace(
    pair_name="XBT/CAD",
    last_error=None,
    data_range="fixture",
)
live_candles = [
    {"timestamp": 1700000000, "close": 50000.00},
    {"timestamp": 1700003600, "close": 50125.00},
    {"timestamp": 1700007200, "close": 49975.00},
]

with (
    patch("dashboard.run_strategy_backtest", return_value=results),
    patch(
        "dashboard.load_kraken_market_data",
        return_value=(market_data, live_candles),
    ),
    patch("dashboard.run_live_market_backtest", return_value=None),
    patch(
        "dashboard.load_historical_btc_cad_data",
        return_value=(market_data, []),
    ),
    patch("dashboard.run_historical_market_backtest", return_value=None),
    patch("dashboard.get_provider_health", return_value=provider_health),
    patch(
        "dashboard._authenticated_user_key",
        side_effect=test_authenticated_user_key,
    ),
):
    dashboard.render_dashboard()
"""


def _select_dashboard_section(page, section):
    """Use the Overview hub and return control instead of sidebar navigation."""
    if not hasattr(page, "get_by_role"):
        page.get_by_text(section.title(), exact=True).click()
        return
    if section == "SYSTEM":
        # System is an operational view, not one of the six user-facing
        # Overview shortcuts.  Navigate through the same query-param route
        # used by the app instead of waiting for a nonexistent card.
        base_url = page.url.split("?", 1)[0]
        page.goto(
            f"{base_url}?section=SYSTEM",
            wait_until="domcontentloaded",
            timeout=30000,
        )
        return
    if section == "OVERVIEW":
        overview_button = page.get_by_role(
            "button", name="Return to Overview", exact=False
        )
        if not overview_button.count():
            overview_button = page.get_by_text(
                "Return to Overview", exact=True
            )
        if overview_button.count():
            for index in range(overview_button.count() - 1, -1, -1):
                if overview_button.nth(index).bounding_box():
                    overview_button.nth(index).click()
                    page.wait_for_selector(
                        ".st-key-overview_nav_grid",
                        state="visible",
                        timeout=10000,
                    )
                    break
        return
    if section == "LIVE MONITOR":
        if not page.locator(".st-key-overview_nav_grid").count():
            _select_dashboard_section(page, "OVERVIEW")
            page.wait_for_selector(
                ".st-key-overview_nav_grid",
                state="visible",
                timeout=10000,
            )
        page.wait_for_selector(
            ".st-key-overview_live_monitor",
            state="attached",
            timeout=20000,
        )
        decision_buttons = page.locator(
            ".st-key-overview_live_monitor [data-testid='stButton'] button"
        )
        for index in range(decision_buttons.count()):
            if decision_buttons.nth(index).bounding_box():
                decision_buttons.nth(index).click()
                page.wait_for_timeout(900)
                return
    if section == "MARKET":
        if not page.locator(".st-key-overview_nav_grid").count():
            _select_dashboard_section(page, "OVERVIEW")
            page.wait_for_selector(
                ".st-key-overview_nav_grid",
                state="visible",
                timeout=10000,
            )
        market_link = page.locator(".overview-market-chart-link")
        if market_link.count() and market_link.first.bounding_box():
            market_link.first.click()
            page.wait_for_timeout(900)
            return
    if not page.locator(".st-key-overview_nav_grid").count():
        _select_dashboard_section(page, "OVERVIEW")
        page.wait_for_selector(
            ".st-key-overview_nav_grid",
            state="visible",
            timeout=10000,
        )
    buttons = page.get_by_role("button", name=section.title(), exact=False)
    if not buttons.count():
        buttons = page.get_by_role("link", name=section.title(), exact=False)
    clicked = False
    for index in range(buttons.count() - 1, -1, -1):
        if buttons.nth(index).bounding_box():
            buttons.nth(index).click()
            page.wait_for_timeout(900)
            clicked = True
            break
    if not clicked:
        page.locator(
            f'.st-key-overview_nav_grid :is(button, a):has-text("{section}")'
        ).last.click()


@unittest.skipIf(
    os.environ.get("RELEASE_CHECK_SKIP_BROWSER") == "1",
    "Browser regressions run in dedicated release workflow jobs.",
)
class DashboardAssistantBrowserTests(unittest.TestCase):
    def test_dashboard_wait_reports_page_and_fixture_diagnostics(self):
        """A failed browser wait should show partial renders and fixture logs."""
        class FakeLocator:
            def inner_text(self):
                return "PARTIAL FIXTURE RENDER"

        class FakePage:
            def locator(self, selector):
                if selector != "body":
                    raise AssertionError(f"unexpected selector: {selector}")
                return FakeLocator()

        page = FakePage()
        output = deque(["fixture started", "fixture failed during render"])

        class FinishedReader:
            def join(self, timeout=None):
                return None

        reader = FinishedReader()

        with self.assertRaises(AssertionError) as failure:
            _dashboard_wait(
                lambda: (_ for _ in ()).throw(TimeoutError("wait expired")),
                page,
                output,
                reader,
                "Dashboard section",
            )

        message = str(failure.exception)
        self.assertIn("Dashboard section did not become ready", message)
        self.assertIn("PARTIAL FIXTURE RENDER", message)
        self.assertIn("fixture failed during render", message)

    def test_fixture_cleanup_stops_child_and_closes_output_reader(self):
        """Fixture teardown must not leave a server or pipe behind."""
        process = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        output, reader = _start_bounded_output_reader(process)
        self.assertEqual(list(output), [])

        cleanup_error = _cleanup_fixture_process(process, reader)

        self.assertEqual(cleanup_error, "")
        self.assertIsNotNone(process.poll())
        self.assertFalse(reader.is_alive())
        self.assertTrue(process.stdout.closed)

    MOBILE_VIEWPORTS = (390, 320)
    LANDSCAPE_TABLET_VIEWPORT = (1024, 768)
    PORTRAIT_TABLET_VIEWPORT = (768, 1024)
    BROWSER_ZOOM_FACTORS = (1.0, 1.25, 1.5)
    MOBILE_BROWSER_PROFILES = (
        (
            "iOS Safari",
            390,
            (
                "Mozilla/5.0 (iPhone; CPU iPhone OS 17_5 like Mac OS X) "
                "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 "
                "Mobile/15E148 Safari/604.1"
            ),
        ),
        (
            "Android Chrome",
            412,
            (
                "Mozilla/5.0 (Linux; Android 14; Pixel 8) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/125.0.0.0 Mobile Safari/537.36"
            ),
        ),
        (
            "Android Firefox",
            390,
            (
                "Mozilla/5.0 (Android 14; Mobile; rv:128.0) "
                "Gecko/128.0 Firefox/128.0"
            ),
        ),
    )
    MOBILE_SECTIONS = (
        ("OVERVIEW", "Paper operations progress"),
        ("LIVE MONITOR", "BTC/CAD market display"),
        ("STRATEGY", "STRATEGY TELEMETRY"),
        ("POSITIONS", "POSITION SNAPSHOT"),
        ("PERFORMANCE", "Historical batch backtest results summary"),
        ("RISK", "RISK MONITOR"),
        ("MARKET", "BTC/CAD market display"),
        ("SETTINGS", "SETTINGS & PREFERENCES"),
    )
    CHART_SECTIONS = (
        ("OVERVIEW", "Paper account trajectory"),
        ("LIVE MONITOR", "BTC/CAD market display"),
        ("MARKET", "BTC/CAD market display"),
        ("PERFORMANCE", "Historical batch backtest results summary"),
    )

    def _assert_mobile_geometry(self, page, expected_viewport):
        geometry = page.evaluate(
            """() => {
                const elements = Array.from(document.querySelectorAll(
                    ".mc-data-metric, .mc-kpi-card, .mc-chart-card, .mc-trades-card, [data-testid='stDataFrame'], [data-testid='stHeading']"
                ));
                return {
                    viewport: window.innerWidth,
                    documentWidth: document.documentElement.scrollWidth,
                    bodyWidth: document.body.scrollWidth,
                    elements: elements.map((element) => {
                        const box = element.getBoundingClientRect();
                        return {
                            text: element.innerText,
                            left: box.left,
                            right: box.right,
                            top: box.top,
                            bottom: box.bottom,
                        };
                    }),
                };
            }"""
        )
        self.assertEqual(geometry["viewport"], expected_viewport)
        self.assertLessEqual(geometry["documentWidth"], expected_viewport + 1)
        self.assertLessEqual(geometry["bodyWidth"], expected_viewport + 1)
        self.assertTrue(geometry["elements"])
        for item in geometry["elements"]:
            self.assertGreaterEqual(item["left"], -1, item["text"])
            self.assertLessEqual(
                item["right"], expected_viewport + 1, item["text"]
            )
            self.assertGreater(item["bottom"], item["top"], item["text"])

    def _assert_tiny_chart_geometry(self, page, expected_viewport):
        geometry = page.evaluate(
            """() => {
                const chartCards = Array.from(
                    document.querySelectorAll(".mc-chart-card")
                );
                const details = [];
                chartCards.forEach((card, cardIndex) => {
                    const visuals = Array.from(
                        card.querySelectorAll("svg, canvas")
                    );
                    const labels = Array.from(card.querySelectorAll(
                        ".mc-chart-axis, .legend, [class*='legend'], " +
                        "[aria-label*='legend']"
                    ));
                    details.push(
                        ...visuals.map((element, index) => ({
                            kind: element.tagName.toLowerCase(),
                            index,
                            cardIndex,
                            text: element.getAttribute("aria-label") || "",
                            box: element.getBoundingClientRect(),
                        })),
                        ...labels.map((element, index) => ({
                            kind: "chart detail",
                            index,
                            cardIndex,
                            text: element.textContent.trim(),
                            box: element.getBoundingClientRect(),
                        })),
                    );
                });
                return {
                    chartCount: chartCards.length,
                    details: details.map(({ kind, index, cardIndex, text, box }) => ({
                        kind,
                        index,
                        cardIndex,
                        text,
                        left: box.left,
                        right: box.right,
                        top: box.top,
                        bottom: box.bottom,
                    })),
                };
            }"""
        )
        if geometry["chartCount"] == 0:
            return
        visuals = [
            item for item in geometry["details"]
            if item["kind"] in {"svg", "canvas"}
        ]
        self.assertTrue(visuals)
        for item in geometry["details"]:
            with self.subTest(
                kind=item["kind"],
                chart=item["cardIndex"],
                detail=item["text"],
            ):
                self.assertGreater(item["right"], item["left"])
                self.assertGreater(item["bottom"], item["top"])
                self.assertGreaterEqual(item["left"], -1)
                self.assertLessEqual(item["right"], expected_viewport + 1)

    def test_direct_detail_urls_survive_fresh_session_refresh(self):
        """Shared detail links keep their destination after a full refresh."""
        from playwright.sync_api import sync_playwright

        destinations = (
            "LIVE MONITOR",
            "STRATEGY",
            "POSITIONS",
            "PERFORMANCE",
            "RISK",
            "OPTIONS REVIEW",
            "MARKET",
            "RESEARCH",
            "BACKTEST",
            "SYSTEM",
            "SETTINGS",
        )
        visible_content = {
            "STRATEGY": "AUTONOMOUS PORTFOLIO DECISION",
            "RESEARCH": "RESEARCH LAB",
            "OPTIONS REVIEW": "DEFINED-RISK OPTIONS",
        }
        project_root = os.path.dirname(os.path.abspath(__file__))

        with tempfile.TemporaryDirectory() as temp_dir:
            wrapper_path = os.path.join(temp_dir, "dashboard_browser_fixture.py")
            with open(wrapper_path, "w", encoding="utf-8") as wrapper_file:
                wrapper_file.write(_browser_dashboard_wrapper())

            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(
                    headless=True,
                    executable_path=shutil.which("chromium"),
                    env={**os.environ, "HOME": temp_dir},
                )
                process = None
                startup_output = []
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
                            "--server.enableCORS",
                            "false",
                            "--browser.gatherUsageStats",
                            "false",
                        ],
                        cwd=project_root,
                        env={
                            **os.environ,
                            "DASHBOARD_TEST_PROVIDER_STATE": "HEALTHY",
                        },
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                    )
                    startup_output, output_reader = _start_bounded_output_reader(
                        process
                    )
                    base_url = f"http://127.0.0.1:{port}"
                    startup_page = browser.new_page(
                        viewport={"width": 1280, "height": 900}
                    )
                    _navigate_fixture_until_ready(
                        lambda: startup_page.goto(
                            base_url,
                            wait_until="domcontentloaded",
                            timeout=3000,
                        ),
                        process,
                        startup_output,
                        output_reader,
                        "Direct-link refresh fixture",
                    )
                    startup_page.close()

                    for destination in destinations:
                        with self.subTest(destination=destination):
                            # A new context gives every shared-link check a
                            # clean Streamlit session, independent of history
                            # and the preceding destination.
                            context = browser.new_context(
                                viewport={"width": 1280, "height": 900}
                            )
                            page = context.new_page()
                            try:
                                url = f"{base_url}/?{urlencode({'section': destination})}"
                                page.goto(
                                    url,
                                    wait_until="domcontentloaded",
                                    timeout=15000,
                                )
                                page.wait_for_selector(
                                    ".mc-title",
                                    state="visible",
                                    timeout=20000,
                                )
                                expected_heading = page.locator(
                                    ".mc-title"
                                ).inner_text().strip()
                                self.assertEqual(expected_heading, destination)
                                if destination in visible_content:
                                    self.assertIn(
                                        visible_content[destination],
                                        page.locator("body").inner_text(),
                                    )

                                page.reload(
                                    wait_until="domcontentloaded",
                                    timeout=15000,
                                )
                                page.wait_for_selector(
                                    ".mc-title",
                                    state="visible",
                                    timeout=20000,
                                )
                                self.assertEqual(
                                    page.locator(".mc-title").inner_text().strip(),
                                    expected_heading,
                                )
                                if destination in visible_content:
                                    self.assertIn(
                                        visible_content[destination],
                                        page.locator("body").inner_text(),
                                    )
                                self.assertIn(
                                    f"section={urlencode({'section': destination}).split('=', 1)[1]}",
                                    page.url,
                                )
                            finally:
                                context.close()
                finally:
                    if process is not None:
                        process.terminate()
                        try:
                            process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait(timeout=10)
                    if output_reader is not None:
                        output_reader.join(timeout=1)
                    browser.close()

    def _assert_landscape_tablet_chart_geometry(self, page, zoom_factor=1.0):
        expected_width, expected_height = self.LANDSCAPE_TABLET_VIEWPORT
        geometry = page.evaluate(
            """zoomFactor => {
                const details = [];
                document.querySelectorAll(".mc-chart-card").forEach(
                    (card, cardIndex) => {
                        const selectors = [
                            ["chart visual", "svg, canvas"],
                            ["legend", ".legend, [class*='legend'], [aria-label*='legend']"],
                            ["axis label", ".mc-chart-axis"],
                        ];
                        selectors.forEach(([kind, selector]) => {
                            card.querySelectorAll(selector).forEach(
                                (element, index) => {
                                    const box = element.getBoundingClientRect();
                                    details.push({
                                        kind,
                                        cardIndex,
                                        index,
                                        text: element.textContent.trim() ||
                                            element.getAttribute("aria-label") || "",
                                        left: box.left,
                                        right: box.right,
                                        top: box.top,
                                        bottom: box.bottom,
                                    });
                                }
                            );
                        });
                    }
                );
                return {
                    viewport: [window.innerWidth, window.innerHeight],
                    effectiveViewportWidth: window.innerWidth / zoomFactor,
                    browserZoomFactor: zoomFactor,
                    chartCount: document.querySelectorAll(".mc-chart-card").length,
                    details,
                };
            }""",
            zoom_factor,
        )
        self.assertEqual(
            geometry["viewport"],
            [expected_width, expected_height],
        )
        self.assertAlmostEqual(
            geometry["browserZoomFactor"],
            zoom_factor,
            msg="Browser zoom fixture did not use the requested zoom level.",
        )
        self.assertGreater(
            geometry["chartCount"],
            0,
            "Landscape-tablet fixture did not render any chart cards.",
        )
        details = geometry["details"]
        self.assertTrue(
            any(item["kind"] == "chart visual" for item in details),
            "Landscape-tablet fixture rendered chart cards without SVG/canvas visuals.",
        )
        self.assertTrue(
            any(item["kind"] == "axis label" for item in details),
            "Landscape-tablet fixture rendered chart cards without axis labels.",
        )
        for item in details:
            with self.subTest(
                detail=item["kind"],
                chart=item["cardIndex"],
                index=item["index"],
                text=item["text"],
            ):
                self.assertGreater(
                    item["right"],
                    item["left"],
                    f"{item['kind']} has no width: {item['text']!r}",
                )
                self.assertGreater(
                    item["bottom"],
                    item["top"],
                    f"{item['kind']} has no height: {item['text']!r}",
                )
                self.assertGreaterEqual(
                    item["left"],
                    -1,
                    (
                        f"{item['kind']} starts left of viewport at "
                        f"{geometry['browserZoomFactor']:.2f}x zoom "
                        f"(effective width {geometry['effectiveViewportWidth']:.1f}px): "
                        f"{item['text']!r}"
                    ),
                )
                self.assertLessEqual(
                    item["right"],
                    expected_width + 1,
                    (
                        f"{item['kind']} exceeds the {expected_width}px visible "
                        f"boundary at {geometry['browserZoomFactor']:.2f}x zoom "
                        f"(effective width {geometry['effectiveViewportWidth']:.1f}px): "
                        f"{item['text']!r}"
                    ),
                )

    def _assert_portrait_chart_geometry(
        self, page, zoom_factor, expected_width=None
    ):
        expected_width = expected_width or self.MOBILE_VIEWPORTS[0]
        expected_height = 844
        geometry = page.evaluate(
            """zoomFactor => {
                const details = [];
                document.querySelectorAll(".mc-chart-card").forEach(
                    (card, cardIndex) => {
                        const selectors = [
                            ["chart visual", "svg, canvas"],
                            ["legend", ".legend, [class*='legend'], [aria-label*='legend']"],
                            ["axis label", ".mc-chart-axis"],
                        ];
                        selectors.forEach(([kind, selector]) => {
                            card.querySelectorAll(selector).forEach(
                                (element, index) => {
                                    const box = element.getBoundingClientRect();
                                    details.push({
                                        kind,
                                        cardIndex,
                                        index,
                                        text: element.textContent.trim() ||
                                            element.getAttribute("aria-label") || "",
                                        left: box.left,
                                        right: box.right,
                                        top: box.top,
                                        bottom: box.bottom,
                                    });
                                }
                            );
                        });
                    }
                );
                return {
                    viewport: [window.innerWidth, window.innerHeight],
                    documentWidth: document.documentElement.scrollWidth,
                    bodyWidth: document.body.scrollWidth,
                    browserZoomFactor: zoomFactor,
                    chartCount: document.querySelectorAll(".mc-chart-card").length,
                    details,
                };
            }""",
            zoom_factor,
        )
        self.assertEqual(
            geometry["viewport"],
            [expected_width, expected_height],
        )
        self.assertAlmostEqual(
            geometry["browserZoomFactor"],
            zoom_factor,
            msg="Portrait browser fixture did not use the requested zoom level.",
        )
        self.assertLessEqual(
            geometry["documentWidth"],
            expected_width + 1,
            "Portrait dashboard page boundary overflows at high zoom.",
        )
        self.assertLessEqual(
            geometry["bodyWidth"],
            expected_width + 1,
            "Portrait dashboard body boundary overflows at high zoom.",
        )
        self.assertGreater(
            geometry["chartCount"],
            0,
            "Portrait fixture did not render any chart cards.",
        )
        details = geometry["details"]
        self.assertTrue(
            any(item["kind"] == "chart visual" for item in details),
            "Portrait fixture rendered chart cards without SVG/canvas visuals.",
        )
        self.assertTrue(
            any(item["kind"] == "axis label" for item in details),
            "Portrait fixture rendered chart cards without axis labels.",
        )
        for item in details:
            with self.subTest(
                detail=item["kind"],
                chart=item["cardIndex"],
                index=item["index"],
                text=item["text"],
            ):
                self.assertGreater(item["right"], item["left"])
                self.assertGreater(item["bottom"], item["top"])
                self.assertGreaterEqual(
                    item["left"],
                    -1,
                    (
                        f"{item['kind']} starts left of viewport at "
                        f"{geometry['browserZoomFactor']:.2f}x zoom: "
                        f"{item['text']!r}"
                    ),
                )
                self.assertLessEqual(
                    item["right"],
                    expected_width + 1,
                    (
                        f"{item['kind']} exceeds the {expected_width}px visible "
                        f"portrait boundary at "
                        f"{geometry['browserZoomFactor']:.2f}x zoom: "
                        f"{item['text']!r}"
                    ),
                )

    def test_overview_shortcuts_support_keyboard_navigation_and_activation(self):
        """Overview shortcuts stay reachable, visible, and keyboard-activatable."""
        from playwright.sync_api import sync_playwright

        shortcuts = (
            ("Positions", "POSITIONS"),
            ("Strategy", "STRATEGY"),
            ("Performance", "PERFORMANCE"),
            ("Risk", "RISK"),
            ("Options Review", "OPTIONS REVIEW"),
            ("Settings", "SETTINGS"),
        )
        project_root = os.path.dirname(os.path.abspath(__file__))

        with tempfile.TemporaryDirectory() as temp_dir:
            wrapper_path = os.path.join(temp_dir, "dashboard_browser_fixture.py")
            with open(wrapper_path, "w", encoding="utf-8") as wrapper_file:
                wrapper_file.write(_browser_dashboard_wrapper())

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
                            "--server.enableCORS",
                            "false",
                            "--browser.gatherUsageStats",
                            "false",
                        ],
                        cwd=project_root,
                        env={
                            **os.environ,
                            "DASHBOARD_TEST_PROVIDER_STATE": "HEALTHY",
                        },
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                    )
                    startup_output, output_reader = _start_bounded_output_reader(
                        process
                    )
                    page = browser.new_page(viewport={"width": 1280, "height": 900})
                    _navigate_fixture_until_ready(
                        lambda: page.goto(
                            f"http://127.0.0.1:{port}",
                            wait_until="domcontentloaded",
                            timeout=3000,
                        ),
                        process,
                        startup_output,
                        output_reader,
                        "Keyboard navigation fixture",
                    )
                    _dashboard_wait(
                        lambda: page.wait_for_selector(
                            ".st-key-overview_nav_grid",
                            state="visible",
                            timeout=20000,
                        ),
                        page,
                        startup_output,
                        output_reader,
                        "Overview navigation",
                    )
                    _dashboard_wait(
                        lambda: page.wait_for_function(
                            """() => document.querySelectorAll(
                                '.st-key-overview_nav_grid a'
                            ).length >= 6""",
                            timeout=20000,
                        ),
                        page,
                        startup_output,
                        output_reader,
                        "Overview shortcuts",
                    )
                    buttons = page.locator(
                        ".st-key-overview_nav_grid a:visible"
                    )
                    self.assertEqual(buttons.count(), len(shortcuts))
                    self.assertEqual(
                        [
                            buttons.nth(index).inner_text().strip().splitlines()[-1]
                            for index in range(buttons.count())
                        ],
                        [label.upper() for label, _ in shortcuts],
                    )
                    # The destination label is the complete accessible name;
                    # Material icons are CSS decoration, not spoken button text.
                    for label, _ in shortcuts:
                        with self.subTest(accessible_name=label):
                            accessible_buttons = page.get_by_role(
                                "link", name=label, exact=True
                            )
                            self.assertEqual(accessible_buttons.count(), 1)
                            self.assertEqual(
                                accessible_buttons.get_attribute("aria-label"), None
                            )
                            self.assertEqual(
                                accessible_buttons.locator(
                                    '[data-testid="stIconMaterial"]'
                                ).count(),
                                0,
                            )

                    def ensure_overview():
                        return_button = page.get_by_role(
                            "button", name="Return to Overview", exact=False
                        )
                        for return_index in range(return_button.count() - 1, -1, -1):
                            candidate = return_button.nth(return_index)
                            if candidate.bounding_box():
                                candidate.click()
                                _dashboard_wait(
                                    lambda: page.wait_for_function(
                                        "() => document.body.innerText.includes('Paper operations progress')",
                                        timeout=20000,
                                    ),
                                    page,
                                    startup_output,
                                    output_reader,
                                    "Overview content",
                                )
                                break
                        else:
                            self.assertIn(
                                 "Paper operations progress",
                                page.locator("body").inner_text(),
                                "Overview content is not active",
                            )
                        _dashboard_wait(
                            lambda: page.wait_for_selector(
                                ".st-key-overview_nav_grid",
                                state="visible",
                                timeout=20000,
                            ),
                            page,
                            startup_output,
                            output_reader,
                            "Overview navigation",
                        )
                        _dashboard_wait(
                            lambda: page.wait_for_function(
                                """() => Array.from(document.querySelectorAll(
                                    '.st-key-overview_nav_grid a'
                                )).filter(button => {
                                    const box = button.getBoundingClientRect();
                                    return box.width > 0 && box.height > 0;
                                }).length >= 6""",
                                timeout=20000,
                            ),
                            page,
                            startup_output,
                            output_reader,
                            "Visible overview shortcuts",
                        )

                    # Tab follows the visual/DOM order and every shortcut receives
                    # a keyboard-visible focus ring. Start at the first shortcut
                    # so unrelated Streamlit controls do not affect this flow.
                    buttons.first.focus()
                    for index, (label, _) in enumerate(shortcuts):
                        if index:
                            page.keyboard.press("Tab")
                        active = page.evaluate(
                            """() => {
                                const active = document.activeElement;
                                const grid = document.querySelector(
                                    '.st-key-overview_nav_grid'
                                );
                                const lines = (active?.innerText || "")
                                    .trim().split(/\\n+/);
                                return {
                                    label: lines[lines.length - 1] || "",
                                    inGrid: Boolean(grid?.contains(active)),
                                    focusVisible: Boolean(
                                        active?.matches(':focus-visible')
                                    ),
                                    outline: getComputedStyle(active).outlineStyle,
                                    borderColor: getComputedStyle(active).borderColor,
                                };
                            }"""
                        )
                        self.assertEqual(active["label"], label.upper())
                        self.assertTrue(active["inGrid"], label)
                        self.assertTrue(active["focusVisible"], label)
                        self.assertTrue(
                            active["outline"] != "none"
                            or active["borderColor"] != "rgb(157, 167, 255)",
                            f"{label} has no visible keyboard focus treatment",
                        )

                    # The assistant is rendered in its own component iframe and
                    # must never displace or capture shortcut keyboard traversal.
                    self.assertEqual(
                        page.locator(
                            ".st-key-overview_nav_grid iframe"
                        ).count(),
                        0,
                    )

                    for key in ("Enter",):
                        for label, expected_heading in shortcuts:
                            with self.subTest(key=key, shortcut=label):
                                # Use a fresh browser context for each activation:
                                # Streamlit reruns replace the page DOM, while a
                                # fresh session starts deterministically on Overview.
                                activation_context = browser.new_context(
                                    viewport={"width": 1280, "height": 900}
                                )
                                activation_page = activation_context.new_page()
                                try:
                                    _navigate_fixture_until_ready(
                                        lambda: activation_page.goto(
                                            f"http://127.0.0.1:{port}",
                                            wait_until="domcontentloaded",
                                            timeout=3000,
                                        ),
                                        process,
                                        startup_output,
                                        output_reader,
                                        "Keyboard activation fixture",
                                    )
                                    _dashboard_wait(
                                        lambda: activation_page.wait_for_function(
                                            """() => Array.from(
                                                document.querySelectorAll(
                                                    '.st-key-overview_nav_grid a'
                                                )
                                            ).filter(button => {
                                                const box = button.getBoundingClientRect();
                                                return box.width > 0 && box.height > 0;
                                            }).length >= 6""",
                                            timeout=20000,
                                        ),
                                        activation_page,
                                        startup_output,
                                        output_reader,
                                        "Keyboard activation shortcuts",
                                    )
                                    shortcut_candidates = activation_page.locator(
                                        ".st-key-overview_nav_grid a"
                                    )
                                    shortcut = None
                                    for candidate_index in range(
                                        shortcut_candidates.count() - 1, -1, -1
                                    ):
                                        candidate = shortcut_candidates.nth(candidate_index)
                                        candidate_lines = (
                                            candidate.inner_text().strip().splitlines()
                                        )
                                        if (
                                            candidate.bounding_box()
                                            and candidate_lines
                                            and candidate_lines[-1].strip().upper()
                                            == label.upper()
                                        ):
                                            shortcut = candidate
                                            break
                                    self.assertIsNotNone(shortcut, label)
                                    shortcut.focus()
                                    activation_page.keyboard.press(key)
                                    _dashboard_wait(
                                        lambda: activation_page.wait_for_function(
                                            """heading => document.body &&
                                            document.body.innerText.includes(heading)""",
                                            arg=expected_heading,
                                            timeout=20000,
                                        ),
                                        activation_page,
                                        startup_output,
                                        output_reader,
                                        f"{label} destination",
                                    )
                                    self.assertIn(
                                        expected_heading,
                                        activation_page.locator("body").inner_text(),
                                    )
                                finally:
                                    activation_page.close()
                                    activation_context.close()
                finally:
                    if page is not None:
                        page.close()
                    if process is not None:
                        process.terminate()
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
                    browser.close()

    def test_overview_shortcuts_preserve_browser_back_and_forward_history(self):
        """Every Overview query destination round-trips through browser history."""
        from playwright.sync_api import sync_playwright

        destinations = (
            ("Positions", "POSITIONS", "Current paper position"),
            ("Strategy", "STRATEGY", "Strategy decision"),
            ("Performance", "PERFORMANCE", "Historical batch backtest results summary"),
            ("Risk", "RISK", "Guardrail status"),
            ("Options Review", "OPTIONS REVIEW", "Candidate review"),
            ("Settings", "SETTINGS", "Dashboard preferences"),
        )
        project_root = os.path.dirname(os.path.abspath(__file__))

        with tempfile.TemporaryDirectory() as temp_dir:
            wrapper_path = os.path.join(temp_dir, "dashboard_browser_fixture.py")
            with open(wrapper_path, "w", encoding="utf-8") as wrapper_file:
                wrapper_file.write(_browser_dashboard_wrapper())

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
                            "--server.enableCORS",
                            "false",
                            "--browser.gatherUsageStats",
                            "false",
                        ],
                        cwd=project_root,
                        env={
                            **os.environ,
                            "DASHBOARD_TEST_PROVIDER_STATE": "HEALTHY",
                        },
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                    )
                    startup_output, output_reader = _start_bounded_output_reader(
                        process
                    )
                    page = browser.new_page(viewport={"width": 1280, "height": 900})
                    base_url = f"http://127.0.0.1:{port}"
                    _navigate_fixture_until_ready(
                        lambda: page.goto(
                            base_url,
                            wait_until="domcontentloaded",
                            timeout=3000,
                        ),
                        process,
                        startup_output,
                        output_reader,
                        "Overview history fixture",
                    )
                    _dashboard_wait(
                        lambda: page.wait_for_selector(
                            ".st-key-overview_nav_grid",
                            state="visible",
                            timeout=20000,
                        ),
                        page,
                        startup_output,
                        output_reader,
                        "Overview navigation",
                    )

                    for index, (label, destination, expected_heading) in enumerate(
                        destinations
                    ):
                        with self.subTest(destination=destination):
                            _dashboard_wait(
                                lambda: _click_dashboard_link_until_ready(
                                    page, label
                                ),
                                page,
                                startup_output,
                                output_reader,
                                f"{label} link",
                            )
                            _dashboard_wait(
                                lambda: page.wait_for_function(
                                    """destination => new URL(window.location.href)
                                    .searchParams.get("section") === destination""",
                                    arg=destination,
                                    timeout=20000,
                                ),
                                page,
                                startup_output,
                                output_reader,
                                f"{label} URL navigation",
                            )
                            _dashboard_wait(
                                lambda: page.wait_for_function(
                                    """heading => document.body &&
                                    document.body.innerText.includes(heading)""",
                                    arg=expected_heading,
                                    timeout=20000,
                                ),
                                page,
                                startup_output,
                                output_reader,
                                f"{label} destination",
                            )
                            _dashboard_wait(
                                lambda: _traverse_dashboard_history(page, "back"),
                                page,
                                startup_output,
                                output_reader,
                                f"{label} browser back",
                            )
                            _dashboard_wait(
                                lambda: page.wait_for_selector(
                                    ".st-key-overview_nav_grid",
                                    state="visible",
                                    timeout=20000,
                                ),
                                page,
                                startup_output,
                                output_reader,
                                "Overview navigation after back",
                            )
                            _dashboard_wait(
                                lambda: page.wait_for_function(
                                    """() => document.body &&
                                    document.body.innerText.includes(
                                        "Paper operations progress"
                                    )""",
                                    timeout=20000,
                                ),
                                page,
                                startup_output,
                                output_reader,
                                "Overview content after back",
                            )
                            self.assertIsNone(
                                page.evaluate(
                                    "() => new URL(window.location.href)"
                                    '.searchParams.get("section")'
                                )
                            )
                            self.assertIn(
                                "Paper operations progress",
                                page.locator("body").inner_text(),
                            )

                            _dashboard_wait(
                                lambda: _traverse_dashboard_history(page, "forward"),
                                page,
                                startup_output,
                                output_reader,
                                f"{label} browser forward",
                            )
                            _dashboard_wait(
                                lambda: page.wait_for_function(
                                    """destination => new URL(window.location.href)
                                    .searchParams.get("section") === destination""",
                                    arg=destination,
                                    timeout=20000,
                                ),
                                page,
                                startup_output,
                                output_reader,
                                f"{label} URL forward navigation",
                            )
                            _dashboard_wait(
                                lambda: page.wait_for_function(
                                    """heading => document.body &&
                                    document.body.innerText.includes(heading)""",
                                    arg=expected_heading,
                                    timeout=20000,
                                ),
                                page,
                                startup_output,
                                output_reader,
                                f"{label} destination after forward",
                            )
                            if index < len(destinations) - 1:
                                _dashboard_wait(
                                    lambda: _traverse_dashboard_history(
                                        page, "back"
                                    ),
                                    page,
                                    startup_output,
                                    output_reader,
                                    "Overview navigation after forward",
                                )
                                _dashboard_wait(
                                    lambda: page.wait_for_selector(
                                        ".st-key-overview_nav_grid",
                                        state="visible",
                                        timeout=20000,
                                    ),
                                    page,
                                    startup_output,
                                    output_reader,
                                    "Overview navigation after forward",
                                )
                finally:
                    if page is not None:
                        page.close()
                    if process is not None:
                        process.terminate()
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
                        if "output_reader" in locals():
                            output_reader.join(timeout=1)
                    browser.close()

    def test_crashed_browser_fixture_reports_startup_output(self):
        """A local fixture crash should expose its startup log, not just timeout."""
        project_root = os.path.dirname(os.path.abspath(__file__))
        startup_marker = "INTENTIONAL_FIXTURE_STARTUP_FAILURE"

        with tempfile.TemporaryDirectory() as temp_dir:
            wrapper_path = os.path.join(temp_dir, "dashboard_browser_fixture.py")
            with open(wrapper_path, "w", encoding="utf-8") as wrapper_file:
                wrapper_file.write(
                    "import os\n"
                    f"print({startup_marker!r}, flush=True)\n"
                    f"raise RuntimeError({startup_marker!r})\n"
                )

            process = subprocess.Popen(
                [
                    sys.executable,
                    wrapper_path,
                ],
                cwd=project_root,
                env={**os.environ, "HOME": temp_dir},
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            startup_output, output_reader = _start_bounded_output_reader(process)
            try:
                process.wait(timeout=10)
                with self.assertRaises(AssertionError) as failure:
                    raise AssertionError(
                        "Streamlit fixture exited before navigation completed:\n"
                        f"{_fixture_output(startup_output, output_reader)}"
                    )
            finally:
                _cleanup_fixture_process(process, output_reader)

        self.assertIn(startup_marker, str(failure.exception))

    def test_unavailable_browser_fixture_reports_startup_output_on_timeout(self):
        """A fixture that never becomes available should expose its startup log."""
        project_root = os.path.dirname(os.path.abspath(__file__))
        startup_marker = "INTENTIONAL_FIXTURE_STARTUP_TIMEOUT"

        with tempfile.TemporaryDirectory() as temp_dir:
            wrapper_path = os.path.join(temp_dir, "dashboard_browser_fixture.py")
            with open(wrapper_path, "w", encoding="utf-8") as wrapper_file:
                wrapper_file.write(
                    "import time\n"
                    f"print({startup_marker!r}, flush=True)\n"
                    "time.sleep(10)\n"
                )

            process = subprocess.Popen(
                [
                    sys.executable,
                    wrapper_path,
                ],
                cwd=project_root,
                env={**os.environ, "HOME": temp_dir},
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            startup_output, output_reader = _start_bounded_output_reader(process)
            try:
                def navigate_unavailable():
                    raise RuntimeError("fixture is not ready")

                with self.assertRaises(AssertionError) as failure:
                    _navigate_fixture_until_ready(
                        navigate_unavailable,
                        process,
                        startup_output,
                        output_reader,
                        "Streamlit fixture",
                        timeout=0.1,
                    )
            finally:
                _cleanup_fixture_process(process, output_reader)

        self.assertIn(startup_marker, str(failure.exception))

    def test_mobile_browser_keeps_dashboard_sections_visible(self):
        from playwright.sync_api import sync_playwright

        project_root = os.path.dirname(os.path.abspath(__file__))
        with tempfile.TemporaryDirectory() as temp_dir:
            wrapper_path = os.path.join(temp_dir, "dashboard_browser_fixture.py")
            with open(wrapper_path, "w", encoding="utf-8") as wrapper_file:
                wrapper_file.write(_browser_dashboard_wrapper())

            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(
                    headless=True,
                    executable_path=shutil.which("chromium"),
                    env={**os.environ, "HOME": temp_dir},
                )
                process = None
                page = None
                try:
                    for viewport in self.MOBILE_VIEWPORTS:
                        with self.subTest(viewport=viewport):
                            port = _find_free_port()
                            environment = {
                                **os.environ,
                                "DASHBOARD_TEST_PROVIDER_STATE": "HEALTHY",
                            }
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
                                    "--server.enableCORS",
                                    "false",
                                    "--browser.gatherUsageStats",
                                    "false",
                                ],
                                cwd=project_root,
                                env=environment,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT,
                                text=True,
                            )
                            startup_output, output_reader = _start_bounded_output_reader(
                                process
                            )
                            page = browser.new_page(
                                viewport={"width": viewport, "height": 844}
                            )
                            _navigate_fixture_until_ready(
                                lambda: page.goto(
                                    f"http://127.0.0.1:{port}",
                                    wait_until="domcontentloaded",
                                    timeout=3000,
                                ),
                                process,
                                startup_output,
                                output_reader,
                                "Streamlit fixture",
                            )

                            for section, expected_heading in self.MOBILE_SECTIONS:
                                with self.subTest(
                                    viewport=viewport, section=section
                                ):
                                    _select_dashboard_section(page, section)
                                    try:
                                        page.wait_for_function(
                                            "heading => document.body.innerText.includes(heading)",
                                            arg=expected_heading,
                                            timeout=20000,
                                        )
                                    except Exception as error:
                                        raise AssertionError(
                                            f"{expected_heading} did not render: "
                                            f"{page.locator('body').inner_text()}"
                                        ) from error
                                    rendered_text = page.locator("body").inner_text()
                                    self.assertIn(expected_heading, rendered_text)
                                    self._assert_mobile_geometry(page, viewport)
                                    if viewport == 320:
                                        self._assert_tiny_chart_geometry(
                                            page, viewport
                                        )
                            page.close()
                            page = None
                            process.terminate()
                            try:
                                process.wait(timeout=5)
                            except subprocess.TimeoutExpired:
                                process.kill()
                                process.wait()
                finally:
                    if page is not None:
                        page.close()
                    if process is not None:
                        process.terminate()
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
                    browser.close()

    def _run_orbit_summary_snapshot_comparison(
        self,
        *,
        browser_engine="chromium",
        snapshot_viewports=ORBIT_SNAPSHOT_VIEWPORTS,
        baseline_prefix="orbit-summary",
        user_agent=None,
        browser_zoom_factors=(1.0,),
    ):
        """Compare the Overview shell at approved widths and zoom levels."""
        from io import BytesIO

        from PIL import Image
        from playwright.sync_api import sync_playwright

        project_root = os.path.dirname(os.path.abspath(__file__))
        update_baselines = os.environ.get("ORBIT_SNAPSHOT_UPDATE") == "1"
        update_approved = os.environ.get("ORBIT_SNAPSHOT_UPDATE_APPROVED") == "1"
        if update_baselines and not update_approved:
            self.fail(
                "Baseline updates require explicit review approval: set both "
                "ORBIT_SNAPSHOT_UPDATE=1 and ORBIT_SNAPSHOT_UPDATE_APPROVED=1."
            )
        diff_dir = os.environ.get(
            "ORBIT_SNAPSHOT_DIFF_DIR",
            os.path.join(project_root, "artifacts", "orbit-summary-diffs"),
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            wrapper_path = os.path.join(temp_dir, "dashboard_browser_fixture.py")
            with open(wrapper_path, "w", encoding="utf-8") as wrapper_file:
                wrapper_file.write(_browser_dashboard_wrapper())
            fixture_data_dir = os.path.join(temp_dir, "observation-data")
            _write_orbit_observation_fixture(fixture_data_dir)

            with sync_playwright() as playwright:
                browser_type = getattr(playwright, browser_engine)
                browser_kwargs = {
                    "headless": True,
                    "env": {**os.environ, "HOME": temp_dir},
                }
                if browser_engine == "chromium":
                    browser_kwargs["executable_path"] = shutil.which("chromium")
                browser = browser_type.launch(
                    **browser_kwargs,
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
                            "--server.enableCORS",
                            "false",
                            "--browser.gatherUsageStats",
                            "false",
                        ],
                        cwd=project_root,
                        env={
                            **os.environ,
                            "DASHBOARD_TEST_PROVIDER_STATE": "HEALTHY",
                            "OBSERVATION_DATA_DIR": fixture_data_dir,
                        },
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                    )
                    startup_output, output_reader = _start_bounded_output_reader(process)

                    for snapshot_name, viewport_width, viewport_height in snapshot_viewports:
                        for browser_zoom in browser_zoom_factors:
                            if page is not None:
                                page.close()
                            page_kwargs = {
                                "viewport": {
                                    "width": viewport_width,
                                    "height": viewport_height,
                                }
                            }
                            if user_agent:
                                page_kwargs["user_agent"] = user_agent
                            page = browser.new_page(**page_kwargs)
                            _navigate_fixture_until_ready(
                                lambda: page.goto(
                                    f"http://127.0.0.1:{port}",
                                    wait_until="domcontentloaded",
                                    timeout=3000,
                                ),
                                process,
                                startup_output,
                                output_reader,
                                "Orbit Summary fixture",
                            )
                            page.wait_for_function(
                                "heading => document.body.innerText.includes(heading)",
                                 arg="Paper operations progress",
                                timeout=20000,
                            )
                            page.wait_for_function(
                                """() => document.querySelectorAll(
                                     '.st-key-overview_nav_grid a'
                                     ).length >= 6""",
                                timeout=20000,
                            )
                            # Streamlit component iframes can be attached before
                            # their document has hydrated.  Wait for the
                            # assistant's actual root so the screenshot and the
                            # structural assertion below observe the same
                            # rendered state across browser engines.
                            page.frame_locator("iframe").locator("#orb").wait_for(
                                state="visible",
                                timeout=20000,
                            )
                            self.assertEqual(
                                page.locator('[data-testid="stSidebar"]').count(),
                                0,
                            )
                            # Playwright does not expose Firefox's browser zoom
                            # setting. CSS zoom gives the same reduced effective
                            # layout width while retaining a deterministic
                            # 1280x900 screenshot surface in hosted CI.
                            page.evaluate(
                                """zoomFactor => {
                                    document.documentElement.style.zoom =
                                        `${zoomFactor * 100}%`;
                                }""",
                                browser_zoom,
                            )
                            page.add_style_tag(
                                content="""
                                    *, *::before, *::after {
                                        animation: none !important;
                                        transition: none !important;
                                        caret-color: transparent !important;
                                    }
                                """
                            )

                            screenshot = page.screenshot(full_page=False)
                            with Image.open(BytesIO(screenshot)) as image:
                                self.assertEqual(
                                    image.size,
                                    (viewport_width, viewport_height),
                                    "Orbit snapshot did not use the requested viewport.",
                                )
                            self.assertGreater(
                                len(screenshot),
                                10_000,
                                "Orbit snapshot is suspiciously empty.",
                            )
                            baseline_path = os.path.join(
                                ORBIT_SNAPSHOT_BASELINE_DIR,
                                (
                                    f"{baseline_prefix}-{snapshot_name}"
                                    f"-zoom-{browser_zoom:g}.png"
                                    if browser_zoom != 1.0
                                    else f"{baseline_prefix}-{snapshot_name}.png"
                                ),
                            )
                            diff_path = os.path.join(
                                diff_dir,
                                (
                                    f"orbit-summary-{snapshot_name}"
                                    f"-zoom-{browser_zoom:g}-diff.png"
                                    if browser_zoom != 1.0
                                    else f"orbit-summary-{snapshot_name}-diff.png"
                                ),
                            )
                            if update_baselines:
                                os.makedirs(ORBIT_SNAPSHOT_BASELINE_DIR, exist_ok=True)
                                with open(baseline_path, "wb") as baseline_file:
                                    baseline_file.write(screenshot)
                            else:
                                _compare_orbit_snapshot_to_baseline(
                                    screenshot,
                                    baseline_path,
                                    diff_path,
                                )

                            overview = page.locator(".orbit-overview")
                            self.assertEqual(overview.count(), 1)
                            self.assertEqual(page.locator(".orbit-shell").count(), 1)
                            self.assertEqual(
                                page.locator(".orbit-observation-card").count(), 1
                            )
                            self.assertEqual(page.locator(".orbit-mission").count(), 0)
                            self.assertEqual(page.locator(".orbit-core").count(), 0)
                            self.assertEqual(
                                page.locator(".st-key-overview_market .mc-chart-card").count(),
                                1,
                            )
                            self.assertGreaterEqual(
                                page.locator(".orbit-panel").count(),
                                4,
                                "Orbit Summary lost one or more progress cards.",
                            )
                            self.assertGreaterEqual(
                                page.locator(".orbit-progress").count(),
                                2,
                                "Orbit Summary lost its mission/progress indicators.",
                            )
                            assistant_frames = [
                                frame
                                for frame in page.frames
                                if frame.locator("#orb").count()
                            ]
                            self.assertEqual(
                                len(assistant_frames),
                                1,
                                "Orbit Summary lost its floating voice assistant.",
                            )
                            self.assertEqual(
                                page.locator(".mc-breadcrumb").count(),
                                0,
                                "The legacy breadcrumb returned to Overview.",
                            )
                            self.assertEqual(
                                page.locator(".mc-navbar-title, .mc-title").count(),
                                0,
                                "The duplicate legacy title shell returned to Overview.",
                            )
                            rendered_text = overview.inner_text()
                            self.assertNotIn("COMMAND STATUS", rendered_text)
                            self.assertNotIn("OPEN KOVA", rendered_text)

                            geometry = page.evaluate(
                                """zoomFactor => {
                                    const overview = document.querySelector(
                                        ".orbit-overview"
                                    );
                                    const box = overview.getBoundingClientRect();
                                    return {
                                        viewport: window.innerWidth,
                                        documentWidth: document.documentElement.scrollWidth,
                                        bodyWidth: document.body.scrollWidth,
                                        left: box.left,
                                        right: box.right,
                                        top: box.top,
                                        bottom: box.bottom,
                                    };
                                }""",
                                browser_zoom,
                            )
                            self.assertEqual(geometry["viewport"], viewport_width)
                            # Firefox exposes the CSS layout width before zoom
                            # when zooming out, but caps scrollWidth at the
                            # physical viewport when zooming in.
                            effective_width = max(
                                viewport_width,
                                viewport_width / browser_zoom,
                            )
                            self.assertLessEqual(
                                geometry["documentWidth"],
                                effective_width + 1,
                                "Orbit Summary overflows the document horizontally.",
                            )
                            self.assertLessEqual(
                                geometry["bodyWidth"],
                                effective_width + 1,
                                "Orbit Summary overflows the body horizontally.",
                            )
                            self.assertGreater(geometry["right"], geometry["left"])
                            self.assertGreater(geometry["bottom"], geometry["top"])
                            page.screenshot(
                                path=os.path.join(
                                    temp_dir,
                                    (
                                        f"orbit-summary-{snapshot_name}"
                                        f"-zoom-{browser_zoom:g}.png"
                                    ),
                                ),
                                full_page=False,
                            )
                finally:
                    if page is not None:
                        page.close()
                    if process is not None:
                        process.terminate()
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
                        output_reader.join(timeout=1)
                        process.stdout.close()
                    browser.close()

    def test_orbit_summary_snapshots_protect_responsive_composition(self):
        """Capture the Overview shell at release-critical desktop/mobile widths."""
        self._run_orbit_summary_snapshot_comparison()

    def test_orbit_summary_webkit_mobile_snapshot_protects_safari_composition(self):
        """Compare the iPhone-sized Overview shell in the WebKit engine."""
        if os.environ.get("RUN_WEBKIT_BROWSER_TESTS") != "1":
            self.skipTest(
                "Hosted WebKit is required; set RUN_WEBKIT_BROWSER_TESTS=1 to run."
            )
        self._run_orbit_summary_snapshot_comparison(
            browser_engine="webkit",
            snapshot_viewports=(("mobile-390", 390, 844),),
            baseline_prefix="orbit-summary-webkit",
            user_agent=self.MOBILE_BROWSER_PROFILES[0][2],
        )

    def test_orbit_summary_webkit_portrait_tablet_snapshot_protects_safari_composition(
        self,
    ):
        """Compare the approved portrait-tablet Overview shell in WebKit."""
        if os.environ.get("RUN_WEBKIT_BROWSER_TESTS") != "1":
            self.skipTest(
                "Hosted WebKit is required; set RUN_WEBKIT_BROWSER_TESTS=1 to run."
            )
        self._run_orbit_summary_snapshot_comparison(
            browser_engine="webkit",
            snapshot_viewports=(("portrait-tablet", 768, 1024),),
            baseline_prefix="orbit-summary-webkit",
            user_agent=self.MOBILE_BROWSER_PROFILES[0][2],
        )

    def test_orbit_summary_webkit_landscape_tablet_snapshot_protects_safari_composition(
        self,
    ):
        """Compare the approved landscape-tablet Overview shell in WebKit."""
        if os.environ.get("RUN_WEBKIT_BROWSER_TESTS") != "1":
            self.skipTest(
                "Hosted WebKit is required; set RUN_WEBKIT_BROWSER_TESTS=1 to run."
            )
        self._run_orbit_summary_snapshot_comparison(
            browser_engine="webkit",
            snapshot_viewports=(("landscape-tablet", 1024, 768),),
            baseline_prefix="orbit-summary-webkit",
            user_agent=self.MOBILE_BROWSER_PROFILES[0][2],
        )

    def test_orbit_summary_webkit_desktop_zoom_snapshots_protect_wrapping(self):
        """Compare WebKit desktop composition at representative browser zoom."""
        if os.environ.get("RUN_WEBKIT_BROWSER_TESTS") != "1":
            self.skipTest(
                "Hosted WebKit is required; set RUN_WEBKIT_BROWSER_TESTS=1 to run."
            )
        self._run_orbit_summary_snapshot_comparison(
            browser_engine="webkit",
            snapshot_viewports=(("desktop", 1280, 900),),
            baseline_prefix="orbit-summary-webkit",
            browser_zoom_factors=(0.8, 1.25),
        )

    def test_orbit_summary_firefox_mobile_snapshot_protects_firefox_composition(self):
        """Compare the Android-sized Overview shell in the Firefox engine."""
        if os.environ.get("RUN_FIREFOX_BROWSER_TESTS") != "1":
            self.skipTest(
                "Hosted Firefox is required; set RUN_FIREFOX_BROWSER_TESTS=1 to run."
            )
        self._run_orbit_summary_snapshot_comparison(
            browser_engine="firefox",
            snapshot_viewports=(("mobile-390", 390, 844),),
            baseline_prefix="orbit-summary-firefox",
            user_agent=self.MOBILE_BROWSER_PROFILES[2][2],
        )

    def test_orbit_summary_firefox_desktop_snapshot_protects_firefox_typography(self):
        """Compare the desktop Overview shell in the hosted Firefox engine."""
        if os.environ.get("RUN_FIREFOX_BROWSER_TESTS") != "1":
            self.skipTest(
                "Hosted Firefox is required; set RUN_FIREFOX_BROWSER_TESTS=1 to run."
            )
        self._run_orbit_summary_snapshot_comparison(
            browser_engine="firefox",
            snapshot_viewports=(("desktop", 1280, 900),),
            baseline_prefix="orbit-summary-firefox",
        )

    def test_orbit_summary_firefox_desktop_zoom_snapshots_protect_wrapping(self):
        """Compare Firefox desktop composition at representative browser zoom."""
        if os.environ.get("RUN_FIREFOX_BROWSER_TESTS") != "1":
            self.skipTest(
                "Hosted Firefox is required; set RUN_FIREFOX_BROWSER_TESTS=1 to run."
            )
        self._run_orbit_summary_snapshot_comparison(
            browser_engine="firefox",
            snapshot_viewports=(("desktop", 1280, 900),),
            baseline_prefix="orbit-summary-firefox",
            browser_zoom_factors=(0.8, 1.25),
        )

    def test_orbit_summary_firefox_landscape_tablet_snapshot_protects_responsive_composition(
        self,
    ):
        """Compare the approved landscape-tablet Overview shell in Firefox."""
        if os.environ.get("RUN_FIREFOX_BROWSER_TESTS") != "1":
            self.skipTest(
                "Hosted Firefox is required; set RUN_FIREFOX_BROWSER_TESTS=1 to run."
            )
        self._run_orbit_summary_snapshot_comparison(
            browser_engine="firefox",
            snapshot_viewports=(("landscape-tablet", 1024, 768),),
            baseline_prefix="orbit-summary-firefox",
        )

    def test_orbit_summary_firefox_portrait_tablet_snapshot_protects_responsive_composition(
        self,
    ):
        """Compare the approved portrait-tablet Overview shell in Firefox."""
        if os.environ.get("RUN_FIREFOX_BROWSER_TESTS") != "1":
            self.skipTest(
                "Hosted Firefox is required; set RUN_FIREFOX_BROWSER_TESTS=1 to run."
            )
        self._run_orbit_summary_snapshot_comparison(
            browser_engine="firefox",
            snapshot_viewports=(("portrait-tablet", 768, 1024),),
            baseline_prefix="orbit-summary-firefox",
        )

    def test_mobile_browser_keeps_provider_health_states_visible(self):
        from playwright.sync_api import sync_playwright

        project_root = os.path.dirname(os.path.abspath(__file__))
        with tempfile.TemporaryDirectory() as temp_dir:
            wrapper_path = os.path.join(temp_dir, "dashboard_browser_fixture.py")
            with open(wrapper_path, "w", encoding="utf-8") as wrapper_file:
                wrapper_file.write(_browser_dashboard_wrapper())

            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(
                    headless=True,
                    executable_path=shutil.which("chromium"),
                    env={**os.environ, "HOME": temp_dir},
                )
                try:
                    for viewport in self.MOBILE_VIEWPORTS:
                        for state in ("HEALTHY", "DEGRADED", "UNAVAILABLE"):
                            with self.subTest(
                                viewport=viewport, provider_state=state
                            ):
                                port = _find_free_port()
                                environment = {
                                    **os.environ,
                                    "DASHBOARD_TEST_PROVIDER_STATE": state,
                                }
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
                                        "--server.enableCORS",
                                        "false",
                                        "--browser.gatherUsageStats",
                                        "false",
                                    ],
                                    cwd=project_root,
                                    env=environment,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT,
                                    text=True,
                                )
                                startup_output, output_reader = (
                                    _start_bounded_output_reader(process)
                                )
                                page = browser.new_page(
                                    viewport={"width": viewport, "height": 844}
                                )
                                _navigate_fixture_until_ready(
                                    lambda: page.goto(
                                        f"http://127.0.0.1:{port}",
                                        wait_until="domcontentloaded",
                                        timeout=3000,
                                    ),
                                    process,
                                    startup_output,
                                    output_reader,
                                    "Streamlit fixture",
                                )

                                try:
                                    _select_dashboard_section(page, "SYSTEM")
                                    page.get_by_text(
                                        "SYSTEM HEALTH", exact=True
                                    ).wait_for(timeout=10000)
                                    page.get_by_text(
                                        "Provider diagnostics", exact=True
                                    ).click()
                                    page.get_by_text(
                                        f"PROVIDER: {state}", exact=True
                                    ).wait_for(timeout=10000)

                                    self.assertIn(
                                        f"PROVIDER: {state}",
                                        page.locator("body").inner_text(),
                                    )
                                    rendered_text = page.locator("body").inner_text()
                                    for text in (
                                        "PROVIDER",
                                        "Managed provider",
                                        "REQUESTS",
                                        "7",
                                        "SUCCESS RATE",
                                        "LAST OUTCOME",
                                        "LATEST FAILURE CATEGORY",
                                        "Failure category counts",
                                    ):
                                        self.assertIn(
                                            text,
                                            rendered_text,
                                        )

                                    geometry = page.evaluate(
                                        """() => ({
                                            viewport: window.innerWidth,
                                            documentWidth: document.documentElement.scrollWidth,
                                            bodyWidth: document.body.scrollWidth,
                                            values: [...document.querySelectorAll(
                                                '.mc-data-label, .mc-data-value'
                                            )].map((element) => {
                                                const box = element.getBoundingClientRect();
                                                return {
                                                    text: element.innerText,
                                                    left: box.left,
                                                    right: box.right,
                                                    top: box.top,
                                                    bottom: box.bottom,
                                                };
                                            }),
                                        })"""
                                    )
                                    self.assertEqual(geometry["viewport"], viewport)
                                    self.assertLessEqual(
                                        geometry["documentWidth"],
                                        geometry["viewport"] + 1,
                                    )
                                    self.assertLessEqual(
                                        geometry["bodyWidth"],
                                        geometry["viewport"] + 1,
                                    )
                                    self.assertTrue(geometry["values"])
                                    for item in geometry["values"]:
                                        self.assertGreaterEqual(item["left"], -1)
                                        self.assertLessEqual(
                                            item["right"],
                                            viewport + 1,
                                        )
                                        self.assertGreater(item["bottom"], item["top"])
                                finally:
                                    page.close()
                                    process.terminate()
                                    try:
                                        process.wait(timeout=5)
                                    except subprocess.TimeoutExpired:
                                        process.kill()
                                        process.wait()
                finally:
                    browser.close()

    def test_landscape_tablet_browser_keeps_chart_details_readable(self):
        from playwright.sync_api import sync_playwright

        project_root = os.path.dirname(os.path.abspath(__file__))
        viewport_width, viewport_height = self.LANDSCAPE_TABLET_VIEWPORT
        with tempfile.TemporaryDirectory() as temp_dir:
            wrapper_path = os.path.join(temp_dir, "dashboard_browser_fixture.py")
            with open(wrapper_path, "w", encoding="utf-8") as wrapper_file:
                wrapper_file.write(_browser_dashboard_wrapper())

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
                    environment = {
                        **os.environ,
                        "DASHBOARD_TEST_PROVIDER_STATE": "HEALTHY",
                    }
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
                            "--server.enableCORS",
                            "false",
                            "--browser.gatherUsageStats",
                            "false",
                        ],
                        cwd=project_root,
                        env=environment,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                    )
                    startup_output, output_reader = _start_bounded_output_reader(process)
                    page = browser.new_page(
                        viewport={
                            "width": viewport_width,
                            "height": viewport_height,
                        }
                    )
                    _navigate_fixture_until_ready(
                        lambda: page.goto(
                            f"http://127.0.0.1:{port}",
                            wait_until="domcontentloaded",
                            timeout=3000,
                        ),
                        process,
                        startup_output,
                        output_reader,
                        "Streamlit landscape-tablet fixture",
                    )

                    for zoom_factor in self.BROWSER_ZOOM_FACTORS:
                        with self.subTest(
                            viewport=self.LANDSCAPE_TABLET_VIEWPORT,
                            browser_zoom=zoom_factor,
                        ):
                            # CSS zoom is deterministic in headless Chromium and
                            # models the reduced effective content width caused by
                            # browser zoom without changing the physical viewport.
                            page.evaluate(
                                """zoomFactor => {
                                    document.documentElement.style.zoom =
                                        `${zoomFactor * 100}%`;
                                }""",
                                zoom_factor,
                            )
                            for section in ("PERFORMANCE",):
                                with self.subTest(
                                    viewport=self.LANDSCAPE_TABLET_VIEWPORT,
                                    browser_zoom=zoom_factor,
                                    section=section,
                                ):
                                    _select_dashboard_section(page, section)
                                    try:
                                        page.wait_for_function(
                                            "heading => document.body.innerText.includes(heading)",
                                            arg=(
                                                "Historical batch backtest results summary"
                                                if section == "PERFORMANCE"
                                                else "HISTORICAL BATCH BACKTEST"
                                            ),
                                            timeout=20000,
                                        )
                                    except Exception as error:
                                        raise AssertionError(
                                            f"{section} did not render at "
                                            f"{zoom_factor:.2f}x browser zoom: "
                                            f"{page.locator('body').inner_text()}"
                                        ) from error
                                    if section == "PERFORMANCE":
                                        page.wait_for_selector(
                                            ".mc-chart-card",
                                            state="attached",
                                            timeout=20000,
                                        )
                                    self._assert_landscape_tablet_chart_geometry(
                                        page, zoom_factor
                                    )
                finally:
                    if page is not None:
                        page.close()
                    if process is not None:
                        process.terminate()
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
                    browser.close()

    def test_portrait_browser_keeps_chart_details_readable_across_zoom_levels(self):
        from playwright.sync_api import sync_playwright

        project_root = os.path.dirname(os.path.abspath(__file__))
        viewport_width, viewport_height = self.MOBILE_VIEWPORTS[0], 844
        with tempfile.TemporaryDirectory() as temp_dir:
            wrapper_path = os.path.join(temp_dir, "dashboard_browser_fixture.py")
            with open(wrapper_path, "w", encoding="utf-8") as wrapper_file:
                wrapper_file.write(_browser_dashboard_wrapper())

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
                    environment = {
                        **os.environ,
                        "DASHBOARD_TEST_PROVIDER_STATE": "HEALTHY",
                    }
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
                            "--server.enableCORS",
                            "false",
                            "--browser.gatherUsageStats",
                            "false",
                        ],
                        cwd=project_root,
                        env=environment,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                    )
                    startup_output, output_reader = _start_bounded_output_reader(process)
                    page = browser.new_page(
                        viewport={
                            "width": viewport_width,
                            "height": viewport_height,
                        }
                    )
                    _navigate_fixture_until_ready(
                        lambda: page.goto(
                            f"http://127.0.0.1:{port}",
                            wait_until="domcontentloaded",
                            timeout=3000,
                        ),
                        process,
                        startup_output,
                        output_reader,
                        "Streamlit portrait fixture",
                    )

                    for zoom_factor in self.BROWSER_ZOOM_FACTORS:
                        page.evaluate(
                            """zoomFactor => {
                                document.documentElement.style.zoom =
                                    `${zoomFactor * 100}%`;
                            }""",
                            zoom_factor,
                        )
                        for section, expected_heading in self.CHART_SECTIONS:
                            with self.subTest(
                                viewport=(viewport_width, viewport_height),
                                browser_zoom=zoom_factor,
                                section=section,
                            ):
                                _select_dashboard_section(page, section)
                                try:
                                    page.wait_for_function(
                                        "heading => document.body.innerText.includes(heading)",
                                        arg=expected_heading,
                                        timeout=20000,
                                    )
                                except Exception as error:
                                    raise AssertionError(
                                        f"{section} did not render at "
                                        f"{zoom_factor:.2f}x browser zoom: "
                                        f"{page.locator('body').inner_text()}"
                                    ) from error
                                self._assert_portrait_chart_geometry(
                                    page, zoom_factor
                                )
                finally:
                    if page is not None:
                        page.close()
                    if process is not None:
                        process.terminate()
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
                    browser.close()

    def test_narrowest_portrait_browser_keeps_chart_details_with_high_zoom(self):
        from playwright.sync_api import sync_playwright

        project_root = os.path.dirname(os.path.abspath(__file__))
        viewport_width, viewport_height = self.MOBILE_VIEWPORTS[1], 844
        zoom_factor = self.BROWSER_ZOOM_FACTORS[-1]
        with tempfile.TemporaryDirectory() as temp_dir:
            wrapper_path = os.path.join(temp_dir, "dashboard_browser_fixture.py")
            with open(wrapper_path, "w", encoding="utf-8") as wrapper_file:
                wrapper_file.write(_browser_dashboard_wrapper())

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
                    environment = {
                        **os.environ,
                        "DASHBOARD_TEST_PROVIDER_STATE": "HEALTHY",
                    }
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
                            "--server.enableCORS",
                            "false",
                            "--browser.gatherUsageStats",
                            "false",
                        ],
                        cwd=project_root,
                        env=environment,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                    )
                    startup_output, output_reader = _start_bounded_output_reader(process)
                    page = browser.new_page(
                        viewport={
                            "width": viewport_width,
                            "height": viewport_height,
                        }
                    )
                    _navigate_fixture_until_ready(
                        lambda: page.goto(
                            f"http://127.0.0.1:{port}",
                            wait_until="domcontentloaded",
                            timeout=3000,
                        ),
                        process,
                        startup_output,
                        output_reader,
                        "Streamlit narrow portrait fixture",
                    )

                    page.evaluate(
                        """zoomFactor => {
                            document.documentElement.style.zoom =
                                `${zoomFactor * 100}%`;
                        }""",
                        zoom_factor,
                    )
                    for section, expected_heading in (
                        ("PERFORMANCE", "Historical batch backtest results summary"),
                    ):
                        with self.subTest(
                            viewport=(viewport_width, viewport_height),
                            browser_zoom=zoom_factor,
                            section=section,
                        ):
                            _select_dashboard_section(page, section)
                            try:
                                page.wait_for_function(
                                    "heading => document.body.innerText.includes(heading)",
                                    arg=expected_heading,
                                    timeout=20000,
                                )
                            except Exception as error:
                                raise AssertionError(
                                    f"{section} did not render at "
                                    f"{viewport_width}px and "
                                    f"{zoom_factor:.2f}x browser zoom: "
                                    f"{page.locator('body').inner_text()}"
                                ) from error
                            self._assert_portrait_chart_geometry(
                                page, zoom_factor, expected_width=viewport_width
                            )
                finally:
                    if page is not None:
                        page.close()
                    if process is not None:
                        process.terminate()
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
                    browser.close()

    def test_mobile_browser_profiles_keep_chart_details_readable(self):
        """Exercise mobile browser layout with representative Safari/Chrome profiles."""
        from playwright.sync_api import sync_playwright

        project_root = os.path.dirname(os.path.abspath(__file__))
        with tempfile.TemporaryDirectory() as temp_dir:
            wrapper_path = os.path.join(temp_dir, "dashboard_browser_fixture.py")
            with open(wrapper_path, "w", encoding="utf-8") as wrapper_file:
                wrapper_file.write(_browser_dashboard_wrapper())

            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(
                    headless=True,
                    executable_path=shutil.which("chromium"),
                    env={**os.environ, "HOME": temp_dir},
                )
                process = None
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
                            "--server.enableCORS",
                            "false",
                            "--browser.gatherUsageStats",
                            "false",
                        ],
                        cwd=project_root,
                        env={
                            **os.environ,
                            "DASHBOARD_TEST_PROVIDER_STATE": "HEALTHY",
                        },
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                    )
                    startup_output, output_reader = _start_bounded_output_reader(process)
                    for profile_name, viewport_width, user_agent in (
                        self.MOBILE_BROWSER_PROFILES
                    ):
                        with self.subTest(
                            browser_profile=profile_name,
                            viewport_width=viewport_width,
                        ):
                            context = browser.new_context(
                                viewport={"width": viewport_width, "height": 844},
                                user_agent=user_agent,
                                is_mobile=True,
                                has_touch=True,
                                device_scale_factor=2,
                            )
                            page = context.new_page()
                            try:
                                _navigate_fixture_until_ready(
                                    lambda: page.goto(
                                        f"http://127.0.0.1:{port}",
                                        wait_until="domcontentloaded",
                                        timeout=3000,
                                    ),
                                    process,
                                    startup_output,
                                    output_reader,
                                    f"{profile_name} fixture",
                                )

                                self.assertIn(
                                    "Mobile",
                                    page.evaluate("navigator.userAgent"),
                                )
                                for section, expected_heading in self.CHART_SECTIONS:
                                    with self.subTest(
                                        browser_profile=profile_name,
                                        section=section,
                                    ):
                                        _select_dashboard_section(page, section)
                                        try:
                                            page.wait_for_function(
                                                (
                                                    "heading => "
                                                    "document.body.innerText.includes(heading)"
                                                ),
                                                arg=expected_heading,
                                                timeout=20000,
                                            )
                                        except Exception as error:
                                            raise AssertionError(
                                                f"{expected_heading} did not render for "
                                                f"{profile_name}: "
                                                f"{page.locator('body').inner_text()}"
                                            ) from error
                                        self._assert_portrait_chart_geometry(
                                            page,
                                            zoom_factor=1.0,
                                            expected_width=viewport_width,
                                        )
                            finally:
                                page.close()
                                context.close()
                finally:
                    if process is not None:
                        process.terminate()
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
                    browser.close()


class DashboardOverviewFocusTests(unittest.TestCase):
    def test_hosted_auth_appearance_restores_for_two_accounts(self):
        """Exercise real hosted auth when approved storage states are supplied."""
        hosted_url = os.environ.get("KOVA_HOSTED_DASHBOARD_URL")
        account_a_state = os.environ.get("KOVA_HOSTED_AUTH_STORAGE_STATE_A")
        account_b_state = os.environ.get("KOVA_HOSTED_AUTH_STORAGE_STATE_B")
        if not hosted_url or not account_a_state or not account_b_state:
            self.skipTest(
                "Hosted auth validation requires KOVA_HOSTED_DASHBOARD_URL and "
                "two approved Playwright storage-state files."
            )
        for state_path in (account_a_state, account_b_state):
            if not os.path.isfile(state_path):
                self.skipTest(f"Hosted auth storage state is missing: {state_path}")

        from playwright.sync_api import sync_playwright

        expected = (
            (account_a_state, "Light", "#f4f7fb"),
            (account_b_state, "Dark", "#030c1d"),
        )
        settings_url = hosted_url.rstrip("/") + "/?section=SETTINGS"
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(
                headless=True,
                executable_path=shutil.which("chromium"),
            )
            try:
                for storage_state, appearance, background in expected:
                    context = browser.new_context(storage_state=storage_state)
                    page = context.new_page()
                    try:
                        page.goto(settings_url, wait_until="domcontentloaded", timeout=30000)
                        page.wait_for_function(
                            "() => document.body.innerText.includes('SETTINGS & PREFERENCES')",
                            timeout=30000,
                        )
                        page.get_by_text(appearance, exact=True).click()
                        page.wait_for_timeout(750)
                    finally:
                        page.close()
                        context.close()

                    fresh_context = browser.new_context(storage_state=storage_state)
                    fresh_page = fresh_context.new_page()
                    try:
                        fresh_page.goto(
                            settings_url,
                            wait_until="domcontentloaded",
                            timeout=30000,
                        )
                        fresh_page.wait_for_function(
                            "() => document.body.innerText.includes('SETTINGS & PREFERENCES')",
                            timeout=30000,
                        )
                        selected = fresh_page.evaluate(
                            """() => Array.from(
                                document.querySelectorAll("input[type='radio']")
                            ).find(input => input.checked)?.closest("label")
                                ?.innerText.trim() || ''"""
                        )
                        current_background = fresh_page.evaluate(
                            """() => getComputedStyle(
                                document.documentElement
                            ).getPropertyValue("--mc-bg").trim()"""
                        )
                        self.assertEqual(selected, appearance)
                        self.assertEqual(current_background, background)
                    finally:
                        fresh_page.close()
                        fresh_context.close()

                anonymous = browser.new_context()
                page = anonymous.new_page()
                try:
                    page.goto(settings_url, wait_until="domcontentloaded", timeout=30000)
                    page.wait_for_function(
                        "() => document.body.innerText.includes('SETTINGS & PREFERENCES')",
                        timeout=30000,
                    )
                    current_background = page.evaluate(
                        """() => getComputedStyle(
                            document.documentElement
                        ).getPropertyValue("--mc-bg").trim()"""
                    )
                    self.assertEqual(current_background, "#030c1d")
                finally:
                    page.close()
                    anonymous.close()
            finally:
                browser.close()

    def test_authenticated_appearance_restores_across_fresh_browser_contexts(self):
        """Verify account appearance state survives a new browser session."""
        from playwright.sync_api import sync_playwright

        expected_backgrounds = {
            "Light": "#f4f7fb",
            "Dark": "#030c1d",
        }
        project_root = os.path.dirname(os.path.abspath(__file__))

        with tempfile.TemporaryDirectory() as temp_dir:
            wrapper_path = os.path.join(temp_dir, "dashboard_browser_fixture.py")
            preferences_path = os.path.join(temp_dir, "dashboard_preferences.json")
            auth_path = os.path.join(temp_dir, "dashboard_test_auth.json")
            with open(wrapper_path, "w", encoding="utf-8") as wrapper_file:
                wrapper_file.write(_browser_dashboard_wrapper())
            with open(auth_path, "w", encoding="utf-8") as auth_file:
                json.dump({}, auth_file)

            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(
                    headless=True,
                    executable_path=shutil.which("chromium"),
                    env={**os.environ, "HOME": temp_dir},
                )
                process = None
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
                            "--server.enableCORS",
                            "false",
                            "--browser.gatherUsageStats",
                            "false",
                        ],
                        cwd=project_root,
                        env={
                            **os.environ,
                            "DASHBOARD_TEST_PROVIDER_STATE": "HEALTHY",
                            "KOVA_DASHBOARD_PREFERENCES_PATH": preferences_path,
                            "KOVA_TEST_AUTH_PATH": auth_path,
                        },
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                    )
                    startup_output, output_reader = _start_bounded_output_reader(
                        process
                    )
                    base_url = f"http://127.0.0.1:{port}"
                    with open(preferences_path, "w", encoding="utf-8") as preferences_file:
                        json.dump(
                            {
                                "users": {
                                    "sub:browser-account-a": {"appearance": "Light"},
                                    "sub:browser-account-b": {"appearance": "Dark"},
                                }
                            },
                            preferences_file,
                        )

                    def open_settings(account=None):
                        with open(auth_path, "w", encoding="utf-8") as auth_file:
                            json.dump(
                                {} if account is None else {"user": account},
                                auth_file,
                            )
                        context_options = {
                            "viewport": {"width": 1280, "height": 900}
                        }
                        context = browser.new_context(**context_options)
                        page = context.new_page()
                        url = base_url
                        _navigate_fixture_until_ready(
                            lambda: page.goto(
                                url,
                                wait_until="domcontentloaded",
                                timeout=3000,
                            ),
                            process,
                            startup_output,
                            output_reader,
                            "Appearance restore fixture",
                        )
                        page.wait_for_selector(
                            ".st-key-overview_nav_grid",
                            state="visible",
                            timeout=20000,
                        )
                        _select_dashboard_section(page, "SETTINGS")
                        page.wait_for_function(
                            """() => document.body.innerText.includes(
                                "SETTINGS & PREFERENCES"
                            )""",
                            timeout=20000,
                        )
                        return context, page

                    def current_background(page):
                        return page.evaluate(
                            """() => getComputedStyle(
                                document.documentElement
                            ).getPropertyValue("--mc-bg").trim()"""
                        )

                    def selected_appearance(page):
                        return page.evaluate(
                            """() => Array.from(
                                document.querySelectorAll("input[type='radio']")
                            ).find(input => input.checked)?.closest("label")
                                ?.innerText.trim() || ''"""
                        )

                    # A newly created context for each account must restore its
                    # own durable value, as it would after sign-in.
                    for account, appearance in (
                        ("account-a", "Light"),
                        ("account-b", "Dark"),
                    ):
                        context, page = open_settings(account)
                        try:
                            self.assertEqual(
                                selected_appearance(page),
                                appearance,
                            )
                            self.assertEqual(
                                current_background(page),
                                expected_backgrounds[appearance],
                            )
                        finally:
                            page.close()
                            context.close()

                    # An anonymous session must not inherit either value.
                    context, page = open_settings()
                    try:
                        self.assertEqual(
                            current_background(page),
                            expected_backgrounds["Dark"],
                        )
                    finally:
                        page.close()
                        context.close()
                finally:
                    if process is not None:
                        process.terminate()
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
                    browser.close()

    def test_authenticated_appearance_preferences_are_isolated_and_restored(self):
        account_a = SimpleNamespace(is_logged_in=True, sub="account-a", email=None)
        account_b = SimpleNamespace(is_logged_in=True, sub="account-b", email=None)

        with tempfile.TemporaryDirectory() as temp_dir:
            preferences_path = Path(temp_dir) / "dashboard_preferences.json"
            with patch.object(dashboard, "DASHBOARD_PREFERENCES_PATH", preferences_path):
                with patch.object(dashboard.st, "user", account_a):
                    with patch.object(
                        dashboard.st,
                        "session_state",
                        {"dashboard_appearance": "Light"},
                    ):
                        dashboard._save_dashboard_appearance_preference()

                with patch.object(dashboard.st, "user", account_b):
                    with patch.object(
                        dashboard.st,
                        "session_state",
                        {"dashboard_appearance": "System Default"},
                    ):
                        dashboard._save_dashboard_appearance_preference()

                with patch.object(dashboard.st, "user", account_a):
                    self.assertEqual(
                        dashboard._load_saved_dashboard_appearance(), "Light"
                    )
                with patch.object(dashboard.st, "user", account_b):
                    self.assertEqual(
                        dashboard._load_saved_dashboard_appearance(),
                        "System Default",
                    )

                with preferences_path.open(encoding="utf-8") as preferences_file:
                    preferences = json.load(preferences_file)
                self.assertEqual(
                    set(preferences["users"]),
                    {"sub:account-a", "sub:account-b"},
                )

    def test_anonymous_appearance_is_session_only(self):
        anonymous_user = SimpleNamespace(is_logged_in=False, sub=None, email=None)

        with tempfile.TemporaryDirectory() as temp_dir:
            preferences_path = Path(temp_dir) / "dashboard_preferences.json"
            with patch.object(dashboard, "DASHBOARD_PREFERENCES_PATH", preferences_path):
                with patch.object(dashboard.st, "user", anonymous_user):
                    with patch.object(
                        dashboard.st,
                        "session_state",
                        {"dashboard_appearance": "Light"},
                    ):
                        dashboard._save_dashboard_appearance_preference()
                    self.assertIsNone(
                        dashboard._load_saved_dashboard_appearance()
                    )
            self.assertFalse(preferences_path.exists())

    def test_invalid_or_incomplete_appearance_records_fall_back_to_dark(self):
        authenticated_user = SimpleNamespace(
            is_logged_in=True, sub="account-a", email=None
        )
        invalid_values = (
            {},
            {"users": []},
            {"users": {"sub:account-a": []}},
            {"users": {"sub:account-a": {}}},
            {"users": {"sub:account-a": {"appearance": "Neon"}}},
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            preferences_path = Path(temp_dir) / "dashboard_preferences.json"
            with patch.object(dashboard, "DASHBOARD_PREFERENCES_PATH", preferences_path):
                with patch.object(dashboard.st, "user", authenticated_user):
                    for stored_preferences in invalid_values:
                        with self.subTest(stored_preferences=stored_preferences):
                            with preferences_path.open("w", encoding="utf-8") as preferences_file:
                                json.dump(stored_preferences, preferences_file)
                            restored = (
                                dashboard._load_saved_dashboard_appearance()
                                or "Dark"
                            )
                            self.assertEqual(restored, "Dark")

    def test_overview_uses_compact_entry_points_for_deeper_views(self):
        overview_source = inspect.getsource(dashboard.render_mc_overview_page)
        navigation_source = inspect.getsource(
            dashboard.render_mc_overview_navigation
        )
        navigation_renderer_source = inspect.getsource(dashboard.render_mc_navigation)
        live_monitor_source = inspect.getsource(
            dashboard.render_mc_overview_live_monitor
        )

        for removed_renderer in (
            "render_mc_position_snapshot",
            "render_mc_ai_decision",
        ):
            self.assertNotIn(removed_renderer, overview_source)
        self.assertNotIn("st.sidebar", navigation_renderer_source)
        self.assertNotIn("Current mission", overview_source)
        self.assertNotIn("Market snapshot", overview_source)
        for destination in (
            "POSITIONS",
            "STRATEGY",
            "PERFORMANCE",
            "RISK",
            "OPTIONS REVIEW",
            "SETTINGS",
        ):
            self.assertIn(f'"{destination}"', navigation_source)
            self.assertIn('href="?section={destination}"', navigation_source)
        self.assertIn('target="_self"', navigation_source)
        market_source = inspect.getsource(dashboard.render_mc_overview_market)
        settings_source = inspect.getsource(dashboard.render_mc_settings_page)
        diagnostics_source = inspect.getsource(
            dashboard.render_mc_pre_live_diagnostics
        )
        strategy_source = inspect.getsource(
            dashboard.render_mc_autonomous_portfolio_decision
        )
        research_source = inspect.getsource(dashboard.render_mc_research_lab)
        self.assertIn("section=MARKET", market_source)
        self.assertIn("render_mc_line_chart", market_source)
        self.assertIn("overview-market-chart-link", market_source)
        self.assertIn("?section=MARKET", market_source)
        self.assertIn("render_mc_system_health", settings_source)
        self.assertIn("render_mc_backtest_page", settings_source)
        self.assertIn("render_mc_research_catalogue", settings_source)
        research_provider_source = inspect.getsource(
            dashboard.render_mc_research_status
        )
        for contract_field in (
            "contract_status",
            "Contract status",
            "Contract reason",
            "adapter is reviewed",
        ):
            self.assertIn(contract_field, research_provider_source)
        for diagnostic in (
            "Paper mode",
            "Evidence reconciliation",
            "Options boundary",
            "Authenticated runner controls",
            "cannot interrupt the active genuine observation",
        ):
            self.assertIn(diagnostic, diagnostics_source)
        self.assertNotIn("paper_observation_runner", diagnostics_source)
        self.assertIn('"LIVE MONITOR"', live_monitor_source)
        self.assertIn("render_mc_line_chart", live_monitor_source)
        self.assertNotIn("render_mc_live_market", live_monitor_source)
        for field in (
            "Genuine paper decision",
            "Historical council score",
            "Historical regime context",
            "Configured allocation ceiling",
            "Paper sizing capacity",
            "Concentration / diversification",
            "Risk Governor",
            "Live execution",
            "promote into production",
        ):
            self.assertIn(field, strategy_source)
        for field in (
            "Average market benchmark",
            "Average net strategy",
            "Average net delta vs benchmark",
            "Net fees",
            "Slippage",
            "RESEARCH-ONLY",
            "live or genuine-observation result",
        ):
            self.assertIn(field, research_source)

    def test_overview_entry_points_remain_mobile_safe_and_accessible(self):
        theme_source = inspect.getsource(dashboard.inject_mission_control_theme)
        overview_theme_source = inspect.getsource(
            dashboard.render_orbit_overview_styles
        )
        navigation_source = inspect.getsource(
            dashboard.render_mc_overview_navigation
        )
        self.assertIn(".st-key-overview_nav_grid", overview_theme_source)
        self.assertIn("Explore dashboard details", navigation_source)
        self.assertIn('class="overview-nav-link"', navigation_source)
        self.assertIn('target="_self"', navigation_source)
        self.assertIn("stSidebarCollapseButton", theme_source)
        self.assertIn('content: "K O V A"', theme_source)
        self.assertIn(
            'content: "Knowledge-Oriented Virtual Assistant"',
            theme_source,
        )


class DashboardAssistantHealthTests(unittest.TestCase):
    def test_top_left_assistant_widget_navigates_to_ai_assistant(self):
        source = (os.path.dirname(__file__) + "/dashboard.py")
        with open(source, encoding="utf-8") as dashboard_file:
            dashboard_source = dashboard_file.read()
        self.assertNotIn("render_mc_assistant_widget", dashboard_source)
        self.assertNotIn("OPEN KOVA", dashboard_source)

    def test_orbit_assistant_trigger_navigates_without_session_state_error(self):
        source = (os.path.dirname(__file__) + "/dashboard.py")
        with open(source, encoding="utf-8") as dashboard_file:
            dashboard_source = dashboard_file.read()
        self.assertNotIn("render_mc_orbit_assistant_trigger", dashboard_source)
        self.assertNotIn("AI ASSISTANT", dashboard_source)

    def test_narrow_layout_keeps_provider_health_telemetry_readable(self):
        provider_health_states = (
            ("HEALTHY", "SUCCESS", "UNKNOWN", "UNKNOWN"),
            ("DEGRADED", "FAILURE", "Provider outage", "Rate limit: 2 · Provider outage: 1"),
            ("UNAVAILABLE", "FAILURE", "Timeout", "Timeout: 1"),
        )

        for availability, outcome, category, counts in provider_health_states:
            with self.subTest(availability=availability):
                provider_health = {
                    "provider": "Managed provider",
                    "availability": availability,
                    "requests": 7,
                    "successes": 4,
                    "failures": 3,
                    "success_rate_percent": 57.1,
                    "last_latency_ms": 1200.0,
                    "last_outcome": outcome,
                    "last_failure_category": (
                        None
                        if category == "UNKNOWN"
                        else (
                            "provider_outage"
                            if availability == "DEGRADED"
                            else "timeout"
                        )
                    ),
                    "failure_categories": (
                        {}
                        if counts == "UNKNOWN"
                        else {"timeout": 1}
                        if counts == "Timeout: 1"
                        else {"rate_limit": 2, "provider_outage": 1}
                    ),
                }
                with patch(
                    "dashboard.get_provider_health",
                    return_value=provider_health,
                ):
                    app = AppTest.from_function(
                        render_assistant_health_view
                    ).run()

                rendered_markdown = "\n".join(
                    item.value for item in app.markdown
                )
                rendered_metrics = {
                    item.label: item.value
                    for item in app.metric
                }
                self.assertIn(
                    "@media (max-width: 640px)", rendered_markdown
                )
                self.assertIn(
                    "flex: 1 1 100% !important", rendered_markdown
                )
                self.assertIn(
                    f"PROVIDER: {availability}", rendered_markdown
                )
                self.assertIn(
                    '<div class="mc-data-label">Provider</div>'
                    '<div class="mc-data-value">Managed provider</div>',
                    rendered_markdown,
                )
                self.assertIn(
                    '<div class="mc-data-label">Requests</div>'
                    '<div class="mc-data-value">7</div>',
                    rendered_markdown,
                )
                self.assertIn(
                    '<div class="mc-data-label">Last outcome</div>'
                    f'<div class="mc-data-value">{outcome}</div>',
                    rendered_markdown,
                )
                self.assertIn(
                    '<div class="mc-data-label">'
                    "Latest failure category</div>"
                    f'<div class="mc-data-value">{category}</div>',
                    rendered_markdown,
                )
                self.assertEqual(
                    rendered_metrics["Failure category counts"],
                    counts,
                )

    def test_full_dashboard_navigation_keeps_provider_health_readable(self):
        health_states = (
            ("HEALTHY", "SUCCESS", "UNKNOWN", "UNKNOWN"),
            ("DEGRADED", "FAILURE", "Provider outage", "Rate limit: 2 · Provider outage: 1"),
            ("UNAVAILABLE", "FAILURE", "Timeout", "Timeout: 1"),
        )
        provider_template = {
            "provider": "Managed provider",
            "requests": 1,
            "successes": 1,
            "failures": 0,
            "success_rate_percent": 100.0,
            "last_latency_ms": 1200.0,
            "failure_categories": {},
        }
        dashboard_results = {
            "evaluation_history": [{}],
        }
        market_data = SimpleNamespace(pair_name="XBT/CAD", last_error=None)

        for availability, outcome, category, counts in health_states:
            with self.subTest(availability=availability):
                provider_health = {
                    **provider_template,
                    "availability": availability,
                    "last_outcome": outcome,
                    "last_failure_category": (
                        None
                        if category == "UNKNOWN"
                        else (
                            "provider_outage"
                            if availability == "DEGRADED"
                            else "timeout"
                        )
                    ),
                    "failure_categories": (
                        {}
                        if counts == "UNKNOWN"
                        else {"timeout": 1}
                        if counts == "Timeout: 1"
                        else {"rate_limit": 2, "provider_outage": 1}
                    ),
                }
                with (
                    patch(
                        "dashboard.run_strategy_backtest",
                        return_value=dashboard_results,
                    ),
                    patch(
                        "dashboard.load_kraken_market_data",
                        return_value=(market_data, []),
                    ),
                    patch(
                        "dashboard.run_live_market_backtest",
                        return_value=None,
                    ),
                    patch(
                        "dashboard.load_historical_btc_cad_data",
                        return_value=(market_data, []),
                    ),
                    patch(
                        "dashboard.run_historical_market_backtest",
                        return_value=None,
                    ),
                    patch(
                        "dashboard.render_mc_overview_page",
                        return_value=None,
                    ),
                    patch(
                        "dashboard.get_provider_health",
                        return_value=provider_health,
                    ),
                ):
                    app = AppTest.from_function(
                        render_full_dashboard
                    ).run()
                    self.assertEqual(len(app.sidebar.radio), 0)
                    self.assertIn(
                        "overview_nav_grid",
                        inspect.getsource(dashboard.render_mc_overview_navigation),
                    )
                    continue

                rendered_markdown = "\n".join(
                    item.value for item in app.markdown
                )
                rendered_metrics = {
                    item.label: item.value
                    for item in app.metric
                }
                self.assertIn("KOVA", rendered_markdown)
                self.assertIn(
                    f"PROVIDER: {availability}",
                    rendered_markdown,
                )
                self.assertIn(
                    (
                        '<div class="mc-data-label">Last outcome</div>'
                        f'<div class="mc-data-value">{outcome}</div>'
                    ),
                    rendered_markdown,
                )
                self.assertIn(
                    (
                        '<div class="mc-data-label">'
                        "Latest failure category</div>"
                        f'<div class="mc-data-value">{category}</div>'
                    ),
                    rendered_markdown,
                )
                self.assertEqual(
                    rendered_metrics["Failure category counts"],
                    counts,
                )

    def test_health_view_keeps_every_provider_state_readable(self):
        health_states = (
            {
                "name": "healthy",
                "availability": "HEALTHY",
                "requests": 1,
                "successes": 1,
                "failures": 0,
                "success_rate_percent": 100.0,
                "last_outcome": "SUCCESS",
                "last_failure_category": None,
                "failure_categories": {},
                "category_label": "UNKNOWN",
                "counts_label": "UNKNOWN",
            },
            {
                "name": "timeout",
                "availability": "UNAVAILABLE",
                "requests": 1,
                "successes": 0,
                "failures": 1,
                "success_rate_percent": 0.0,
                "last_outcome": "FAILURE",
                "last_failure_category": "timeout",
                "failure_categories": {"timeout": 1},
                "category_label": "Timeout",
                "counts_label": "Timeout: 1",
            },
            {
                "name": "network error",
                "availability": "UNAVAILABLE",
                "requests": 1,
                "successes": 0,
                "failures": 1,
                "success_rate_percent": 0.0,
                "last_outcome": "FAILURE",
                "last_failure_category": "network_error",
                "failure_categories": {"network_error": 1},
                "category_label": "Network error",
                "counts_label": "Network error: 1",
            },
            {
                "name": "rate limit",
                "availability": "UNAVAILABLE",
                "requests": 1,
                "successes": 0,
                "failures": 1,
                "success_rate_percent": 0.0,
                "last_outcome": "FAILURE",
                "last_failure_category": "rate_limit",
                "failure_categories": {"rate_limit": 1},
                "category_label": "Rate limit",
                "counts_label": "Rate limit: 1",
            },
            {
                "name": "provider outage",
                "availability": "DEGRADED",
                "requests": 7,
                "successes": 4,
                "failures": 3,
                "success_rate_percent": 57.1,
                "last_outcome": "FAILURE",
                "last_failure_category": "provider_outage",
                "failure_categories": {
                    "rate_limit": 2,
                    "provider_outage": 1,
                },
                "category_label": "Provider outage",
                "counts_label": "Rate limit: 2 · Provider outage: 1",
            },
            {
                "name": "response validation",
                "availability": "UNAVAILABLE",
                "requests": 1,
                "successes": 0,
                "failures": 1,
                "success_rate_percent": 0.0,
                "last_outcome": "FAILURE",
                "last_failure_category": "response_validation",
                "failure_categories": {"response_validation": 1},
                "category_label": "Response validation",
                "counts_label": "Response validation: 1",
            },
        )

        for state in health_states:
            with self.subTest(state=state["name"]):
                provider_health = {
                    "provider": "Managed provider",
                    "availability": state["availability"],
                    "requests": state["requests"],
                    "successes": state["successes"],
                    "failures": state["failures"],
                    "success_rate_percent": state["success_rate_percent"],
                    "last_latency_ms": 1200.0,
                    "last_outcome": state["last_outcome"],
                    "last_failure_category": state["last_failure_category"],
                    "failure_categories": state["failure_categories"],
                }

                with patch(
                    "dashboard.get_provider_health",
                    return_value=provider_health,
                ):
                    app = AppTest.from_function(
                        render_assistant_health_view
                    ).run()

                rendered_markdown = "\n".join(
                    item.value for item in app.markdown
                )
                rendered_metrics = {
                    item.label: item.value
                    for item in app.metric
                }
                self.assertIn(
                    f"PROVIDER: {state['availability']}",
                    rendered_markdown,
                )
                for label, value in (
                    ("Last outcome", state["last_outcome"]),
                    ("Latest failure category", state["category_label"]),
                ):
                    self.assertIn(
                        (
                            f'<div class="mc-data-label">{label}</div>'
                            f'<div class="mc-data-value">{value}</div>'
                        ),
                        rendered_markdown,
                    )
                self.assertEqual(
                    rendered_metrics["Failure category counts"],
                    state["counts_label"],
                )


    @staticmethod
    def _capture_webkit_failure_screenshot(page, screenshot_path):
        if screenshot_path and page is not None:
            screenshot_dir = os.path.dirname(screenshot_path)
            if screenshot_dir:
                os.makedirs(screenshot_dir, exist_ok=True)
            page.screenshot(path=screenshot_path, full_page=True)

    @classmethod
    def _navigate_webkit_page(cls, page, url, screenshot_path=None):
        try:
            page.goto(
                url,
                wait_until="domcontentloaded",
                timeout=3000,
            )
        except BaseException:
            cls._capture_webkit_failure_screenshot(page, screenshot_path)
            raise

    @classmethod
    def _run_webkit_chart_checks(cls, page, expected_width, screenshot_path=None):
        try:
            for section, expected_heading in (
                ("OVERVIEW", "Paper account trajectory"),
                ("PERFORMANCE", "Historical batch backtest results summary"),
            ):
                _select_dashboard_section(page, section)
                page.wait_for_function(
                    "heading => document.body.innerText.includes(heading)",
                    arg=expected_heading,
                    timeout=20000,
                )
                DashboardAssistantBrowserTests()._assert_portrait_chart_geometry(
                    page, 1.0, expected_width=expected_width
                )
                unittest.TestCase().assertTrue(
                    page.locator(
                        ".mc-chart-card .legend, "
                        ".mc-chart-card [class*='legend'], "
                        ".mc-chart-card [aria-label*='legend']"
                    ).count(),
                    f"{section} chart rendered without a legend.",
                )
        except BaseException:
            cls._capture_webkit_failure_screenshot(page, screenshot_path)
            raise

    def test_webkit_chart_failure_captures_configured_diagnostic_screenshot(self):
        class FakeLocator:
            def count(self):
                return 1

        class FakePage:
            def get_by_text(self, section, exact):
                return self

            def click(self):
                return None

            def wait_for_function(self, expression, arg, timeout):
                return None

            def locator(self, selector):
                return FakeLocator()

            def screenshot(self, path, full_page):
                with open(path, "wb") as screenshot_file:
                    screenshot_file.write(b"deterministic webkit diagnostic")

        with tempfile.TemporaryDirectory() as temp_dir:
            screenshot_path = os.path.join(
                temp_dir, "artifacts", "webkit-failure", "chart.png"
            )
            with patch.object(
                DashboardAssistantBrowserTests,
                "_assert_portrait_chart_geometry",
                side_effect=AssertionError("forced chart assertion failure"),
            ) as chart_assertion:
                with self.assertRaisesRegex(
                    AssertionError, "forced chart assertion failure"
                ):
                    self._run_webkit_chart_checks(
                        FakePage(),
                        expected_width=390,
                        screenshot_path=screenshot_path,
                    )

            self.assertTrue(os.path.isfile(screenshot_path))
            self.assertGreater(os.path.getsize(screenshot_path), 0)
            chart_assertion.assert_called_once()

    def test_webkit_navigation_failure_captures_configured_diagnostic_screenshot(self):
        class FakePage:
            def goto(self, url, wait_until, timeout):
                raise RuntimeError("forced WebKit navigation failure")

            def screenshot(self, path, full_page):
                with open(path, "wb") as screenshot_file:
                    screenshot_file.write(b"deterministic webkit navigation diagnostic")

        with tempfile.TemporaryDirectory() as temp_dir:
            screenshot_path = os.path.join(
                temp_dir, "artifacts", "webkit-failure", "navigation.png"
            )
            with self.assertRaisesRegex(
                RuntimeError, "forced WebKit navigation failure"
            ):
                self._navigate_webkit_page(
                    FakePage(),
                    "http://127.0.0.1:1",
                    screenshot_path=screenshot_path,
                )

            self.assertTrue(os.path.isfile(screenshot_path))
            self.assertGreater(os.path.getsize(screenshot_path), 0)

    def test_webkit_iphone_chart_details_are_readable(self):
        """Run the deterministic chart fixture in a real WebKit engine."""
        if os.environ.get("RUN_WEBKIT_BROWSER_TESTS") != "1":
            self.skipTest(
                "Hosted WebKit is required; set RUN_WEBKIT_BROWSER_TESTS=1 to run."
            )

        from playwright.sync_api import sync_playwright

        project_root = os.path.dirname(os.path.abspath(__file__))
        viewport_width, viewport_height = (
            DashboardAssistantBrowserTests.MOBILE_VIEWPORTS[0],
            844,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            wrapper_path = os.path.join(temp_dir, "dashboard_browser_fixture.py")
            with open(wrapper_path, "w", encoding="utf-8") as wrapper_file:
                wrapper_file.write(_browser_dashboard_wrapper())

            with sync_playwright() as playwright:
                browser = playwright.webkit.launch(
                    headless=True,
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
                            "--server.enableCORS",
                            "false",
                            "--browser.gatherUsageStats",
                            "false",
                        ],
                        cwd=project_root,
                        env={
                            **os.environ,
                            "DASHBOARD_TEST_PROVIDER_STATE": "HEALTHY",
                        },
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                    )
                    startup_output, output_reader = _start_bounded_output_reader(
                        process
                    )
                    page = browser.new_page(
                        viewport={
                            "width": viewport_width,
                            "height": viewport_height,
                        },
                        user_agent=DashboardAssistantBrowserTests.MOBILE_BROWSER_PROFILES[0][2],
                    )
                    _navigate_fixture_until_ready(
                        lambda: self._navigate_webkit_page(
                            page,
                            f"http://127.0.0.1:{port}",
                            screenshot_path=os.environ.get(
                                "WEBKIT_FAILURE_SCREENSHOT_PATH"
                            ),
                        ),
                        process,
                        startup_output,
                        output_reader,
                        "Streamlit WebKit fixture",
                    )

                    self._run_webkit_chart_checks(
                        page,
                        expected_width=viewport_width,
                        screenshot_path=os.environ.get(
                            "WEBKIT_FAILURE_SCREENSHOT_PATH"
                        ),
                    )
                finally:
                    if page is not None:
                        page.close()
                    if process is not None:
                        process.terminate()
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
                        if "output_reader" in locals():
                            output_reader.join(timeout=1)
                    browser.close()


if __name__ == "__main__":
    unittest.main()
