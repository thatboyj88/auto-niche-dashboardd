import os
import shutil
import socket
import subprocess
import time
import unittest
from urllib.request import urlopen


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def wait_for_health(port, process):
    deadline = time.time() + 30
    while time.time() < deadline:
        if process.poll() is not None:
            raise AssertionError("Orbit API server exited before becoming healthy.")
        try:
            with urlopen(f"http://127.0.0.1:{port}/api/healthz", timeout=1) as response:
                if response.status == 200:
                    return
        except OSError:
            time.sleep(0.1)
    raise AssertionError("Orbit API server did not become healthy.")


class OrbitRouteBrowserTests(unittest.TestCase):
    def test_standalone_orbit_renders_at_iphone_chrome_dimensions(self):
        from playwright.sync_api import sync_playwright

        port = free_port()
        process = subprocess.Popen(
            ["pnpm", "run", "dev"],
            cwd=os.path.join(os.path.dirname(__file__), "artifacts", "api-server"),
            env={**os.environ, "PORT": str(port), "NODE_ENV": "development"},
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            wait_for_health(port, process)
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(
                    headless=True,
                    executable_path=shutil.which("chromium"),
                )
                page = browser.new_page(
                    viewport={"width": 430, "height": 932},
                    user_agent=(
                        "Mozilla/5.0 (iPhone; CPU iPhone OS 17_5 like Mac OS X) "
                        "AppleWebKit/605.1.15 (KHTML, like Gecko) "
                        "CriOS/125.0.6422.99 Mobile/15E148 Safari/604.1"
                    ),
                    device_scale_factor=3,
                    is_mobile=True,
                    has_touch=True,
                )
                page.goto(
                    f"http://127.0.0.1:{port}/orbit/",
                    wait_until="networkidle",
                )
                self.assertEqual(page.title(), "Kova — Orbit Summary")
                self.assertEqual(page.viewport_size, {"width": 430, "height": 932})
                self.assertLessEqual(
                    page.evaluate("document.documentElement.scrollWidth"),
                    430,
                )
                self.assertEqual(page.locator(".bottom-nav").count(), 1)
                self.assertEqual(page.locator(".orb-large").count(), 1)
                self.assertIn("UNAVAILABLE", page.locator("body").inner_text())
                self.assertNotIn("$25.00", page.locator("#paperEquity").inner_text())

                page.locator("#kovaButton").click()
                self.assertEqual(page.locator("#kovaDrawer.open").count(), 1)
                page.locator("#closeKova").click()
                self.assertEqual(page.locator("#kovaDrawer.open").count(), 0)
                browser.close()
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

    def test_direct_and_prefixed_routes_serve_the_same_document(self):
        port = free_port()
        process = subprocess.Popen(
            ["pnpm", "run", "dev"],
            cwd=os.path.join(os.path.dirname(__file__), "artifacts", "api-server"),
            env={**os.environ, "PORT": str(port), "NODE_ENV": "development"},
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            wait_for_health(port, process)
            documents = []
            for route in ("/orbit/", "/api/orbit/"):
                with urlopen(f"http://127.0.0.1:{port}{route}", timeout=5) as response:
                    self.assertEqual(response.status, 200)
                    documents.append(response.read())
            self.assertEqual(documents[0], documents[1])
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

    @unittest.skipUnless(
        os.environ.get("RUN_WEBKIT_BROWSER_TESTS") == "1",
        "Hosted WebKit is required; set RUN_WEBKIT_BROWSER_TESTS=1 to run.",
    )
    def test_standalone_orbit_renders_at_iphone_safari_dimensions(self):
        from playwright.sync_api import sync_playwright

        port = free_port()
        process = subprocess.Popen(
            ["pnpm", "run", "dev"],
            cwd=os.path.join(os.path.dirname(__file__), "artifacts", "api-server"),
            env={**os.environ, "PORT": str(port), "NODE_ENV": "development"},
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            wait_for_health(port, process)
            with sync_playwright() as playwright:
                browser = playwright.webkit.launch()
                page = browser.new_page(
                    viewport={"width": 430, "height": 932},
                    user_agent=(
                        "Mozilla/5.0 (iPhone; CPU iPhone OS 17_5 like Mac OS X) "
                        "AppleWebKit/605.1.15 (KHTML, like Gecko) "
                        "Version/17.5 Mobile/15E148 Safari/604.1"
                    ),
                    device_scale_factor=3,
                    is_mobile=True,
                    has_touch=True,
                )
                page.goto(
                    f"http://127.0.0.1:{port}/orbit/",
                    wait_until="networkidle",
                )
                self.assertEqual(page.title(), "Kova — Orbit Summary")
                self.assertLessEqual(
                    page.evaluate("document.documentElement.scrollWidth"),
                    430,
                )
                self.assertEqual(page.locator(".bottom-nav").count(), 1)
                browser.close()
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()