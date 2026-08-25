import http.server
import os
import shutil
import threading
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent
COMPONENT = ROOT / "kova_voice_component" / "index.html"


class KovaVoiceAssistantContractTests(unittest.TestCase):
    def test_orb_has_no_visible_glyph_and_exposes_accessible_label(self):
        source = COMPONENT.read_text(encoding="utf-8")
        self.assertIn('id="orb"', source)
        self.assertIn('aria-label="KOVA voice assistant.', source)
        self.assertIn('<span class="orb-core" aria-hidden="true"></span>', source)
        self.assertIn('<span class="orb-sheen" aria-hidden="true"></span>', source)
        self.assertNotIn(">K<", source)
        self.assertNotIn(">◉<", source)
        self.assertNotIn('class="panel"', source)
        self.assertNotIn('class="quick-mic"', source)

    def test_voice_states_and_safety_interactions_are_present(self):
        source = COMPONENT.read_text(encoding="utf-8")
        for state in ("listening", "thinking", "speaking", "idle"):
            self.assertIn(f"state-{state}", source)
        self.assertIn("speechSynthesis.cancel()", source)
        self.assertIn("event.key.toLowerCase() === \"k\"", source)
        self.assertIn("event.error === \"not-allowed\"", source)
        self.assertIn("type: \"user_message\"", source)
        self.assertIn("isStreamlitMessage: true", source)
        self.assertIn("target.closest?.", source)
        self.assertIn('setAttribute("allowtransparency", "true")', source)
        self.assertIn("background: transparent !important", source)
        self.assertIn('window.frameElement.style.position = "fixed"', source)

    def test_component_is_mounted_after_every_dashboard_page(self):
        source = (ROOT / "dashboard.py").read_text(encoding="utf-8")
        self.assertIn("def render_kova_voice_assistant(", source)
        self.assertIn("render_kova_voice_assistant(", source)
        self.assertIn('key="kova_voice_assistant"', source)
        navigation = source.split("NAVIGATION_ITEMS =", 1)[1].split(")", 1)[0]
        self.assertNotIn("AI ASSISTANT", navigation)
        self.assertNotIn("render_mc_ai_assistant(", source.split("def render_dashboard", 1)[1])
        self.assertNotIn("render_mc_assistant_widget()", source.split("def render_dashboard", 1)[1])


@unittest.skipUnless(
    shutil.which("chromium")
    and os.environ.get("RELEASE_CHECK_SKIP_BROWSER") != "1",
    "Browser regressions run in dedicated release workflow jobs.",
)
class KovaVoiceAssistantBrowserTests(unittest.TestCase):
    def test_orb_is_floating_responsive_and_keyboard_safe(self):
        from playwright.sync_api import sync_playwright

        class QuietHandler(http.server.SimpleHTTPRequestHandler):
            def log_message(self, *_args):
                pass

        server = http.server.ThreadingHTTPServer(
            ("127.0.0.1", 0),
            lambda *args, **kwargs: QuietHandler(
                *args,
                directory=str(COMPONENT.parent),
                **kwargs,
            ),
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(
                    headless=True,
                    executable_path=shutil.which("chromium"),
                )
                page = browser.new_page(
                    viewport={"width": 430, "height": 932},
                    is_mobile=True,
                    has_touch=True,
                )
                page.add_init_script(
                    """
                    class FakeRecognition {
                      start() { this.onstart?.(); }
                      stop() { this.onend?.(); }
                      abort() {}
                    }
                    window.SpeechRecognition = FakeRecognition;
                    window.SpeechSynthesisUtterance = class {
                      constructor(text) { this.text = text; }
                    };
                    Object.defineProperty(window, "speechSynthesis", {value: {
                      cancel() {},
                      getVoices() { return []; },
                      speak(utterance) { utterance.onstart?.(); }
                    }, configurable: true});
                    """
                )
                page.goto(
                    f"http://127.0.0.1:{server.server_address[1]}/index.html",
                    wait_until="domcontentloaded",
                )
                self.assertEqual(page.locator("#orb").inner_text(), "")
                self.assertLessEqual(
                    page.evaluate("document.documentElement.scrollWidth"),
                    430,
                )
                page.locator("#orb").click(force=True)
                page.locator("#orb").click(force=True)
                self.assertEqual(page.locator("#status").inner_text(), "LISTENING")

                page.evaluate("() => { document.body.tabIndex = -1; document.body.focus(); }")
                page.keyboard.press("k")
                self.assertEqual(page.locator("#status").inner_text(), "IDLE")

                page.evaluate(
                    """() => {
                      const input = document.createElement('input');
                      input.id = 'editing-probe';
                      document.body.appendChild(input);
                      input.focus();
                    }"""
                )
                page.keyboard.press("k")
                self.assertEqual(page.locator("#status").inner_text(), "IDLE")

                page.wait_for_timeout(500)
                page.locator("#orb").click(force=True)
                page.evaluate("() => document.body.focus()")
                page.keyboard.press("k")
                self.assertEqual(page.locator("#status").inner_text(), "LISTENING")

                page.evaluate(
                    """() => window.dispatchEvent(new MessageEvent('message', {data: {
                      type: 'streamlit:render',
                      args: {response: {id: 'response-1', text: 'Status is read-only.'}}
                    }}))"""
                )
                page.wait_for_timeout(50)
                self.assertEqual(page.locator("#status").inner_text(), "SPEAKING")
                page.locator("#orb").click(force=True)
                self.assertEqual(page.locator("#status").inner_text(), "LISTENING")
                self.assertEqual(page.locator(".panel").count(), 0)
                self.assertEqual(page.locator(".quick-mic").count(), 0)
                browser.close()
        finally:
            server.shutdown()
            server.server_close()