import json
import unittest
from pathlib import Path


ROOT = Path(__file__).parent


class MobilePwaTests(unittest.TestCase):
    def test_manifest_is_safe_installable_dark_shell(self):
        manifest = json.loads(
            (ROOT / "static" / "manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["display"], "standalone")
        self.assertEqual(manifest["orientation"], "any")
        self.assertEqual(manifest["background_color"], "#030c1d")
        self.assertEqual(manifest["theme_color"], "#030c1d")
        self.assertEqual(manifest["start_url"], "/")
        self.assertEqual(manifest["name"], "Kova")
        self.assertEqual(manifest["short_name"], "Kova")
        self.assertEqual(
            manifest["description"],
            "Kova BTC/CAD Operations Centre",
        )
        self.assertEqual(len(manifest["icons"]), 9)
        for icon in manifest["icons"]:
            self.assertTrue((ROOT / icon["src"].removeprefix("/app/")).exists())
            self.assertEqual(icon["type"], "image/png")
            self.assertIn(icon["purpose"], ("any", "any maskable"))

    def test_icons_keep_operations_centre_brand_mark(self):
        orb = (ROOT / "static" / "kova" / "kova-orb.svg").read_text(
            encoding="utf-8"
        )
        self.assertIn("kova-source.png", orb)
        self.assertEqual(
            (ROOT / "static" / "kova" / "kova-source.png").read_bytes(),
            (ROOT / "attached_assets"
             / "A32F156B-98B5-4352-8CB9-97BD388E3A32_1787371033452.png").read_bytes(),
        )

    def test_kova_raster_assets_are_square_opaque_pngs(self):
        from PIL import Image

        for size in (72, 96, 128, 152, 192, 256, 384, 512, 1024):
            with self.subTest(size=size):
                image = Image.open(
                    ROOT / "static" / "kova" / f"kova-pwa-{size}.png"
                )
                self.assertEqual(image.size, (size, size))
                self.assertEqual(image.format, "PNG")
                self.assertIn(image.mode, ("RGB", "RGBA"))
                if image.mode == "RGBA":
                    self.assertEqual(image.getextrema()[3], (255, 255))

    def test_dashboard_declares_ios_metadata_and_periodic_read_only_refresh(self):
        dashboard = (ROOT / "dashboard.py").read_text(encoding="utf-8")
        for marker in (
            'rel="manifest"',
            "apple-mobile-web-app-capable",
            "apple-mobile-web-app-status-bar-style",
            'apple-mobile-web-app-title" content="Kova"',
            'page_title="Kova"',
            "@st.fragment(run_every=\"30s\")",
            "Genuine paper engine",
        ):
            self.assertIn(marker, dashboard)
