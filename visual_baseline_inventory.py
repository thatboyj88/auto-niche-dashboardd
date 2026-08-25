"""Inventory the reviewed screenshot set used by browser snapshot tests."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_TEST_SOURCE = PROJECT_ROOT / "test_dashboard_ai_operations_assistant.py"
DEFAULT_BASELINE_DIR = PROJECT_ROOT / "visual_baselines" / "orbit_summary"

# Keep this manifest reviewed and explicit.  A test method may expand to more
# than one screenshot when it exercises multiple zoom factors.
BASELINE_MANIFEST = {
    "test_orbit_summary_snapshots_protect_responsive_composition": (
        "orbit-summary-desktop.png",
        "orbit-summary-mobile-320.png",
        "orbit-summary-mobile-390.png",
        "orbit-summary-mobile-430.png",
    ),
    "test_orbit_summary_webkit_mobile_snapshot_protects_safari_composition": (
        "orbit-summary-webkit-mobile-390.png",
    ),
    "test_orbit_summary_webkit_portrait_tablet_snapshot_protects_safari_composition": (
        "orbit-summary-webkit-portrait-tablet.png",
    ),
    "test_orbit_summary_webkit_landscape_tablet_snapshot_protects_safari_composition": (
        "orbit-summary-webkit-landscape-tablet.png",
    ),
    "test_orbit_summary_webkit_desktop_zoom_snapshots_protect_wrapping": (
        "orbit-summary-webkit-desktop-zoom-0.8.png",
        "orbit-summary-webkit-desktop-zoom-1.25.png",
    ),
    "test_orbit_summary_firefox_mobile_snapshot_protects_firefox_composition": (
        "orbit-summary-firefox-mobile-390.png",
    ),
    "test_orbit_summary_firefox_desktop_snapshot_protects_firefox_typography": (
        "orbit-summary-firefox-desktop.png",
    ),
    "test_orbit_summary_firefox_desktop_zoom_snapshots_protect_wrapping": (
        "orbit-summary-firefox-desktop-zoom-0.8.png",
        "orbit-summary-firefox-desktop-zoom-1.25.png",
    ),
    "test_orbit_summary_firefox_landscape_tablet_snapshot_protects_responsive_composition": (
        "orbit-summary-firefox-landscape-tablet.png",
    ),
    "test_orbit_summary_firefox_portrait_tablet_snapshot_protects_responsive_composition": (
        "orbit-summary-firefox-portrait-tablet.png",
    ),
}


def discover_snapshot_tests(test_source: Path = DEFAULT_TEST_SOURCE) -> set[str]:
    """Find browser test methods that invoke the snapshot comparison helper."""
    tree = ast.parse(test_source.read_text(encoding="utf-8"), filename=str(test_source))
    found = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name.startswith("test_") and any(
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == "_run_orbit_summary_snapshot_comparison"
            for call in ast.walk(node)
        ):
            found.add(node.name)
    return found


def validate_inventory(
    baseline_dir: Path = DEFAULT_BASELINE_DIR,
    test_source: Path = DEFAULT_TEST_SOURCE,
) -> list[str]:
    """Return safe, actionable inventory errors without reading image data."""
    discovered = discover_snapshot_tests(test_source)
    expected_tests = set(BASELINE_MANIFEST)
    errors = []
    for test_name in sorted(discovered - expected_tests):
        errors.append(f"UNREGISTERED snapshot test: test={test_name}")
    for test_name in sorted(expected_tests - discovered):
        errors.append(f"STALE manifest test: test={test_name}")

    expected_files = {
        filename for filenames in BASELINE_MANIFEST.values() for filename in filenames
    }
    for test_name, filenames in BASELINE_MANIFEST.items():
        for filename in filenames:
            path = baseline_dir / filename
            if not path.is_file():
                errors.append(f"MISSING baseline: test={test_name} file={path}")

    actual_files = {
        path.name for path in baseline_dir.iterdir() if path.is_file()
    } if baseline_dir.is_dir() else set()
    for filename in sorted(actual_files - expected_files):
        errors.append(f"UNEXPECTED baseline: file={baseline_dir / filename}")
    return errors


def main() -> int:
    errors = validate_inventory()
    if errors:
        print("\n".join(errors))
        return 1
    print(
        "VISUAL_BASELINE_INVENTORY_OK "
        f"tests={len(BASELINE_MANIFEST)} "
        f"files={sum(len(files) for files in BASELINE_MANIFEST.values())}"
    )
    return 0


if __name__ == "__main__":
    if sys.argv[1:]:
        raise SystemExit("usage: python -m visual_baseline_inventory")
    raise SystemExit(main())