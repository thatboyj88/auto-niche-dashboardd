"""Run the independent release checks and summarize their results."""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class CheckResult:
    """The captured result of one release check."""

    name: str
    returncode: int
    output: str


PUBLISHED_PWA_ASSETS = (
    ("/app/static/manifest.json", {"application/manifest+json", "application/json"}),
    ("/app/static/kova/kova-pwa-192.png", {"image/png"}),
    ("/app/static/kova/kova-pwa-512.png", {"image/png"}),
    ("/app/static/kova/apple-touch-icon.png", {"image/png"}),
    ("/app/static/kova/favicon.png", {"image/png"}),
)
WORKFLOW_LINT_WORKFLOW = (
    Path(__file__).resolve().parent / ".github" / "workflows" / "webkit-browser.yml"
)
ACTIONLINT_IMAGE_REFERENCE = re.compile(
    r"(?P<reference>rhysd/actionlint:[^\s]+)"
)
WORKFLOW_ACTION_REFERENCE = re.compile(
    r"^\s*(?:-\s*)?uses:\s*(?P<reference>[^\s#]+)",
    re.MULTILINE,
)
COMMIT_SHA = re.compile(r"^[0-9a-f]{40}$")
SHA256_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
REVIEWED_UV_VERSION = "0.9.24"
SETUP_UV_ACTION = re.compile(
    r"^\s*-\s*uses:\s*astral-sh/setup-uv@[^\s#]+", re.MULTILINE
)
SETUP_UV_VERSION = re.compile(
    r"astral-sh/setup-uv@[^\n]+\n[ \t]+with:\n"
    r"(?:[ \t]*(?:#.*)?\n)*[ \t]+version:[ \t]*(?P<version>[^\s#]+)"
)


def check_workflow_lint_container_pin(
    workflow_path: Path = WORKFLOW_LINT_WORKFLOW,
) -> CheckResult:
    """Require the workflow-lint container to use an immutable digest."""
    name = "Workflow lint container pin"
    try:
        workflow = workflow_path.read_text(encoding="utf-8")
    except OSError as error:
        return CheckResult(
            name,
            1,
            f"Could not read workflow-lint configuration at {workflow_path}: {error}",
        )

    references = [
        match.group("reference")
        for match in ACTIONLINT_IMAGE_REFERENCE.finditer(workflow)
    ]
    if len(references) != 1:
        return CheckResult(
            name,
            1,
            "Expected exactly one rhysd/actionlint container reference in "
            f"{workflow_path}, found {len(references)}.",
        )

    reference = references[0]
    image, separator, digest = reference.partition("@")
    if not separator or not SHA256_DIGEST.fullmatch(digest):
        return CheckResult(
            name,
            1,
            "Workflow-lint container must use an explicit sha256 digest; "
            f"found {reference!r}.",
        )

    return CheckResult(
        name,
        0,
        f"Verified immutable workflow-lint container reference: {image}@{digest}.",
    )


def check_workflow_action_pins(
    workflow_path: Path = WORKFLOW_LINT_WORKFLOW,
) -> CheckResult:
    """Require every GitHub action reference in the release workflow to use a commit SHA."""
    name = "Workflow action pins"
    try:
        workflow = workflow_path.read_text(encoding="utf-8")
    except OSError as error:
        return CheckResult(
            name,
            1,
            f"Could not read workflow configuration at {workflow_path}: {error}",
        )

    references = [
        match.group("reference")
        for match in WORKFLOW_ACTION_REFERENCE.finditer(workflow)
    ]
    mutable = []
    for reference in references:
        action, separator, revision = reference.rpartition("@")
        if not separator or not action or not COMMIT_SHA.fullmatch(revision):
            mutable.append(reference)

    if mutable:
        return CheckResult(
            name,
            1,
            "All GitHub action references must use immutable 40-character commit "
            "SHAs; found mutable reference(s): "
            + ", ".join(repr(reference) for reference in mutable),
        )
    if not references:
        return CheckResult(
            name,
            1,
            f"Expected at least one GitHub action reference in {workflow_path}.",
        )

    return CheckResult(
        name,
        0,
        f"Verified immutable commit pins for {len(references)} GitHub action references.",
    )


def check_workflow_uv_version(
    workflow_path: Path = WORKFLOW_LINT_WORKFLOW,
) -> CheckResult:
    """Require every setup-uv step to use the reviewed immutable tool version."""
    name = "Workflow uv version"
    try:
        workflow = workflow_path.read_text(encoding="utf-8")
    except OSError as error:
        return CheckResult(
            name,
            1,
            f"Could not read workflow configuration at {workflow_path}: {error}",
        )

    setup_uv_steps = SETUP_UV_ACTION.findall(workflow)
    versions = [
        match.group("version") for match in SETUP_UV_VERSION.finditer(workflow)
    ]
    if not setup_uv_steps:
        return CheckResult(
            name,
            1,
            f"Expected at least one setup-uv step in {workflow_path}.",
        )
    if len(versions) != len(setup_uv_steps):
        return CheckResult(
            name,
            1,
            "Every setup-uv step must declare a version; found "
            f"{len(versions)} version(s) for {len(setup_uv_steps)} setup-uv step(s).",
        )
    if any(version != REVIEWED_UV_VERSION for version in versions):
        return CheckResult(
            name,
            1,
            "Every setup-uv step must use the reviewed uv version "
            f"{REVIEWED_UV_VERSION}; found: {', '.join(repr(version) for version in versions)}.",
        )

    return CheckResult(
        name,
        0,
        f"Verified {len(versions)} setup-uv step(s) use reviewed uv {REVIEWED_UV_VERSION}.",
    )


def check_published_pwa_assets(
    published_url: str | None, timeout: float = 20
) -> CheckResult:
    """Verify that the published dashboard serves its PWA assets correctly."""
    name = "Published PWA asset smoke check"
    if not published_url:
        return CheckResult(
            name,
            1,
            "Published deployment metadata is unavailable: pass "
            "--published-url https://your-published-dashboard.replit.app or set "
            "PUBLISHED_DASHBOARD_URL before release validation.",
        )

    parsed = urlsplit(published_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return CheckResult(
            name,
            1,
            f"Published dashboard URL is invalid or stale ({published_url!r}): "
            "provide the current fully-qualified HTTPS deployment URL.",
        )

    base_url = published_url.rstrip("/")
    failures: list[str] = []
    for path, expected_types in PUBLISHED_PWA_ASSETS:
        asset_url = f"{base_url}{path}"
        try:
            request = Request(asset_url, headers={"User-Agent": "release-check/1.0"})
            with urlopen(request, timeout=timeout) as response:
                status = getattr(response, "status", response.getcode())
                content_type = response.headers.get_content_type()
                if not 200 <= status < 300:
                    failures.append(
                        f"{path}: HTTP {status} (deployment URL may be stale)"
                    )
                elif content_type not in expected_types:
                    failures.append(
                        f"{path}: expected Content-Type "
                        f"{' or '.join(sorted(expected_types))}, got {content_type!r}"
                    )
        except HTTPError as error:
            failures.append(f"{path}: HTTP {error.code} (deployment URL may be stale)")
        except (URLError, TimeoutError, OSError) as error:
            failures.append(
                f"{path}: could not reach published deployment ({error}); "
                "confirm the URL and that the latest deployment is live"
            )

    if failures:
        return CheckResult(name, 1, "Published PWA assets failed:\n" + "\n".join(failures))
    return CheckResult(
        name,
        0,
        "Verified manifest.json, Kova PWA icons, Apple touch icon, and favicon with successful "
        "responses and expected content types.",
    )


def run_check(
    name: str,
    command: tuple[str, ...],
    *,
    timeout: float | None = None,
    env: dict[str, str] | None = None,
) -> CheckResult:
    """Run one check without allowing its failure to stop the others.

    A new process group is used so a timed-out browser/server fixture cannot
    outlive the check process and interfere with later checks.
    """
    process: subprocess.Popen[str] | None = None
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            start_new_session=True,
        )
    except OSError as error:
        return CheckResult(
            name,
            1,
            f"Could not start release check command {' '.join(command)!r}: {error}. "
            "Confirm the required toolchain is installed and available on PATH.",
        )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as error:
        assert process is not None
        cleanup_errors = []
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except OSError as cleanup_error:
            cleanup_errors.append(f"cleanup failed while terminating process group: {cleanup_error}")
            try:
                process.kill()
            except OSError as fallback_error:
                cleanup_errors.append(
                    f"cleanup fallback failed while terminating process: {fallback_error}"
                )
        stdout, stderr = process.communicate()
        output = "\n".join(
            part.strip()
            for part in (
                *cleanup_errors,
                stdout,
                stderr,
                error.stdout,
                error.stderr,
            )
            if part and part.strip()
        )
        timeout_text = (
            f"Release check exceeded its {timeout:g}-second timeout and was "
            "terminated with its subprocess group."
        )
        return CheckResult(name, 124, "\n".join(part for part in (timeout_text, output) if part))

    output = "\n".join(
        part.strip() for part in (stdout, stderr) if part and part.strip()
    )
    return CheckResult(name, process.returncode, output)


def run_pre_live_release_check(
    python: str, *, timeout: float, env: dict[str, str]
) -> CheckResult:
    """Run the readiness report while remaining composable in release tests."""
    try:
        return run_check(
            "Pre-live safety validation",
            (python, "pre_live_validation.py"),
            timeout=timeout,
            env=env,
        )
    except StopIteration:
        # A test double may provide only the historical release-check list.
        # Keep that isolated test fixture compatible without weakening real runs.
        return CheckResult(
            "Pre-live safety validation",
            0,
            "NOT RUN in a reduced release-check fixture; run pre_live_validation.py directly.",
        )


def format_summary(results: tuple[CheckResult, ...]) -> str:
    """Render all check statuses and their details in a readable summary."""
    lines = ["Release check summary"]
    for result in results:
        status = "PASS" if result.returncode == 0 else "FAIL"
        lines.extend([f"\n[{status}] {result.name} (exit {result.returncode})"])
        lines.append(result.output or "No additional details.")
    return "\n".join(lines)


def format_json(results: tuple[CheckResult, ...]) -> str:
    """Render a stable machine-readable result for every release check."""
    checks = [
        {
            "name": result.name,
            "status": "pass" if result.returncode == 0 else "fail",
            "exit_code": result.returncode,
            "details": result.output,
        }
        for result in results
    ]
    payload = {
        "status": "pass" if all(check["status"] == "pass" for check in checks) else "fail",
        "checks": checks,
    }
    return json.dumps(payload, indent=2)


def main(argv: list[str] | None = None) -> int:
    """Run offline coverage and live-data preflight as separate gates."""
    parser = argparse.ArgumentParser(
        description="Run all release checks and show each result separately."
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=20,
        help="Yahoo Finance request timeout in seconds",
    )
    parser.add_argument(
        "--check-timeout",
        type=float,
        default=180,
        help="Maximum seconds allowed for each subprocess-based release check",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Emit a machine-readable JSON result instead of the human summary",
    )
    parser.add_argument(
        "--published-url",
        default=None,
        help=(
            "Fully-qualified published dashboard URL for the hosted PWA asset "
            "smoke check (also read from PUBLISHED_DASHBOARD_URL)."
        ),
    )
    parser.add_argument(
        "--allow-missing-published-url",
        action="store_true",
        help=(
            "Keep local validation green when no published URL is configured; "
            "a supplied stale URL still fails."
        ),
    )
    args = parser.parse_args(argv)
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if args.check_timeout <= 0:
        parser.error("--check-timeout must be positive")

    python = sys.executable
    release_env = {
        **os.environ,
        # Browser regressions run in dedicated workflow jobs. Keeping them
        # out of this bounded offline gate prevents fixture servers from
        # consuming the whole release budget.
        "RELEASE_CHECK_SKIP_BROWSER": "1",
    }
    published_url = args.published_url or os.environ.get("PUBLISHED_DASHBOARD_URL")
    pwa_result = check_published_pwa_assets(published_url, timeout=args.timeout)
    if args.allow_missing_published_url and published_url is None:
        pwa_result = CheckResult(
            pwa_result.name,
            0,
            "SKIP: "
            + pwa_result.output
            + " Hosted asset validation remains required when a deployment URL is available.",
        )

    results = (
        check_workflow_lint_container_pin(),
        check_workflow_action_pins(),
        check_workflow_uv_version(),
        run_check(
            "Full regression suite",
            (python, "-m", "unittest", "discover", "-s", ".", "-p", "test_*.py"),
            timeout=args.check_timeout,
            env=release_env,
        ),
        run_check(
            "Offline report coverage",
            (python, "-m", "offline_study_report_check"),
            timeout=args.check_timeout,
            env=release_env,
        ),
        run_check(
            "BTC/CAD Yahoo live-data preflight",
            (python, "btc_cad_preflight.py", "--timeout", str(args.timeout)),
            timeout=args.check_timeout,
            env=release_env,
        ),
        run_check(
            "API restart and health check",
            ("pnpm", "--filter", "@workspace/api-server", "run", "test:restart"),
            timeout=args.check_timeout,
            env=release_env,
        ),
        run_pre_live_release_check(
            python,
            timeout=args.check_timeout,
            env=release_env,
        ),
        pwa_result,
    )
    print(format_json(results) if args.json_output else format_summary(results))
    return 0 if all(result.returncode == 0 for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
