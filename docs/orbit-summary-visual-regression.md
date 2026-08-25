# Orbit Summary visual regression

The responsive browser check compares the Overview shell against the approved
screenshots in `visual_baselines/orbit_summary/` at these release-critical
sizes:

- desktop: 1280 × 900
- mobile-320: 320 × 844
- mobile-390: 390 × 844
- mobile-430: 430 × 844

CI runs the comparison on every push and pull request. If a screenshot changes
enough to exceed the comparison tolerance, the check fails and uploads a
focused diff image as the `orbit-summary-visual-diffs` workflow artifact.

Firefox mobile runs as a separate hosted regression against the same
deterministic fixture. Its approved `390 × 844` screenshot is intentionally
stored separately from Chromium and WebKit baselines so browser-specific
typography and layout drift is visible.

## Intentionally updating a baseline

Baseline changes require an explicit local review. First run the check without
update flags and inspect the current screenshots or the uploaded CI diff. Only
after confirming that the visual change is intentional, regenerate all four
approved images with:

```sh
ORBIT_SNAPSHOT_UPDATE=1 \
ORBIT_SNAPSHOT_UPDATE_APPROVED=1 \
uv run python -m unittest \
  test_dashboard_ai_operations_assistant.DashboardAssistantBrowserTests.test_orbit_summary_snapshots_protect_responsive_composition
```

Review the resulting PNGs in `visual_baselines/orbit_summary/`, commit them
alongside the intentional UI change, and rerun the comparison without the
update flags. `ORBIT_SNAPSHOT_UPDATE=1` is rejected unless the separate
`ORBIT_SNAPSHOT_UPDATE_APPROVED=1` review acknowledgment is also present.

Firefox mobile baseline updates use the same explicit approval requirement:

```sh
RUN_FIREFOX_BROWSER_TESTS=1 \
ORBIT_SNAPSHOT_UPDATE=1 \
ORBIT_SNAPSHOT_UPDATE_APPROVED=1 \
uv run python -m unittest \
  test_dashboard_ai_operations_assistant.DashboardAssistantBrowserTests.test_orbit_summary_firefox_mobile_snapshot_protects_firefox_composition
```

Review the resulting `orbit-summary-firefox-mobile-390.png` before committing
it. A Firefox comparison failure uploads a focused diff as the
`firefox-orbit-summary-visual-diff` workflow artifact.
