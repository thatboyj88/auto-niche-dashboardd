# Hosted workflow verification

## Published workflow

- **Repository:** [`thatboyj88/auto-niche-dashboardd`](https://github.com/thatboyj88/auto-niche-dashboardd)
- **Default branch:** `main`
- **Workflow:** [WebKit browser regression](https://github.com/thatboyj88/auto-niche-dashboardd/blob/main/.github/workflows/webkit-browser.yml)
- **Publishing commit:** [`921f627`](https://github.com/thatboyj88/auto-niche-dashboardd/commit/921f627d1906cf441bf07fd4a763840d479fd07f)

GitHub Actions registered the workflow as active after the publishing commit.

## Dispatch evidence

On 2026-08-25, a `workflow_dispatch` run was accepted and completed:

- **Run:** [32803470771](https://github.com/thatboyj88/auto-niche-dashboardd/actions/runs/32803470771)
- **Result:** `failure`
- **Artifacts:** [open artifacts](https://github.com/thatboyj88/auto-niche-dashboardd/actions/runs/32803470771/artifacts)

The run result confirms workflow discovery and hosted execution. Its failure does
not invoke the drift alert because the notification job deliberately runs only
for failed `schedule` events.

## Alert-link safety

`test_observation_notifications.py` verifies that a browser-drift alert includes
the full GitHub run and artifact web links, while webhook content is excluded
from the message payload. The same contract was exercised with the dispatched
run URL above using a mocked webhook transport, so no test notification or
credential material was sent.