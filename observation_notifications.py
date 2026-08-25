"""Optional, non-blocking Slack milestone notifications."""

from __future__ import annotations

import json
import os
from urllib.error import URLError
from urllib.request import Request, urlopen


RECONCILIATION_FAILURE_EVENT = "OBSERVATION_RECONCILIATION_FAILURE"
PERSISTENCE_FAILURE_EVENT = "OBSERVATION_PERSISTENCE_UNAVAILABLE"
PERSISTENCE_RECOVERY_EVENT = "OBSERVATION_PERSISTENCE_RECOVERED"
BROWSER_DRIFT_FAILURE_EVENT = "BROWSER_DRIFT_FAILURE"


class ObservationNotifier:
    """Send safe operational milestones without affecting trading behavior."""

    def __init__(self, webhook_url: str | None = None, timeout: float = 10):
        self.webhook_url = webhook_url or os.getenv(
            "BTC_CAD_PREFLIGHT_SLACK_WEBHOOK_URL"
        )
        self.timeout = timeout

    def notify(self, event: str, status: dict) -> bool:
        if not self.webhook_url:
            return False
        if event == RECONCILIATION_FAILURE_EVENT:
            text = (
                f"BTC/CAD paper observation: {event}\n"
                f"Safety state: {status.get('status', 'UNKNOWN')}\n"
                f"Reason: {status.get('last_error', 'unknown reconciliation failure')}"
            )
        elif event == PERSISTENCE_FAILURE_EVENT:
            text = (
                f"BTC/CAD paper observation: {event}\n"
                "Observation progress is paused until evidence is durably committed.\n"
                f"Operation: {status.get('operation', 'unknown')}\n"
                f"Error code: {status.get('error_code', 'PAPER_EVIDENCE_STORAGE_UNAVAILABLE')}\n"
                f"Reason: {status.get('last_error', 'unknown storage failure')}"
            )
        elif event == PERSISTENCE_RECOVERY_EVENT:
            text = (
                f"BTC/CAD paper observation: {event}\n"
                "Evidence storage is healthy and paper observation has resumed."
            )
        elif event == BROWSER_DRIFT_FAILURE_EVENT:
            text = (
                "AI Trading Dashboard: scheduled browser drift checks failed.\n"
                f"Failed workflow run: <{status.get('run_url', '')}|open run>\n"
                "Focused visual diffs: "
                f"<{status.get('artifacts_url', '')}|open artifacts>"
            )
        else:
            text = (
                f"BTC/CAD paper observation: {event}\n"
                f"Status: {status.get('status', 'UNKNOWN')}\n"
                f"Genuine signals: {status.get('signal_count', 0)}\n"
                f"Genuine completed trades: {status.get('trade_count', 0)}\n"
                f"Data health: {status.get('last_data_health', 'UNKNOWN')}"
            )
        payload = {
            "text": text
        }
        request = Request(
            self.webhook_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                return 200 <= response.status < 300
        except (OSError, URLError):
            # Notification failure must never stop or change paper execution.
            return False