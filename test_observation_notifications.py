import json
import unittest
from urllib.error import URLError
from unittest.mock import patch

from observation_notifications import (
    BROWSER_DRIFT_FAILURE_EVENT,
    PERSISTENCE_FAILURE_EVENT,
    RECONCILIATION_FAILURE_EVENT,
    ObservationNotifier,
)


class ObservationNotificationTests(unittest.TestCase):
    @patch("observation_notifications.urlopen")
    def test_scheduled_browser_drift_alert_links_run_and_visual_diffs(
        self, urlopen
    ):
        response = urlopen.return_value.__enter__.return_value
        response.status = 204
        webhook_secret = "secret-token-that-must-not-be-forwarded"
        notifier = ObservationNotifier(
            webhook_url=f"https://hooks.example.test/{webhook_secret}"
        )

        self.assertTrue(
            notifier.notify(
                BROWSER_DRIFT_FAILURE_EVENT,
                {
                    "run_url": (
                        "https://github.com/example/project/actions/runs/123"
                    ),
                    "artifacts_url": (
                        "https://github.com/example/project/actions/runs/123/artifacts"
                    ),
                },
            )
        )

        request = urlopen.call_args.args[0]
        payload = json.loads(request.data.decode("utf-8"))
        self.assertIn(
            "<https://github.com/example/project/actions/runs/123|open run>",
            payload["text"],
        )
        self.assertIn(
            "<https://github.com/example/project/actions/runs/123/artifacts|open artifacts>",
            payload["text"],
        )
        self.assertNotIn(webhook_secret, payload["text"])

    @patch(
        "observation_notifications.urlopen",
        side_effect=URLError("connection refused"),
    )
    def test_scheduled_browser_drift_alert_returns_false_when_delivery_fails(
        self, urlopen
    ):
        notifier = ObservationNotifier(webhook_url="https://hooks.example.test")

        self.assertFalse(
            notifier.notify(
                BROWSER_DRIFT_FAILURE_EVENT,
                {
                    "run_url": "https://github.com/example/project/actions/runs/123",
                    "artifacts_url": (
                        "https://github.com/example/project/actions/runs/123/artifacts"
                    ),
                },
            )
        )
        urlopen.assert_called_once()

    @patch("observation_notifications.urlopen")
    def test_reconciliation_alert_contains_only_terminal_state_and_reason(
        self, urlopen
    ):
        response = urlopen.return_value.__enter__.return_value
        response.status = 200
        notifier = ObservationNotifier(webhook_url="https://hooks.example.test")

        self.assertTrue(
            notifier.notify(
                RECONCILIATION_FAILURE_EVENT,
                {
                    "status": "STOPPED_SAFETY_FAILURE",
                    "last_error": (
                        "paper engine totals do not reconcile with persisted "
                        "observation evidence"
                    ),
                    "signal_count": 12,
                    "trade_count": 4,
                },
            )
        )

        request = urlopen.call_args.args[0]
        payload = json.loads(request.data.decode("utf-8"))
        self.assertIn("STOPPED_SAFETY_FAILURE", payload["text"])
        self.assertIn("paper engine totals do not reconcile", payload["text"])
        self.assertNotIn("12", payload["text"])
        self.assertNotIn("4", payload["text"])

    @patch("observation_notifications.urlopen")
    def test_persistence_outage_alert_contains_stable_code_and_safe_reason(
        self, urlopen
    ):
        response = urlopen.return_value.__enter__.return_value
        response.status = 200
        notifier = ObservationNotifier(webhook_url="https://hooks.example.test")

        self.assertTrue(
            notifier.notify(
                PERSISTENCE_FAILURE_EVENT,
                {
                    "status": "WAITING_FOR_PERSISTENCE",
                    "operation": "paper_transition_commit",
                    "error_code": "PAPER_EVIDENCE_STORAGE_UNAVAILABLE",
                    "last_error": "disk full",
                },
            )
        )

        request = urlopen.call_args.args[0]
        payload = json.loads(request.data.decode("utf-8"))
        self.assertIn("PAPER_EVIDENCE_STORAGE_UNAVAILABLE", payload["text"])
        self.assertIn("Reason: disk full", payload["text"])