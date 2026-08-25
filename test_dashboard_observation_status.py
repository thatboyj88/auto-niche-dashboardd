import json
import os
import tempfile
import unittest
from unittest.mock import patch

from dashboard import load_live_observation_status


class DashboardObservationStatusTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_dir = self.temp_dir.name

    def tearDown(self):
        self.temp_dir.cleanup()

    def _write(self, name, value):
        with open(os.path.join(self.data_dir, name), "w", encoding="utf-8") as handle:
            json.dump(value, handle)

    def test_dashboard_reads_controller_engine_and_paper_store(self):
        self._write(
            "observation_controller.json",
            {
                "status": "RUNNING",
                "started_at": "2026-08-21T04:16:56+00:00",
                "last_cycle_at": "2026-08-21T04:20:00+00:00",
                "last_data_health": "HEALTHY",
                "cycles": 2,
                "healthy_cycles": 2,
                "unhealthy_cycles": 0,
            },
        )
        self._write(
            "paper_engine_state.json",
            {
                "capital": 25.0,
                "position": 0.0,
                "genuine_signals": 1,
                "genuine_completed_trades": 0,
                "persistence_health": {
                    "status": "UNAVAILABLE",
                    "error_code": "PAPER_EVIDENCE_STORAGE_UNAVAILABLE",
                    "last_error": "disk full",
                    "operation": "paper_transition_commit",
                },
            },
        )
        with open(
            os.path.join(self.data_dir, "observations.jsonl"),
            "w",
            encoding="utf-8",
        ) as handle:
            handle.write(
                json.dumps(
                    {
                        "dataset": "PAPER_OPERATIONAL",
                        "record_type": "SIGNAL",
                    }
                )
                + "\n"
            )

        with patch.dict(
            os.environ,
            {
                "OBSERVATION_DATA_DIR": self.data_dir,
                "OBSERVATION_CONTROLLER_STATE_PATH": os.path.join(
                    self.data_dir, "observation_controller.json"
                ),
                "PAPER_ENGINE_STATE_PATH": os.path.join(
                    self.data_dir, "paper_engine_state.json"
                ),
                "OBSERVATION_STORE_PATH": os.path.join(
                    self.data_dir, "observations.jsonl"
                ),
            },
            clear=False,
        ):
            status = load_live_observation_status()

        self.assertTrue(status["available"])
        self.assertEqual(status["status"], "RUNNING")
        self.assertEqual(status["signals"], 1)
        self.assertEqual(status["trades"], 0)
        self.assertEqual(status["healthy_ratio"], 1)
        self.assertEqual(status["paper_storage"]["status"], "UNAVAILABLE")
        self.assertEqual(
            status["paper_storage"]["error_code"],
            "PAPER_EVIDENCE_STORAGE_UNAVAILABLE",
        )
        self.assertEqual(status["paper_storage"]["last_error"], "disk full")

    def test_dashboard_does_not_assume_storage_is_healthy_without_runner_signal(self):
        self._write(
            "paper_engine_state.json",
            {
                "capital": 25.0,
                "persistence_health": {
                    "status": "UNKNOWN",
                    "error_code": None,
                    "last_error": None,
                    "operation": None,
                },
            },
        )
        with patch.dict(
            os.environ,
            {
                "OBSERVATION_DATA_DIR": self.data_dir,
                "PAPER_ENGINE_STATE_PATH": os.path.join(
                    self.data_dir, "paper_engine_state.json"
                ),
            },
            clear=False,
        ):
            status = load_live_observation_status()

        self.assertEqual(status["paper_storage"]["status"], "UNKNOWN")

    def test_dashboard_does_not_create_missing_runtime_files(self):
        missing_dir = os.path.join(self.data_dir, "missing")
        with patch.dict(
            os.environ,
            {"OBSERVATION_DATA_DIR": missing_dir},
            clear=False,
        ):
            status = load_live_observation_status()

        self.assertFalse(status["available"])
        self.assertFalse(os.path.exists(missing_dir))

    def test_dashboard_exposes_blocked_controller_restore(self):
        controller_path = os.path.join(self.data_dir, "observation_controller.json")
        with open(controller_path, "w", encoding="utf-8") as handle:
            handle.write("{not valid json")

        with patch.dict(
            os.environ,
            {
                "OBSERVATION_DATA_DIR": self.data_dir,
                "OBSERVATION_CONTROLLER_STATE_PATH": controller_path,
            },
            clear=False,
        ):
            status = load_live_observation_status()

        self.assertTrue(status["available"])
        self.assertEqual(status["status"], "BLOCKED_RESTORE")
        self.assertEqual(status["runner_status"], "BLOCKED_RESTORE")
        self.assertEqual(status["controller_state_path"], controller_path)
        self.assertEqual(
            status["controller_restore_error"],
            "observation controller state cannot be restored",
        )
        self.assertNotIn("milestone", status)

    def test_dashboard_marks_a_stale_observation_heartbeat_without_writing(self):
        self._write(
            "observation_controller.json",
            {
                "status": "RUNNING",
                "started_at": "2026-08-22T00:00:00+00:00",
                "last_cycle_at": "2026-08-22T00:00:00+00:00",
                "cycles": 1,
                "healthy_cycles": 1,
                "unhealthy_cycles": 0,
            },
        )
        with patch.dict(
            os.environ,
            {
                "OBSERVATION_DATA_DIR": self.data_dir,
                "OBSERVATION_STALE_AFTER_SECONDS": "60",
            },
            clear=False,
        ), patch("dashboard.datetime") as fake_datetime:
            fake_datetime.fromisoformat.side_effect = __import__(
                "datetime"
            ).datetime.fromisoformat
            fake_datetime.now.return_value = __import__(
                "datetime"
            ).datetime(2026, 8, 22, 0, 2, tzinfo=__import__("datetime").timezone.utc)
            status = load_live_observation_status()
        self.assertTrue(status["cycle_is_stale"])
        self.assertEqual(status["runner_status"], "STALE")