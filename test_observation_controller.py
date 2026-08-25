import os
import tempfile
import unittest

from observation_controller import (
    ObservationController,
    ObservationControlError,
    ObservationControllerError,
    ObservationCriteria,
    apply_paper_control,
    observation_control_lock,
)
from observation_store import ObservationStore
from paper_observation_adapter import PaperObservationAdapter


class ObservationControllerTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.criteria = ObservationCriteria(
            min_completed_trades=2,
            min_observation_days=1,
            max_observation_days=3,
            min_healthy_ratio=0.75,
        )
        self.controller = ObservationController(
            self.criteria,
            state_path=os.path.join(self.temp_dir.name, "controller.json"),
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_requires_genuine_trade_count_and_health(self):
        self.controller.start(started_at="2026-08-01T00:00:00+00:00")
        status = self.controller.record_cycle(
            data_health="HEALTHY",
            engine_status={
                "genuine_signals": 3,
                "genuine_completed_trades": 2,
            },
            observed_at="2026-08-01T12:00:00+00:00",
        )
        self.assertEqual(status["status"], "RUNNING")

        status = self.controller.record_cycle(
            data_health="HEALTHY",
            engine_status={
                "genuine_signals": 3,
                "genuine_completed_trades": 2,
            },
            observed_at="2026-08-02T00:00:00+00:00",
        )
        self.assertEqual(status["status"], "COMPLETED")

    def test_maximum_duration_does_not_fake_completion(self):
        self.controller.start(started_at="2026-08-01T00:00:00+00:00")
        status = self.controller.record_cycle(
            data_health="HEALTHY",
            engine_status={
                "genuine_signals": 1,
                "genuine_completed_trades": 0,
            },
            observed_at="2026-08-04T00:00:00+00:00",
        )
        self.assertEqual(status["status"], "STOPPED_INSUFFICIENT_EVIDENCE")
        self.assertEqual(status["trade_count"], 0)

    def test_restart_restores_controller_state(self):
        self.controller.start(started_at="2026-08-01T00:00:00+00:00")
        self.controller.record_cycle(
            data_health="DEGRADED",
            engine_status={
                "genuine_signals": 1,
                "genuine_completed_trades": 0,
            },
            observed_at="2026-08-01T01:00:00+00:00",
        )
        restored = ObservationController(
            self.criteria,
            state_path=self.controller.state_path,
        )
        self.assertEqual(restored.status()["cycles"], 1)
        self.assertEqual(restored.status()["unhealthy_cycles"], 1)

    def test_corrupted_json_state_cannot_be_restored(self):
        with open(self.controller.state_path, "w", encoding="utf-8") as handle:
            handle.write("{not valid json")

        with self.assertRaisesRegex(
            ObservationControllerError,
            "observation controller state cannot be restored",
        ):
            ObservationController(self.criteria, state_path=self.controller.state_path)

    def test_unreadable_json_state_cannot_be_restored(self):
        with open(self.controller.state_path, "wb") as handle:
            handle.write(b"\xff\xfe")

        with self.assertRaisesRegex(
            ObservationControllerError,
            "observation controller state cannot be restored",
        ):
            ObservationController(self.criteria, state_path=self.controller.state_path)

    def test_incomplete_json_state_cannot_be_restored(self):
        with open(self.controller.state_path, "w", encoding="utf-8") as handle:
            handle.write('{"status": "RUNNING", "cycles": "one"}')

        with self.assertRaisesRegex(
            ObservationControllerError,
            "observation controller state cannot be restored",
        ):
            ObservationController(self.criteria, state_path=self.controller.state_path)

    def test_safety_stop_is_terminal(self):
        self.controller.start(started_at="2026-08-01T00:00:00+00:00")
        status = self.controller.stop_for_safety("market data unavailable")
        self.assertEqual(status["status"], "STOPPED_SAFETY_FAILURE")
        with self.assertRaises(ObservationControllerError):
            self.controller.start()

    def test_operator_controls_pause_resume_and_stop(self):
        self.controller.start(started_at="2026-08-01T00:00:00+00:00")
        self.assertEqual(self.controller.pause()["status"], "PAUSED")
        self.assertEqual(self.controller.start()["status"], "RUNNING")
        self.assertEqual(self.controller.stop()["status"], "STOPPED_MANUAL")
        with self.assertRaises(ObservationControllerError):
            self.controller.start()

    def test_operator_control_requires_authentication_confirmation_and_risk(self):
        lock_path = os.path.join(self.temp_dir.name, "controls.lock")
        kwargs = dict(
            authenticated=False,
            confirmed=True,
            risk_governor=lambda _action: True,
            criteria=self.criteria,
            state_path=self.controller.state_path,
            lock_path=lock_path,
        )
        with self.assertRaisesRegex(ObservationControlError, "authenticated"):
            apply_paper_control("START", **kwargs)
        kwargs["authenticated"] = True
        kwargs["confirmed"] = False
        with self.assertRaisesRegex(ObservationControlError, "confirmation"):
            apply_paper_control("START", **kwargs)
        kwargs["confirmed"] = True
        kwargs["risk_governor"] = lambda _action: False
        with self.assertRaisesRegex(ObservationControlError, "Risk Governor"):
            apply_paper_control("START", **kwargs)

    def test_operator_control_rejects_lock_conflict(self):
        lock_path = os.path.join(self.temp_dir.name, "controls.lock")
        with observation_control_lock(lock_path):
            with self.assertRaisesRegex(ObservationControlError, "busy"):
                apply_paper_control(
                    "START",
                    authenticated=True,
                    confirmed=True,
                    risk_governor=lambda _action: True,
                    criteria=self.criteria,
                    state_path=self.controller.state_path,
                    lock_path=lock_path,
                )

    def test_operator_control_rejects_stale_running_state(self):
        self.controller.start(started_at="2026-08-01T00:00:00+00:00")
        self.controller.state["last_cycle_at"] = "2026-08-01T00:00:00+00:00"
        self.controller._save()
        with self.assertRaisesRegex(ObservationControlError, "stale"):
            apply_paper_control(
                "PAUSE",
                authenticated=True,
                confirmed=True,
                risk_governor=lambda _action: True,
                criteria=self.criteria,
                state_path=self.controller.state_path,
                lock_path=os.path.join(self.temp_dir.name, "controls.lock"),
                stale_after_seconds=1,
            )

    def test_invalid_criteria_are_rejected(self):
        with self.assertRaises(ValueError):
            ObservationCriteria(
                min_completed_trades=0,
                min_observation_days=1,
                max_observation_days=3,
                min_healthy_ratio=0.75,
            )

    def test_completion_reconciles_genuine_jsonl_evidence(self):
        store = ObservationStore(os.path.join(self.temp_dir.name, "observations.jsonl"))
        adapter = PaperObservationAdapter(store)
        controller = ObservationController(
            ObservationCriteria(20, 7, 14, 0.95),
            state_path=os.path.join(self.temp_dir.name, "verified-controller.json"),
            observation_store=store,
        )
        controller.start(started_at="2026-08-01T00:00:00+00:00")

        for index in range(20):
            timestamp = f"2026-08-{index + 1:02d}T00:00:00+00:00"
            adapter.record_signal(
                signal_id=f"signal-{index}",
                observed_at=timestamp,
                symbol="BTC/CAD",
                strategy_score=80,
                entry_eligible=True,
                market_data_timestamp=timestamp,
                data_health="HEALTHY",
            )
            adapter.record_trade(
                trade_id=f"trade-{index}",
                signal_id=f"signal-{index}",
                entry_at=timestamp,
                exit_at=timestamp,
                entry_price=100,
                exit_price=101,
                profit_loss=1,
                fees=0,
                slippage=0,
                exit_reason="TAKE PROFIT",
            )
            status = controller.record_cycle(
                data_health="HEALTHY",
                engine_status={
                    "genuine_signals": index + 1,
                    "genuine_completed_trades": index + 1,
                },
                observed_at=(
                    "2026-08-01T00:00:00+00:00"
                    if index < 19
                    else "2026-08-08T00:00:00+00:00"
                ),
            )

        self.assertEqual(status["status"], "COMPLETED")
        self.assertEqual(status["trade_count"], 20)
        self.assertEqual(status["observation_days"], 7.0)
        self.assertEqual(status["healthy_ratio"], 1.0)
        self.assertTrue(status["evidence_reconciled"])

    def test_mismatched_jsonl_evidence_stops_safely(self):
        store = ObservationStore(os.path.join(self.temp_dir.name, "mismatch.jsonl"))
        controller = ObservationController(
            ObservationCriteria(1, 1, 2, 0.5),
            state_path=os.path.join(self.temp_dir.name, "mismatch-controller.json"),
            observation_store=store,
        )
        controller.start(started_at="2026-08-01T00:00:00+00:00")

        status = controller.record_cycle(
            data_health="HEALTHY",
            engine_status={
                "genuine_signals": 1,
                "genuine_completed_trades": 1,
            },
            observed_at="2026-08-02T00:00:00+00:00",
        )

        self.assertEqual(status["status"], "STOPPED_SAFETY_FAILURE")
        self.assertFalse(status["evidence_reconciled"])
        self.assertEqual(
            status["safety_failure_code"], "EVIDENCE_RECONCILIATION_FAILURE"
        )