import json
import os
import subprocess
import sys
import textwrap
from unittest.mock import Mock, patch
import tempfile
import unittest

from observation_controller import (
    ObservationController,
    ObservationControllerError,
    ObservationCriteria,
)
from observation_store import ObservationStore
from paper_observation_adapter import PaperObservationAdapter
from paper_observation_runner import (
    ObservationRunnerError,
    PaperObservationRunner,
    SingleRunnerLock,
)
from incremental_paper_engine import IncrementalPaperEngine, IncrementalPaperEngineError
from observation_notifications import (
    PERSISTENCE_FAILURE_EVENT,
    PERSISTENCE_RECOVERY_EVENT,
    RECONCILIATION_FAILURE_EVENT,
)


class FakeMarketData:
    def __init__(self, candles, health):
        self.candles = candles
        self.health = health
        self.loads = 0

    def load(self):
        self.loads += 1
        return self.candles


class RecordingNotifier:
    def __init__(self):
        self.events = []

    def notify(self, event, status):
        self.events.append((event, status))
        return True


class FailingNotifier:
    def notify(self, _event, _status):
        raise OSError("notification endpoint unavailable")


class PaperObservationRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.store = ObservationStore(os.path.join(self.temp_dir.name, "obs.jsonl"))
        self.engine = IncrementalPaperEngine(
            adapter=PaperObservationAdapter(self.store),
            state_path=os.path.join(self.temp_dir.name, "engine.json"),
        )
        self.controller = ObservationController(
            ObservationCriteria(1, 1, 2, 0.5),
            state_path=os.path.join(self.temp_dir.name, "controller.json"),
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_runner_requires_explicit_completion_criteria(self):
        old = {
            name: os.environ.pop(name, None)
            for name in (
                "OBSERVATION_MIN_COMPLETED_TRADES",
                "OBSERVATION_MIN_DAYS",
                "OBSERVATION_MAX_DAYS",
                "OBSERVATION_MIN_HEALTHY_RATIO",
            )
        }
        try:
            with self.assertRaises(ObservationRunnerError):
                PaperObservationRunner(
                    market_data=FakeMarketData([], {"status": "UNAVAILABLE"}),
                    engine=self.engine,
                    lock_path=os.path.join(self.temp_dir.name, "runner.lock"),
                )
        finally:
            for name, value in old.items():
                if value is not None:
                    os.environ[name] = value

    def test_fresh_runner_fails_closed_on_corrupted_controller_state(self):
        state_path = os.path.join(self.temp_dir.name, "corrupted-controller.json")
        with open(state_path, "w", encoding="utf-8") as handle:
            handle.write("{not valid json")

        old = {
            name: os.environ.get(name)
            for name, value in (
                ("OBSERVATION_MIN_COMPLETED_TRADES", "1"),
                ("OBSERVATION_MIN_DAYS", "1"),
                ("OBSERVATION_MAX_DAYS", "2"),
                ("OBSERVATION_MIN_HEALTHY_RATIO", "0.5"),
            )
        }
        try:
            for name, value in (
                ("OBSERVATION_MIN_COMPLETED_TRADES", "1"),
                ("OBSERVATION_MIN_DAYS", "1"),
                ("OBSERVATION_MAX_DAYS", "2"),
                ("OBSERVATION_MIN_HEALTHY_RATIO", "0.5"),
            ):
                os.environ[name] = value
            os.environ["OBSERVATION_CONTROLLER_STATE_PATH"] = state_path

            runner = PaperObservationRunner(
                market_data=FakeMarketData([], {"status": "UNAVAILABLE"}),
                engine=self.engine,
                lock_path=os.path.join(self.temp_dir.name, "runner.lock"),
            )
            status = runner.status()
            self.assertEqual(status["runner"], "BLOCKED_RESTORE")
            self.assertEqual(status["controller"]["status"], "BLOCKED_RESTORE")
            self.assertEqual(status["controller"]["state_path"], state_path)
            self.assertIn("cannot be restored", status["controller"]["last_error"])
        finally:
            for name, value in old.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
            os.environ.pop("OBSERVATION_CONTROLLER_STATE_PATH", None)

    def test_corrupted_controller_state_is_visible_as_blocked_health(self):
        state_path = os.path.join(self.temp_dir.name, "corrupted-controller.json")
        with open(state_path, "w", encoding="utf-8") as handle:
            handle.write("{not valid json")

        old_state_path = os.environ.get("OBSERVATION_CONTROLLER_STATE_PATH")
        old_criteria = {
            name: os.environ.get(name)
            for name in (
                "OBSERVATION_MIN_COMPLETED_TRADES",
                "OBSERVATION_MIN_DAYS",
                "OBSERVATION_MAX_DAYS",
                "OBSERVATION_MIN_HEALTHY_RATIO",
            )
        }
        try:
            os.environ["OBSERVATION_CONTROLLER_STATE_PATH"] = state_path
            for name, value in (
                ("OBSERVATION_MIN_COMPLETED_TRADES", "1"),
                ("OBSERVATION_MIN_DAYS", "1"),
                ("OBSERVATION_MAX_DAYS", "2"),
                ("OBSERVATION_MIN_HEALTHY_RATIO", "0.5"),
            ):
                os.environ[name] = value
            runner = PaperObservationRunner(
                market_data=FakeMarketData(
                    [{"close": 100}], {"status": "HEALTHY"}
                ),
                engine=self.engine,
                notifier=RecordingNotifier(),
                lock_path=os.path.join(self.temp_dir.name, "blocked.lock"),
            )
            status = runner.run_cycle()
            self.assertEqual(status["runner"], "BLOCKED_RESTORE")
            self.assertEqual(status["controller"]["status"], "BLOCKED_RESTORE")
            self.assertEqual(status["controller"]["state_path"], state_path)
            self.assertIn("cannot be restored", status["controller"]["last_error"])
            self.assertEqual(runner.market_data.loads, 0)
            self.assertEqual(runner.notifier.events, [])
        finally:
            if old_state_path is None:
                os.environ.pop("OBSERVATION_CONTROLLER_STATE_PATH", None)
            else:
                os.environ["OBSERVATION_CONTROLLER_STATE_PATH"] = old_state_path
            for name, value in old_criteria.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value

    def test_unhealthy_cycle_does_not_initialize_engine_or_write_observations(self):
        runner = PaperObservationRunner(
            market_data=FakeMarketData([], {"status": "UNAVAILABLE"}),
            engine=self.engine,
            controller=self.controller,
            lock_path=os.path.join(self.temp_dir.name, "runner.lock"),
        )

        status = runner.run_cycle()

        self.assertEqual(status["controller"]["status"], "RUNNING")
        self.assertIsNone(status["engine"]["started_at"])
        self.assertEqual(self.store.read_records(), [])

    def test_single_runner_lock_rejects_second_process_in_same_process(self):
        path = os.path.join(self.temp_dir.name, "runner.lock")
        with SingleRunnerLock(path):
            with self.assertRaises(ObservationRunnerError):
                with SingleRunnerLock(path):
                    pass

    def test_runner_subprocess_ignores_orphaned_lock_file(self):
        path = os.path.join(self.temp_dir.name, "runner.lock")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("lock file left behind by an earlier process")

        self.controller.state["status"] = "COMPLETED"
        self.controller._save()
        environment = os.environ.copy()
        environment.update(
            {
                "OBSERVATION_MIN_COMPLETED_TRADES": "1",
                "OBSERVATION_MIN_DAYS": "1",
                "OBSERVATION_MAX_DAYS": "2",
                "OBSERVATION_MIN_HEALTHY_RATIO": "0.5",
                "OBSERVATION_CONTROLLER_STATE_PATH": self.controller.state_path.as_posix(),
                "OBSERVATION_RUNNER_LOCK_PATH": path,
            }
        )

        process = subprocess.run(
            [sys.executable, "paper_observation_runner.py", "--once"],
            cwd=os.path.dirname(__file__) or ".",
            env=environment,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )

        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(process.stdout, "")
        self.assertEqual(process.stderr, "")
        self.assertTrue(os.path.isfile(path))

    def test_single_runner_lock_rejects_second_process_and_releases_after_exit(self):
        path = os.path.join(self.temp_dir.name, "runner.lock")
        holder_script = textwrap.dedent(
            """
            import sys
            from paper_observation_runner import SingleRunnerLock

            with SingleRunnerLock(sys.argv[1]):
                print("lock-held", flush=True)
                sys.stdin.read(1)
            """
        )
        contender_script = textwrap.dedent(
            """
            import sys
            from paper_observation_runner import ObservationRunnerError, SingleRunnerLock

            try:
                with SingleRunnerLock(sys.argv[1]):
                    raise SystemExit("lock unexpectedly available")
            except ObservationRunnerError as error:
                print(error)
            """
        )
        reacquire_script = textwrap.dedent(
            """
            import sys
            from paper_observation_runner import SingleRunnerLock

            with SingleRunnerLock(sys.argv[1]):
                print("lock-reacquired")
            """
        )
        environment = os.environ.copy()
        first_process = subprocess.Popen(
            [sys.executable, "-c", holder_script, path],
            cwd=os.path.dirname(__file__) or ".",
            env=environment,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            self.assertEqual(first_process.stdout.readline().strip(), "lock-held")

            second_process = subprocess.run(
                [sys.executable, "-c", contender_script, path],
                cwd=os.path.dirname(__file__) or ".",
                env=environment,
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            self.assertEqual(second_process.returncode, 0, second_process.stderr)
            self.assertEqual(
                second_process.stdout.strip(),
                "another observation runner is already active",
            )

            first_process.stdin.write("release\n")
            first_process.stdin.flush()
            first_process.wait(timeout=10)
            first_stderr = first_process.stderr.read()
            self.assertEqual(first_process.returncode, 0, first_stderr)
            first_process.stdin.close()
            first_process.stdout.close()
            first_process.stderr.close()

            reacquired_process = subprocess.run(
                [sys.executable, "-c", reacquire_script, path],
                cwd=os.path.dirname(__file__) or ".",
                env=environment,
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            self.assertEqual(reacquired_process.returncode, 0, reacquired_process.stderr)
            self.assertEqual(reacquired_process.stdout.strip(), "lock-reacquired")
        finally:
            if first_process.poll() is None:
                first_process.kill()
                first_process.wait(timeout=10)
            for stream in (
                first_process.stdin,
                first_process.stdout,
                first_process.stderr,
            ):
                if stream is not None and not stream.closed:
                    stream.close()

    def test_runner_process_failure_releases_lock_for_restart(self):
        path = os.path.join(self.temp_dir.name, "runner.lock")
        failing_script = textwrap.dedent(
            """
            import sys
            from pathlib import Path
            from paper_observation_runner import PaperObservationRunner

            class FailingRunner(PaperObservationRunner):
                def run_cycle(self):
                    print("lock-held", flush=True)
                    raise RuntimeError("unexpected cycle failure")

            runner = object.__new__(FailingRunner)
            runner.lock_path = Path(sys.argv[1])
            runner.poll_seconds = 1
            runner.run_forever()
            """
        )
        reacquire_script = textwrap.dedent(
            """
            import sys
            from paper_observation_runner import SingleRunnerLock

            with SingleRunnerLock(sys.argv[1]):
                print("lock-reacquired", flush=True)
            """
        )
        environment = os.environ.copy()

        failed_process = subprocess.run(
            [sys.executable, "-c", failing_script, path],
            cwd=os.path.dirname(__file__) or ".",
            env=environment,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )

        self.assertNotEqual(failed_process.returncode, 0)
        self.assertIn("RuntimeError: unexpected cycle failure", failed_process.stderr)
        self.assertEqual(failed_process.stdout.strip(), "lock-held")

        restarted_process = subprocess.run(
            [sys.executable, "-c", reacquire_script, path],
            cwd=os.path.dirname(__file__) or ".",
            env=environment,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )

        self.assertEqual(restarted_process.returncode, 0, restarted_process.stderr)
        self.assertEqual(restarted_process.stdout.strip(), "lock-reacquired")

    def test_forced_runner_termination_releases_lock_for_immediate_restart(self):
        path = os.path.join(self.temp_dir.name, "runner.lock")
        running_script = textwrap.dedent(
            """
            import sys
            import time
            from pathlib import Path
            from paper_observation_runner import PaperObservationRunner

            class SlowRunner(PaperObservationRunner):
                def run_cycle(self):
                    print("lock-held", flush=True)
                    time.sleep(60)

            runner = object.__new__(SlowRunner)
            runner.lock_path = Path(sys.argv[1])
            runner.poll_seconds = 1
            runner.run_forever()
            """
        )
        contender_script = textwrap.dedent(
            """
            import sys
            from paper_observation_runner import ObservationRunnerError, SingleRunnerLock

            try:
                with SingleRunnerLock(sys.argv[1]):
                    print("lock-reacquired", flush=True)
            except ObservationRunnerError as error:
                print(error, flush=True)
            """
        )
        environment = os.environ.copy()
        running_process = subprocess.Popen(
            [sys.executable, "-c", running_script, path],
            cwd=os.path.dirname(__file__) or ".",
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            self.assertEqual(running_process.stdout.readline().strip(), "lock-held")

            blocked_process = subprocess.run(
                [sys.executable, "-c", contender_script, path],
                cwd=os.path.dirname(__file__) or ".",
                env=environment,
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            self.assertEqual(blocked_process.returncode, 0, blocked_process.stderr)
            self.assertEqual(
                blocked_process.stdout.strip(),
                "another observation runner is already active",
            )

            running_process.terminate()
            running_process.wait(timeout=10)

            restarted_process = subprocess.run(
                [sys.executable, "-c", contender_script, path],
                cwd=os.path.dirname(__file__) or ".",
                env=environment,
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            self.assertEqual(restarted_process.returncode, 0, restarted_process.stderr)
            self.assertEqual(restarted_process.stdout.strip(), "lock-reacquired")
        finally:
            if running_process.poll() is None:
                running_process.kill()
                running_process.wait(timeout=10)
            for stream in (
                running_process.stdout,
                running_process.stderr,
            ):
                if stream is not None and not stream.closed:
                    stream.close()

    def test_runner_cli_reports_lock_conflict_without_traceback(self):
        path = os.path.join(self.temp_dir.name, "runner.lock")
        holder_script = textwrap.dedent(
            """
            import sys
            from paper_observation_runner import SingleRunnerLock

            with SingleRunnerLock(sys.argv[1]):
                print("lock-held", flush=True)
                sys.stdin.read(1)
            """
        )
        environment = os.environ.copy()
        environment.update(
            {
                "OBSERVATION_MIN_COMPLETED_TRADES": "1",
                "OBSERVATION_MIN_DAYS": "1",
                "OBSERVATION_MAX_DAYS": "2",
                "OBSERVATION_MIN_HEALTHY_RATIO": "0.5",
                "OBSERVATION_CONTROLLER_STATE_PATH": os.path.join(
                    self.temp_dir.name, "controller.json"
                ),
                "OBSERVATION_RUNNER_LOCK_PATH": path,
            }
        )
        first_process = subprocess.Popen(
            [sys.executable, "-c", holder_script, path],
            cwd=os.path.dirname(__file__) or ".",
            env=environment,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            self.assertEqual(first_process.stdout.readline().strip(), "lock-held")

            second_process = subprocess.run(
                [sys.executable, "paper_observation_runner.py", "--once"],
                cwd=os.path.dirname(__file__) or ".",
                env=environment,
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )

            self.assertNotEqual(second_process.returncode, 0)
            self.assertEqual(
                second_process.stderr,
                "another observation runner is already active\n",
            )
            self.assertNotIn("Traceback", second_process.stderr)
            self.assertEqual(second_process.stdout, "")
        finally:
            if first_process.poll() is None:
                first_process.stdin.write("release\n")
                first_process.stdin.flush()
                first_process.wait(timeout=10)
            for stream in (
                first_process.stdin,
                first_process.stdout,
                first_process.stderr,
            ):
                if stream is not None and not stream.closed:
                    stream.close()

    def test_runner_reports_stale_heartbeat_without_mutating_observation(self):
        self.controller.start(started_at="2026-08-22T00:00:00+00:00")
        self.controller.state["last_cycle_at"] = "2026-08-22T00:00:00+00:00"
        self.controller._save()
        with patch.dict(
            os.environ, {"OBSERVATION_STALE_AFTER_SECONDS": "60"}, clear=False
        ):
            runner = PaperObservationRunner(
                market_data=FakeMarketData([], {"status": "UNAVAILABLE"}),
                engine=self.engine,
                controller=self.controller,
                lock_path=os.path.join(self.temp_dir.name, "runner.lock"),
            )
        with patch("paper_observation_runner.datetime") as fake_datetime:
            fake_datetime.now.return_value = __import__(
                "datetime"
            ).datetime(2026, 8, 22, 0, 2, tzinfo=__import__("datetime").timezone.utc)
            fake_datetime.fromisoformat.side_effect = __import__(
                "datetime"
            ).datetime.fromisoformat
            status = runner.status()
        self.assertEqual(status["runner"], "STALE")
        self.assertEqual(status["controller"]["status"], "RUNNING")

    def test_reconciliation_failure_notifies_once_and_exposes_safe_reason(self):
        notifier = RecordingNotifier()
        runner = PaperObservationRunner(
            market_data=FakeMarketData([{"close": 100}], {"status": "HEALTHY"}),
            engine=self.engine,
            controller=self.controller,
            notifier=notifier,
            lock_path=os.path.join(self.temp_dir.name, "runner.lock"),
        )
        self.engine.status = lambda: {
            "started_at": "2026-08-01T00:00:00+00:00",
            "genuine_signals": 1,
            "genuine_completed_trades": 1,
        }
        self.engine.initialize = lambda _candles: None
        self.engine.process = lambda _candles, data_health: None

        first = runner.run_cycle()
        second = runner.run_cycle()

        self.assertEqual(first["controller"]["status"], "STOPPED_SAFETY_FAILURE")
        reconciliation_events = [
            event_status
            for event_status in notifier.events
            if event_status[0] == RECONCILIATION_FAILURE_EVENT
        ]
        self.assertEqual(len(reconciliation_events), 1)
        event, status = reconciliation_events[0]
        self.assertEqual(event, RECONCILIATION_FAILURE_EVENT)
        self.assertEqual(
            status["last_error"],
            "paper engine totals do not reconcile with persisted observation evidence",
        )
        self.assertEqual(second["controller"]["status"], "STOPPED_SAFETY_FAILURE")

    def test_reconciliation_alert_stays_deduplicated_after_runner_restart(self):
        first_notifier = RecordingNotifier()
        first_runner = PaperObservationRunner(
            market_data=FakeMarketData([{"close": 100}], {"status": "HEALTHY"}),
            engine=self.engine,
            controller=self.controller,
            notifier=first_notifier,
            lock_path=os.path.join(self.temp_dir.name, "runner.lock"),
        )
        self.engine.status = lambda: {
            "started_at": "2026-08-01T00:00:00+00:00",
            "genuine_signals": 1,
            "genuine_completed_trades": 1,
        }
        self.engine.initialize = lambda _candles: None
        self.engine.process = lambda _candles, data_health: None

        first_status = first_runner.run_cycle()

        second_controller = ObservationController(
            ObservationCriteria(1, 1, 2, 0.5),
            state_path=os.path.join(self.temp_dir.name, "controller.json"),
        )
        second_engine = IncrementalPaperEngine(
            adapter=PaperObservationAdapter(self.store),
            state_path=os.path.join(self.temp_dir.name, "engine-restarted.json"),
        )
        second_notifier = RecordingNotifier()
        second_runner = PaperObservationRunner(
            market_data=FakeMarketData([{"close": 200}], {"status": "HEALTHY"}),
            engine=second_engine,
            controller=second_controller,
            notifier=second_notifier,
            lock_path=os.path.join(self.temp_dir.name, "runner-restarted.lock"),
        )

        second_status = second_runner.run_cycle()

        first_reconciliation_events = [
            event for event in first_notifier.events
            if event[0] == RECONCILIATION_FAILURE_EVENT
        ]
        second_reconciliation_events = [
            event for event in second_notifier.events
            if event[0] == RECONCILIATION_FAILURE_EVENT
        ]
        self.assertEqual(first_status["controller"]["status"], "STOPPED_SAFETY_FAILURE")
        self.assertEqual(
            first_status["controller"]["safety_failure_code"],
            "EVIDENCE_RECONCILIATION_FAILURE",
        )
        self.assertEqual(
            [event[0] for event in first_reconciliation_events],
            [RECONCILIATION_FAILURE_EVENT],
        )
        self.assertEqual(second_controller.status()["status"], "STOPPED_SAFETY_FAILURE")
        self.assertEqual(second_status["controller"]["status"], "STOPPED_SAFETY_FAILURE")
        self.assertEqual(second_reconciliation_events, [])

    def test_restored_persistence_outage_is_announced_once_after_runner_restart(self):
        self.engine.state.update(
            {
                "status": "WAITING_FOR_PERSISTENCE",
                "started_at": "2026-08-23T00:00:00+00:00",
                "persistence_health": {
                    "status": "UNAVAILABLE",
                    "error_code": "PAPER_EVIDENCE_STORAGE_UNAVAILABLE",
                    "last_error": "disk full",
                    "operation": "paper_transition_commit",
                },
            }
        )
        self.engine._save_state()
        restarted_engine = IncrementalPaperEngine(
            adapter=PaperObservationAdapter(self.store),
            state_path=self.engine.state_path,
        )
        notifier = RecordingNotifier()
        controller = Mock()
        controller.status.return_value = {"status": "RUNNING"}
        controller.record_cycle.return_value = {"status": "RUNNING"}
        runner = PaperObservationRunner(
            market_data=FakeMarketData([], {"status": "UNAVAILABLE"}),
            engine=restarted_engine,
            controller=controller,
            notifier=notifier,
            lock_path=os.path.join(self.temp_dir.name, "runner.lock"),
        )

        runner.run_cycle()
        runner.run_cycle()

        self.assertEqual(
            [event[0] for event in notifier.events],
            [PERSISTENCE_FAILURE_EVENT],
        )
        outage_status = notifier.events[0][1]
        self.assertEqual(outage_status["last_error"], "disk full")
        self.assertEqual(outage_status["operation"], "paper_transition_commit")

    def test_notification_delivery_failure_does_not_stop_restored_outage_cycle(self):
        self.engine.state.update(
            {
                "status": "WAITING_FOR_PERSISTENCE",
                "started_at": "2026-08-23T00:00:00+00:00",
                "persistence_health": {
                    "status": "UNAVAILABLE",
                    "error_code": "PAPER_EVIDENCE_STORAGE_UNAVAILABLE",
                    "last_error": "disk full",
                    "operation": "paper_transition_commit",
                },
            }
        )
        self.engine._save_state()
        restarted_engine = IncrementalPaperEngine(
            adapter=PaperObservationAdapter(self.store),
            state_path=self.engine.state_path,
        )
        controller = Mock()
        controller.status.return_value = {"status": "RUNNING"}
        controller.record_cycle.return_value = {"status": "RUNNING"}
        runner = PaperObservationRunner(
            market_data=FakeMarketData([], {"status": "UNAVAILABLE"}),
            engine=restarted_engine,
            controller=controller,
            notifier=FailingNotifier(),
            lock_path=os.path.join(self.temp_dir.name, "runner.lock"),
        )

        status = runner.run_cycle()

        self.assertEqual(status["controller"]["status"], "RUNNING")
        self.assertEqual(status["engine"]["status"], "WAITING_FOR_PERSISTENCE")

    def test_storage_guidance_survives_real_process_restart_and_recovers_once(self):
        """Exercise outage restore and recovery across actual Python processes."""
        state_path = os.path.join(self.temp_dir.name, "process-engine.json")
        store_path = os.path.join(self.temp_dir.name, "process-observations.jsonl")
        events_path = os.path.join(self.temp_dir.name, "process-events.jsonl")
        process_script = textwrap.dedent(
            """
            import json
            import os
            from incremental_paper_engine import IncrementalPaperEngine
            from paper_observation_adapter import PaperObservationAdapter
            from observation_store import ObservationStore
            from paper_observation_runner import PaperObservationRunner

            state_path = os.environ["TEST_STATE_PATH"]
            store = ObservationStore(os.environ["TEST_STORE_PATH"])
            engine = IncrementalPaperEngine(
                adapter=PaperObservationAdapter(store),
                state_path=state_path,
            )

            class Market:
                health = {"status": os.environ.get("TEST_MARKET_HEALTH", "UNAVAILABLE")}
                def load(self):
                    return []

            class Controller:
                observation_store = None
                def status(self):
                    return {"status": "RUNNING"}
                def record_cycle(self, **_kwargs):
                    return {"status": "RUNNING"}

            class Notifier:
                def notify(self, event, status):
                    if os.environ.get("TEST_FAIL_NOTIFIER") == "1":
                        raise OSError("notification endpoint unavailable")
                    with open(os.environ["TEST_EVENTS_PATH"], "a", encoding="utf-8") as handle:
                        handle.write(json.dumps({"event": event, "status": status}) + "\\n")

            mode = os.environ["TEST_PROCESS_MODE"]
            if mode == "seed":
                engine.state.update({
                    "status": "WAITING_FOR_PERSISTENCE",
                    "started_at": "2026-08-23T00:00:00+00:00",
                    "persistence_health": {
                        "status": "UNAVAILABLE",
                        "error_code": "PAPER_EVIDENCE_STORAGE_UNAVAILABLE",
                        "last_error": "disk full",
                        "operation": "paper_transition_commit",
                    },
                })
                engine._save_state()
                runner = PaperObservationRunner(
                    market_data=Market(),
                    engine=engine,
                    controller=Controller(),
                    notifier=Notifier(),
                    lock_path=state_path + ".lock",
                )
                runner.run_cycle()
            elif mode == "restart":
                def durable_recovery():
                    engine.state["status"] = "RUNNING"
                    engine.state["persistence_health"] = {
                        "status": "HEALTHY",
                        "error_code": None,
                        "last_error": None,
                        "operation": None,
                    }
                    engine._save_state()
                engine.process = lambda _candles, data_health: durable_recovery()
                runner = PaperObservationRunner(
                    market_data=Market(),
                    engine=engine,
                    controller=Controller(),
                    notifier=Notifier(),
                    lock_path=state_path + ".lock",
                )
                runner.run_cycle()
                runner.run_cycle()
            else:
                raise SystemExit("unknown test process mode")
            """
        )

        def run_process(mode, *, fail_notifier=False, market_health="UNAVAILABLE"):
            environment = os.environ.copy()
            environment.update(
                {
                    "TEST_PROCESS_MODE": mode,
                    "TEST_STATE_PATH": state_path,
                    "TEST_STORE_PATH": store_path,
                    "TEST_EVENTS_PATH": events_path,
                    "TEST_FAIL_NOTIFIER": "1" if fail_notifier else "0",
                    "TEST_MARKET_HEALTH": market_health,
                }
            )
            return subprocess.run(
                [sys.executable, "-c", process_script],
                cwd=os.path.dirname(__file__) or ".",
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )

        first_process = run_process("seed", fail_notifier=True)
        self.assertEqual(first_process.returncode, 0, first_process.stderr)

        restarted_process = run_process("restart", market_health="HEALTHY")
        self.assertEqual(restarted_process.returncode, 0, restarted_process.stderr)
        with open(events_path, encoding="utf-8") as handle:
            events = [json.loads(line) for line in handle if line.strip()]
        self.assertEqual(
            [event["event"] for event in events],
            [PERSISTENCE_FAILURE_EVENT, PERSISTENCE_RECOVERY_EVENT],
        )
        self.assertEqual(events[0]["status"]["last_error"], "disk full")
        self.assertEqual(events[0]["status"]["operation"], "paper_transition_commit")
        self.assertEqual(events[1]["status"], {
            "status": "HEALTHY",
            "error_code": None,
            "last_error": None,
            "operation": None,
        })

    def test_storage_recovery_notifies_healthy_state_without_stale_outage_details(self):
        notifier = RecordingNotifier()
        engine = Mock()
        engine.adapter = self.engine.adapter
        storage = {"status": "HEALTHY"}
        engine.status.side_effect = lambda: {
            "started_at": "2026-08-23T00:00:00+00:00",
            "persistence_health": dict(storage),
        }

        def process(_candles, data_health):
            if storage["status"] == "HEALTHY":
                storage.update(
                    {
                        "status": "UNAVAILABLE",
                        "error_code": "PAPER_EVIDENCE_STORAGE_UNAVAILABLE",
                        "last_error": "disk full",
                        "operation": "paper_transition_commit",
                    }
                )
                raise IncrementalPaperEngineError(
                    "paper observation paused: evidence storage is unavailable"
                )
            storage.update(
                {
                    "status": "HEALTHY",
                    "error_code": None,
                    "last_error": None,
                    "operation": None,
                }
            )

        engine.process.side_effect = process
        controller = Mock()
        controller.status.return_value = {"status": "RUNNING"}
        controller.record_cycle.return_value = {"status": "RUNNING"}
        runner = PaperObservationRunner(
            market_data=FakeMarketData([{"close": 100}], {"status": "HEALTHY"}),
            engine=engine,
            controller=controller,
            notifier=notifier,
            lock_path=os.path.join(self.temp_dir.name, "runner.lock"),
        )

        runner.run_cycle()
        runner.run_cycle()

        self.assertEqual(
            [event[0] for event in notifier.events],
            [PERSISTENCE_FAILURE_EVENT, PERSISTENCE_RECOVERY_EVENT],
        )
        recovery_status = notifier.events[-1][1]
        self.assertEqual(recovery_status["status"], "HEALTHY")
        self.assertIsNone(recovery_status["error_code"])
        self.assertIsNone(recovery_status["last_error"])
        self.assertIsNone(recovery_status["operation"])