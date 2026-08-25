import os
import json
import stat
import tempfile
import unittest
from unittest.mock import patch

from observation_store import (
    HISTORICAL,
    PAPER_OPERATIONAL,
    SIGNAL,
    TRADE,
    ObservationStore,
    ObservationStoreError,
)


class ObservationStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.temp_dir.name, "nested", "observations.jsonl")
        self.store = ObservationStore(self.path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_store_is_created_with_private_permissions(self):
        mode = stat.S_IMODE(os.stat(self.path).st_mode)
        self.assertEqual(mode, 0o600)

    def test_historical_and_paper_records_remain_separate(self):
        self.store.append(
            dataset=HISTORICAL,
            record_type=TRADE,
            payload={"net_profit_loss": 1.0},
            occurred_at="2026-01-01T00:00:00+00:00",
        )
        self.store.append(
            dataset=PAPER_OPERATIONAL,
            record_type=SIGNAL,
            payload={"strategy_score": 80},
            occurred_at="2026-02-01T00:00:00+00:00",
        )

        summary = self.store.summary()
        self.assertEqual(summary["historical"]["TRADE"], 1)
        self.assertEqual(summary["paper_operational"]["SIGNAL"], 1)
        self.assertEqual(summary["paper_observation_count"], 1)

    def test_idempotency_key_prevents_duplicate_records(self):
        first = self.store.append(
            dataset=PAPER_OPERATIONAL,
            record_type=TRADE,
            payload={"trade_number": 1},
            idempotency_key="paper-trade-1",
        )
        second = self.store.append(
            dataset=PAPER_OPERATIONAL,
            record_type=TRADE,
            payload={"trade_number": 1, "changed": True},
            idempotency_key="paper-trade-1",
        )

        self.assertEqual(first, second)
        self.assertEqual(len(self.store.read_records()), 1)

    def test_invalid_dataset_and_type_fail_closed(self):
        with self.assertRaises(ObservationStoreError):
            self.store.append(
                dataset="BACKTEST",
                record_type=TRADE,
                payload={},
            )
        with self.assertRaises(ObservationStoreError):
            self.store.append(
                dataset=PAPER_OPERATIONAL,
                record_type="ORDER",
                payload={},
            )

    def test_sensitive_payload_fields_are_rejected(self):
        with self.assertRaises(ObservationStoreError):
            self.store.append(
                dataset=PAPER_OPERATIONAL,
                record_type=SIGNAL,
                payload={"nested": {"api_key": "must-not-persist"}},
            )

    def test_corrupt_record_fails_loudly_instead_of_being_silently_skipped(self):
        with open(self.path, "w", encoding="utf-8") as handle:
            handle.write('{"dataset":"PAPER_OPERATIONAL"}\n')

        with self.assertRaises(ObservationStoreError):
            self.store.read_records()

    def test_interrupted_paper_transition_replays_without_duplicate_evidence(self):
        state_path = os.path.join(self.temp_dir.name, "engine-state.json")
        state = {"last_processed_timestamp": 2, "genuine_signals": 1}
        record = self.store.build_record(
            dataset=PAPER_OPERATIONAL,
            record_type=SIGNAL,
            payload={"signal_id": "signal-1"},
            occurred_at="2026-08-01T00:00:00+00:00",
            idempotency_key="signal:1",
        )

        def crash(point):
            if point == "after_evidence":
                raise RuntimeError("injected process crash")

        self.store._test_transaction_failpoint = crash
        with self.assertRaisesRegex(RuntimeError, "injected process crash"):
            self.store.commit_paper_transition(
                state_path=state_path, state=state, records=[record]
            )
        self.assertFalse(os.path.exists(state_path))
        self.assertEqual(len(self.store.read_records()), 1)

        restarted_store = ObservationStore(self.path)
        restarted_store.recover_paper_transition(state_path)
        self.assertEqual(len(restarted_store.read_records()), 1)
        with open(state_path, encoding="utf-8") as handle:
            self.assertEqual(json.load(handle), state)
        self.assertFalse(os.path.exists(state_path + ".txn"))

    def test_crash_before_apply_replays_transaction(self):
        state_path = os.path.join(self.temp_dir.name, "engine-state.json")
        state = {"genuine_signals": 2}
        self.store._test_transaction_failpoint = lambda point: (
            (_ for _ in ()).throw(RuntimeError("injected process crash"))
            if point == "after_journal" else None
        )
        with self.assertRaises(RuntimeError):
            self.store.commit_paper_transition(
                state_path=state_path, state=state, records=[]
            )
        restarted_store = ObservationStore(self.path)
        restarted_store.recover_paper_transition(state_path)
        with open(state_path, encoding="utf-8") as handle:
            self.assertEqual(json.load(handle), state)

    def _transition_fixture(self):
        state_path = os.path.join(self.temp_dir.name, "engine-state.json")
        state = {"last_processed_timestamp": 2, "genuine_signals": 1}
        record = self.store.build_record(
            dataset=PAPER_OPERATIONAL,
            record_type=SIGNAL,
            payload={"signal_id": "signal-fault"},
            occurred_at="2026-08-01T00:00:00+00:00",
            idempotency_key="signal:fault",
        )
        return state_path, state, record

    def test_journal_write_failure_leaves_no_partial_transition(self):
        state_path, _, record = self._transition_fixture()
        with patch("observation_store.os.fsync", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                self.store.commit_paper_transition(
                    state_path=state_path, state={"last_processed_timestamp": 2},
                    records=[record],
                )

        self.assertEqual(self.store.read_records(), [])
        self.assertFalse(os.path.exists(state_path))
        self.assertFalse(os.path.exists(state_path + ".txn"))
        self.assertFalse(os.path.exists(state_path + ".txn.tmp"))
        restarted = ObservationStore(self.path)
        self.assertEqual(restarted.read_records(dataset=PAPER_OPERATIONAL), [])

    def test_journal_write_error_leaves_no_partial_transition(self):
        state_path, _, record = self._transition_fixture()
        with patch("observation_store.json.dump", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                self.store.commit_paper_transition(
                    state_path=state_path, state={"last_processed_timestamp": 2},
                    records=[record],
                )

        self.assertEqual(self.store.read_records(), [])
        self.assertFalse(os.path.exists(state_path))
        self.assertFalse(os.path.exists(state_path + ".txn"))
        self.assertFalse(os.path.exists(state_path + ".txn.tmp"))

    def test_evidence_fsync_failure_keeps_journal_for_deterministic_recovery(self):
        state_path, state, record = self._transition_fixture()
        calls = 0
        original_fsync = os.fsync

        def fail_evidence_fsync(descriptor):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("disk full")
            return original_fsync(descriptor)

        with patch("observation_store.os.fsync", side_effect=fail_evidence_fsync):
            with self.assertRaises(OSError):
                self.store.commit_paper_transition(
                    state_path=state_path, state=state, records=[record]
                )

        self.assertEqual(self.store.read_records(), [])
        self.assertFalse(os.path.exists(state_path))
        self.assertTrue(os.path.exists(state_path + ".txn"))
        self.assertFalse(os.path.exists(self.path + ".txn-tmp"))
        restarted = ObservationStore(self.path)
        restarted.recover_paper_transition(state_path)
        self.assertEqual(restarted.read_records(dataset=PAPER_OPERATIONAL), [record])
        with open(state_path, encoding="utf-8") as handle:
            self.assertEqual(json.load(handle), state)

    def test_replace_failure_after_evidence_is_recoverable_without_duplicates(self):
        state_path, state, record = self._transition_fixture()
        calls = 0
        original_replace = os.replace

        def fail_state_replace(source, destination):
            nonlocal calls
            calls += 1
            if calls == 3:
                raise OSError("disk full")
            return original_replace(source, destination)

        with patch("observation_store.os.replace", side_effect=fail_state_replace):
            with self.assertRaises(OSError):
                self.store.commit_paper_transition(
                    state_path=state_path, state=state, records=[record]
                )

        self.assertEqual(len(self.store.read_records(dataset=PAPER_OPERATIONAL)), 1)
        self.assertFalse(os.path.exists(state_path))
        self.assertTrue(os.path.exists(state_path + ".txn"))
        restarted = ObservationStore(self.path)
        restarted.recover_paper_transition(state_path)
        restarted.recover_paper_transition(state_path)
        self.assertEqual(len(restarted.read_records(dataset=PAPER_OPERATIONAL)), 1)
        with open(state_path, encoding="utf-8") as handle:
            self.assertEqual(json.load(handle), state)

    def test_evidence_replace_failure_retries_from_journal(self):
        state_path, state, record = self._transition_fixture()
        calls = 0
        original_replace = os.replace

        def fail_evidence_replace(source, destination):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("disk full")
            return original_replace(source, destination)

        with patch("observation_store.os.replace", side_effect=fail_evidence_replace):
            with self.assertRaises(OSError):
                self.store.commit_paper_transition(
                    state_path=state_path, state=state, records=[record]
                )

        self.assertEqual(self.store.read_records(dataset=PAPER_OPERATIONAL), [])
        self.assertFalse(os.path.exists(state_path))
        self.assertTrue(os.path.exists(state_path + ".txn"))
        restarted = ObservationStore(self.path)
        restarted.recover_paper_transition(state_path)
        self.assertEqual(len(restarted.read_records(dataset=PAPER_OPERATIONAL)), 1)
        with open(state_path, encoding="utf-8") as handle:
            self.assertEqual(json.load(handle), state)