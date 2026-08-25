"""Persistent, source-separated observation storage for the V2 foundation.

This module is intentionally not imported by the strategy or paper execution
path.  A future observation adapter can write records explicitly, while the
current frozen paper system remains untouched.
"""

from __future__ import annotations

import json
import os
import secrets
import fcntl
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HISTORICAL = "HISTORICAL"
PAPER_OPERATIONAL = "PAPER_OPERATIONAL"
SIGNAL = "SIGNAL"
TRADE = "TRADE"
SCHEMA_VERSION = 1
_FORBIDDEN_PAYLOAD_KEYS = {
    "api_key",
    "authorization",
    "conversation",
    "credential",
    "password",
    "prompt",
    "secret",
    "token",
}


class ObservationStoreError(ValueError):
    """Raised when an observation record cannot be safely persisted/read."""


class ObservationPersistenceError(OSError, ObservationStoreError):
    """Raised when a durable observation commit cannot be completed."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _record_id() -> str:
    return secrets.token_hex(16)


class ObservationStore:
    """Append-only JSONL store with strict dataset separation and idempotency."""

    def __init__(self, path: str | os.PathLike[str] | None = None):
        configured = path or os.getenv(
            "OBSERVATION_STORE_PATH",
            ".data/observations.jsonl",
        )
        self.path = Path(configured)
        self._persistence_health = {
            "status": "HEALTHY",
            "error_code": None,
            "last_error": None,
            "operation": None,
        }
        self.lock_path = self.path.with_suffix(f"{self.path.suffix}.lock")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            self.path.chmod(0o600)
        else:
            self.path.touch(mode=0o600)

    def append(
        self,
        *,
        dataset: str,
        record_type: str,
        payload: dict[str, Any],
        occurred_at: str | None = None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        self._validate_dataset(dataset)
        self._validate_record_type(record_type)
        if not isinstance(payload, dict):
            raise ObservationStoreError("payload must be an object")
        self._validate_payload_safety(payload)

        record = {
            "schema_version": SCHEMA_VERSION,
            "record_id": _record_id(),
            "dataset": dataset,
            "record_type": record_type,
            "occurred_at": occurred_at or _utc_now(),
            "idempotency_key": idempotency_key,
            "payload": payload,
        }
        self._validate_record(record)
        # The idempotency check and append must occur under one process-shared
        # lock. JSONL remains a deliberately small, append-only store; this
        # prevents duplicate genuine evidence when runners overlap during a
        # restart without introducing an unreviewed database migration.
        with self.lock_path.open("a+", encoding="utf-8") as lock_handle:
            os.fchmod(lock_handle.fileno(), 0o600)
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            try:
                records = self.read_records()
                if idempotency_key:
                    for existing in records:
                        if existing.get("idempotency_key") == idempotency_key:
                            return existing
                try:
                    with self.path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(record, separators=(",", ":")) + "\n")
                        handle.flush()
                        os.fsync(handle.fileno())
                except OSError as error:
                    self._mark_persistence_failure(error, "observation_write")
                    raise ObservationPersistenceError(
                        "observation storage is unavailable during durable write"
                    ) from error
                self._clear_persistence_health()
            finally:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        self.path.chmod(0o600)
        return record

    def build_record(
        self,
        *,
        dataset: str,
        record_type: str,
        payload: dict[str, Any],
        occurred_at: str | None = None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        """Build and validate a record without making it durable."""
        self._validate_dataset(dataset)
        self._validate_record_type(record_type)
        if not isinstance(payload, dict):
            raise ObservationStoreError("payload must be an object")
        self._validate_payload_safety(payload)
        record = {
            "schema_version": SCHEMA_VERSION,
            "record_id": _record_id(),
            "dataset": dataset,
            "record_type": record_type,
            "occurred_at": occurred_at or _utc_now(),
            "idempotency_key": idempotency_key,
            "payload": payload,
        }
        self._validate_record(record)
        return record

    def commit_paper_transition(
        self,
        *,
        state_path: str | os.PathLike[str],
        state: dict[str, Any],
        records: list[dict[str, Any]],
    ) -> None:
        """Durably commit paper evidence and its corresponding engine state.

        A fsynced intent is the recovery boundary. Replaying it is safe because
        evidence is keyed idempotently and the state replacement is deterministic.
        """
        state_path = Path(state_path)
        journal_path = state_path.with_suffix(f"{state_path.suffix}.txn")
        for record in records:
            self._validate_record(record)
            if record["dataset"] != PAPER_OPERATIONAL:
                raise ObservationStoreError("paper transition contains non-paper evidence")
        journal = {"state_path": str(state_path), "state": state, "records": records}
        try:
            with self.lock_path.open("a+", encoding="utf-8") as lock_handle:
                os.fchmod(lock_handle.fileno(), 0o600)
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
                try:
                    self._write_json_fsync(journal_path, journal)
                    self._transaction_failpoint("after_journal")
                    self._apply_transition(state_path, state, records)
                    journal_path.unlink(missing_ok=True)
                    self._fsync_directory(journal_path.parent)
                finally:
                    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        except ObservationPersistenceError:
            raise
        except OSError as error:
            self._mark_persistence_failure(error, "paper_transition_commit")
            raise ObservationPersistenceError(
                "observation storage is unavailable during paper commit"
            ) from error
        self._clear_persistence_health()

    def recover_paper_transition(self, state_path: str | os.PathLike[str]) -> None:
        """Replay an interrupted transition before loading engine state."""
        state_path = Path(state_path)
        journal_path = state_path.with_suffix(f"{state_path.suffix}.txn")
        if not journal_path.exists():
            return
        try:
            with self.lock_path.open("a+", encoding="utf-8") as lock_handle:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
                try:
                    try:
                        with journal_path.open("r", encoding="utf-8") as handle:
                            journal = json.load(handle)
                    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
                        raise ObservationStoreError("paper transition cannot be recovered safely") from error
                    if journal.get("state_path") != str(state_path) or not isinstance(journal.get("state"), dict):
                        raise ObservationStoreError("paper transition cannot be recovered safely")
                    records = journal.get("records")
                    if not isinstance(records, list):
                        raise ObservationStoreError("paper transition cannot be recovered safely")
                    self._apply_transition(state_path, journal["state"], records)
                    journal_path.unlink()
                    self._fsync_directory(journal_path.parent)
                finally:
                    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        except OSError as error:
            self._mark_persistence_failure(error, "paper_transition_recovery")
            raise ObservationPersistenceError(
                "observation storage is unavailable during paper recovery"
            ) from error
        self._clear_persistence_health()

    def persistence_health(self) -> dict[str, Any]:
        """Return the operator-facing state of durable observation storage."""
        return dict(self._persistence_health)

    def _mark_persistence_failure(self, error: OSError, operation: str) -> None:
        self._persistence_health = {
            "status": "UNAVAILABLE",
            "error_code": "PAPER_EVIDENCE_STORAGE_UNAVAILABLE",
            "last_error": str(error) or error.__class__.__name__,
            "operation": operation,
        }

    def mark_persistence_failure(self, error: OSError, operation: str) -> None:
        """Record an outage from a related durable paper-state operation."""
        self._mark_persistence_failure(error, operation)

    def _clear_persistence_health(self) -> None:
        self._persistence_health = {
            "status": "HEALTHY",
            "error_code": None,
            "last_error": None,
            "operation": None,
        }

    def _apply_transition(
        self, state_path: Path, state: dict[str, Any], records: list[dict[str, Any]]
    ) -> None:
        for record in records:
            self._validate_record(record)
            if record["dataset"] != PAPER_OPERATIONAL:
                raise ObservationStoreError("paper transition contains non-paper evidence")
        existing = self.read_records()
        existing_keys = {
            record.get("idempotency_key")
            for record in existing
            if record.get("idempotency_key")
        }
        additions = [
            record for record in records
            if not record.get("idempotency_key")
            or record.get("idempotency_key") not in existing_keys
        ]
        if additions:
            self._write_records_fsync(existing + additions)
        self._transaction_failpoint("after_evidence")
        self._write_json_fsync(state_path, state)
        state_path.chmod(0o600)

    def _write_records_fsync(self, records: list[dict[str, Any]]) -> None:
        temporary = self.path.with_suffix(f"{self.path.suffix}.txn-tmp")
        try:
            with temporary.open("w", encoding="utf-8") as handle:
                for record in records:
                    handle.write(json.dumps(record, separators=(",", ":")) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            temporary.chmod(0o600)
            os.replace(temporary, self.path)
        finally:
            # A failed write/fsync/replace must not leave a second, plausible
            # evidence file behind. The journal remains the only recovery
            # record, and a later restart can retry the whole transition.
            temporary.unlink(missing_ok=True)
        self.path.chmod(0o600)

    @staticmethod
    def _write_json_fsync(path: Path, value: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(f"{path.suffix}.tmp")
        try:
            with temporary.open("w", encoding="utf-8") as handle:
                json.dump(value, handle, separators=(",", ":"))
                handle.flush()
                os.fsync(handle.fileno())
            temporary.chmod(0o600)
            os.replace(temporary, path)
        finally:
            # If replace did not happen, discard the uncommitted candidate.
            # If it did happen, unlink is a no-op.
            temporary.unlink(missing_ok=True)
        path.chmod(0o600)

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _transaction_failpoint(self, point: str) -> None:
        hook = getattr(self, "_test_transaction_failpoint", None)
        if hook:
            hook(point)

    def read_records(
        self,
        *,
        dataset: str | None = None,
        record_type: str | None = None,
    ) -> list[dict[str, Any]]:
        if dataset is not None:
            self._validate_dataset(dataset)
        if record_type is not None:
            self._validate_record_type(record_type)

        records = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ObservationStoreError(
                        f"invalid observation record at line {line_number}"
                    ) from error
                self._validate_record(record)
                if dataset and record["dataset"] != dataset:
                    continue
                if record_type and record["record_type"] != record_type:
                    continue
                records.append(record)
        return records

    def summary(self) -> dict[str, Any]:
        records = self.read_records()
        counts = {
            HISTORICAL: {SIGNAL: 0, TRADE: 0},
            PAPER_OPERATIONAL: {SIGNAL: 0, TRADE: 0},
        }
        for record in records:
            counts[record["dataset"]][record["record_type"]] += 1
        paper_records = [
            record for record in records if record["dataset"] == PAPER_OPERATIONAL
        ]
        return {
            "schema_version": SCHEMA_VERSION,
            "historical": counts[HISTORICAL],
            "paper_operational": counts[PAPER_OPERATIONAL],
            "paper_observation_count": len(paper_records),
            "first_paper_observation": (
                paper_records[0]["occurred_at"] if paper_records else None
            ),
            "last_paper_observation": (
                paper_records[-1]["occurred_at"] if paper_records else None
            ),
        }

    def paper_counts(self) -> dict[str, int]:
        """Return counts derived strictly from validated paper JSONL records."""
        records = self.read_records(dataset=PAPER_OPERATIONAL)
        return {
            SIGNAL: sum(record["record_type"] == SIGNAL for record in records),
            TRADE: sum(record["record_type"] == TRADE for record in records),
        }

    @staticmethod
    def _validate_dataset(dataset: str) -> None:
        if dataset not in {HISTORICAL, PAPER_OPERATIONAL}:
            raise ObservationStoreError("invalid observation dataset")

    @staticmethod
    def _validate_record_type(record_type: str) -> None:
        if record_type not in {SIGNAL, TRADE}:
            raise ObservationStoreError("invalid observation record type")

    @classmethod
    def _validate_record(cls, record: Any) -> None:
        if not isinstance(record, dict):
            raise ObservationStoreError("observation record must be an object")
        required = {
            "schema_version",
            "record_id",
            "dataset",
            "record_type",
            "occurred_at",
            "payload",
        }
        if not required.issubset(record):
            raise ObservationStoreError("observation record is missing fields")
        if record["schema_version"] != SCHEMA_VERSION:
            raise ObservationStoreError("unsupported observation schema")
        if not isinstance(record["record_id"], str) or not record["record_id"]:
            raise ObservationStoreError("observation record id is invalid")
        cls._validate_dataset(record["dataset"])
        cls._validate_record_type(record["record_type"])
        if not isinstance(record["occurred_at"], str) or not record["occurred_at"]:
            raise ObservationStoreError("observation timestamp is invalid")
        if not isinstance(record["payload"], dict):
            raise ObservationStoreError("observation payload is invalid")
        cls._validate_payload_safety(record["payload"])

    @classmethod
    def _validate_payload_safety(cls, payload: Any) -> None:
        if isinstance(payload, dict):
            for key, value in payload.items():
                normalized = str(key).lower().replace("-", "_")
                if normalized in _FORBIDDEN_PAYLOAD_KEYS:
                    raise ObservationStoreError(
                        "observation payload contains a forbidden field"
                    )
                cls._validate_payload_safety(value)
        elif isinstance(payload, (list, tuple)):
            for value in payload:
                cls._validate_payload_safety(value)