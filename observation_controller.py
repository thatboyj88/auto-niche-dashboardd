"""Evidence-based observation-period controller.

The controller does not decide trades. It only tracks genuine engine output,
data-health cycles, and explicitly supplied completion criteria.
"""

from __future__ import annotations

import json
import os
import fcntl
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from observation_store import (
    SIGNAL,
    TRADE,
    ObservationStore,
    ObservationStoreError,
)

RECONCILIATION_FAILURE_CODE = "EVIDENCE_RECONCILIATION_FAILURE"
MANUAL_STOP_CODE = "MANUAL_OPERATOR_STOP"
CONTROL_ACTIONS = frozenset({"START", "PAUSE", "STOP"})


class ObservationControllerError(RuntimeError):
    """Raised when observation state cannot be safely continued."""


class ObservationControlError(ObservationControllerError):
    """Raised when an operator control cannot be authorized safely."""


@dataclass(frozen=True)
class ObservationCriteria:
    min_completed_trades: int
    min_observation_days: int
    max_observation_days: int
    min_healthy_ratio: float

    def __post_init__(self):
        if self.min_completed_trades < 1:
            raise ValueError("min_completed_trades must be positive")
        if self.min_observation_days < 1:
            raise ValueError("min_observation_days must be positive")
        if self.max_observation_days < self.min_observation_days:
            raise ValueError("max_observation_days must cover minimum days")
        if not 0 < self.min_healthy_ratio <= 1:
            raise ValueError("min_healthy_ratio must be in (0, 1]")


class ObservationController:
    """Persisted controller that completes only on genuine evidence."""

    def __init__(
        self,
        criteria: ObservationCriteria,
        *,
        state_path: str | os.PathLike[str] | None = None,
        observation_store: ObservationStore | None = None,
    ):
        self.criteria = criteria
        self.state_path = Path(
            state_path
            or os.getenv(
                "OBSERVATION_CONTROLLER_STATE_PATH",
                ".data/observation_controller.json",
            )
        )
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.observation_store = observation_store
        self.state = self._load_state()

    def start(self, *, started_at: str | None = None) -> dict[str, Any]:
        if self.state["status"] in {
            "COMPLETED",
            "STOPPED_INSUFFICIENT_EVIDENCE",
            "STOPPED_SAFETY_FAILURE",
            "STOPPED_MANUAL",
        }:
            raise ObservationControllerError(
                f"observation is terminal: {self.state['status']}"
            )
        if self.state["started_at"] is None:
            self.state["started_at"] = started_at or self._now_iso()
        self.state["status"] = "RUNNING"
        self._save()
        return self.status()

    def pause(self) -> dict[str, Any]:
        if self.state["status"] != "RUNNING":
            raise ObservationControllerError(
                f"cannot pause while {self.state['status']}"
            )
        self.state["status"] = "PAUSED"
        self._save()
        return self.status()

    def stop(self) -> dict[str, Any]:
        if self.state["status"] in {
            "COMPLETED",
            "STOPPED_INSUFFICIENT_EVIDENCE",
            "STOPPED_SAFETY_FAILURE",
            "STOPPED_MANUAL",
        }:
            raise ObservationControllerError(
                f"observation is terminal: {self.state['status']}"
            )
        self.state["status"] = "STOPPED_MANUAL"
        self.state["last_error"] = "stopped by authenticated operator"
        self.state["safety_failure_code"] = MANUAL_STOP_CODE
        self._save()
        return self.status()

    def record_cycle(
        self,
        *,
        data_health: str,
        engine_status: dict[str, Any],
        observed_at: str | None = None,
    ) -> dict[str, Any]:
        if self.state["status"] != "RUNNING":
            raise ObservationControllerError(
                f"cannot record cycle while {self.state['status']}"
            )
        timestamp = observed_at or self._now_iso()
        self.state["cycles"] += 1
        self.state["last_cycle_at"] = timestamp
        self.state["last_data_health"] = data_health
        self.state["signal_count"] = int(engine_status.get("genuine_signals", 0))
        engine_trade_count = int(
            engine_status.get("genuine_completed_trades", 0)
        )
        if self.observation_store is not None:
            try:
                evidence_counts = self.observation_store.paper_counts()
            except ObservationStoreError as error:
                return self.stop_for_safety(
                    "persisted observation evidence cannot be validated",
                    failure_code=RECONCILIATION_FAILURE_CODE,
                )
            if (
                engine_trade_count != evidence_counts[TRADE]
                or self.state["signal_count"] != evidence_counts[SIGNAL]
            ):
                return self.stop_for_safety(
                    "paper engine totals do not reconcile with persisted observation evidence",
                    failure_code=RECONCILIATION_FAILURE_CODE,
                )
            self.state["signal_count"] = evidence_counts[SIGNAL]
            self.state["trade_count"] = evidence_counts[TRADE]
        else:
            self.state["trade_count"] = engine_trade_count
        if data_health == "HEALTHY":
            self.state["healthy_cycles"] += 1
        elif data_health in {"UNAVAILABLE", "DEGRADED"}:
            self.state["unhealthy_cycles"] += 1
        else:
            self.state["unhealthy_cycles"] += 1

        if self._meets_completion():
            self.state["status"] = "COMPLETED"
        elif self._past_maximum_duration(timestamp):
            self.state["status"] = "STOPPED_INSUFFICIENT_EVIDENCE"
        self._save()
        return self.status()

    def stop_for_safety(
        self, reason: str, *, failure_code: str | None = None
    ) -> dict[str, Any]:
        if not reason.strip():
            raise ValueError("safety stop reason is required")
        self.state["status"] = "STOPPED_SAFETY_FAILURE"
        self.state["last_error"] = reason
        self.state["safety_failure_code"] = failure_code
        self._save()
        return self.status()

    def status(self) -> dict[str, Any]:
        started = self._parse_timestamp(self.state["started_at"])
        last_cycle = self._parse_timestamp(self.state["last_cycle_at"])
        observation_days = 0.0
        if started and last_cycle:
            observation_days = max(
                0.0, (last_cycle - started).total_seconds() / 86400
            )
        ratio = (
            self.state["healthy_cycles"] / self.state["cycles"]
            if self.state["cycles"]
            else 0.0
        )
        return {
            **self.state,
            "state_path": str(self.state_path),
            "criteria": asdict(self.criteria),
            "observation_days": round(observation_days, 4),
            "healthy_ratio": round(ratio, 4),
            "evidence_reconciled": self._evidence_reconciled(),
        }

    def _evidence_reconciled(self) -> bool:
        """Indicate whether persisted paper evidence agrees with controller totals."""
        if self.observation_store is None:
            return True
        try:
            counts = self.observation_store.paper_counts()
        except ObservationStoreError:
            return False
        return (
            counts[SIGNAL] == self.state["signal_count"]
            and counts[TRADE] == self.state["trade_count"]
        )

    def _meets_completion(self) -> bool:
        status = self.status()
        return (
            status["trade_count"] >= self.criteria.min_completed_trades
            and status["observation_days"] >= self.criteria.min_observation_days
            and status["healthy_ratio"] >= self.criteria.min_healthy_ratio
        )

    def _past_maximum_duration(self, timestamp: str) -> bool:
        started = self._parse_timestamp(self.state["started_at"])
        current = self._parse_timestamp(timestamp)
        if not started or not current:
            return False
        return (
            current - started
        ).total_seconds() >= self.criteria.max_observation_days * 86400

    def _load_state(self) -> dict[str, Any]:
        initial = {
            "status": "NOT_STARTED",
            "started_at": None,
            "last_cycle_at": None,
            "last_data_health": None,
            "cycles": 0,
            "healthy_cycles": 0,
            "unhealthy_cycles": 0,
            "signal_count": 0,
            "trade_count": 0,
            "last_error": None,
            "safety_failure_code": None,
        }
        if not self.state_path.exists():
            return initial
        try:
            with self.state_path.open("r", encoding="utf-8") as handle:
                loaded = json.load(handle)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ObservationControllerError(
                "observation controller state cannot be restored"
            ) from error
        if not isinstance(loaded, dict):
            raise ObservationControllerError(
                "observation controller state cannot be restored"
            )
        initial.update(loaded)
        self._validate_state(initial)
        return initial

    @staticmethod
    def _validate_state(state: dict[str, Any]) -> None:
        """Reject persisted state that could not be safely resumed."""
        valid_statuses = {
            "NOT_STARTED",
            "RUNNING",
            "PAUSED",
            "COMPLETED",
            "STOPPED_INSUFFICIENT_EVIDENCE",
            "STOPPED_SAFETY_FAILURE",
            "STOPPED_MANUAL",
        }
        if not isinstance(state["status"], str) or state["status"] not in valid_statuses:
            raise ObservationControllerError(
                "observation controller state cannot be restored"
            )

        nullable_strings = (
            "started_at",
            "last_cycle_at",
            "last_data_health",
            "last_error",
            "safety_failure_code",
        )
        if any(
            state[field] is not None and not isinstance(state[field], str)
            for field in nullable_strings
        ):
            raise ObservationControllerError(
                "observation controller state cannot be restored"
            )

        counters = (
            "cycles",
            "healthy_cycles",
            "unhealthy_cycles",
            "signal_count",
            "trade_count",
        )
        if any(
            not isinstance(state[field], int)
            or isinstance(state[field], bool)
            or state[field] < 0
            for field in counters
        ):
            raise ObservationControllerError(
                "observation controller state cannot be restored"
            )

    def _save(self) -> None:
        self.state_path.touch(mode=0o600, exist_ok=True)
        temporary = self.state_path.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(self.state, handle, separators=(",", ":"))
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(0o600)
        os.replace(temporary, self.state_path)
        self.state_path.chmod(0o600)

    @staticmethod
    def _parse_timestamp(value: str | None) -> datetime | None:
        if not value:
            return None
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as error:
            raise ObservationControllerError(
                "observation controller timestamp is invalid"
            ) from error

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()


@contextmanager
def observation_control_lock(path: str | os.PathLike[str]):
    """Use the same advisory lock as the long-lived runner."""
    lock_path = Path(path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ObservationControlError(
                "paper observation is busy; try again after the current cycle"
            ) from error
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def apply_paper_control(
    action: str,
    *,
    authenticated: bool,
    confirmed: bool,
    risk_governor: callable,
    criteria: ObservationCriteria,
    state_path: str | os.PathLike[str],
    lock_path: str | os.PathLike[str],
    observation_store: ObservationStore | None = None,
    stale_after_seconds: float | None = None,
) -> dict[str, Any]:
    """Authorize and apply one paper-only operator control atomically."""
    action = str(action).upper()
    if action not in CONTROL_ACTIONS:
        raise ObservationControlError("unsupported paper control")
    if not authenticated:
        raise ObservationControlError("authenticated operator access is required")
    if not confirmed:
        raise ObservationControlError("explicit confirmation is required")
    try:
        allowed = bool(risk_governor(action))
    except Exception as error:
        raise ObservationControlError("Risk Governor could not validate control") from error
    if not allowed:
        raise ObservationControlError("Risk Governor rejected paper control")

    with observation_control_lock(lock_path):
        controller = ObservationController(
            criteria, state_path=state_path, observation_store=observation_store
        )
        status = controller.status()
        if not status.get("evidence_reconciled", False):
            raise ObservationControlError("paper evidence is not reconciled")
        if (
            stale_after_seconds is not None
            and status["status"] == "RUNNING"
            and status.get("last_cycle_at")
        ):
            last_cycle = controller._parse_timestamp(status["last_cycle_at"])
            if last_cycle is None or (
                datetime.now(timezone.utc) - last_cycle
            ).total_seconds() > stale_after_seconds:
                raise ObservationControlError("paper observation heartbeat is stale")
        if action == "START":
            return controller.start()
        if action == "PAUSE":
            return controller.pause()
        return controller.stop()