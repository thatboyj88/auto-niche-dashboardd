"""Long-lived, single-instance V2 genuine paper-observation runner."""

from __future__ import annotations

import argparse
import fcntl
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from incremental_paper_engine import IncrementalPaperEngine
from incremental_paper_engine import IncrementalPaperEngineError
from kraken_live_data import KrakenMarketData
from observation_controller import (
    RECONCILIATION_FAILURE_CODE,
    ObservationController,
    ObservationControllerError,
    ObservationCriteria,
)
from observation_notifications import (
    PERSISTENCE_FAILURE_EVENT,
    PERSISTENCE_RECOVERY_EVENT,
    RECONCILIATION_FAILURE_EVENT,
    ObservationNotifier,
)


class ObservationRunnerError(RuntimeError):
    """Raised when the runner cannot safely start or continue."""


CONTROLLER_RESTORE_BLOCKED_STATUS = "BLOCKED_RESTORE"


class SingleRunnerLock:
    """Process-held advisory lock preventing duplicate paper runners."""

    def __init__(self, path: str | os.PathLike[str]):
        self.path = Path(path)
        self.handle = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a+", encoding="utf-8")
        self.handle.seek(0)
        try:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            self.handle.close()
            self.handle = None
            raise ObservationRunnerError(
                "another observation runner is already active"
            ) from error
        os.fchmod(self.handle.fileno(), 0o600)
        return self

    def __exit__(self, _type, _value, _traceback):
        if self.handle is not None:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
            self.handle.close()
            self.handle = None


class PaperObservationRunner:
    """Compose public market data, the incremental engine, and controller."""

    def __init__(
        self,
        *,
        market_data: Any | None = None,
        engine: IncrementalPaperEngine | None = None,
        controller: ObservationController | None = None,
        notifier: ObservationNotifier | None = None,
        lock_path: str | os.PathLike[str] | None = None,
        poll_seconds: float = 300,
    ):
        if poll_seconds <= 0:
            raise ValueError("poll_seconds must be positive")
        self.market_data = market_data or KrakenMarketData(interval=60)
        self.engine = engine or IncrementalPaperEngine()
        self._controller_restore_error: str | None = None
        self._controller_state_path: Path | None = None
        if controller is not None:
            self.controller = controller
        else:
            self._controller_state_path = self._controller_state_path_from_environment()
            try:
                self.controller = self._controller_from_environment(
                    self.engine.adapter.store
                )
            except ObservationControllerError as error:
                # A corrupt state file must continue to block execution, but the
                # runner must remain inspectable by health checks.
                self.controller = None
                self._controller_restore_error = str(error)
        if controller is not None and self.controller.observation_store is None:
            self.controller.observation_store = self.engine.adapter.store
        self.notifier = notifier or ObservationNotifier()
        persistence_health = self.engine.status().get("persistence_health", {})
        self._persistence_alerted = persistence_health.get("status") == "UNAVAILABLE"
        # A restored outage has not been announced by this process yet. Keep
        # this separate from _persistence_alerted so recovery remains tied to
        # the later durable transition, not to process startup.
        self._persistence_notification_pending = self._persistence_alerted
        self.lock_path = Path(
            lock_path
            or os.getenv(
                "OBSERVATION_RUNNER_LOCK_PATH",
                ".data/paper_observation_runner.lock",
            )
        )
        self.poll_seconds = poll_seconds
        stale_after = os.getenv("OBSERVATION_STALE_AFTER_SECONDS")
        self.stale_after_seconds = (
            float(stale_after) if stale_after is not None else None
        )
        if self.stale_after_seconds is not None and self.stale_after_seconds <= 0:
            raise ValueError("OBSERVATION_STALE_AFTER_SECONDS must be positive")

    def run_cycle(self) -> dict[str, Any]:
        if self.controller is None:
            return self.status()

        self._announce_persistence_outage_if_pending()

        if self.controller.status()["status"] in {
            "COMPLETED",
            "STOPPED_INSUFFICIENT_EVIDENCE",
            "STOPPED_SAFETY_FAILURE",
            "STOPPED_MANUAL",
        }:
            return self.status()

        if self.controller.status()["status"] == "PAUSED":
            return self.status()

        if self.controller.status()["status"] == "NOT_STARTED":
            controller_status = self.controller.start()
            self.notifier.notify("OBSERVATION_STARTED", controller_status)

        candles = self.market_data.load()
        health = self.market_data.health
        data_health = health.get("status", "UNAVAILABLE")
        if data_health == "HEALTHY":
            if self.engine.status()["started_at"] is None:
                self.engine.initialize(candles)
            try:
                self.engine.process(candles, data_health=data_health)
            except IncrementalPaperEngineError as error:
                storage_health = self.engine.status().get("persistence_health", {})
                if storage_health.get("status") == "UNAVAILABLE":
                    if not self._persistence_alerted:
                        self._notify(
                            PERSISTENCE_FAILURE_EVENT,
                            {
                                "status": "WAITING_FOR_PERSISTENCE",
                                "error_code": storage_health.get("error_code"),
                                "last_error": storage_health.get("last_error") or str(error),
                                "operation": storage_health.get("operation"),
                            },
                        )
                        self._persistence_alerted = True
                    return self.status()
                raise
            storage_health = self.engine.status().get("persistence_health", {})
            if (
                storage_health.get("status") == "HEALTHY"
                and self._persistence_alerted
            ):
                self._notify(
                    PERSISTENCE_RECOVERY_EVENT,
                    {
                        "status": "HEALTHY",
                        "error_code": None,
                        "last_error": None,
                        "operation": None,
                    },
                )
                self._persistence_alerted = False

        controller_status = self.controller.record_cycle(
            data_health=data_health,
            engine_status=self.engine.status(),
        )
        if controller_status["status"] == "COMPLETED":
            self._notify("OBSERVATION_COMPLETED", controller_status)
        elif controller_status["status"] == "STOPPED_INSUFFICIENT_EVIDENCE":
            self._notify(
                "OBSERVATION_STOPPED_INSUFFICIENT_EVIDENCE",
                controller_status,
            )
        elif (
            controller_status["status"] == "STOPPED_SAFETY_FAILURE"
            and controller_status.get("safety_failure_code")
            == RECONCILIATION_FAILURE_CODE
        ):
            self._notify(
                RECONCILIATION_FAILURE_EVENT,
                controller_status,
            )
        return self.status()

    def _announce_persistence_outage_if_pending(self) -> None:
        if not self._persistence_notification_pending:
            return
        storage_health = self.engine.status().get("persistence_health", {})
        self._notify(
            PERSISTENCE_FAILURE_EVENT,
            {
                "status": "WAITING_FOR_PERSISTENCE",
                "error_code": storage_health.get(
                    "error_code", "PAPER_EVIDENCE_STORAGE_UNAVAILABLE"
                ),
                "last_error": storage_health.get("last_error"),
                "operation": storage_health.get("operation"),
            },
        )
        # Delivery is best-effort. Do not repeatedly page on every poll when
        # the webhook itself is unavailable.
        self._persistence_notification_pending = False

    def _notify(self, event: str, status: dict[str, Any]) -> None:
        try:
            self.notifier.notify(event, status)
        except Exception:
            # Notifications are operational guidance only and must never
            # change or stop paper execution.
            return

    def run_forever(self, *, max_cycles: int | None = None) -> dict[str, Any]:
        cycles = 0
        while True:
            # Release the process lock between polls so an authenticated
            # operator can safely pause or stop between complete cycles.
            with SingleRunnerLock(self.lock_path):
                status = self.run_cycle()
            cycles += 1
            if status["controller"]["status"] == "PAUSED":
                if max_cycles is not None and cycles >= max_cycles:
                    return status
                time.sleep(self.poll_seconds)
                continue
            if status["controller"]["status"] != "RUNNING":
                return status
            if max_cycles is not None and cycles >= max_cycles:
                return status
            time.sleep(self.poll_seconds)

    def status(self) -> dict[str, Any]:
        if self.controller is None:
            controller_status = {
                "status": CONTROLLER_RESTORE_BLOCKED_STATUS,
                "state_path": str(self._controller_state_path),
                "last_error": self._controller_restore_error,
                "error_code": "CONTROLLER_STATE_RESTORE_BLOCKED",
            }
            runner_status = CONTROLLER_RESTORE_BLOCKED_STATUS
        else:
            controller_status = self.controller.status()
            if (
                controller_status["status"] == "RUNNING"
                and self._cycle_is_stale(controller_status.get("last_cycle_at"))
            ):
                runner_status = "STALE"
            else:
                runner_status = (
                    "RUNNING"
                    if controller_status["status"] in {"NOT_STARTED", "RUNNING"}
                    else "STOPPED"
                )
        return {
            "runner": runner_status,
            "controller": controller_status,
            "engine": self.engine.status(),
            "market_data": getattr(self.market_data, "health", {}),
            "paper_storage": self.engine.status().get("persistence_health", {}),
        }

    def _cycle_is_stale(self, last_cycle_at: str | None) -> bool:
        """Report a stale heartbeat without mutating controller or engine state."""
        if self.stale_after_seconds is None or not last_cycle_at:
            return False
        try:
            cycle_time = datetime.fromisoformat(last_cycle_at.replace("Z", "+00:00"))
        except ValueError:
            return True
        return (
            datetime.now(timezone.utc) - cycle_time
        ).total_seconds() > self.stale_after_seconds

    @staticmethod
    def _controller_from_environment(
        observation_store: Any | None = None,
    ) -> ObservationController:
        names = (
            "OBSERVATION_MIN_COMPLETED_TRADES",
            "OBSERVATION_MIN_DAYS",
            "OBSERVATION_MAX_DAYS",
            "OBSERVATION_MIN_HEALTHY_RATIO",
        )
        values = {name: os.getenv(name) for name in names}
        if any(value is None for value in values.values()):
            missing = ", ".join(name for name, value in values.items() if value is None)
            raise ObservationRunnerError(
                f"observation criteria are not configured: {missing}"
            )
        criteria = ObservationCriteria(
            min_completed_trades=int(values[names[0]]),
            min_observation_days=int(values[names[1]]),
            max_observation_days=int(values[names[2]]),
            min_healthy_ratio=float(values[names[3]]),
        )
        return ObservationController(criteria, observation_store=observation_store)

    @staticmethod
    def _controller_state_path_from_environment() -> Path:
        return Path(
            os.getenv(
                "OBSERVATION_CONTROLLER_STATE_PATH",
                ".data/observation_controller.json",
            )
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    runner = PaperObservationRunner(
        poll_seconds=float(os.getenv("OBSERVATION_POLL_SECONDS", "300"))
    )
    try:
        if args.once:
            runner.run_forever(max_cycles=1)
        else:
            runner.run_forever()
    except ObservationRunnerError as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())