"""Explicit adapter for recording genuine paper-operation observations.

This module deliberately has no market-data loop and is not imported by the
strategy, dashboard, or backtest paths. A future operational runner must call
these methods with events produced by the real paper process.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any

from observation_store import PAPER_OPERATIONAL, SIGNAL, TRADE, ObservationStore


class PaperObservationValidationError(ValueError):
    """Raised when an operational observation is incomplete or unsafe."""


def _require_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PaperObservationValidationError(f"{field} must be non-empty text")
    return value


def _require_timestamp(value: Any, field: str) -> str:
    text = _require_text(value, field)
    try:
        datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as error:
        raise PaperObservationValidationError(
            f"{field} must be an ISO-8601 timestamp"
        ) from error
    return text


def _require_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PaperObservationValidationError(f"{field} must be numeric")
    if not math.isfinite(float(value)):
        raise PaperObservationValidationError(f"{field} must be finite")
    return float(value)


def _require_positive_number(value: Any, field: str) -> float:
    number = _require_number(value, field)
    if number <= 0:
        raise PaperObservationValidationError(f"{field} must be positive")
    return number


def _require_non_negative_number(value: Any, field: str) -> float:
    number = _require_number(value, field)
    if number < 0:
        raise PaperObservationValidationError(f"{field} must be non-negative")
    return number


class PaperObservationAdapter:
    """Validated, append-only bridge for genuine paper operational events."""

    def __init__(self, store: ObservationStore | None = None):
        self.store = store or ObservationStore()

    def record_signal(
        self,
        *,
        signal_id: str,
        observed_at: str,
        symbol: str,
        strategy_score: float,
        entry_eligible: bool,
        market_data_timestamp: str,
        data_health: str,
    ) -> dict[str, Any]:
        return self._signal_record(
            signal_id=signal_id,
            observed_at=observed_at,
            symbol=symbol,
            strategy_score=strategy_score,
            entry_eligible=entry_eligible,
            market_data_timestamp=market_data_timestamp,
            data_health=data_health,
            persist=True,
        )

    def prepare_signal(self, **kwargs: Any) -> dict[str, Any]:
        return self._signal_record(**kwargs, persist=False)

    def _signal_record(self, *, persist: bool, **kwargs: Any) -> dict[str, Any]:
        signal_id = kwargs["signal_id"]
        observed_at = kwargs["observed_at"]
        symbol = kwargs["symbol"]
        strategy_score = kwargs["strategy_score"]
        entry_eligible = kwargs["entry_eligible"]
        market_data_timestamp = kwargs["market_data_timestamp"]
        data_health = kwargs["data_health"]
        if not isinstance(entry_eligible, bool):
            raise PaperObservationValidationError("entry_eligible must be boolean")
        payload = {
            "signal_id": _require_text(signal_id, "signal_id"),
            "symbol": _require_text(symbol, "symbol"),
            "strategy_score": _require_number(strategy_score, "strategy_score"),
            "entry_eligible": entry_eligible,
            "market_data_timestamp": _require_timestamp(market_data_timestamp, "market_data_timestamp"),
            "data_health": _require_text(data_health, "data_health"),
        }
        timestamp = _require_timestamp(observed_at, "observed_at")
        if not persist:
            return self.store.build_record(
                dataset=PAPER_OPERATIONAL, record_type=SIGNAL, payload=payload,
                occurred_at=timestamp, idempotency_key=f"signal:{payload['signal_id']}",
            )
        return self.store.append(
            dataset=PAPER_OPERATIONAL, record_type=SIGNAL, payload=payload,
            occurred_at=timestamp, idempotency_key=f"signal:{payload['signal_id']}",
        )

    def record_trade(
        self,
        *,
        trade_id: str,
        signal_id: str,
        entry_at: str,
        exit_at: str,
        entry_price: float,
        exit_price: float,
        profit_loss: float,
        fees: float,
        slippage: float,
        exit_reason: str,
    ) -> dict[str, Any]:
        return self._trade_record(
            trade_id=trade_id, signal_id=signal_id, entry_at=entry_at,
            exit_at=exit_at, entry_price=entry_price, exit_price=exit_price,
            profit_loss=profit_loss, fees=fees, slippage=slippage,
            exit_reason=exit_reason, persist=True,
        )

    def prepare_trade(self, **kwargs: Any) -> dict[str, Any]:
        return self._trade_record(**kwargs, persist=False)

    def _trade_record(self, *, persist: bool, **kwargs: Any) -> dict[str, Any]:
        trade_id, signal_id = kwargs["trade_id"], kwargs["signal_id"]
        entry_at, exit_at = kwargs["entry_at"], kwargs["exit_at"]
        entry_price, exit_price = kwargs["entry_price"], kwargs["exit_price"]
        profit_loss, fees = kwargs["profit_loss"], kwargs["fees"]
        slippage, exit_reason = kwargs["slippage"], kwargs["exit_reason"]
        entry_timestamp = _require_timestamp(entry_at, "entry_at")
        exit_timestamp = _require_timestamp(exit_at, "exit_at")
        payload = {
            "trade_id": _require_text(trade_id, "trade_id"),
            "signal_id": _require_text(signal_id, "signal_id"),
            "entry_at": entry_timestamp, "exit_at": exit_timestamp,
            "entry_price": _require_positive_number(entry_price, "entry_price"),
            "exit_price": _require_positive_number(exit_price, "exit_price"),
            "profit_loss": _require_number(profit_loss, "profit_loss"),
            "fees": _require_non_negative_number(fees, "fees"),
            "slippage": _require_non_negative_number(slippage, "slippage"),
            "exit_reason": _require_text(exit_reason, "exit_reason"),
        }
        if datetime.fromisoformat(exit_timestamp.replace("Z", "+00:00")) < datetime.fromisoformat(entry_timestamp.replace("Z", "+00:00")):
            raise PaperObservationValidationError("exit_at must not precede entry_at")
        if not persist:
            return self.store.build_record(
                dataset=PAPER_OPERATIONAL, record_type=TRADE, payload=payload,
                occurred_at=exit_timestamp, idempotency_key=f"trade:{payload['trade_id']}",
            )
        return self.store.append(
            dataset=PAPER_OPERATIONAL, record_type=TRADE, payload=payload,
            occurred_at=exit_timestamp, idempotency_key=f"trade:{payload['trade_id']}",
        )