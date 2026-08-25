"""Incremental paper-only execution engine for genuine observation.

This engine starts from the latest committed public candle and processes only
new candles supplied by an operational runner. It never backfills historical
candles into PAPER_OPERATIONAL and never exposes an order/exchange interface.
"""

from __future__ import annotations

import json
import math
import os
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

from config import (
    MAX_DAILY_LOSS_PERCENT,
    MAX_POSITION_PERCENT,
    MAX_TRADES_PER_DAY,
    STARTING_CAPITAL,
    STOP_LOSS_PERCENT,
    TAKE_PROFIT_PERCENT,
    PAPER_TRADING,
    LIVE_TRADING,
    FEE_PERCENT,
    SLIPPAGE_PERCENT,
)
from indicators import (
    calculate_average_volume,
    calculate_ema,
    calculate_rsi,
)
from market_data_health import HEALTHY
from observation_store import ObservationPersistenceError, ObservationStore
from paper_observation_adapter import PaperObservationAdapter
from risk_manager import risk_check
from strategy import calculate_strategy_score

class IncrementalPaperEngineError(RuntimeError):
    """Raised when the paper engine cannot safely continue."""


def classify_market_condition(
    *,
    price: float,
    ema21: float,
    ema50: float,
    ema200: float,
    long_term_trend: bool,
    short_term_momentum: bool,
) -> tuple[str, str]:
    """Classify only from indicators available at the observed candle."""
    bullish_alignment = price > ema21 > ema50 > ema200
    bearish_alignment = price < ema21 < ema50 < ema200
    if bullish_alignment and long_term_trend and short_term_momentum:
        return "Bull", "Strong Bull"
    if bullish_alignment or (price > ema21 and ema21 > ema50):
        return "Bull", "Weak Bull"
    if bearish_alignment and not short_term_momentum:
        return "Bear", "Strong Bear"
    if bearish_alignment or (price < ema21 and ema21 < ema50):
        return "Bear", "Weak Bear"
    return "Sideways", "Neutral/Sideways"


class IncrementalPaperEngine:
    """Stateful, one-candle-at-a-time paper execution engine."""

    def __init__(
        self,
        *,
        adapter: PaperObservationAdapter | None = None,
        state_path: str | os.PathLike[str] | None = None,
        starting_capital: float = STARTING_CAPITAL,
    ):
        if not PAPER_TRADING or LIVE_TRADING:
            raise IncrementalPaperEngineError(
                "incremental engine requires PAPER_TRADING=True and LIVE_TRADING=False"
            )
        if starting_capital != STARTING_CAPITAL:
            raise IncrementalPaperEngineError(
                "incremental engine requires the canonical starting capital"
            )
        self.adapter = adapter or PaperObservationAdapter()
        self.state_path = Path(
            state_path
            or os.getenv("PAPER_ENGINE_STATE_PATH", ".data/paper_engine_state.json")
        )
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.adapter.store.recover_paper_transition(self.state_path)
        self.state = self._load_state()

    def initialize(self, candles: list[dict[str, Any]]) -> dict[str, Any]:
        """Arm observation at the newest committed candle without backfill."""
        self._validate_candles(candles)
        if not candles:
            raise IncrementalPaperEngineError("cannot initialize without candles")
        if self.state["started_at"] is None:
            self.state["started_at"] = self._now_iso()
            self.state["last_processed_timestamp"] = candles[-1]["timestamp"]
            self.state["status"] = "ARMED"
            self._save_state()
        return self.status()

    def process(
        self,
        candles: list[dict[str, Any]],
        *,
        data_health: str,
        symbol: str = "BTC/CAD",
    ) -> list[dict[str, Any]]:
        """Process only new candles; unhealthy data is skipped fail-closed."""
        self._validate_candles(candles)
        if not candles:
            return []
        if self.state["started_at"] is None:
            self.initialize(candles)
            return []
        if data_health != HEALTHY:
            self.state["status"] = "WAITING_FOR_HEALTHY_DATA"
            self.state["last_error"] = f"market data health: {data_health}"
            self._save_state()
            return []

        events = []
        for index, candle in enumerate(candles):
            if candle["timestamp"] <= self.state["last_processed_timestamp"]:
                continue
            if index < 200:
                self.state["last_processed_timestamp"] = candle["timestamp"]
                continue
            previous_state = deepcopy(self.state)
            event = self._process_candle(candles[: index + 1], index, symbol)
            self.state["last_processed_timestamp"] = candle["timestamp"]
            if event:
                events.extend(event)
            try:
                self._commit_transition(event)
            except ObservationPersistenceError as error:
                self.state = previous_state
                self.state["status"] = "WAITING_FOR_PERSISTENCE"
                self.state["last_error"] = str(error)
                self.state["persistence_health"] = (
                    self.adapter.store.persistence_health()
                )
                # Keep the outage visible after a runner restart when the
                # failed transition itself left the prior state intact.
                try:
                    self._save_state()
                except ObservationPersistenceError:
                    pass
                raise IncrementalPaperEngineError(
                    "paper observation paused: evidence storage is unavailable"
                ) from error
        self.state["status"] = "RUNNING"
        self.state["last_error"] = None
        self._save_state()
        return events

    def _update_drawdown(self, market_price: float) -> None:
        equity = self.state["capital"] + (
            self.state["position"] * market_price
        )
        self.state["current_equity"] = equity
        self.state["peak_equity"] = max(
            self.state["peak_equity"],
            equity,
        )
        drawdown = max(
            0.0,
            (
                (self.state["peak_equity"] - equity)
                / self.state["peak_equity"] * 100
            )
            if self.state["peak_equity"] > 0
            else 0.0,
        )
        self.state["drawdown_percent"] = drawdown
        self.state["max_drawdown_percent"] = max(
            self.state["max_drawdown_percent"],
            drawdown,
        )

    def status(self) -> dict[str, Any]:
        state = self.state
        return {
            "status": state["status"],
            "started_at": state["started_at"],
            "last_processed_timestamp": state["last_processed_timestamp"],
            "last_signal": state["last_signal"],
            "last_completed_trade": state["last_completed_trade"],
            "genuine_signals": state["genuine_signals"],
            "genuine_completed_trades": state["genuine_completed_trades"],
            "cash": state["capital"],
            "position": state["position"],
            "entry_price": state["entry_price"],
            "current_equity": state["current_equity"],
            "drawdown_percent": state["drawdown_percent"],
            "max_drawdown_percent": state["max_drawdown_percent"],
            "last_error": state["last_error"],
            "persistence_health": state.get(
                "persistence_health", self.adapter.store.persistence_health()
            ),
        }

    def _process_candle(
        self,
        candles: list[dict[str, Any]],
        index: int,
        symbol: str,
    ) -> list[dict[str, Any]]:
        candle = candles[index]
        prices = [item["close"] for item in candles]
        volumes = [item["volume"] for item in candles]
        ema_9 = calculate_ema(prices, 9)
        ema_21 = calculate_ema(prices, 21)
        ema_50 = calculate_ema(prices, 50)
        ema_200 = calculate_ema(prices, 200)
        rsi = calculate_rsi(prices)
        average_volume = calculate_average_volume(volumes)
        if None in (ema_9, ema_21, ema_50, ema_200, rsi, average_volume):
            return []

        score, decision, reasons, conditions = calculate_strategy_score(
            ema_9,
            ema_21,
            ema_50,
            ema_200,
            rsi,
            candle["close"],
            average_volume,
            candle["volume"],
        )
        self._update_drawdown(candle["close"])
        market_condition, market_condition_detail = classify_market_condition(
            price=candle["close"],
            ema21=ema_21,
            ema50=ema_50,
            ema200=ema_200,
            long_term_trend=conditions.get("long_term_trend", False),
            short_term_momentum=conditions.get("short_term_momentum", False),
        )
        signal_id = f"{symbol}:{candle['timestamp']}"
        signal = self.adapter.prepare_signal(
            signal_id=signal_id,
            observed_at=self._timestamp_iso(candle["timestamp"]),
            symbol=symbol,
            strategy_score=score,
            entry_eligible=decision == "BUY CANDIDATE",
            market_data_timestamp=self._timestamp_iso(candle["timestamp"]),
            data_health=HEALTHY,
            market_condition=market_condition,
            market_condition_detail=market_condition_detail,
            drawdown_percent=self.state["drawdown_percent"],
            max_drawdown_percent=self.state["max_drawdown_percent"],
        )
        self.state["genuine_signals"] += 1
        self.state["last_signal"] = signal
        events = [{"type": "SIGNAL", "record": signal}]

        current_day = candle["timestamp"] // 86400
        if current_day != self.state["current_day"]:
            self.state["current_day"] = current_day
            self.state["trades_today"] = 0
            self.state["daily_starting_capital"] = self.state["capital"]

        if self.state["position"] > 0:
            if candle["close"] <= self.state["stop_price"]:
                return events + [self._close(candle, "STOP LOSS", signal_id)]
            if candle["close"] >= self.state["target_price"]:
                return events + [self._close(candle, "TAKE PROFIT", signal_id)]
            return events

        daily_loss = self.state["daily_starting_capital"] - self.state["capital"]
        if decision != "BUY CANDIDATE":
            return events
        if daily_loss >= self.state["daily_starting_capital"] * MAX_DAILY_LOSS_PERCENT:
            return events
        allowed, _ = risk_check(
            self.state["capital"],
            0.0,
            self.state["trades_today"],
            score,
            candle["close"],
            daily_starting_capital=self.state["daily_starting_capital"],
        )
        if not allowed:
            return events

        position_value = self.state["capital"] * MAX_POSITION_PERCENT
        entry_price = candle["close"] * (1 + SLIPPAGE_PERCENT)
        entry_fee = position_value * FEE_PERCENT
        if position_value + entry_fee > self.state["capital"]:
            return events
        self.state.update(
            {
                "position": position_value / entry_price,
                "entry_price": entry_price,
                "entry_at": self._timestamp_iso(candle["timestamp"]),
                "entry_value": position_value,
                "entry_fee": entry_fee,
                "entry_slippage": position_value * SLIPPAGE_PERCENT,
                "stop_price": entry_price * (1 - STOP_LOSS_PERCENT),
                "target_price": entry_price * (1 + TAKE_PROFIT_PERCENT),
                "entry_signal_id": signal_id,
                "entry_score": score,
                "entry_market_condition": market_condition,
                "entry_market_condition_detail": market_condition_detail,
                "trades_today": self.state["trades_today"] + 1,
                "capital": self.state["capital"] - position_value - entry_fee,
            }
        )
        return events

    def _close(
        self,
        candle: dict[str, Any],
        reason: str,
        signal_id: str,
    ) -> dict[str, Any]:
        exit_price = candle["close"] * (1 - SLIPPAGE_PERCENT)
        gross_value = self.state["position"] * exit_price
        exit_fee = gross_value * FEE_PERCENT
        net_value = gross_value - exit_fee
        profit_loss = net_value - self.state["entry_value"]
        trade_id = f"{self.state['entry_at']}:{candle['timestamp']}"
        entry_at = self.state["entry_at"]
        entry_price = self.state["entry_price"]
        entry_fee = self.state["entry_fee"]
        entry_slippage = self.state["entry_slippage"]
        market_condition = self.state.get(
            "entry_market_condition",
            "UNAVAILABLE",
        )
        market_condition_detail = self.state.get(
            "entry_market_condition_detail",
            "UNAVAILABLE",
        )
        self.state["capital"] += net_value
        self.state["genuine_completed_trades"] += 1
        self.state.update(
            {
                "position": 0.0,
                "entry_price": 0.0,
                "entry_at": None,
                "entry_value": 0.0,
                "entry_fee": 0.0,
                "entry_slippage": 0.0,
                "stop_price": 0.0,
                "target_price": 0.0,
                "entry_signal_id": None,
                "entry_score": 0,
            "entry_market_condition": "UNAVAILABLE",
            "entry_market_condition_detail": "UNAVAILABLE",
            }
        )
        self._update_drawdown(candle["close"])
        record = self.adapter.prepare_trade(
            trade_id=trade_id,
            signal_id=signal_id,
            entry_at=entry_at,
            exit_at=self._timestamp_iso(candle["timestamp"]),
            entry_price=entry_price,
            exit_price=exit_price,
            profit_loss=profit_loss,
            fees=entry_fee + exit_fee,
            slippage=entry_slippage + gross_value * SLIPPAGE_PERCENT,
            exit_reason=reason,
            market_condition=market_condition,
            market_condition_detail=market_condition_detail,
            drawdown_percent=self.state["drawdown_percent"],
            max_drawdown_percent=self.state["max_drawdown_percent"],
        )
        self.state["last_completed_trade"] = record
        return {"type": "TRADE", "record": record}

    def _load_state(self) -> dict[str, Any]:
        initial = {
            "status": "STOPPED",
            "started_at": None,
            "last_processed_timestamp": 0,
            "last_signal": None,
            "last_completed_trade": None,
            "genuine_signals": 0,
            "genuine_completed_trades": 0,
            "capital": STARTING_CAPITAL,
            "position": 0.0,
            "entry_price": 0.0,
            "entry_at": None,
            "entry_value": 0.0,
            "entry_fee": 0.0,
            "entry_slippage": 0.0,
            "stop_price": 0.0,
            "target_price": 0.0,
            "entry_signal_id": None,
            "entry_score": 0,
            "current_day": None,
            "trades_today": 0,
            "daily_starting_capital": STARTING_CAPITAL,
            "current_equity": STARTING_CAPITAL,
            "peak_equity": STARTING_CAPITAL,
            "drawdown_percent": 0.0,
            "max_drawdown_percent": 0.0,
            "entry_market_condition": "UNAVAILABLE",
            "entry_market_condition_detail": "UNAVAILABLE",
            "last_error": None,
            "persistence_health": self.adapter.store.persistence_health(),
        }
        if not self.state_path.exists():
            return initial
        try:
            with self.state_path.open("r", encoding="utf-8") as handle:
                state = json.load(handle)
        except (OSError, json.JSONDecodeError) as error:
            raise IncrementalPaperEngineError(
                "paper engine state cannot be restored safely"
            ) from error
        if not isinstance(state, dict) or state.get("capital") is None:
            raise IncrementalPaperEngineError(
                "paper engine state is invalid and cannot be restored"
            )
        initial.update(state)
        self._validate_state(initial)
        return initial

    @staticmethod
    def _validate_state(state: dict[str, Any]) -> None:
        """Reject persisted state that could create invalid paper evidence."""
        statuses = {
            "STOPPED",
            "ARMED",
            "RUNNING",
            "WAITING_FOR_HEALTHY_DATA",
            "WAITING_FOR_PERSISTENCE",
        }
        if state["status"] not in statuses:
            raise IncrementalPaperEngineError("paper engine state is invalid and cannot be restored")
        numeric_non_negative = (
            "last_processed_timestamp",
            "genuine_signals",
            "genuine_completed_trades",
            "capital",
            "position",
            "entry_price",
            "entry_value",
            "entry_fee",
            "entry_slippage",
            "stop_price",
            "target_price",
            "trades_today",
            "daily_starting_capital",
            "current_equity",
            "peak_equity",
            "drawdown_percent",
            "max_drawdown_percent",
        )
        for field in numeric_non_negative:
            value = state[field]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise IncrementalPaperEngineError(
                    "paper engine state is invalid and cannot be restored"
                )
        if state["genuine_completed_trades"] > state["genuine_signals"]:
            raise IncrementalPaperEngineError(
                "paper engine state is invalid and cannot be restored"
            )
        for field in ("started_at", "entry_at"):
            value = state[field]
            if value is not None:
                if not isinstance(value, str):
                    raise IncrementalPaperEngineError(
                        "paper engine state is invalid and cannot be restored"
                    )
                try:
                    datetime.fromisoformat(value.replace("Z", "+00:00"))
                except ValueError as error:
                    raise IncrementalPaperEngineError(
                        "paper engine state is invalid and cannot be restored"
                    ) from error
        if state["position"] > 0 and (
            state["entry_price"] <= 0
            or state["entry_value"] <= 0
            or state["stop_price"] <= 0
            or state["target_price"] <= 0
            or state["entry_at"] is None
            or not isinstance(state["entry_signal_id"], str)
        ):
            raise IncrementalPaperEngineError(
                "paper engine state is invalid and cannot be restored"
            )

    def _save_state(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_suffix(".tmp")
        try:
            try:
                with temporary.open("w", encoding="utf-8") as handle:
                    json.dump(self.state, handle, separators=(",", ":"))
                    handle.flush()
                    os.fsync(handle.fileno())
                temporary.chmod(0o600)
                os.replace(temporary, self.state_path)
            except OSError as error:
                self.adapter.store.mark_persistence_failure(error, "engine_state_commit")
                self.state["persistence_health"] = (
                    self.adapter.store.persistence_health()
                )
                raise ObservationPersistenceError(
                    "observation storage is unavailable during engine state commit"
                ) from error
        finally:
            # The in-memory state is intentionally not treated as committed
            # until replace succeeds. A failed candidate must not be mistaken
            # for recoverable engine state on the next process start.
            temporary.unlink(missing_ok=True)
        self.state_path.chmod(0o600)

    def _commit_transition(self, events: list[dict[str, Any]]) -> None:
        self.adapter.store.commit_paper_transition(
            state_path=self.state_path,
            state=self.state,
            records=[event["record"] for event in events],
        )
        # Only a completed durable transition is allowed to clear a persisted
        # evidence-storage outage.
        self.state["persistence_health"] = self.adapter.store.persistence_health()

    @staticmethod
    def _validate_candles(candles: list[dict[str, Any]]) -> None:
        if not isinstance(candles, list):
            raise IncrementalPaperEngineError("candles must be a list")
        previous = 0
        for candle in candles:
            if not isinstance(candle, dict):
                raise IncrementalPaperEngineError("candle must be an object")
            required = {"timestamp", "open", "high", "low", "close", "volume"}
            if not required.issubset(candle):
                raise IncrementalPaperEngineError("candle is missing required fields")
            if int(candle["timestamp"]) <= previous:
                raise IncrementalPaperEngineError("candles must be strictly ordered")
            numeric = [candle[field] for field in required if field != "timestamp"]
            if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in numeric):
                raise IncrementalPaperEngineError("candle contains invalid numeric data")
            previous = int(candle["timestamp"])

    @staticmethod
    def _timestamp_iso(timestamp: int) -> str:
        from datetime import datetime, timezone

        return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()

    @staticmethod
    def _now_iso() -> str:
        from datetime import datetime, timezone

        return datetime.now(timezone.utc).isoformat()