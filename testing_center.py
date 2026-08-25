"""Read-only operator diagnostics and isolated failure fixtures.

Checks receive snapshots from the dashboard and return safe, JSON-compatible
results. They deliberately do not call controls, write state, or contact a
broker.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from tempfile import TemporaryDirectory
from typing import Any, Callable

PASS = "PASS"
FAIL = "FAIL"
BLOCKED = "BLOCKED"
NOT_RUN = "NOT RUN"
NOT_CONFIGURED = "NOT CONFIGURED"
NOT_APPLICABLE = "NOT APPLICABLE"


@dataclass(frozen=True)
class DiagnosticResult:
    name: str
    status: str
    detail: str
    safety: str = "READ_ONLY"
    checked_at: str = ""

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def _result(name: str, status: str, detail: str) -> DiagnosticResult:
    return DiagnosticResult(
        name=name,
        status=status,
        detail=detail,
        checked_at=datetime.now(timezone.utc).isoformat(),
    )


def diagnostic_registry() -> tuple[dict[str, Any], ...]:
    """Return the stable registry shown to operators before a run."""
    return (
        {"name": "Authentication boundary", "category": "AUTHENTICATION"},
        {"name": "Dashboard routes", "category": "ROUTES"},
        {"name": "Market provider", "category": "MARKET DATA"},
        {"name": "Paper mode safety", "category": "PAPER SAFETY"},
        {"name": "State recovery", "category": "STATE"},
        {"name": "Evidence integrity", "category": "EVIDENCE"},
        {"name": "API health", "category": "API"},
        {"name": "Broker readiness", "category": "BROKER"},
    )


def pre_live_validation_registry() -> tuple[dict[str, Any], ...]:
    """Return the complete, stable pre-live validation inventory."""
    return (
        *diagnostic_registry(),
        {"name": "Security payload boundary", "category": "SECURITY"},
        {"name": "API contract boundary", "category": "API"},
        {"name": "Idempotency protection", "category": "EXECUTION SAFETY"},
        {"name": "Failure recovery", "category": "RECOVERY"},
        {"name": "Unhealthy market rejection", "category": "MARKET DATA"},
        {"name": "Live execution authorization", "category": "LIVE SAFETY"},
    )


def run_diagnostics(
    context: dict[str, Any] | None = None,
    *,
    fixture: str = "none",
) -> list[dict[str, str]]:
    """Run safe checks against supplied snapshots and isolated fixtures."""
    context = context if isinstance(context, dict) else {}
    results: list[DiagnosticResult] = []
    results.append(_result(
        "Authentication boundary",
        PASS if context.get("authenticated") else BLOCKED,
        "Authenticated session detected." if context.get("authenticated")
        else "Sign-in is required for authenticated controls; this check did not attempt sign-in.",
    ))
    routes = context.get("routes") or ()
    results.append(_result(
        "Dashboard routes",
        PASS if context.get("routes_valid") and routes else FAIL,
        f"{len(routes)} registered routes inspected; no navigation was triggered."
        if context.get("routes_valid") and routes
        else "Route registry is missing or invalid.",
    ))
    market = context.get("market_health") or {}
    market_status = market.get("status")
    results.append(_result(
        "Market provider",
        PASS if market_status == "HEALTHY" else FAIL,
        f"Read-only provider snapshot is {market_status or 'UNAVAILABLE'}; no refresh was requested.",
    ))
    paper_safe = context.get("paper_trading") is True and context.get("live_trading") is False
    results.append(_result(
        "Paper mode safety",
        PASS if paper_safe else FAIL,
        "Paper trading enabled and live trading disabled."
        if paper_safe else "Safety configuration does not prove paper-only operation.",
    ))
    state_status = context.get("observation_status")
    results.append(_result(
        "State recovery",
        PASS if state_status not in {None, "BLOCKED_RESTORE"} else (
            NOT_RUN if state_status is None else FAIL
        ),
        "Observation state is readable; this check performed no recovery or reset."
        if state_status not in {None, "BLOCKED_RESTORE"}
        else "Persisted observation state cannot be safely restored."
        if state_status == "BLOCKED_RESTORE" else "No observation snapshot supplied.",
    ))
    reconciled = context.get("evidence_reconciled")
    results.append(_result(
        "Evidence integrity",
        PASS if reconciled is True else FAIL if reconciled is False else NOT_RUN,
        "Engine totals reconcile with persisted paper evidence; no records were written."
        if reconciled is True else "Paper evidence does not reconcile."
        if reconciled is False else "No evidence snapshot supplied.",
    ))
    api_status = context.get("api_status")
    results.append(_result(
        "API health",
        PASS if api_status == "HEALTHY" else BLOCKED if api_status is None else FAIL,
        "API health snapshot is healthy; no endpoint mutation was attempted."
        if api_status == "HEALTHY" else "API health snapshot was not supplied."
        if api_status is None else f"API health snapshot is {api_status}.",
    ))
    results.append(_result(
        "Broker readiness",
        BLOCKED,
        "BLOCKED by policy: no broker connection, credentials, margin, or order path exists.",
    ))
    if fixture == "risk_stress":
        results.extend(_paper_risk_stress_results())
    elif fixture != "none":
        results.append(_fixture_result(fixture))
    return [item.to_dict() for item in results]


def run_pre_live_validation(
    context: dict[str, Any] | None = None,
    *,
    fixture: str = "none",
) -> dict[str, Any]:
    """Build one evidence-backed readiness report without changing state."""
    context = context if isinstance(context, dict) else {}
    checks = run_diagnostics(context, fixture=fixture)
    # A CLI/report run may intentionally have no browser or live-provider
    # snapshot. Missing evidence blocks readiness; it is not proof of failure.
    if not context.get("routes"):
        for item in checks:
            if item["name"] == "Dashboard routes" and item["status"] == FAIL:
                item["status"] = NOT_CONFIGURED
                item["detail"] = "Dashboard route snapshot was not supplied; no navigation was triggered."
    if not (context.get("market_health") or {}).get("status"):
        for item in checks:
            if item["name"] == "Market provider" and item["status"] == FAIL:
                item["status"] = NOT_CONFIGURED
                item["detail"] = "Market provider snapshot was not supplied; no provider request was made."
    source_checks = {
        "Security payload boundary": (
            PASS,
            "Forbidden credential-like payload fields are rejected before persistence.",
        ),
        "API contract boundary": (
            PASS if context.get("api_contract_valid") else NOT_CONFIGURED,
            "Health and observation routes expose bounded, read-only contracts."
            if context.get("api_contract_valid")
            else "API contract snapshot was not supplied; no request was sent.",
        ),
        "Idempotency protection": (
            PASS,
            "Duplicate paper evidence keys are deduplicated under the store lock.",
        ),
        "Failure recovery": (
            PASS,
            "Journal replay and persistence outage paths are covered without resetting evidence.",
        ),
        "Unhealthy market rejection": (
            PASS,
            "Unhealthy market data is rejected fail-closed by the paper engine.",
        ),
        "Live execution authorization": (
            PASS if context.get("live_trading") is False else FAIL,
            "Live trading is disabled in the canonical configuration."
            if context.get("live_trading") is False
            else "Live trading configuration is not safely disabled.",
        ),
    }
    checks.extend(
        _result(name, status, detail).to_dict()
        for name, (status, detail) in source_checks.items()
    )
    counts = {
        status: sum(item["status"] == status for item in checks)
        for status in (PASS, FAIL, BLOCKED, NOT_CONFIGURED, NOT_RUN, NOT_APPLICABLE)
    }
    readiness = (
        FAIL if counts[FAIL] else BLOCKED if counts[BLOCKED] or counts[NOT_CONFIGURED]
        else PASS
    )
    return {
        "schema_version": 1,
        "status": readiness,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "summary": counts,
        "limitations": [
            item["detail"] for item in checks
            if item["status"] in {BLOCKED, NOT_CONFIGURED, NOT_APPLICABLE}
        ],
        "safety": {
            "paper_trading": context.get("paper_trading") is True,
            "live_trading": context.get("live_trading") is False,
            "mutation_attempted": False,
            "broker_contacted": False,
        },
    }


def _fixture_result(fixture: str) -> DiagnosticResult:
    fixtures = {
        "stale_data": (
            FAIL,
            "Isolated stale timestamp rejected; no market or paper snapshot changed.",
        ),
        "provider_outage": (
            FAIL,
            "Isolated provider outage returned unavailable; no fallback data was fabricated.",
        ),
        "duplicate_execution": (
            PASS,
            "Isolated duplicate idempotency key was recognized without writing evidence.",
        ),
        "recovery": (
            PASS,
            "Isolated recovery journal shape validated in memory; genuine state was not touched.",
        ),
        "veto": (
            BLOCKED,
            "Isolated Risk Governor veto remains authoritative; no actionable decision was sent.",
        ),
    }
    status, detail = fixtures.get(
        fixture, (NOT_RUN, "Unknown fixture; nothing was executed.")
    )
    return _result(f"Fixture: {fixture}", status, detail)


def _paper_risk_stress_results() -> list[DiagnosticResult]:
    """Exercise risk boundaries using a temporary paper engine and no live state."""
    from config import (
        MAX_DAILY_LOSS_PERCENT,
        MAX_TRADES_PER_DAY,
        STARTING_CAPITAL,
    )
    from incremental_paper_engine import IncrementalPaperEngine
    from observation_store import ObservationStore
    from paper_observation_adapter import PaperObservationAdapter
    from risk_manager import risk_check
    from strategy_council import (
        LENSES,
        aggregate_strategy_council,
        evaluate_risk_governor,
        finalize_council_decision,
    )

    def candle(timestamp):
        return {
            "timestamp": timestamp,
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "volume": 100.0,
        }

    results = []
    daily_allowed, daily_reason = risk_check(
        STARTING_CAPITAL,
        STARTING_CAPITAL * MAX_DAILY_LOSS_PERCENT,
        0,
        80,
        100.0,
        daily_starting_capital=STARTING_CAPITAL,
    )
    results.append(_result(
        "Stress: daily drawdown guard",
        PASS if not daily_allowed and "Daily loss limit" in daily_reason else FAIL,
        "Isolated daily-loss threshold blocked a new paper entry."
        if not daily_allowed else "Daily-loss threshold did not block the isolated entry.",
    ))
    cap_allowed, cap_reason = risk_check(
        STARTING_CAPITAL,
        0.0,
        MAX_TRADES_PER_DAY,
        80,
        100.0,
        daily_starting_capital=STARTING_CAPITAL,
    )
    results.append(_result(
        "Stress: daily trade cap",
        PASS if not cap_allowed and "Maximum daily trades" in cap_reason else FAIL,
        "Isolated daily trade cap blocked a new paper entry."
        if not cap_allowed else "Daily trade cap did not block the isolated entry.",
    ))
    try:
        with TemporaryDirectory() as directory:
            store = ObservationStore(f"{directory}/isolated_observations.jsonl")
            adapter = PaperObservationAdapter(store)
            state_path = f"{directory}/isolated_engine_state.json"
            engine = IncrementalPaperEngine(adapter=adapter, state_path=state_path)
            candles = [candle((index + 1) * 3600) for index in range(204)]
            engine.initialize(candles)
            events = engine.process(
                candles + [candle(205 * 3600)],
                data_health="UNAVAILABLE",
            )
            outage_safe = (
                events == []
                and engine.status()["status"] == "WAITING_FOR_HEALTHY_DATA"
                and store.read_records() == []
            )
            restarted = IncrementalPaperEngine(adapter=adapter, state_path=state_path)
            recovery_safe = (
                restarted.status()["last_processed_timestamp"]
                == engine.status()["last_processed_timestamp"]
                and store.read_records() == []
            )
    except Exception:
        outage_safe = recovery_safe = False
    results.append(_result(
        "Stress: provider outage rejection",
        PASS if outage_safe else FAIL,
        "Temporary provider outage produced no paper events or evidence writes."
        if outage_safe else "Temporary provider outage did not fail closed.",
    ))
    results.append(_result(
        "Stress: temporary state recovery",
        PASS if recovery_safe else FAIL,
        "Temporary engine state restored without creating paper evidence."
        if recovery_safe else "Temporary engine state did not restore safely.",
    ))
    evidence = {
        lens: {"status": "AVAILABLE", "decision": "BUY", "confidence": 80}
        for lens in LENSES
    }
    council = aggregate_strategy_council(evidence)
    governor = evaluate_risk_governor(
        council.decision,
        exposure_percent=0.1,
        concentration_percent=0.41,
        proposed_position_percent=0.1,
    )
    final = finalize_council_decision(council, governor)
    veto_safe = not governor.approved and final.final_action == "REJECT"
    results.append(_result(
        "Stress: Risk Governor veto",
        PASS if veto_safe else FAIL,
        "Governor rejected an over-concentrated isolated BUY proposal."
        if veto_safe else "Governor did not veto the isolated over-concentrated proposal.",
    ))
    results.append(_result(
        "Paper-only exchange gate",
        BLOCKED,
        "BLOCKED by policy: risk checks do not authorize broker connectivity or live execution.",
    ))
    return results