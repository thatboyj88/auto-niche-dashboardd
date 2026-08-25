"""Auditable multi-lens strategy council with a final paper-only veto gate."""

from dataclasses import dataclass
from typing import Mapping

from config import MAX_POSITION_PERCENT

LENSES = (
    "value", "growth", "quality", "momentum",
    "trend", "dividend", "mean_reversion", "macro",
)
ACTIONABLE = frozenset({"BUY", "SELL", "REBALANCE"})
VALID_DECISIONS = ACTIONABLE | {"HOLD", "WAIT", "REJECT"}
VALID_STATUSES = frozenset({"AVAILABLE", "UNAVAILABLE", "STALE", "INVALID", "CONTRADICTORY"})


@dataclass(frozen=True)
class CouncilResult:
    decision: str
    confidence: float | None
    data_quality: str
    disagreement: str
    rationale: str
    lenses: dict

    def to_dict(self):
        return {
            "decision": self.decision,
            "confidence": self.confidence,
            "data_quality": self.data_quality,
            "disagreement": self.disagreement,
            "rationale": self.rationale,
            "lenses": self.lenses,
        }


@dataclass(frozen=True)
class RiskGovernorResult:
    approved: bool
    veto_reason: str | None
    checks: dict

    def to_dict(self):
        return {
            "approved": self.approved,
            "veto_reason": self.veto_reason,
            "checks": self.checks,
        }


@dataclass(frozen=True)
class FinalCouncilResult:
    council: CouncilResult
    governor: RiskGovernorResult
    final_action: str
    final_reason: str

    def to_dict(self):
        return {
            "council": self.council.to_dict(),
            "governor": self.governor.to_dict(),
            "final_action": self.final_action,
            "final_reason": self.final_reason,
        }


def _finite(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and value == value
        and value not in (float("inf"), float("-inf"))
    )


def normalize_lens_evidence(evidence: Mapping | None) -> dict:
    """Return every requested lens explicitly, never silently omit a lens."""
    source = evidence if isinstance(evidence, Mapping) else {}
    normalized = {}
    for lens in LENSES:
        item = source.get(lens)
        if not isinstance(item, Mapping):
            normalized[lens] = {
                "status": "UNAVAILABLE", "decision": "WAIT",
                "confidence": None, "reason": "No evidence supplied.",
            }
            continue
        status = str(item.get("status", "INVALID")).upper()
        decision = str(item.get("decision", "WAIT")).upper()
        confidence = item.get("confidence")
        valid = (
            status in VALID_STATUSES
            and decision in VALID_DECISIONS
            and (confidence is None or (_finite(confidence) and 0 <= confidence <= 100))
        )
        if not valid:
            status, decision, confidence = "INVALID", "WAIT", None
        normalized[lens] = {
            "status": status,
            "decision": decision if status == "AVAILABLE" else "WAIT",
            "confidence": float(confidence) if confidence is not None else None,
            "reason": str(item.get("reason") or (
                "Lens evidence is unavailable." if status != "AVAILABLE"
                else "Evidence supplied."
            )),
        }
    return normalized


def aggregate_strategy_council(evidence: Mapping | None) -> CouncilResult:
    lenses = normalize_lens_evidence(evidence)
    unavailable = [name for name, item in lenses.items() if item["status"] != "AVAILABLE"]
    available = [item for item in lenses.values() if item["status"] == "AVAILABLE"]
    if unavailable or not available:
        quality = "INVALID" if any(item["status"] == "INVALID" for item in lenses.values()) else "DATA_INSUFFICIENT"
        return CouncilResult(
            "WAIT", None, quality,
            f"{len(unavailable)} of {len(LENSES)} lenses are not available.",
            "Council is blocked until every requested lens has fresh, valid evidence.",
            lenses,
        )
    votes = {}
    for item in available:
        votes[item["decision"]] = votes.get(item["decision"], 0) + 1
    winner, count = max(votes.items(), key=lambda pair: pair[1])
    if len(votes) > 1 and count / len(available) <= 0.75:
        return CouncilResult(
            "WAIT",
            round(sum(item["confidence"] or 0 for item in available) / len(available), 2),
            "CONTRADICTORY",
            "Material lens disagreement prevents a consolidated action.",
            f"Votes disagree across {', '.join(sorted(votes))}; council defaults to WAIT.",
            lenses,
        )
    confidence = round(sum(item["confidence"] or 0 for item in available) / len(available), 2)
    return CouncilResult(
        winner, confidence, "SUFFICIENT",
        "No material disagreement." if len(votes) == 1 else f"{count}/{len(available)} lenses support {winner}.",
        f"{winner} is supported by {count} of {len(available)} complete lenses.",
        lenses,
    )


def evaluate_risk_governor(
    decision: str,
    *,
    exposure_percent,
    concentration_percent,
    proposed_position_percent,
    data_health="HEALTHY",
    paper_trading=True,
    live_trading=False,
    safety_policy=True,
    max_position_percent=MAX_POSITION_PERCENT,
) -> RiskGovernorResult:
    """Evaluate all final safety conditions after council aggregation."""
    checks = {}
    reasons = []
    for name, value in (
        ("exposure", exposure_percent),
        ("concentration", concentration_percent),
        ("proposed_position", proposed_position_percent),
    ):
        checks[name] = _finite(value) and 0 <= value <= 1
        if not checks[name]:
            reasons.append(f"{name} is invalid.")
    checks["data_health"] = data_health == "HEALTHY"
    checks["paper_only"] = paper_trading is True and live_trading is False
    checks["safety_policy"] = safety_policy is True
    if not checks["data_health"]:
        reasons.append(f"market data health is {data_health}.")
    if not checks["paper_only"]:
        reasons.append("paper-only execution policy is not active.")
    if not checks["safety_policy"]:
        reasons.append("safety policy is not active.")
    if checks["concentration"] and concentration_percent > max_position_percent:
        reasons.append(f"concentration {concentration_percent:.1%} exceeds {max_position_percent:.1%}.")
        checks["concentration_limit"] = False
    else:
        checks["concentration_limit"] = True
    if checks["proposed_position"] and proposed_position_percent > max_position_percent:
        reasons.append(f"proposed size {proposed_position_percent:.1%} exceeds {max_position_percent:.1%}.")
        checks["position_limit"] = False
    else:
        checks["position_limit"] = True
    approved = not reasons
    return RiskGovernorResult(approved, "; ".join(reasons) if reasons else None, checks)


def finalize_council_decision(council: CouncilResult, governor: RiskGovernorResult) -> FinalCouncilResult:
    if council.decision not in ACTIONABLE:
        return FinalCouncilResult(council, governor, council.decision, council.rationale)
    if not governor.approved:
        return FinalCouncilResult(
            council, governor, "REJECT",
            f"Risk Governor vetoed {council.decision}: {governor.veto_reason}",
        )
    return FinalCouncilResult(council, governor, council.decision, council.rationale)