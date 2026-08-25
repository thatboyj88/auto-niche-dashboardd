"""Read-only portfolio allocation and risk analysis.

This module deliberately has no file, network, or execution side effects.  It
turns an observed portfolio state into explainable diagnostics and
recommendations; callers must decide separately whether a recommendation is
safe to show or promote.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from math import isfinite

from config import MAX_POSITION_PERCENT, STOP_LOSS_PERCENT


@dataclass(frozen=True)
class PortfolioAnalysis:
    status: str
    total_value: float | None
    current_allocation: dict[str, float]
    proposed_allocation: dict[str, float]
    position_sizing: dict[str, float | str]
    concentration: dict[str, object]
    diversification: dict[str, object]
    risk_adjusted_allocation: dict[str, object]
    recommendations: tuple[str, ...]
    issues: tuple[str, ...]

    def to_dict(self):
        return {
            "status": self.status,
            "total_value": self.total_value,
            "current_allocation": self.current_allocation,
            "proposed_allocation": self.proposed_allocation,
            "position_sizing": self.position_sizing,
            "concentration": self.concentration,
            "diversification": self.diversification,
            "risk_adjusted_allocation": self.risk_adjusted_allocation,
            "recommendations": list(self.recommendations),
            "issues": list(self.issues),
        }


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and isfinite(value)


def analyze_portfolio(
    *,
    cash,
    positions,
    target_allocations=None,
    max_position_percent=MAX_POSITION_PERCENT,
    stop_loss_percent=STOP_LOSS_PERCENT,
    rebalance_threshold=0.05,
    observed_at=None,
    now=None,
    max_age_seconds=900,
):
    """Analyze marked-to-market positions without creating an order intent.

    Each position accepts symbol, quantity, price, and optional asset_class,
    target_weight, expected_return, and volatility. Weights are fractions.
    Invalid or missing critical values produce an unavailable result rather
    than an invented balance.
    """
    issues = []
    if observed_at is not None:
        try:
            observed = datetime.fromisoformat(str(observed_at).replace("Z", "+00:00"))
            if observed.tzinfo is None:
                observed = observed.replace(tzinfo=timezone.utc)
            reference = now or datetime.now(timezone.utc)
            if reference.tzinfo is None:
                reference = reference.replace(tzinfo=timezone.utc)
            age = (reference - observed).total_seconds()
            if age < 0 or not _number(max_age_seconds) or age > max_age_seconds:
                issues.append("Portfolio valuation is stale or has an invalid timestamp.")
        except (TypeError, ValueError, AttributeError):
            issues.append("Portfolio valuation timestamp is invalid.")
    if not _number(cash) or cash < 0:
        issues.append("Cash balance is missing or invalid.")
    if not isinstance(positions, (list, tuple)):
        issues.append("Position data is unavailable.")
        positions = []
    if not _number(max_position_percent) or not 0 < max_position_percent <= 1:
        issues.append("Concentration limit is invalid.")
    if not _number(stop_loss_percent) or not 0 < stop_loss_percent < 1:
        issues.append("Stop-loss risk input is invalid.")

    marked = []
    for raw in positions:
        if not isinstance(raw, dict):
            issues.append("A position record is invalid.")
            continue
        symbol = str(raw.get("symbol") or "").strip().upper()
        quantity, price = raw.get("quantity"), raw.get("price")
        if not symbol or not _number(quantity) or quantity < 0 or not _number(price) or price <= 0:
            issues.append(f"Position {symbol or 'unknown'} is missing a valid quantity or price.")
            continue
        marked.append(
            {
                "symbol": symbol,
                "asset_class": str(raw.get("asset_class") or "unknown").lower(),
                "value": float(quantity * price),
                "expected_return": raw.get("expected_return"),
                "volatility": raw.get("volatility"),
            }
        )

    if issues:
        return PortfolioAnalysis(
            "UNAVAILABLE", None, {}, {}, {}, {"status": "UNAVAILABLE", "breaches": []},
            {"status": "UNAVAILABLE", "effective_assets": None},
            {"status": "UNAVAILABLE", "inputs": {}, "provenance": "No valid marked portfolio."},
            ("No portfolio recommendation is available until inputs are valid.",), tuple(issues),
        )

    total = float(cash) + sum(item["value"] for item in marked)
    if total <= 0:
        return PortfolioAnalysis(
            "EMPTY", 0.0, {"CASH": 1.0}, {}, {
                "status": "EMPTY", "max_position_value": 0.0, "stop_risk_at_max": 0.0,
            },
            {"status": "CLEAR", "limit": max_position_percent, "breaches": []},
            {"status": "LIMITED", "asset_count": 0, "effective_assets": 0.0},
            {"status": "UNAVAILABLE", "inputs": {}, "provenance": "No investable marked assets."},
            ("Maintain cash until a valid, risk-approved paper opportunity exists.",), (),
        )

    values = {"CASH": float(cash)}
    for item in marked:
        values[item["symbol"]] = values.get(item["symbol"], 0.0) + item["value"]
    allocation = {key: value / total for key, value in values.items()}
    proposed = {}
    if target_allocations is not None:
        if not isinstance(target_allocations, dict):
            issues.append("Target allocation data is invalid.")
        elif any(not _number(value) or value < 0 for value in target_allocations.values()):
            issues.append("Target allocation weights must be finite and non-negative.")
        else:
            proposed = {
                str(key).upper(): float(value)
                for key, value in target_allocations.items()
            }
            if sum(proposed.values()) > 1.0 + 1e-9:
                issues.append("Target allocation weights exceed 100%.")
            else:
                proposed.setdefault("CASH", max(0.0, 1 - sum(proposed.values())))
    if issues:
        return PortfolioAnalysis(
            "UNAVAILABLE", None, {}, {}, {}, {"status": "UNAVAILABLE", "breaches": []},
            {"status": "UNAVAILABLE", "effective_assets": None},
            {"status": "UNAVAILABLE", "inputs": {}, "provenance": "No valid marked portfolio."},
            ("No portfolio recommendation is available until inputs are valid.",), tuple(issues),
        )

    breaches = [
        f"{symbol} allocation {weight:.1%} exceeds {max_position_percent:.1%} limit."
        for symbol, weight in allocation.items()
        if symbol != "CASH" and weight > max_position_percent
    ]
    hhi = sum(weight * weight for weight in allocation.values() if weight > 0)
    effective_assets = 1 / hhi if hhi else 0.0
    concentration = {
        "status": "BREACH" if breaches else "CLEAR",
        "limit": max_position_percent,
        "largest_asset": max(
            ((key, value) for key, value in allocation.items() if key != "CASH"),
            key=lambda pair: pair[1],
            default=("CASH", allocation.get("CASH", 1.0)),
        )[0],
        "largest_weight": max(
            (value for key, value in allocation.items() if key != "CASH"),
            default=0.0,
        ),
        "breaches": breaches,
    }
    if target_allocations is None:
        proposed = dict(allocation)
        for symbol, weight in allocation.items():
            if symbol != "CASH" and weight > max_position_percent:
                excess = weight - max_position_percent
                proposed[symbol] = max_position_percent
                proposed["CASH"] = proposed.get("CASH", 0.0) + excess
    diversification = {
        "status": "LIMITED" if len([key for key in allocation if key != "CASH"]) < 2 else "ASSESSED",
        "asset_count": len([key for key in allocation if key != "CASH"]),
        "effective_assets": round(effective_assets, 3),
        "hhi": round(hhi, 4),
    }
    sizing = {
        "status": "AVAILABLE",
        "capital_base": float(cash),
        "max_position_value": float(cash) * max_position_percent,
        "stop_risk_at_max": float(cash) * max_position_percent * stop_loss_percent,
        "stop_loss_percent": stop_loss_percent,
        "explanation": (
            f"Uses observed paper cash × {max_position_percent:.1%} ceiling; "
            f"stop risk is sized at {stop_loss_percent:.1%}."
        ),
    }

    risk_inputs = []
    scores = {}
    for item in marked:
        expected, volatility = item["expected_return"], item["volatility"]
        if _number(expected) and _number(volatility) and volatility > 0:
            score = float(expected) / float(volatility)
            scores[item["symbol"]] = score
            risk_inputs.append({
                "symbol": item["symbol"], "expected_return": float(expected),
                "volatility": float(volatility), "return_to_risk": round(score, 4),
            })
    risk_adjusted = {
        "status": "AVAILABLE" if risk_inputs else "UNAVAILABLE",
        "inputs": risk_inputs,
        "uncertainty": (
            "Estimates are incomplete; ranking is not available."
            if not risk_inputs
            else "Uncertainty is not modeled; estimates are caller-supplied and informational."
        ),
        "provenance": "Caller-supplied estimates only; no estimates are fabricated.",
        "ranking": [symbol for symbol, _ in sorted(scores.items(), key=lambda pair: pair[1], reverse=True)],
    }

    recommendations = []
    if breaches:
        recommendations.append("VETO proposed increases: concentration breach requires Risk Governor review.")
    if not marked:
        recommendations.append("Keep the portfolio in cash until a valid paper position is observed.")
    elif not proposed:
        recommendations.append("No target allocation supplied; rebalance amounts are intentionally not proposed.")
    else:
        drifts = {
            symbol: proposed.get(symbol, 0.0) - allocation.get(symbol, 0.0)
            for symbol in set(allocation) | set(proposed)
        }
        material = {symbol: drift for symbol, drift in drifts.items() if abs(drift) >= rebalance_threshold}
        if material:
            recommendations.append("Review the material allocation drift before any paper-only rebalance.")
        else:
            recommendations.append("Current allocation is within the rebalance tolerance.")
    if diversification["status"] == "LIMITED":
        recommendations.append("Diversification is limited: this analysis covers fewer than two invested assets.")

    return PortfolioAnalysis(
        "OK", total, allocation, proposed, sizing, concentration, diversification,
        risk_adjusted, tuple(recommendations), tuple(breaches),
    )