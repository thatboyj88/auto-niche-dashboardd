"""Universal, fail-closed investment decision records.

This module is deliberately independent from paper execution and observation
storage.  It gives future stock, ETF, option, portfolio, and strategy-council
features one validated record format without creating an execution interface.
"""

from __future__ import annotations

import math
import json
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Mapping, Sequence


class AssetType(str, Enum):
    STOCK = "STOCK"
    ETF = "ETF"
    OPTION = "OPTION"
    DEFINED_RISK_OPTION_STRATEGY = "DEFINED_RISK_OPTION_STRATEGY"
    CASH = "CASH"
    CRYPTO = "CRYPTO"


class Decision(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    WAIT = "WAIT"
    REBALANCE = "REBALANCE"
    REJECT = "REJECT"


class Direction(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    NEUTRAL = "NEUTRAL"


class DataQuality(str, Enum):
    SUFFICIENT = "SUFFICIENT"
    DATA_INSUFFICIENT = "DATA_INSUFFICIENT"
    STALE = "STALE"
    INVALID = "INVALID"


ACTIONABLE_DECISIONS = frozenset(
    {Decision.BUY, Decision.SELL, Decision.REBALANCE}
)
OPTION_ASSET_TYPES = frozenset(
    {AssetType.OPTION, AssetType.DEFINED_RISK_OPTION_STRATEGY}
)
SUPPORTED_ASSET_TYPES = frozenset(AssetType)


class InvestmentDecisionValidationError(ValueError):
    """Raised when a completed decision record is internally inconsistent."""


class PublicOptionQuoteProviderError(RuntimeError):
    """Raised when the public quote source cannot provide a safe snapshot."""


class OptionStrategy(str, Enum):
    """The deliberately small, defined-risk-only options strategy vocabulary."""

    LONG_CALL = "LONG_CALL"
    LONG_PUT = "LONG_PUT"
    COVERED_CALL = "COVERED_CALL"
    CASH_SECURED_PUT = "CASH_SECURED_PUT"
    BULL_CALL_SPREAD = "BULL_CALL_SPREAD"
    BEAR_PUT_SPREAD = "BEAR_PUT_SPREAD"
    PROTECTIVE_PUT = "PROTECTIVE_PUT"
    COLLAR = "COLLAR"


@dataclass(frozen=True)
class NormalizedOptionContract:
    """A validated quote snapshot. This is analysis data, never an order."""

    underlying: str
    option_type: str
    strike: float
    expiration: str
    bid: float
    ask: float
    underlying_price: float
    multiplier: int
    observed_at: str

    def __post_init__(self) -> None:
        if not self.underlying.strip():
            raise InvestmentDecisionValidationError("underlying must be non-empty text")
        if self.option_type not in {"CALL", "PUT"}:
            raise InvestmentDecisionValidationError("option_type must be CALL or PUT")
        for name in ("strike", "bid", "ask", "underlying_price"):
            value = _finite_number(getattr(self, name), name)
            if value <= 0:
                raise InvestmentDecisionValidationError(f"{name} must be positive")
        if self.ask < self.bid:
            raise InvestmentDecisionValidationError("ask cannot be below bid")
        if isinstance(self.multiplier, bool) or not isinstance(self.multiplier, int):
            raise InvestmentDecisionValidationError("multiplier must be an integer")
        if self.multiplier <= 0:
            raise InvestmentDecisionValidationError("multiplier must be positive")
        try:
            expiration = datetime.fromisoformat(self.expiration.replace("Z", "+00:00"))
            observed = datetime.fromisoformat(self.observed_at.replace("Z", "+00:00"))
        except (AttributeError, TypeError, ValueError) as error:
            raise InvestmentDecisionValidationError(
                "expiration and observed_at must be ISO-8601 timestamps"
            ) from error
        if expiration.tzinfo is None or observed.tzinfo is None:
            raise InvestmentDecisionValidationError(
                "expiration and observed_at must include a timezone"
            )

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2

    @property
    def spread(self) -> float:
        return self.ask - self.bid

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result.update(mid=self.mid, spread=self.spread)
        return result


@dataclass(frozen=True)
class OptionStrategyAnalysis:
    """At-expiration payoff bounds and paper-trade cost estimates."""

    strategy: OptionStrategy
    underlying: str
    break_even: float | None
    maximum_profit: float
    maximum_loss: float
    exposure: float
    cost: float
    slippage: float
    days_to_expiration: int
    contracts: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "strategy", _enum_value(
            self.strategy, OptionStrategy, "strategy"
        ))
        for field in (
            "maximum_profit", "maximum_loss", "exposure", "cost", "slippage",
            "break_even",
        ):
            value = getattr(self, field)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, (int, float))
                or math.isnan(float(value))
            ):
                raise InvestmentDecisionValidationError(f"{field} must be numeric")
        if self.maximum_loss < 0 or self.cost < 0 or self.slippage < 0:
            raise InvestmentDecisionValidationError(
                "maximum_loss, cost, and slippage must be non-negative"
            )
        if self.days_to_expiration < 0 or self.contracts <= 0:
            raise InvestmentDecisionValidationError(
                "days_to_expiration and contracts are invalid"
            )

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["strategy"] = self.strategy.value
        return result


def normalize_option_contract(
    contract: Mapping[str, Any],
    *,
    now: datetime | None = None,
    max_data_age_seconds: float = 900,
) -> NormalizedOptionContract:
    """Normalize one quote and reject missing, stale, or contradictory fields."""
    if not isinstance(contract, Mapping):
        raise InvestmentDecisionValidationError("option contract must be an object")
    required = (
        "underlying", "option_type", "strike", "expiration", "bid", "ask",
        "underlying_price", "observed_at",
    )
    missing = [field for field in required if contract.get(field) is None]
    if missing:
        raise InvestmentDecisionValidationError(
            f"Required option data is missing: {', '.join(missing)}"
        )
    try:
        limit = _finite_number(max_data_age_seconds, "max_data_age_seconds")
        if limit < 0:
            raise InvestmentDecisionValidationError(
                "max_data_age_seconds must be non-negative"
            )
        current = now or datetime.now(timezone.utc)
        if current.tzinfo is None:
            current = current.replace(tzinfo=timezone.utc)
        observed = datetime.fromisoformat(str(contract["observed_at"]).replace("Z", "+00:00"))
        expiration = datetime.fromisoformat(str(contract["expiration"]).replace("Z", "+00:00"))
        if observed.tzinfo is None or expiration.tzinfo is None:
            raise InvestmentDecisionValidationError(
                "expiration and observed_at must include a timezone"
            )
        if expiration <= current:
            raise InvestmentDecisionValidationError("option expiration must be in the future")
        age = (current - observed).total_seconds()
        if age < 0 or age >= limit:
            raise InvestmentDecisionValidationError("option quote is stale or from the future")
        option_type = contract["option_type"]
        if not isinstance(option_type, str):
            raise InvestmentDecisionValidationError("option_type must be CALL or PUT")
        option_type = option_type.upper()
        for field in ("strike", "bid", "ask", "underlying_price"):
            value = _finite_number(contract[field], field)
            if value <= 0:
                raise InvestmentDecisionValidationError(f"{field} must be positive")
        multiplier = contract.get("multiplier", 100)
        if isinstance(multiplier, bool) or not isinstance(multiplier, int) or multiplier <= 0:
            raise InvestmentDecisionValidationError("multiplier must be a positive integer")
        normalized = NormalizedOptionContract(
            underlying=str(contract["underlying"]).strip(),
            option_type=option_type,
            strike=float(contract["strike"]),
            expiration=str(contract["expiration"]),
            bid=float(contract["bid"]),
            ask=float(contract["ask"]),
            underlying_price=float(contract["underlying_price"]),
            multiplier=multiplier,
            observed_at=str(contract["observed_at"]),
        )
        return normalized
    except (InvestmentDecisionValidationError, TypeError, ValueError, OverflowError) as error:
        if isinstance(error, InvestmentDecisionValidationError):
            raise
        raise InvestmentDecisionValidationError(f"Invalid option contract: {error}") from error


def analyze_defined_risk_option_strategy(
    strategy: OptionStrategy | str,
    *,
    contracts: Sequence[NormalizedOptionContract],
    stock_price: float | None = None,
    quantity: int = 1,
    now: datetime | None = None,
) -> OptionStrategyAnalysis:
    """Analyze supported paper candidates; no brokerage or execution is possible."""
    strategy = _enum_value(strategy, OptionStrategy, "strategy")
    if not contracts or any(not isinstance(item, NormalizedOptionContract) for item in contracts):
        raise InvestmentDecisionValidationError("normalized option contracts are required")
    if isinstance(quantity, bool) or not isinstance(quantity, int) or quantity <= 0:
        raise InvestmentDecisionValidationError("quantity must be a positive integer")
    first = contracts[0]
    if any(item.underlying != first.underlying for item in contracts):
        raise InvestmentDecisionValidationError("option legs must share an underlying")
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    expiration = datetime.fromisoformat(first.expiration.replace("Z", "+00:00"))
    dte = max(0, (expiration.date() - current.date()).days)
    multiplier = first.multiplier
    if any(item.multiplier != multiplier or item.expiration != first.expiration for item in contracts):
        raise InvestmentDecisionValidationError("option legs must share expiration and multiplier")

    price = first.underlying_price if stock_price is None else _finite_number(
        stock_price, "stock_price"
    )
    if price <= 0:
        raise InvestmentDecisionValidationError("stock_price must be positive")
    # All values below are per position, including the contract multiplier.
    if strategy == OptionStrategy.LONG_CALL:
        call = _one_type(contracts, "CALL")
        premium = call.mid
        cost, slip = premium * multiplier, call.spread / 2 * multiplier
        result = (price + premium, math.inf, cost, price * multiplier, cost, slip)
    elif strategy == OptionStrategy.LONG_PUT:
        put = _one_type(contracts, "PUT")
        premium = put.mid
        cost, slip = premium * multiplier, put.spread / 2 * multiplier
        result = (put.strike - premium, max(0, put.strike - premium) * multiplier,
                  cost, price * multiplier, cost, slip)
    elif strategy == OptionStrategy.COVERED_CALL:
        call = _one_type(contracts, "CALL")
        premium = call.mid
        result = (price - premium, (call.strike - price + premium) * multiplier,
                  max(0, price - premium) * multiplier, price * multiplier,
                  (price - premium) * multiplier, call.spread / 2 * multiplier)
    elif strategy == OptionStrategy.CASH_SECURED_PUT:
        put = _one_type(contracts, "PUT")
        premium = put.mid
        result = (put.strike - premium, premium * multiplier,
                  max(0, put.strike - premium) * multiplier, put.strike * multiplier,
                  (put.strike - premium) * multiplier, put.spread / 2 * multiplier)
    elif strategy in {OptionStrategy.BULL_CALL_SPREAD, OptionStrategy.BEAR_PUT_SPREAD}:
        long_type = "CALL" if strategy == OptionStrategy.BULL_CALL_SPREAD else "PUT"
        long_leg, short_leg = _ordered_spread_legs(contracts, long_type, strategy)
        debit = long_leg.mid - short_leg.mid
        if debit <= 0:
            raise InvestmentDecisionValidationError("spread must have a positive net debit")
        width = abs(long_leg.strike - short_leg.strike)
        max_profit = max(0, width - debit) * multiplier
        breakeven = (long_leg.strike + debit if long_type == "CALL"
                     else long_leg.strike - debit)
        result = (breakeven, max_profit, debit * multiplier, width * multiplier,
                  debit * multiplier, (long_leg.spread + short_leg.spread) / 2 * multiplier)
    else:
        if strategy == OptionStrategy.PROTECTIVE_PUT:
            if len(contracts) != 1:
                raise InvestmentDecisionValidationError("protective put requires one put leg")
            put = _one_type(contracts, "PUT")
            net = put.mid
            result = (price + net, math.inf, max(0, price + net - put.strike) * multiplier,
                      price * multiplier, (price + net) * multiplier,
                      put.spread / 2 * multiplier)
        else:
            if len(contracts) != 2:
                raise InvestmentDecisionValidationError("collar requires one put and one call leg")
            put = _one_type(contracts, "PUT")
            call = _one_type(contracts, "CALL")
            net = put.mid - call.mid
            result = (price + net, max(0, call.strike - price - net) * multiplier,
                      max(0, price + net - put.strike) * multiplier, price * multiplier,
                      (price + net) * multiplier, (put.spread + call.spread) / 2 * multiplier)
    scaled = tuple(
        value if index == 0 else value * quantity
        for index, value in enumerate(result)
    )
    return OptionStrategyAnalysis(
        strategy, first.underlying, scaled[0], scaled[1], scaled[2], scaled[3],
        scaled[4], scaled[5], dte, quantity
    )


def review_defined_risk_option_candidates(
    candidates: Sequence[Mapping[str, Any]],
    *,
    now: datetime | None = None,
    max_data_age_seconds: float = 900,
) -> list[dict[str, Any]]:
    """Review quote-only option candidates without dropping rejected candidates.

    Each result is intentionally JSON-safe and contains either an analysis or a
    visible rejection. This is a presentation boundary, not an order boundary.
    """
    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
        raise InvestmentDecisionValidationError("option candidates must be a sequence")

    reviewed: list[dict[str, Any]] = []
    for candidate in candidates:
        result: dict[str, Any] = {
            "status": "REJECTED",
            "strategy": str(candidate.get("strategy", "UNKNOWN"))
            if isinstance(candidate, Mapping) else "UNKNOWN",
            "instrument": str(candidate.get("instrument", "UNKNOWN"))
            if isinstance(candidate, Mapping) else "UNKNOWN",
            "analysis": None,
            "rejection_reason": None,
        }
        try:
            if not isinstance(candidate, Mapping):
                raise InvestmentDecisionValidationError("candidate must be an object")
            strategy = _enum_value(candidate.get("strategy"), OptionStrategy, "strategy")
            raw_contracts = candidate.get("contracts")
            if not isinstance(raw_contracts, Sequence) or isinstance(
                raw_contracts, (str, bytes)
            ):
                raise InvestmentDecisionValidationError("candidate contracts are required")
            contracts = [
                normalize_option_contract(
                    contract, now=now, max_data_age_seconds=max_data_age_seconds
                )
                for contract in raw_contracts
            ]
            analysis = analyze_defined_risk_option_strategy(
                strategy,
                contracts=contracts,
                stock_price=candidate.get("stock_price"),
                quantity=candidate.get("quantity", 1),
                now=now,
            )
            result.update(
                status="ACCEPTED",
                strategy=analysis.strategy.value,
                instrument=str(candidate.get("instrument") or analysis.underlying),
                analysis=analysis.to_dict(),
            )
        except (InvestmentDecisionValidationError, TypeError, ValueError) as error:
            result["rejection_reason"] = str(error)
        reviewed.append(result)
    return reviewed


def fetch_public_option_quote_candidates(
    symbol: str,
    *,
    timeout: float = 10,
    opener=urlopen,
) -> dict[str, Any]:
    """Fetch quote-only option candidates from Yahoo Finance's public chain.

    The adapter intentionally returns raw legs to the review boundary. It does
    not fill missing values, synthesize timestamps, or discard malformed legs;
    ``review_defined_risk_option_candidates`` remains responsible for visibly
    rejecting unsafe data.
    """
    if not isinstance(symbol, str) or not symbol.strip():
        return {
            "available": False,
            "source": "Yahoo Finance public options chain",
            "symbol": str(symbol or ""),
            "candidates": [],
            "error": "An option symbol is required.",
            "fetched_at": _now_iso(),
        }
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
        return {
            "available": False,
            "source": "Yahoo Finance public options chain",
            "symbol": symbol.strip().upper(),
            "candidates": [],
            "error": "Quote provider timeout must be positive.",
            "fetched_at": _now_iso(),
        }

    normalized_symbol = symbol.strip().upper()
    url = (
        "https://query1.finance.yahoo.com/v7/finance/options/"
        f"{quote(normalized_symbol, safe='')}"
    )
    request = Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "Kova-read-only-options-review/1.0",
        },
    )
    try:
        with opener(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        result = payload["optionChain"]["result"][0]
        expiration_timestamp = result["expirationDates"][0]
        expiration = datetime.fromtimestamp(
            int(expiration_timestamp), tz=timezone.utc
        ).isoformat()
        underlying_price = result["quote"]["regularMarketPrice"]
        candidates = []
        for option_type, key in (("CALL", "calls"), ("PUT", "puts")):
            for leg in result.get("options", [{}])[0].get(key, []):
                # lastTradeDate is required: using fetch time would make stale
                # or missing market observations appear fresh.
                observed_timestamp = leg.get("lastTradeDate")
                observed_at = (
                    datetime.fromtimestamp(
                        int(observed_timestamp), tz=timezone.utc
                    ).isoformat()
                    if observed_timestamp is not None
                    else None
                )
                candidates.append(
                    {
                        "instrument": leg.get("contractSymbol")
                        or f"{normalized_symbol} {expiration} {option_type}",
                        "strategy": (
                            "LONG_CALL" if option_type == "CALL" else "LONG_PUT"
                        ),
                        "contracts": [
                            {
                                "underlying": normalized_symbol,
                                "option_type": option_type,
                                "strike": leg.get("strike"),
                                "expiration": expiration,
                                "bid": leg.get("bid"),
                                "ask": leg.get("ask"),
                                "underlying_price": underlying_price,
                                "multiplier": 100,
                                "observed_at": observed_at,
                            }
                        ],
                    }
                )
        if not candidates:
            raise PublicOptionQuoteProviderError(
                "Yahoo Finance returned no option legs for the nearest expiration."
            )
        return {
            "available": True,
            "source": "Yahoo Finance public options chain",
            "symbol": normalized_symbol,
            "expiration": expiration,
            "candidates": candidates,
            "error": None,
            "fetched_at": _now_iso(),
        }
    except (HTTPError, URLError, TimeoutError, OSError, KeyError, IndexError,
            TypeError, ValueError, AttributeError, json.JSONDecodeError) as error:
        message = (
            "Yahoo Finance option quotes are unavailable."
            if isinstance(error, (HTTPError, URLError, TimeoutError, OSError))
            else f"Yahoo Finance returned an invalid option chain: {error}"
        )
        return {
            "available": False,
            "source": "Yahoo Finance public options chain",
            "symbol": normalized_symbol,
            "candidates": [],
            "error": message,
            "fetched_at": _now_iso(),
        }
    except PublicOptionQuoteProviderError as error:
        return {
            "available": False,
            "source": "Yahoo Finance public options chain",
            "symbol": normalized_symbol,
            "candidates": [],
            "error": str(error),
            "fetched_at": _now_iso(),
        }


def evaluate_defined_risk_option_candidate(
    analysis: OptionStrategyAnalysis,
    *,
    instrument: str,
    thesis: str,
    proposed_decision: Decision | str = Decision.WAIT,
    risk_veto: Callable[[], tuple[bool, str]] | None = None,
    risk_approved: bool | None = None,
    **decision_fields: Any,
) -> InvestmentDecisionRecord:
    """Create a decision record from analysis, with Risk Governor final authority."""
    if not isinstance(analysis, OptionStrategyAnalysis):
        return _failure_record(
            instrument=str(instrument or "UNKNOWN"),
            underlying="UNKNOWN",
            asset_type=AssetType.DEFINED_RISK_OPTION_STRATEGY,
            strategy="UNKNOWN",
            direction=Direction.NEUTRAL,
            thesis=str(thesis or "No validated thesis available."),
            reason="A validated option strategy analysis is required.",
            data_quality=DataQuality.INVALID,
            observed_at=None,
        )
    fields = {
        "instrument": instrument,
        "underlying": analysis.underlying,
        "asset_type": AssetType.DEFINED_RISK_OPTION_STRATEGY,
        "strategy": analysis.strategy.value,
        "direction": Direction.LONG,
        "thesis": thesis,
        "proposed_decision": proposed_decision,
        "maximum_loss": analysis.maximum_loss,
        "estimated_transaction_cost": decision_fields.pop(
            "estimated_transaction_cost", analysis.cost
        ),
        "estimated_slippage": decision_fields.pop(
            "estimated_slippage", analysis.slippage
        ),
        "risk_veto": risk_veto,
        "risk_approved": risk_approved,
        **decision_fields,
    }
    return evaluate_investment_candidate(**fields)


def _one_type(contracts: Sequence[NormalizedOptionContract], option_type: str):
    legs = [item for item in contracts if item.option_type == option_type]
    if len(legs) != 1:
        raise InvestmentDecisionValidationError(
            f"{option_type} strategy requires exactly one {option_type} leg"
        )
    return legs[0]


def _ordered_spread_legs(contracts, option_type, strategy):
    legs = [item for item in contracts if item.option_type == option_type]
    if len(legs) != 2:
        raise InvestmentDecisionValidationError("vertical spread requires exactly two like-type legs")
    legs.sort(key=lambda item: item.strike)
    if strategy == OptionStrategy.BULL_CALL_SPREAD:
        return legs[0], legs[1]
    return legs[1], legs[0]


def _finite_number(value: Any, field: str, *, allow_none: bool = False) -> float | None:
    if value is None and allow_none:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvestmentDecisionValidationError(f"{field} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise InvestmentDecisionValidationError(f"{field} must be finite")
    return number


def _text(value: Any, field: str, *, allow_none: bool = False) -> str | None:
    if value is None and allow_none:
        return None
    if not isinstance(value, str) or not value.strip():
        raise InvestmentDecisionValidationError(f"{field} must be non-empty text")
    return value.strip()


def _enum_value(value: Any, enum_type: type[Enum], field: str):
    try:
        return value if isinstance(value, enum_type) else enum_type(value)
    except (TypeError, ValueError) as error:
        raise InvestmentDecisionValidationError(f"{field} is invalid") from error


@dataclass(frozen=True)
class InvestmentDecisionRecord:
    """Validated, serializable description of one investment decision."""

    instrument: str
    underlying: str
    asset_type: AssetType
    strategy: str
    direction: Direction
    thesis: str
    expected_return: float | None
    expected_risk: float | None
    maximum_loss: float | None
    liquidity: float | None
    estimated_transaction_cost: float | None
    estimated_slippage: float | None
    confidence: float | None
    time_horizon: str
    portfolio_impact: float | None
    concentration_impact: float | None
    correlation_impact: float | None
    market_regime_compatibility: str
    risk_score: float | None
    decision: Decision
    rejection_reason: str | None
    data_quality: DataQuality
    observed_at: str
    risk_approved: bool | None = None
    strategy_score: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "asset_type", _enum_value(self.asset_type, AssetType, "asset_type")
        )
        object.__setattr__(
            self, "direction", _enum_value(self.direction, Direction, "direction")
        )
        object.__setattr__(
            self, "decision", _enum_value(self.decision, Decision, "decision")
        )
        object.__setattr__(
            self,
            "data_quality",
            _enum_value(self.data_quality, DataQuality, "data_quality"),
        )

        for field in ("instrument", "underlying", "strategy", "thesis", "time_horizon",
                      "market_regime_compatibility", "observed_at"):
            _text(getattr(self, field), field)
        _text(self.rejection_reason, "rejection_reason", allow_none=True)

        for field in (
            "expected_return",
            "expected_risk",
            "maximum_loss",
            "liquidity",
            "estimated_transaction_cost",
            "estimated_slippage",
            "confidence",
            "portfolio_impact",
            "concentration_impact",
            "correlation_impact",
            "risk_score",
            "strategy_score",
        ):
            _finite_number(getattr(self, field), field, allow_none=True)

        if self.maximum_loss is not None and self.maximum_loss < 0:
            raise InvestmentDecisionValidationError("maximum_loss must be non-negative")
        for field in ("liquidity", "confidence", "risk_score"):
            value = getattr(self, field)
            if value is not None and not 0 <= value <= 100:
                raise InvestmentDecisionValidationError(f"{field} must be between 0 and 100")
        for field in ("estimated_transaction_cost", "estimated_slippage"):
            value = getattr(self, field)
            if value is not None and value < 0:
                raise InvestmentDecisionValidationError(f"{field} must be non-negative")

        try:
            datetime.fromisoformat(self.observed_at.replace("Z", "+00:00"))
        except ValueError as error:
            raise InvestmentDecisionValidationError(
                "observed_at must be an ISO-8601 timestamp"
            ) from error

        if self.decision == Decision.REJECT and not self.rejection_reason:
            raise InvestmentDecisionValidationError(
                "REJECT decisions require rejection_reason"
            )
        if self.decision != Decision.REJECT and self.rejection_reason:
            raise InvestmentDecisionValidationError(
                "rejection_reason is only valid for REJECT decisions"
            )
        if self.data_quality in {
            DataQuality.DATA_INSUFFICIENT,
            DataQuality.STALE,
            DataQuality.INVALID,
        } and self.decision in ACTIONABLE_DECISIONS:
            raise InvestmentDecisionValidationError(
                "degraded data cannot produce an actionable decision"
            )
        if (
            self.asset_type in OPTION_ASSET_TYPES
            and self.maximum_loss is None
            and not (
                self.decision == Decision.REJECT
                and self.data_quality != DataQuality.SUFFICIENT
            )
        ):
            raise InvestmentDecisionValidationError(
                "option decisions require a calculable maximum_loss"
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe record for read-only display or future evidence."""
        result = asdict(self)
        for field in ("asset_type", "direction", "decision", "data_quality"):
            result[field] = result[field].value
        return result


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_record_timestamp(value: Any) -> str:
    if isinstance(value, str) and value.strip():
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            if math.isfinite(float(value)):
                return datetime.fromtimestamp(
                    float(value), tz=timezone.utc
                ).isoformat()
        except (OverflowError, OSError, ValueError):
            pass
    return _now_iso()


def _failure_record(
    *,
    instrument: str,
    underlying: str,
    asset_type: AssetType,
    strategy: str,
    direction: Direction,
    thesis: str,
    reason: str,
    data_quality: DataQuality,
    observed_at: str | None,
    maximum_loss: float | None = None,
) -> InvestmentDecisionRecord:
    return InvestmentDecisionRecord(
        instrument=instrument or "UNKNOWN",
        underlying=underlying or "UNKNOWN",
        asset_type=asset_type,
        strategy=strategy or "UNKNOWN",
        direction=direction,
        thesis=thesis or "No validated thesis available.",
        expected_return=None,
        expected_risk=None,
        maximum_loss=(
            maximum_loss
            if maximum_loss is not None
            else (None if asset_type in OPTION_ASSET_TYPES else 0.0)
        ),
        liquidity=None,
        estimated_transaction_cost=None,
        estimated_slippage=None,
        confidence=None,
        time_horizon="UNKNOWN",
        portfolio_impact=None,
        concentration_impact=None,
        correlation_impact=None,
        market_regime_compatibility="UNKNOWN",
        risk_score=None,
        decision=Decision.REJECT,
        rejection_reason=reason,
        data_quality=data_quality,
        observed_at=_safe_record_timestamp(observed_at),
    )


def evaluate_investment_candidate(
    *,
    instrument: str,
    underlying: str | None = None,
    asset_type: AssetType | str,
    strategy: str,
    direction: Direction | str,
    thesis: str,
    proposed_decision: Decision | str = Decision.WAIT,
    expected_return: float | None = None,
    expected_risk: float | None = None,
    maximum_loss: float | None = None,
    liquidity: float | None = None,
    estimated_transaction_cost: float | None = None,
    estimated_slippage: float | None = None,
    confidence: float | None = None,
    time_horizon: str = "UNKNOWN",
    portfolio_impact: float | None = None,
    concentration_impact: float | None = None,
    correlation_impact: float | None = None,
    market_regime_compatibility: str = "UNKNOWN",
    risk_score: float | None = None,
    risk_approved: bool | None = None,
    strategy_score: float | None = None,
    rejection_reason: str | None = None,
    data_quality: DataQuality | str = DataQuality.SUFFICIENT,
    observed_at: str | None = None,
    now: datetime | None = None,
    max_data_age_seconds: float | None = None,
    risk_veto: Callable[[], tuple[bool, str]] | None = None,
) -> InvestmentDecisionRecord:
    """Evaluate a candidate and return a record; this function never executes."""
    raw_asset_type = asset_type
    try:
        resolved_asset_type = _enum_value(asset_type, AssetType, "asset_type")
        resolved_direction = _enum_value(direction, Direction, "direction")
        resolved_quality = _enum_value(data_quality, DataQuality, "data_quality")
        resolved_decision = _enum_value(
            proposed_decision, Decision, "proposed_decision"
        )
    except InvestmentDecisionValidationError as error:
        try:
            fallback_asset = AssetType(raw_asset_type)
        except (TypeError, ValueError):
            fallback_asset = AssetType.CASH
        return _failure_record(
            instrument=str(instrument or "UNKNOWN"),
            underlying=str(underlying or instrument or "UNKNOWN"),
            asset_type=fallback_asset,
            strategy=str(strategy or "UNKNOWN"),
            direction=Direction.NEUTRAL,
            thesis=str(thesis or "No validated thesis available."),
            reason=str(error),
            data_quality=DataQuality.INVALID,
            observed_at=observed_at,
        )

    resolved_underlying = underlying or instrument
    missing_fields = [
        field
        for field, value in {
            "instrument": instrument,
            "underlying": resolved_underlying,
            "strategy": strategy,
            "thesis": thesis,
            "time_horizon": time_horizon,
            "market_regime_compatibility": market_regime_compatibility,
            "observed_at": observed_at,
            "expected_return": expected_return,
            "expected_risk": expected_risk,
            "maximum_loss": maximum_loss,
            "liquidity": liquidity,
            "estimated_transaction_cost": estimated_transaction_cost,
            "estimated_slippage": estimated_slippage,
            "confidence": confidence,
            "portfolio_impact": portfolio_impact,
            "concentration_impact": concentration_impact,
            "correlation_impact": correlation_impact,
            "risk_score": risk_score,
        }.items()
        if value is None or (isinstance(value, str) and not value.strip())
    ]
    if resolved_asset_type == AssetType.CASH and maximum_loss is None:
        maximum_loss = 0.0
        missing_fields.remove("maximum_loss")

    try:
        for field, value in {
            "expected_return": expected_return,
            "expected_risk": expected_risk,
            "maximum_loss": maximum_loss,
            "liquidity": liquidity,
            "estimated_transaction_cost": estimated_transaction_cost,
            "estimated_slippage": estimated_slippage,
            "confidence": confidence,
            "portfolio_impact": portfolio_impact,
            "concentration_impact": concentration_impact,
            "correlation_impact": correlation_impact,
            "risk_score": risk_score,
            "strategy_score": strategy_score,
        }.items():
            _finite_number(value, field, allow_none=True)
        if maximum_loss is not None and maximum_loss < 0:
            raise InvestmentDecisionValidationError("maximum_loss must be non-negative")
        if any(
            value is not None and not 0 <= value <= 100
            for value in (liquidity, confidence, risk_score)
        ):
            raise InvestmentDecisionValidationError(
                "liquidity, confidence, and risk_score must be between 0 and 100"
            )
    except InvestmentDecisionValidationError as error:
        return _failure_record(
            instrument=instrument,
            underlying=resolved_underlying,
            asset_type=resolved_asset_type,
            strategy=strategy,
            direction=resolved_direction,
            thesis=thesis,
            reason=str(error),
            data_quality=DataQuality.INVALID,
            observed_at=observed_at,
        )

    timestamp = observed_at or _now_iso()
    if missing_fields:
        return _failure_record(
            instrument=instrument,
            underlying=resolved_underlying,
            asset_type=resolved_asset_type,
            strategy=strategy,
            direction=resolved_direction,
            thesis=thesis,
            reason=f"Required decision data is missing: {', '.join(missing_fields)}.",
            data_quality=DataQuality.DATA_INSUFFICIENT,
            observed_at=timestamp,
        )

    try:
        observed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except (AttributeError, TypeError, ValueError):
        return _failure_record(
            instrument=instrument,
            underlying=resolved_underlying,
            asset_type=resolved_asset_type,
            strategy=strategy,
            direction=resolved_direction,
            thesis=thesis,
            reason="observed_at must be an ISO-8601 timestamp.",
            data_quality=DataQuality.INVALID,
            observed_at=_now_iso(),
        )
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    age_limit = max_data_age_seconds
    if age_limit is not None and (
        isinstance(age_limit, bool)
        or not isinstance(age_limit, (int, float))
        or not math.isfinite(float(age_limit))
        or age_limit < 0
    ):
        return _failure_record(
            instrument=instrument,
            underlying=resolved_underlying,
            asset_type=resolved_asset_type,
            strategy=strategy,
            direction=resolved_direction,
            thesis=thesis,
            reason="Maximum decision-data age is invalid.",
            data_quality=DataQuality.INVALID,
            observed_at=timestamp,
        )
    if resolved_quality != DataQuality.SUFFICIENT:
        return _failure_record(
            instrument=instrument,
            underlying=resolved_underlying,
            asset_type=resolved_asset_type,
            strategy=strategy,
            direction=resolved_direction,
            thesis=thesis,
            reason=f"Decision data quality is {resolved_quality.value}.",
            data_quality=resolved_quality,
            observed_at=timestamp,
        )
    if age_limit is not None and (
        (current - observed).total_seconds() >= age_limit
    ):
        return _failure_record(
            instrument=instrument,
            underlying=resolved_underlying,
            asset_type=resolved_asset_type,
            strategy=strategy,
            direction=resolved_direction,
            thesis=thesis,
            reason="Critical decision data is stale.",
            data_quality=DataQuality.STALE,
            observed_at=timestamp,
        )

    if resolved_decision in ACTIONABLE_DECISIONS:
        if risk_veto is not None:
            try:
                approved, veto_reason = risk_veto()
            except Exception as error:
                return _failure_record(
                    instrument=instrument,
                    underlying=resolved_underlying,
                    asset_type=resolved_asset_type,
                    strategy=strategy,
                    direction=resolved_direction,
                    thesis=thesis,
                    reason=f"Risk Governor could not complete: {error}",
                    data_quality=DataQuality.INVALID,
                    observed_at=timestamp,
                )
            if not approved:
                return _failure_record(
                    instrument=instrument,
                    underlying=resolved_underlying,
                    asset_type=resolved_asset_type,
                    strategy=strategy,
                    direction=resolved_direction,
                    thesis=thesis,
                    reason=veto_reason or "Risk Governor rejected the candidate.",
                    data_quality=resolved_quality,
                    observed_at=timestamp,
                    maximum_loss=maximum_loss,
                )
        elif risk_approved is not True:
            return _failure_record(
                instrument=instrument,
                underlying=resolved_underlying,
                asset_type=resolved_asset_type,
                strategy=strategy,
                direction=resolved_direction,
                thesis=thesis,
                reason="Risk approval is required before an actionable decision.",
                data_quality=resolved_quality,
                observed_at=timestamp,
                maximum_loss=maximum_loss,
            )

    try:
        return InvestmentDecisionRecord(
            instrument=instrument,
            underlying=resolved_underlying,
            asset_type=resolved_asset_type,
            strategy=strategy,
            direction=resolved_direction,
            thesis=thesis,
            expected_return=expected_return,
            expected_risk=expected_risk,
            maximum_loss=maximum_loss,
            liquidity=liquidity,
            estimated_transaction_cost=estimated_transaction_cost,
            estimated_slippage=estimated_slippage,
            confidence=confidence,
            time_horizon=time_horizon,
            portfolio_impact=portfolio_impact,
            concentration_impact=concentration_impact,
            correlation_impact=correlation_impact,
            market_regime_compatibility=market_regime_compatibility,
            risk_score=risk_score,
            decision=resolved_decision,
            rejection_reason=(
                rejection_reason or "Candidate was rejected by the evaluator."
                if resolved_decision == Decision.REJECT
                else None
            ),
            data_quality=resolved_quality,
            observed_at=timestamp,
            risk_approved=risk_approved,
            strategy_score=strategy_score,
        )
    except InvestmentDecisionValidationError as error:
        return _failure_record(
            instrument=instrument,
            underlying=resolved_underlying,
            asset_type=resolved_asset_type,
            strategy=strategy,
            direction=resolved_direction,
            thesis=thesis,
            reason=str(error),
            data_quality=DataQuality.INVALID,
            observed_at=timestamp,
        )


def adapt_btc_cad_strategy_evaluation(
    evaluation: dict[str, Any],
    *,
    observed_at: str | None = None,
) -> InvestmentDecisionRecord:
    """Adapt existing BTC/CAD strategy output without executing or fabricating data."""
    if not isinstance(evaluation, dict):
        return _failure_record(
            instrument="BTC/CAD",
            underlying="BTC",
            asset_type=AssetType.CRYPTO,
            strategy="FROZEN BTC/CAD STRATEGY",
            direction=Direction.LONG,
            thesis="No strategy evaluation was available.",
            reason="Strategy evaluation must be an object.",
            data_quality=DataQuality.INVALID,
            observed_at=observed_at,
        )
    decision = (
        Decision.BUY
        if evaluation.get("decision") in {"BUY", "BUY CANDIDATE"}
        else Decision.WAIT
    )
    return evaluate_investment_candidate(
        instrument="BTC/CAD",
        underlying="BTC",
        asset_type=AssetType.CRYPTO,
        strategy="FROZEN BTC/CAD STRATEGY",
        direction=Direction.LONG,
        thesis="Existing strategy output adapted for read-only decision analysis.",
        proposed_decision=decision,
        expected_return=evaluation.get("expected_return"),
        expected_risk=evaluation.get("expected_risk"),
        maximum_loss=evaluation.get("maximum_loss"),
        liquidity=evaluation.get("liquidity"),
        estimated_transaction_cost=evaluation.get("estimated_transaction_cost"),
        estimated_slippage=evaluation.get("estimated_slippage"),
        confidence=evaluation.get("confidence"),
        time_horizon=evaluation.get("time_horizon"),
        portfolio_impact=evaluation.get("portfolio_impact"),
        concentration_impact=evaluation.get("concentration_impact"),
        correlation_impact=evaluation.get("correlation_impact"),
        market_regime_compatibility=evaluation.get("market_regime_compatibility"),
        risk_score=evaluation.get("risk_score"),
        strategy_score=evaluation.get("strategy_score"),
        risk_approved=False,
        data_quality=evaluation.get("data_quality", DataQuality.DATA_INSUFFICIENT),
        observed_at=_safe_record_timestamp(
            observed_at or evaluation.get("timestamp")
        ),
    )