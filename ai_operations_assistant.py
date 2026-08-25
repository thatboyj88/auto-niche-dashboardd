"""Read-only AI Operations Assistant boundary.

This module deliberately accepts application results as input and returns only
small JSON-safe summaries.  It has no imports from trading or execution code
and exposes no mutation tools.
"""

from __future__ import annotations

import json
import os
import re
import socket
import time
from dataclasses import dataclass, field
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

UNKNOWN = "UNKNOWN"
ASSISTANT_NAME = "Kova"
UNAVAILABLE_MESSAGE = (
    "ASSISTANT UNAVAILABLE\n\n"
    "The configured AI provider is not available right now. "
    "The trading dashboard and paper systems are unaffected."
)
REFUSAL_MESSAGE = (
    "REFUSED\n\n"
    "I am a read-only operations assistant. I cannot place trades, enable "
    "live trading, change configuration, run experiments, edit files, run "
    "shell commands, access credentials, or perform any other mutation."
)

_MUTATION_PATTERN = re.compile(
    r"\b(place|execute|submit|buy|sell|trade|enable|disable|change|edit|"
    r"modify|update|set|run|launch|start|stop|delete|remove|write|shell|"
    r"command|credential|password|secret|api key|wallet|experiment)\b",
    re.IGNORECASE,
)


def _value(value):
    if value is None:
        return UNKNOWN
    if isinstance(value, (str, int, float, bool)):
        return value
    return UNKNOWN


def _serializable(value):
    if value is None:
        return UNKNOWN
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serializable(item) for item in value]
    return UNKNOWN


def get_status_context(results, live_candles, historical_results):
    return {
        "strategy": "READY" if results else UNKNOWN,
        "paper_trading": "ENABLED" if results is not None else UNKNOWN,
        "live_trading": "DISABLED",
        "live_market_data": "AVAILABLE" if live_candles else "UNAVAILABLE",
        "historical_research": (
            "AVAILABLE" if historical_results is not None else UNKNOWN
        ),
    }


def get_market_context(market_data, candles):
    latest = candles[-1] if candles else None
    previous = candles[-2] if len(candles) > 1 else None
    return {
        "source": _value(getattr(market_data, "pair_name", None) or "Kraken XBT/CAD"),
        "exchange": "Kraken",
        "timeframe": "60 minutes",
        "candles_loaded": len(candles) if candles else 0,
        "latest_timestamp": _value(latest.get("timestamp") if latest else None),
        "latest_close": _value(latest.get("close") if latest else None),
        "previous_close": _value(previous.get("close") if previous else None),
        "last_error": _value(getattr(market_data, "last_error", None)),
    }


def get_strategy_context(latest_evaluation, results):
    evaluation = latest_evaluation or {}
    return {
        "latest_score": _value(evaluation.get("strategy_score")),
        "latest_decision": _value(evaluation.get("decision")),
        "latest_rsi": _value(evaluation.get("rsi")),
        "evaluations": _value(results.get("evaluations") if results else None),
        "highest_score": _value(results.get("highest_score") if results else None),
        "score_at_least_80": _value(
            results.get("score_80_or_more") if results else None
        ),
    }


def get_performance_context(results, historical_results):
    aggregate = historical_results.get("aggregate", {}) if historical_results else {}
    return {
        "paper_starting_capital": _value(
            results.get("starting_capital") if results else None
        ),
        "paper_ending_capital": _value(
            results.get("ending_capital") if results else None
        ),
        "paper_profit": _value(results.get("profit") if results else None),
        "paper_return_percent": _value(
            (
                results["profit"] / results["starting_capital"] * 100
                if results and results.get("starting_capital")
                else None
            )
        ),
        "trades": _value(results.get("trades") if results else None),
        "win_rate": _value(results.get("win_rate") if results else None),
        "max_drawdown": _value(results.get("max_drawdown") if results else None),
        "historical_periods": (
            len(historical_results.get("periods", []))
            if historical_results
            else UNKNOWN
        ),
        "historical_average_return": _value(aggregate.get("average_return")),
        "historical_worst_drawdown": _value(aggregate.get("worst_drawdown")),
    }


def get_position_context(account_status=None):
    if not account_status:
        return {"status": UNKNOWN, "source": "No account snapshot supplied"}
    return _serializable(account_status)


def get_risk_context():
    from config import (
        MAX_DAILY_LOSS_PERCENT,
        MAX_POSITION_PERCENT,
        MAX_TRADES_PER_DAY,
        STOP_LOSS_PERCENT,
        TAKE_PROFIT_PERCENT,
    )

    return {
        "max_position_percent": MAX_POSITION_PERCENT,
        "max_daily_loss_percent": MAX_DAILY_LOSS_PERCENT,
        "max_trades_per_day": MAX_TRADES_PER_DAY,
        "stop_loss_percent": STOP_LOSS_PERCENT,
        "take_profit_percent": TAKE_PROFIT_PERCENT,
    }


def get_configuration_context():
    from config import LIVE_TRADING, PAPER_TRADING, STARTING_CAPITAL

    return {
        "starting_capital": STARTING_CAPITAL,
        "paper_trading": PAPER_TRADING,
        "live_trading": LIVE_TRADING,
        "scope": "READ_ONLY",
    }


def get_research_context(historical_results):
    if not historical_results:
        return {"status": UNKNOWN, "regimes": UNKNOWN}
    regimes = {}
    for regime in ("Bull", "Bear", "Sideways"):
        periods = historical_results.get("regime_summary", {}).get(regime, [])
        regimes[regime] = {
            "completed_periods": len(periods),
            "average_market_return": (
                sum(p["market_return"] for p in periods) / len(periods)
                if periods
                else UNKNOWN
            ),
            "average_strategy_return": (
                sum(p["return_percent"] for p in periods) / len(periods)
                if periods
                else UNKNOWN
            ),
        }
    return {
        "status": "AVAILABLE",
        "completed_periods": len(historical_results.get("periods", [])),
        "regimes": regimes,
        "method": "independent, non-compounded paper periods",
    }


def build_assistant_context(
    results,
    latest_evaluation,
    market_data,
    live_candles,
    historical_results,
    account_status=None,
):
    context = {
        "status": get_status_context(results, live_candles, historical_results),
        "market": get_market_context(market_data, live_candles),
        "strategy": get_strategy_context(latest_evaluation, results),
        "position": get_position_context(account_status),
        "performance": get_performance_context(results, historical_results),
        "risk": get_risk_context(),
        "configuration": get_configuration_context(),
        "research": get_research_context(historical_results),
    }
    return _serializable(context)


class ProviderError(RuntimeError):
    """Raised when a provider cannot return a safe answer."""

    def __init__(self, message, category="provider_error"):
        super().__init__(message)
        self.category = category


FAILURE_CATEGORY_LABELS = {
    "not_configured": "Not configured",
    "timeout": "Timeout",
    "network_error": "Network error",
    "response_validation": "Response validation",
    "rate_limit": "Rate limit",
    "provider_outage": "Provider outage",
    "provider_http_error": "Provider HTTP error",
    "provider_error": "Provider error",
}


def format_failure_category(category):
    """Return a concise, safe label for provider health telemetry."""
    if not category or category == UNKNOWN:
        return UNKNOWN
    return FAILURE_CATEGORY_LABELS.get(category, "Provider error")


def format_failure_category_counts(categories):
    """Format aggregate category counts without exposing provider details."""
    if not categories:
        return UNKNOWN
    return " · ".join(
        f"{format_failure_category(category)}: {count}"
        for category, count in categories.items()
        if isinstance(count, int) and count >= 0
    ) or UNKNOWN


@dataclass
class ProviderHealth:
    """Aggregate, prompt-free health telemetry for one provider instance."""

    provider: str
    requests: int = 0
    successes: int = 0
    failures: int = 0
    last_latency_ms: float | None = None
    last_outcome: str = "NOT CALLED"
    last_failure_category: str | None = None
    _failure_categories: dict = field(default_factory=dict, repr=False)

    def record_success(self, latency_ms):
        self.requests += 1
        self.successes += 1
        self.last_latency_ms = round(latency_ms, 1)
        self.last_outcome = "SUCCESS"
        self.last_failure_category = None

    def record_failure(self, category, latency_ms):
        self.requests += 1
        self.failures += 1
        self.last_latency_ms = round(latency_ms, 1)
        self.last_outcome = "FAILURE"
        self.last_failure_category = category
        self._failure_categories[category] = (
            self._failure_categories.get(category, 0) + 1
        )

    def snapshot(self):
        if not self.requests:
            availability = "NOT CALLED"
        elif self.failures == 0:
            availability = "HEALTHY"
        elif self.successes:
            availability = "DEGRADED"
        else:
            availability = "UNAVAILABLE"
        return {
            "provider": self.provider,
            "availability": availability,
            "requests": self.requests,
            "successes": self.successes,
            "failures": self.failures,
            "success_rate_percent": round(
                self.successes / self.requests * 100, 1
            ) if self.requests else UNKNOWN,
            "last_latency_ms": (
                self.last_latency_ms
                if self.last_latency_ms is not None
                else UNKNOWN
            ),
            "last_outcome": self.last_outcome,
            "last_failure_category": (
                self.last_failure_category or UNKNOWN
            ),
            "failure_categories": dict(self._failure_categories),
        }


_MANAGED_PROVIDER_HEALTH = ProviderHealth("Managed OpenAI")
_LOCAL_PROVIDER_HEALTH = ProviderHealth("Local read-only fallback")


class UnavailableProvider:
    health = ProviderHealth("Unavailable provider")

    def answer(self, question, context, history):
        self.health.record_failure("not_configured", 0)
        raise ProviderError("No AI provider is configured", "not_configured")


class ReadOnlySummaryProvider:
    """Deterministic local provider for safe status and summary questions."""

    def __init__(self, health=None):
        self.health = health or ProviderHealth("Local read-only fallback")

    def answer(self, question, context, history):
        started = time.monotonic()
        lowered = question.lower()
        if "status" in lowered or "mode" in lowered:
            answer = (
                "FACT\n\n"
                f"Paper trading: {context['status']['paper_trading']}\n"
                f"Live trading: {context['status']['live_trading']}\n"
                f"Market data: {context['status']['live_market_data']}"
            )
        elif "performance" in lowered or "return" in lowered or "profit" in lowered:
            performance = context["performance"]
            answer = (
                "FACT\n\n"
                f"Paper profit: {performance['paper_profit']}\n"
                f"Paper return: {performance['paper_return_percent']}%\n"
                f"Win rate: {performance['win_rate']}%"
            )
        elif "risk" in lowered or "limit" in lowered:
            risk = context["risk"]
            answer = (
                "FACT\n\n"
                f"Position limit: {risk['max_position_percent']:.0%}\n"
                f"Daily loss limit: {risk['max_daily_loss_percent']:.0%}\n"
                f"Daily trade limit: {risk['max_trades_per_day']}"
            )
        elif "research" in lowered or "evidence" in lowered:
            research = context["research"]
            answer = (
                "FACT\n\n"
                f"Research status: {research['status']}\n"
                f"Completed periods: {research.get('completed_periods', UNKNOWN)}"
            )
        elif "market" in lowered or "price" in lowered:
            market = context["market"]
            answer = (
                "FACT\n\n"
                f"Source: {market['source']}\n"
                f"Latest close: {market['latest_close']}\n"
                f"Timestamp: {market['latest_timestamp']}"
            )
        else:
            answer = "UNKNOWN\n\nThe approved dashboard context does not answer that question."
        self.health.record_success((time.monotonic() - started) * 1000)
        return answer


@dataclass
class OpenAICompatibleProvider:
    endpoint: str
    api_key: str
    model: str = "gpt-5.6-luna"
    health: ProviderHealth = field(
        default_factory=lambda: ProviderHealth("OpenAI-compatible provider")
    )

    def answer(self, question, context, history):
        system = (
            "You are Kova, a concise read-only trading operations assistant. "
            "Recognize Kova as your assistant name when the user addresses "
            "you directly. "
            "Use only the approved JSON context below. Label every response "
            "with exactly one of FACT, ANALYSIS, or UNKNOWN as its first line. "
            "FACT states values present in the context. ANALYSIS is a clearly "
            "qualified inference from those values. Use UNKNOWN when the "
            "context does not answer the question. Never invent missing "
            "values, request credentials, or perform actions. You are "
            "read-only: do not place trades, change configuration, run code, "
            "or modify files."
        )
        messages = [{"role": "system", "content": system}]
        messages.extend(
            message for message in history[-8:]
            if isinstance(message, dict)
            and message.get("role") in {"user", "assistant"}
            and isinstance(message.get("content"), str)
        )
        messages.append({
            "role": "user",
            "content": f"Context:\n{json.dumps(context)}\n\nQuestion: {question}",
        })
        request = Request(
            self.endpoint,
            data=json.dumps({
                "model": self.model,
                "messages": messages,
                "max_tokens": 500,
            }).encode(),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        started = time.monotonic()
        try:
            with urlopen(request, timeout=15) as response:
                payload = json.loads(response.read())
            answer = payload["choices"][0]["message"]["content"].strip()
            if not re.match(r"^(FACT|ANALYSIS|UNKNOWN)\b", answer):
                raise ProviderError(
                    "Provider response was not grounded",
                    "response_validation",
                )
            self.health.record_success((time.monotonic() - started) * 1000)
            return answer
        except ProviderError as error:
            self.health.record_failure(
                error.category,
                (time.monotonic() - started) * 1000,
            )
            raise
        except (TimeoutError, socket.timeout) as error:
            self.health.record_failure(
                "timeout",
                (time.monotonic() - started) * 1000,
            )
            raise ProviderError(str(error), "timeout") from error
        except HTTPError as error:
            category = (
                "rate_limit"
                if error.code == 429
                else "provider_outage"
                if 500 <= error.code <= 599
                else "provider_http_error"
            )
            self.health.record_failure(
                category,
                (time.monotonic() - started) * 1000,
            )
            raise ProviderError(str(error), category) from error
        except URLError as error:
            category = (
                "timeout"
                if isinstance(error.reason, (TimeoutError, socket.timeout))
                else "network_error"
            )
            self.health.record_failure(
                category,
                (time.monotonic() - started) * 1000,
            )
            raise ProviderError(str(error), category) from error
        except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as error:
            self.health.record_failure(
                "response_validation",
                (time.monotonic() - started) * 1000,
            )
            raise ProviderError(str(error), "response_validation") from error
        except Exception as error:
            self.health.record_failure(
                "provider_error",
                (time.monotonic() - started) * 1000,
            )
            raise ProviderError(str(error), "provider_error") from error


def get_default_provider():
    # Replit's managed integration injects these values without putting the
    # credential in source code. Keep the explicit endpoint variables as a
    # backwards-compatible option for private OpenAI-compatible providers.
    endpoint = (
        os.getenv("AI_INTEGRATIONS_OPENAI_BASE_URL")
        or os.getenv("AI_OPERATIONS_ASSISTANT_ENDPOINT")
    )
    api_key = (
        os.getenv("AI_INTEGRATIONS_OPENAI_API_KEY")
        or os.getenv("AI_OPERATIONS_ASSISTANT_API_KEY")
    )
    if endpoint and api_key:
        if not endpoint.rstrip("/").endswith("/chat/completions"):
            endpoint = endpoint.rstrip("/") + "/chat/completions"
        return OpenAICompatibleProvider(
            endpoint,
            api_key,
            model=os.getenv("AI_OPERATIONS_ASSISTANT_MODEL", "gpt-5.6-luna"),
            health=_MANAGED_PROVIDER_HEALTH,
        )
    return ReadOnlySummaryProvider(health=_LOCAL_PROVIDER_HEALTH)


def get_provider_health():
    """Return aggregate provider telemetry without prompts or context."""
    return get_default_provider().health.snapshot()


def answer_question(question, context, history=None, provider=None):
    question = (question or "").strip()
    if not question:
        return "UNKNOWN\n\nPlease enter an operations question."
    if _MUTATION_PATTERN.search(question):
        return REFUSAL_MESSAGE
    provider = provider or get_default_provider()
    try:
        answer = provider.answer(question, context, history or [])
    except Exception:
        return UNAVAILABLE_MESSAGE
    return answer or "UNKNOWN\n\nNo grounded answer is available."