"""Read-only research-provider adapters with honest, normalized status.

Providers are deliberately opt-in.  A provider is never reported as connected
unless its supported API is configured and a request has returned usable data.
No adapter scrapes pages, bypasses paywalls, or accepts credentials in code.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import urlencode
from urllib.request import Request, urlopen

UNCONFIGURED = "UNCONFIGURED"
CONFIGURED = "CONFIGURED"
AVAILABLE = "AVAILABLE"
STALE = "STALE"
FAILED = "FAILED"
PARTIAL = "PARTIAL"
CONTRACT_VALID = "VALID"
CONTRACT_INVALID = "INVALID"
CONTRACT_NOT_APPLICABLE = "NOT_APPLICABLE"

RESEARCH_DOMAINS = (
    "fundamental",
    "valuation",
    "quality",
    "growth",
    "value",
    "momentum",
    "trend",
    "dividend",
    "news_event",
    "global_opportunity",
    "manipulation_fraud",
)


@dataclass(frozen=True)
class ProviderSpec:
    name: str
    api_name: str
    credential_env: str
    domains: tuple[str, ...]
    endpoint: str
    notes: str


PROVIDER_SPECS = (
    ProviderSpec(
        "SEC EDGAR",
        "sec_edgar",
        "SEC_API_USER_AGENT",
        ("fundamental", "quality", "growth", "value", "dividend", "manipulation_fraud"),
        "https://data.sec.gov/submissions/CIK0000320193.json",
        "Public SEC API; a descriptive User-Agent is required.",
    ),
    ProviderSpec(
        "Federal Reserve FRED",
        "fred",
        "FRED_API_KEY",
        ("global_opportunity", "trend"),
        "https://api.stlouisfed.org/fred/series/observations",
        "Official FRED API; an API key is required.",
    ),
    ProviderSpec(
        "Finnhub",
        "finnhub",
        "FINNHUB_API_KEY",
        ("news_event", "global_opportunity"),
        "https://finnhub.io/api/v1/company-news",
        "Licensed API access; an API key is required.",
    ),
    ProviderSpec(
        "Alpha Vantage",
        "alpha_vantage",
        "ALPHA_VANTAGE_API_KEY",
        ("valuation", "momentum", "trend", "dividend"),
        "https://www.alphavantage.co/query",
        "Official API access; an API key is required.",
    ),
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _base_result(spec: ProviderSpec) -> dict:
    configured = bool(os.getenv(spec.credential_env, "").strip())
    return {
        "provider": spec.name,
        "api": spec.api_name,
        "domains": list(spec.domains),
        "source": spec.endpoint,
        "fetched_at": None,
        "freshness_seconds": None,
        "quality": "unknown",
        "uncertainty": "not assessed",
        "contract_status": CONTRACT_NOT_APPLICABLE,
        # Configuration is not connectivity: no request has succeeded yet.
        "status": CONFIGURED if configured else UNCONFIGURED,
        "configured": configured,
        "partial": False,
        "error": None if configured else f"missing {spec.credential_env}",
        "notes": spec.notes,
        "data": None,
    }


def provider_payload_contract(spec: ProviderSpec, data) -> tuple[bool, str]:
    """Validate only the stable envelope needed by each approved adapter.

    This deliberately returns a reason, not the payload.  Provider responses
    can contain credentials, account identifiers, or other sensitive values.
    Error/throttle envelopes are recognized separately so they remain
    unavailable rather than being mistaken for schema drift.
    """
    if not isinstance(data, (dict, list)) or data in ({}, []):
        return False, "empty or non-JSON provider payload"
    if isinstance(data, dict) and any(
        key in data for key in ("error", "Error Message", "Note", "error_code")
    ):
        return False, "provider error or throttling envelope"
    if spec.api_name == "sec_edgar":
        valid = isinstance(data.get("filings"), dict) and isinstance(
            data.get("name"), str
        )
    elif spec.api_name == "fred":
        valid = isinstance(data.get("observations"), list)
    elif spec.api_name == "finnhub":
        valid = isinstance(data, list) and all(isinstance(item, dict) for item in data)
    elif spec.api_name == "alpha_vantage":
        valid = isinstance(data, dict) and any(
            key.startswith("Time Series") or key in {"data", "monthly", "annualReports"}
            for key in data
        )
    else:
        valid = False
    return (True, "expected provider envelope") if valid else (
        False,
        "provider envelope does not match approved adapter contract",
    )


def provider_catalog() -> list[dict]:
    """Return provider status without making network calls or exposing secrets."""
    return [_base_result(spec) for spec in PROVIDER_SPECS]


def normalize_result(spec: ProviderSpec, data, *, fetched_at=None, now=None) -> dict:
    """Attach common provenance and freshness fields to a provider payload."""
    result = _base_result(spec)
    result["data"] = data
    contract_valid, contract_reason = provider_payload_contract(spec, data)
    result["contract_status"] = (
        CONTRACT_VALID if contract_valid else CONTRACT_INVALID
    )
    result["fetched_at"] = fetched_at or _utc_now()
    try:
        stamp = datetime.fromisoformat(result["fetched_at"].replace("Z", "+00:00"))
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=timezone.utc)
        current = time.time() if now is None else float(now)
        age = max(0.0, current - stamp.timestamp())
        result["freshness_seconds"] = round(age, 1)
        result["status"] = STALE if age > 86400 else AVAILABLE
    except (TypeError, ValueError, AttributeError):
        result["status"] = PARTIAL
        result["uncertainty"] = "invalid provider timestamp"
    provider_error = (
        isinstance(data, dict)
        and any(key in data for key in ("error", "Error Message", "Note"))
    )
    if provider_error:
        result["status"] = PARTIAL
        result["partial"] = True
        result["quality"] = "provider error or throttling response"
        result["uncertainty"] = "provider did not return the requested dataset"
        result["contract_status"] = CONTRACT_INVALID
    elif data is None or data == {} or data == []:
        result["status"] = PARTIAL
        result["quality"] = "insufficient"
        result["uncertainty"] = "provider returned no usable data"
    elif not contract_valid:
        result["status"] = PARTIAL
        result["partial"] = True
        result["quality"] = "provider contract mismatch"
        result["uncertainty"] = contract_reason
    else:
        result["quality"] = "provider response received"
        result["uncertainty"] = "provider-specific; not independently validated"
    return result


def run_contract_checks() -> list[dict]:
    """Run deterministic adapter-contract checks without network or secrets."""
    fixtures = {
        "sec_edgar": {"name": "fixture", "filings": {"recent": {}}},
        "fred": {"observations": [{"date": "2026-08-24", "value": "1.2"}]},
        "finnhub": [{"headline": "fixture", "datetime": 1787529600}],
        "alpha_vantage": {"Time Series (Daily)": {"2026-08-24": {"4. close": "1"}}},
    }
    checks = []
    for spec in PROVIDER_SPECS:
        valid, reason = provider_payload_contract(spec, fixtures[spec.api_name])
        checks.append({"provider": spec.api_name, "ok": valid, "reason": reason})
    return checks


def _run_contract_check_command() -> int:
    checks = run_contract_checks()
    failed = [check for check in checks if not check["ok"]]
    if failed:
        for check in failed:
            print(f"CONTRACT_DRIFT provider={check['provider']} reason={check['reason']}")
        return 1
    print(f"RESEARCH_PROVIDER_CONTRACTS_OK providers={len(checks)}")
    return 0


def _request_json(spec: ProviderSpec, params: dict | None = None, *, timeout=10) -> dict:
    headers = {"Accept": "application/json"}
    if spec.api_name == "sec_edgar":
        headers["User-Agent"] = os.environ[spec.credential_env]
    query = urlencode(params or {})
    url = f"{spec.endpoint}?{query}" if query else spec.endpoint
    with urlopen(Request(url, headers=headers), timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_provider(provider: str, *, params: dict | None = None, timeout=10) -> dict:
    """Fetch one approved provider without falling back to synthetic data."""
    spec = next((item for item in PROVIDER_SPECS if item.api_name == provider), None)
    if spec is None:
        raise ValueError(f"unsupported research provider: {provider}")
    result = _base_result(spec)
    if not result["configured"]:
        return result
    try:
        payload = _request_json(spec, params, timeout=timeout)
        return normalize_result(spec, payload)
    except Exception as exc:  # network/provider errors must be visible, not fatal
        result["status"] = FAILED
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["uncertainty"] = "provider request failed"
        return result


def research_readiness(results: list[dict] | None = None) -> dict:
    """Fail closed for portfolio scoring and strategy-council consumers."""
    results = provider_catalog() if results is None else list(results)
    covered = {domain for item in results if item.get("status") == AVAILABLE
               and item.get("contract_status") in (CONTRACT_VALID, CONTRACT_NOT_APPLICABLE)
               for domain in item.get("domains", [])}
    missing = [domain for domain in RESEARCH_DOMAINS if domain not in covered]
    return {
        "ready": not missing,
        "critical": True,
        "missing_domains": missing,
        "reason": None if not missing else "critical research inputs are unavailable",
    }


if __name__ == "__main__":
    if sys.argv[1:] != ["--contract-check"]:
        raise SystemExit("usage: python -m research_providers --contract-check")
    raise SystemExit(_run_contract_check_command())