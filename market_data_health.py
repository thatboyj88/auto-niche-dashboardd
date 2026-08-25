"""Read-only market-data integrity and freshness diagnostics."""

from __future__ import annotations

import math
import time


HEALTHY = "HEALTHY"
DEGRADED = "DEGRADED"
UNAVAILABLE = "UNAVAILABLE"

_REQUIRED_FIELDS = ("timestamp", "open", "high", "low", "close", "volume")


def normalize_source_metadata(
    *,
    source,
    fetched_at=None,
    freshness_seconds=None,
    quality="unknown",
    uncertainty="not assessed",
    status=UNAVAILABLE,
    error=None,
):
    """Common provenance shape shared by market and research source displays."""
    return {
        "source": source,
        "fetched_at": fetched_at,
        "freshness_seconds": freshness_seconds,
        "quality": quality,
        "uncertainty": uncertainty,
        "status": status,
        "error": error,
    }


def inspect_candles(
    candles,
    *,
    interval_minutes=60,
    now_timestamp=None,
    stale_after_seconds=None,
    provider_available=True,
    provider_error=None,
):
    """Return a JSON-safe health snapshot without changing candle data."""
    candles = list(candles or [])
    interval_seconds = max(float(interval_minutes), 1.0) * 60
    stale_after = (
        float(stale_after_seconds)
        if stale_after_seconds is not None
        else interval_seconds * 2
    )
    now = time.time() if now_timestamp is None else float(now_timestamp)
    issues = []
    timestamps = []

    if not provider_available or provider_error:
        issues.append("provider unavailable")

    for index, candle in enumerate(candles):
        if not isinstance(candle, dict) or not all(
            field in candle for field in _REQUIRED_FIELDS
        ):
            issues.append(f"candle {index + 1} missing required fields")
            continue

        try:
            values = {
                field: float(candle[field])
                for field in _REQUIRED_FIELDS
            }
        except (TypeError, ValueError):
            issues.append(f"candle {index + 1} has non-numeric values")
            continue

        if not all(math.isfinite(value) for value in values.values()):
            issues.append(f"candle {index + 1} has non-finite values")
            continue

        timestamp = int(values["timestamp"])
        timestamps.append(timestamp)
        if timestamp <= 0:
            issues.append(f"candle {index + 1} has an invalid timestamp")
        if min(
            values["open"],
            values["high"],
            values["low"],
            values["close"],
            values["volume"],
        ) <= 0:
            issues.append(f"candle {index + 1} has non-positive OHLCV")
        if values["high"] < max(values["open"], values["close"]):
            issues.append(f"candle {index + 1} has an invalid high")
        if values["low"] > min(values["open"], values["close"]):
            issues.append(f"candle {index + 1} has an invalid low")
        if values["high"] < values["low"]:
            issues.append(f"candle {index + 1} has high below low")

    if len(timestamps) != len(set(timestamps)):
        issues.append("duplicate timestamps")

    ordered_timestamps = timestamps
    for previous, current in zip(
        ordered_timestamps,
        ordered_timestamps[1:],
    ):
        if current <= previous:
            issues.append("timestamp regression")
        elif current - previous > interval_seconds * 1.5:
            issues.append("excessive timestamp gap")

    latest_timestamp = max(timestamps) if timestamps else None
    data_age_seconds = (
        max(0.0, now - latest_timestamp)
        if latest_timestamp is not None
        else None
    )
    if latest_timestamp is not None and data_age_seconds > stale_after:
        issues.append("stale market data")

    if not candles or not timestamps or not provider_available:
        status = UNAVAILABLE
    elif issues:
        status = DEGRADED
    else:
        status = HEALTHY

    return {
        "status": status,
        "issues": list(dict.fromkeys(issues)),
        "candle_count": len(candles),
        "latest_timestamp": latest_timestamp,
        "data_age_seconds": (
            round(data_age_seconds, 1)
            if data_age_seconds is not None
            else None
        ),
        "interval_minutes": interval_minutes,
        "last_known_good_timestamp": (
            latest_timestamp if status == HEALTHY else None
        ),
        "provider_error": provider_error,
    }