"""Validate one fixed Yahoo Finance BTC/CAD period before using its evidence.

This module is deliberately data-only.  It neither imports nor changes strategy,
risk, paper-trading, Kraken, wallet, or order-execution code.
"""

import argparse
import json
import math
import os
import sys
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from yahoo_btc_cad_data import YahooBTCADMarketData


REQUESTED_CANDLE_COUNT = 365
NOTIFICATION_WEBHOOK_ENV = "BTC_CAD_PREFLIGHT_SLACK_WEBHOOK_URL"
SLACK_WEBHOOK_HOST = "hooks.slack.com"
SLACK_TIMEOUT_SECONDS = 10
REQUIRED_OHLCV_FIELDS = ("open", "high", "low", "close", "volume")


class BTCADPreflightError(RuntimeError):
    """Raised when the public BTC/CAD evidence cannot be validated."""


def _coerce_date(value):
    if hasattr(value, "year") and hasattr(value, "month"):
        return value
    return datetime.strptime(value, "%Y-%m-%d").date()


def _requested_dates(start_date=None):
    start_date = _coerce_date(
        start_date or YahooBTCADMarketData.ANCHORED_SAMPLE_START
    )
    return [
        start_date + timedelta(days=offset)
        for offset in range(REQUESTED_CANDLE_COUNT)
    ]


def _first_candle_date(candles):
    if not isinstance(candles, list):
        return None, "Yahoo Finance returned a non-list candle response"
    if not candles:
        return None, "Yahoo Finance returned no candles"

    first_candle = candles[0]
    if not isinstance(first_candle, dict):
        return None, "Candle 1 is not an OHLCV object"
    if "timestamp" not in first_candle:
        return None, "Candle 1 is missing timestamp"

    timestamp = first_candle["timestamp"]
    if type(timestamp) is not int:
        return None, "Candle 1 has a non-integer timestamp"

    try:
        return (
            datetime.fromtimestamp(timestamp, tz=timezone.utc).date(),
            None,
        )
    except (OverflowError, OSError, ValueError):
        return None, "Candle 1 has an invalid timestamp"


def _validate_candles(candles, expected_start_date=None):
    """Return an error message when the requested Yahoo OHLCV period is invalid."""
    if not isinstance(candles, list):
        return "Yahoo Finance returned a non-list candle response"

    if len(candles) != REQUESTED_CANDLE_COUNT:
        return (
            "Yahoo Finance returned "
            f"{len(candles)} candles; the requested period requires exactly "
            f"{REQUESTED_CANDLE_COUNT}"
        )

    expected_dates = _requested_dates(expected_start_date)
    previous_timestamp = None

    for index, candle in enumerate(candles):
        candle_number = index + 1
        if not isinstance(candle, dict):
            return f"Candle {candle_number} is not an OHLCV object"

        if "timestamp" not in candle:
            return f"Candle {candle_number} is missing timestamp"

        timestamp = candle["timestamp"]
        if type(timestamp) is not int:
            return f"Candle {candle_number} has a non-integer timestamp"
        if previous_timestamp is not None and timestamp <= previous_timestamp:
            return (
                f"Candle timestamps are not chronological and unique at "
                f"candle {candle_number}"
            )

        try:
            candle_date = datetime.fromtimestamp(
                timestamp,
                tz=timezone.utc,
            ).date()
        except (OverflowError, OSError, ValueError):
            return f"Candle {candle_number} has an invalid timestamp"

        if candle_date != expected_dates[index]:
            return (
                "Requested 365-candle period is incomplete or shifted at "
                f"candle {candle_number}: expected "
                f"{expected_dates[index].isoformat()}, received "
                f"{candle_date.isoformat()}"
            )

        values = {}
        for field in REQUIRED_OHLCV_FIELDS:
            if field not in candle:
                return f"Candle {candle_number} is missing {field}"
            value = candle[field]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return f"Candle {candle_number} has a malformed {field} value"
            if not math.isfinite(value) or value <= 0:
                return (
                    f"Candle {candle_number} has a non-positive or non-finite "
                    f"{field} value"
                )
            values[field] = float(value)

        previous_timestamp = timestamp

    return None


def _classify_regime(candles):
    # Import lazily to avoid a module cycle when MultiPeriodBacktester uses
    # this preflight as its gate.
    from multi_period_backtest import MultiPeriodBacktester

    return MultiPeriodBacktester.classify_regime(candles)


def validate_period(candles, expected_start_date=None, period=None, source=None):
    """Validate one independent 365-candle period without running a strategy."""
    result = {
        "ok": False,
        "candle_count": len(candles) if isinstance(candles, list) else 0,
        "start_date": None,
        "end_date": None,
    }
    if period is not None:
        result["period"] = period
    if source is not None:
        result["source"] = source

    if expected_start_date is None:
        expected_start_date, validation_error = _first_candle_date(candles)
    else:
        try:
            expected_start_date = _coerce_date(expected_start_date)
            validation_error = None
        except (TypeError, ValueError):
            validation_error = "Requested period has an invalid start date"

    if validation_error:
        result["failure"] = validation_error
        return result

    expected_end_date = (
        expected_start_date +
        timedelta(days=REQUESTED_CANDLE_COUNT - 1)
    )
    result["start_date"] = expected_start_date.isoformat()
    result["end_date"] = expected_end_date.isoformat()

    validation_error = _validate_candles(candles, expected_start_date)
    if validation_error:
        result["failure"] = validation_error
        return result

    regime, market_return = _classify_regime(candles)
    result.update({
        "ok": True,
        "market_return": market_return,
        "regime": regime,
    })
    return result


def run_preflight(market_data=None):
    """
    Validate the existing Yahoo fixed BTC/CAD sample and classify its market.

    The fixed 2019-08-20 through 2020-08-18 window is requested through the
    existing public Yahoo data source.  No strategy or trading components are
    loaded or called.
    """
    market_data = market_data or YahooBTCADMarketData()
    candles = market_data.load_anchored_sample()
    result = validate_period(
        candles,
        expected_start_date=YahooBTCADMarketData.ANCHORED_SAMPLE_START,
    )
    if not result["ok"]:
        provider_error = getattr(market_data, "last_anchored_error", None)
        if provider_error:
            result["failure"] = (
                f"{result['failure']}. Provider detail: {provider_error}"
            )
    return result


def _notification_text(result):
    safety_note = (
        "This preflight only validates public Yahoo Finance BTC/CAD data. "
        "No trades, orders, Kraken private APIs, wallets, or strategy changes "
        "were used."
    )
    context = ""
    if result.get("period"):
        context = f"\nPeriod: {result['period']}"
    if result.get("source"):
        context += f"\nSource: {result['source']}"

    if result["ok"]:
        return (
            "PASS: AI Trading Bot BTC/CAD Yahoo preflight succeeded.\n"
            f"Validated {result['candle_count']} daily candles from "
            f"{result['start_date']} through {result['end_date']}.\n"
            f"Market return: {result['market_return']:+.2f}% "
            f"({result['regime']}).{context}\n"
            f"{safety_note}"
        )
    return (
        "FAIL: AI Trading Bot BTC/CAD Yahoo preflight failed.\n"
        f"Reason: {result['failure']}.{context}\n"
        f"{safety_note}"
    )


def send_slack_notification(result, webhook_url=None):
    """Post the preflight result exclusively to a Slack incoming webhook."""
    webhook_url = webhook_url or os.environ.get(NOTIFICATION_WEBHOOK_ENV)
    if not webhook_url:
        raise BTCADPreflightError(
            f"{NOTIFICATION_WEBHOOK_ENV} is not configured"
        )

    parsed_url = urlparse(webhook_url)
    if (
        parsed_url.scheme != "https"
        or parsed_url.hostname != SLACK_WEBHOOK_HOST
    ):
        raise BTCADPreflightError(
            f"{NOTIFICATION_WEBHOOK_ENV} must be an HTTPS Slack webhook URL"
        )

    request = Request(
        webhook_url,
        data=json.dumps({"text": _notification_text(result)}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=SLACK_TIMEOUT_SECONDS) as response:
        status = response.status if response.status is not None else response.getcode()
        if status < 200 or status >= 300:
            raise BTCADPreflightError(
                f"Slack notification returned HTTP status {status}"
            )


def main(argv=None, market_data=None, notify=send_slack_notification):
    parser = argparse.ArgumentParser(
        description=(
            "Validate the fixed public Yahoo Finance BTC/CAD 365-candle "
            "period and notify maintainers through Slack."
        )
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=20,
        help="Yahoo Finance request timeout in seconds",
    )
    args = parser.parse_args(argv)

    if args.timeout <= 0:
        parser.error("--timeout must be positive")

    if market_data is None:
        market_data = YahooBTCADMarketData(timeout=args.timeout)

    try:
        result = run_preflight(market_data)
    except Exception as error:
        result = {
            "ok": False,
            "failure": f"Yahoo Finance request or validation failed: {error}",
            "candle_count": 0,
        }

    try:
        notify(result)
    except Exception as error:
        print(
            "FAIL: Slack notification could not be delivered: "
            f"{error}",
            file=sys.stderr,
        )
        return 1

    if not result["ok"]:
        print(
            f"FAIL: BTC/CAD Yahoo preflight failed: {result['failure']}",
            file=sys.stderr,
        )
        return 1

    print(
        "PASS: BTC/CAD Yahoo preflight validated "
        f"{result['candle_count']} daily candles from "
        f"{result['start_date']} through {result['end_date']}; "
        f"market return {result['market_return']:+.2f}% "
        f"({result['regime']})."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())