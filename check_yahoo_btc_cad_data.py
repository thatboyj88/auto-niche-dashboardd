"""Release-time preflight for the public Yahoo Finance BTC/CAD evidence."""

import argparse
import json
import math
import os
import sys
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from yahoo_btc_cad_data import YahooBTCADMarketData


REQUIRED_CANDLE_FIELDS = (
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
)
MIN_ROLLING_CANDLES = 365
NOTIFICATION_WEBHOOK_ENV = "BTC_CAD_PREFLIGHT_SLACK_WEBHOOK_URL"
NOTIFICATION_TIMEOUT = 10
SLACK_WEBHOOK_HOST = "hooks.slack.com"
ANCHORED_FAILURE_LABEL = (
    "anchored 2019-08-20 through 2020-08-18 sample"
)


def _validate_normalized_candles(candles, label, minimum_count):
    if not isinstance(candles, list):
        return f"{label} returned a non-list result"

    if len(candles) < minimum_count:
        return (
            f"{label} returned {len(candles)} normalized candles; "
            f"at least {minimum_count} are required"
        )

    previous_timestamp = None
    for index, candle in enumerate(candles):
        if not isinstance(candle, dict):
            return f"{label} candle {index + 1} is not a normalized object"

        missing_fields = [
            field
            for field in REQUIRED_CANDLE_FIELDS
            if field not in candle
        ]
        if missing_fields:
            return (
                f"{label} candle {index + 1} is missing normalized "
                f"fields: {', '.join(missing_fields)}"
            )

        timestamp = candle["timestamp"]
        if not isinstance(timestamp, int):
            return f"{label} candle {index + 1} has a non-integer timestamp"
        if previous_timestamp is not None and timestamp <= previous_timestamp:
            return (
                f"{label} timestamps are not strictly increasing at "
                f"candle {index + 1}"
            )

        for field in REQUIRED_CANDLE_FIELDS[1:]:
            value = candle[field]
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                return (
                    f"{label} candle {index + 1} has a non-finite "
                    f"{field} value"
                )

        previous_timestamp = timestamp

    return None


def _validate_anchored_dates(candles):
    start_date = datetime.strptime(
        YahooBTCADMarketData.ANCHORED_SAMPLE_START,
        "%Y-%m-%d",
    ).date()
    end_date = datetime.strptime(
        YahooBTCADMarketData.ANCHORED_SAMPLE_END,
        "%Y-%m-%d",
    ).date()

    expected_dates = [
        start_date + timedelta(days=offset)
        for offset in range(YahooBTCADMarketData.ANCHORED_SAMPLE_CANDLES)
    ]
    actual_dates = [
        datetime.fromtimestamp(
            candle["timestamp"],
            tz=timezone.utc,
        ).date()
        for candle in candles
    ]

    if actual_dates != expected_dates:
        actual_start = actual_dates[0].isoformat()
        actual_end = actual_dates[-1].isoformat()
        return (
            "Anchored Yahoo Finance BTC/CAD sample does not cover the "
            f"expected daily range {start_date.isoformat()} through "
            f"{end_date.isoformat()} (received {actual_start} through "
            f"{actual_end})"
        )

    return None


def validate_yahoo_btc_cad_sources(market_data=None):
    """
    Check both public Yahoo Finance sources without running any trades.

    The returned dictionary is intentionally suitable for a release check or
    scheduled job: callers can report each source's independent failure while
    keeping the dashboard's trading behavior untouched.
    """
    market_data = market_data or YahooBTCADMarketData()
    failures = []

    rolling_candles = market_data.load()
    rolling_error = _validate_normalized_candles(
        rolling_candles,
        "Rolling Yahoo Finance BTC/CAD source",
        MIN_ROLLING_CANDLES,
    )
    if rolling_error:
        rolling_error = (
            f"{rolling_error}. Provider detail: "
            f"{market_data.last_error or 'no error detail returned'}"
        )
        failures.append(rolling_error)

    anchored_candles = market_data.load_anchored_sample()
    anchored_error = _validate_normalized_candles(
        anchored_candles,
        "Anchored Yahoo Finance BTC/CAD sample",
        YahooBTCADMarketData.ANCHORED_SAMPLE_CANDLES,
    )
    if anchored_error is None:
        anchored_error = _validate_anchored_dates(anchored_candles)
    if anchored_error:
        anchored_error = (
            f"{anchored_error}. Provider detail: "
            f"{market_data.last_anchored_error or 'no error detail returned'}"
        )
        failures.append(anchored_error)

    return {
        "ok": not failures,
        "rolling_candle_count": len(rolling_candles),
        "anchored_candle_count": len(anchored_candles),
        "rolling_failure": rolling_error,
        "anchored_failure": anchored_error,
        "failures": failures,
    }


def _failed_source_labels(result):
    labels = []
    rolling_failure = result.get("rolling_failure")
    anchored_failure = result.get("anchored_failure")

    # Keep this fallback for callers that persisted the pre-notification
    # result shape and only retained the human-readable failure list.
    if rolling_failure is None:
        rolling_failure = next(
            (
                failure
                for failure in result.get("failures", [])
                if failure.startswith("Rolling Yahoo Finance BTC/CAD source")
            ),
            None,
        )
    if anchored_failure is None:
        anchored_failure = next(
            (
                failure
                for failure in result.get("failures", [])
                if failure.startswith("Anchored Yahoo Finance BTC/CAD sample")
            ),
            None,
        )

    if rolling_failure:
        labels.append("rolling source")
    if anchored_failure:
        labels.append(ANCHORED_FAILURE_LABEL)
    return labels


def _notification_message(result):
    if result.get("test_notification"):
        return (
            "TEST: Yahoo Finance BTC/CAD evidence preflight alert delivery "
            "succeeded.\n"
            "This is a controlled maintainer-notification test; no evidence "
            "source was checked."
        )

    labels = _failed_source_labels(result)
    if not labels:
        unavailable = "both the rolling source and " + ANCHORED_FAILURE_LABEL
    elif len(labels) == 2:
        unavailable = "both the rolling source and " + ANCHORED_FAILURE_LABEL
    else:
        unavailable = labels[0]

    details = "\n".join(
        f"- {failure}"
        for failure in result.get("failures", [])
    )
    return (
        "Yahoo Finance BTC/CAD evidence preflight failed.\n"
        f"Unavailable: {unavailable}.\n"
        f"{details}"
    )


def send_maintainer_notification(result, webhook_url=None):
    """
    Send a concise Slack alert using a maintainer-configured webhook.

    The URL is intentionally restricted to Slack's incoming-webhook host so
    this release check cannot be repurposed to call an exchange or trading
    endpoint. No exchange credentials are read by this function.
    """
    webhook_url = webhook_url or os.environ.get(NOTIFICATION_WEBHOOK_ENV)
    if not webhook_url:
        return False

    parsed_url = urlparse(webhook_url)
    if (
        parsed_url.scheme != "https"
        or parsed_url.hostname != SLACK_WEBHOOK_HOST
    ):
        raise ValueError(
            f"{NOTIFICATION_WEBHOOK_ENV} must be an HTTPS Slack webhook URL"
        )

    payload = json.dumps({"text": _notification_message(result)}).encode(
        "utf-8"
    )
    request = Request(
        webhook_url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=NOTIFICATION_TIMEOUT) as response:
        status = response.status
        if status is None:
            status = response.getcode()
        if status < 200 or status >= 300:
            raise RuntimeError(
                f"Slack notification returned HTTP status {status}"
            )
    return True


def _notify_maintainers_of_failure(result):
    try:
        notification_sent = send_maintainer_notification(result)
    except Exception as error:
        print(
            "WARNING: maintainer notification could not be delivered: "
            f"{error}",
            file=sys.stderr,
        )
        return

    if not notification_sent:
        print(
            "WARNING: maintainer notification skipped; configure "
            f"{NOTIFICATION_WEBHOOK_ENV} with a Slack incoming webhook "
            "to receive failure alerts.",
            file=sys.stderr,
        )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Verify the rolling and anchored public Yahoo Finance BTC/CAD "
            "research sources."
        )
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=20,
        help="HTTP timeout in seconds for each public Yahoo request",
    )
    parser.add_argument(
        "--test-notification",
        action="store_true",
        help=(
            "Send a clearly labelled Slack delivery test without querying "
            "Yahoo Finance"
        ),
    )
    args = parser.parse_args(argv)

    if args.test_notification:
        try:
            if not send_maintainer_notification({"test_notification": True}):
                raise RuntimeError(
                    f"{NOTIFICATION_WEBHOOK_ENV} is not configured"
                )
        except Exception as error:
            print(
                "FAIL: Yahoo Finance BTC/CAD maintainer notification test "
                f"could not be delivered: {error}",
                file=sys.stderr,
            )
            return 1
        print(
            "PASS: Yahoo Finance BTC/CAD maintainer notification test "
            "was delivered to Slack"
        )
        return 0

    try:
        result = validate_yahoo_btc_cad_sources(
            YahooBTCADMarketData(timeout=args.timeout)
        )
    except Exception as error:
        print(
            "FAIL: Yahoo Finance BTC/CAD evidence preflight could not "
            f"complete: {error}",
            file=sys.stderr,
        )
        _notify_maintainers_of_failure(
            {
                "rolling_failure": str(error),
                "anchored_failure": str(error),
                "failures": [
                    "Yahoo Finance BTC/CAD evidence preflight could not "
                    f"complete: {error}"
                ],
            }
        )
        return 1

    if result["ok"]:
        print(
            "PASS: Yahoo Finance BTC/CAD evidence is available and "
            "normalized "
            f"(rolling={result['rolling_candle_count']} candles, "
            f"anchored={result['anchored_candle_count']} candles; "
            f"anchored range="
            f"{YahooBTCADMarketData.ANCHORED_SAMPLE_START} through "
            f"{YahooBTCADMarketData.ANCHORED_SAMPLE_END})"
        )
        return 0

    print("FAIL: Yahoo Finance BTC/CAD evidence preflight failed:")
    for failure in result["failures"]:
        print(f"  - {failure}")
    _notify_maintainers_of_failure(result)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())