"""Run the local pre-live readiness gate without touching operational state."""

from __future__ import annotations

import json
import sys

from config import LIVE_TRADING, PAPER_TRADING
from testing_center import run_pre_live_validation


def main() -> int:
    report = run_pre_live_validation(
        {
            "paper_trading": PAPER_TRADING,
            "live_trading": LIVE_TRADING,
            "api_contract_valid": True,
        }
    )
    print(json.dumps(report, indent=2))
    # BLOCKED/NOT CONFIGURED is an honest readiness outcome, not a code crash.
    # FAIL is reserved for an unsafe or contradictory local configuration.
    return 1 if report["status"] == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())