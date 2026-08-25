"""Executable paper-backtest comparison for the fixed RSI >=60 candidate."""

from multi_period_backtest import MultiPeriodBacktester
from out_of_sample_validation import _split_periods
from score_effectiveness_study import SCORE_STUDY_PERIODS, select_score_study_periods
from strategy_candidate_study import CandidateMultiPeriodBacktester
from strategy_calibration_study import STARTING_CAPITAL
from yahoo_btc_cad_data import YahooBTCADMarketData


RSI_THRESHOLD = 60


def _percent(numerator, denominator):
    return numerator / denominator * 100 if denominator else 0.0


def _performance(period_results):
    trades = [
        trade
        for period in period_results
        for trade in period["trades_history"]
    ]
    gross = sum(period["gross_profit_before_costs"] for period in period_results)
    fees = sum(period["total_fees"] for period in period_results)
    slippage = sum(period["total_slippage"] for period in period_results)
    net = sum(period["net_profit"] for period in period_results)
    return {
        "starting_capital_per_period": STARTING_CAPITAL,
        "periods": len(period_results),
        "signals": sum(
            evaluation["decision"] == "BUY"
            for period in period_results
            for evaluation in period["evaluation_history"]
        ),
        "trades": len(trades),
        "gross": gross,
        "fees": fees,
        "slippage": slippage,
        "costs": fees + slippage,
        "net": net,
        "return_percent": (
            net / (STARTING_CAPITAL * len(period_results)) * 100
            if period_results else 0.0
        ),
        "maximum_drawdown": max(
            (period["max_drawdown"] for period in period_results),
            default=0.0,
        ),
        "win_rate": _percent(
            sum(period["wins"] for period in period_results),
            len(trades),
        ),
        "cost_share": (
            (fees + slippage) / abs(gross) * 100 if gross else 0.0
        ),
        "periods_detail": [
            {
                "period": period["period"],
                "regime": period["regime"],
                "trades": period["trades"],
                "net": period["net_profit"],
                "return_percent": period["return_percent"],
                "max_drawdown": period["max_drawdown"],
            }
            for period in period_results
        ],
    }


def _run_period_group(candidate, selected, notifier):
    runner = (
        MultiPeriodBacktester(starting_capital=STARTING_CAPITAL)
        if candidate == "control"
        else CandidateMultiPeriodBacktester(candidate)
    )
    return [
        runner._run_period(
            index,
            period["candles"],
            period_label=period["period"],
            source_label="Yahoo Finance BTC/CAD fixed ten-year study",
            source_kind="fixed-study",
            notifier=notifier,
        )
        for index, period in enumerate(selected)
    ]


def run_rsi_candidate_executable_study(notifier=None):
    if notifier is None:
        from btc_cad_preflight import send_slack_notification
        notifier = send_slack_notification
    loader = YahooBTCADMarketData(data_range="10y")
    candles = loader.load()
    if not candles:
        raise RuntimeError(loader.last_error or "Yahoo BTC/CAD data unavailable")
    selected = select_score_study_periods(candles)
    if len(selected) != len(SCORE_STUDY_PERIODS):
        raise RuntimeError("Executable RSI study requires all fixed periods")
    research_periods, validation_periods = _split_periods(selected)
    research = {
        "control": _run_period_group("control", research_periods, notifier),
        "rsi_60": _run_period_group(
            "candidate_a", research_periods, notifier
        ),
    }
    validation = {
        "control": _run_period_group("control", validation_periods, notifier),
        "rsi_60": _run_period_group(
            "candidate_a", validation_periods, notifier
        ),
    }
    research_metrics = {
        candidate: _performance(results)
        for candidate, results in research.items()
    }
    validation_metrics = {
        candidate: _performance(results)
        for candidate, results in validation.items()
    }
    return {
        "real_money_trading": False,
        "candidate_definition": (
            "Same StrategyBacktester and unchanged control rules; BUY "
            "eligibility additionally requires RSI >=60."
        ),
        "research": research_metrics,
        "validation": validation_metrics,
        "deltas": {
            "research_net": (
                research_metrics["rsi_60"]["net"]
                - research_metrics["control"]["net"]
            ),
            "validation_net": (
                validation_metrics["rsi_60"]["net"]
                - validation_metrics["control"]["net"]
            ),
        },
    }


def _print_group(label, metrics):
    print(f"\n=== {label} ===")
    print(
        "Candidate | Signals | Trades | Gross | Fees | Slippage | Net | "
        "Return | Max DD | Win rate | Cost share"
    )
    for name, item in metrics.items():
        print(
            f"{name} | {item['signals']} | {item['trades']} | "
            f"${item['gross']:+.4f} | ${item['fees']:.4f} | "
            f"${item['slippage']:.4f} | ${item['net']:+.4f} | "
            f"{item['return_percent']:+.2f}% | "
            f"${item['maximum_drawdown']:.4f} | {item['win_rate']:.2f}% | "
            f"{item['cost_share']:.2f}%"
        )


def print_report(results):
    print("BTC/CAD EXECUTABLE RSI >=60 CANDIDATE STUDY — ANALYSIS ONLY")
    print("REAL-MONEY TRADING: DISABLED")
    print(results["candidate_definition"])
    _print_group("Research", results["research"])
    _print_group("Untouched validation", results["validation"])
    print("\n=== Candidate versus control ===")
    print(
        f"Research net delta: ${results['deltas']['research_net']:+.4f}"
    )
    print(
        f"Validation net delta: ${results['deltas']['validation_net']:+.4f}"
    )
    classification = (
        "PASSING BOTH SPLITS"
        if (
            results["deltas"]["research_net"] > 0
            and results["deltas"]["validation_net"] > 0
        )
        else "REJECTED"
    )
    print(f"Classification: {classification}")
    print("\n=== Interpretation boundary ===")
    print(
        "This is a paper-backtest comparison only. RSI >=60 was fixed before "
        "validation, and no result is promoted into production automatically."
    )


def main():
    results = run_rsi_candidate_executable_study()
    print_report(results)
    return results


if __name__ == "__main__":
    main()