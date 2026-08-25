from contextlib import contextmanager

import strategy_backtest as strategy_backtest_module
from multi_period_backtest import MultiPeriodBacktester
from out_of_sample_validation import (
    _split_periods,
)
from score_effectiveness_study import (
    SCORE_STUDY_PERIODS,
    select_score_study_periods,
)
from strategy_calibration_study import (
    FORWARD_HORIZONS,
    STARTING_CAPITAL,
    StrategyCalibrationStudy,
)
from yahoo_btc_cad_data import YahooBTCADMarketData


RSI_ENTRY_FLOOR = 60
MIN_VALIDATION_SIGNALS_FOR_PROMOTION = 20
MAX_VALIDATION_COST_SHARE_PERCENT = 50.0
MIN_RESEARCH_PERIODS_FOR_PROMOTION = 2
MAX_PROMOTION_EVIDENCE_AGE_DAYS = 30
RESEARCH_LIFECYCLE_STAGES = (
    "RESEARCH", "CANDIDATE", "BACKTEST", "STRESS_TEST", "PAPER_TEST", "VALIDATE",
)
_ORIGINAL_SCORE_FUNCTION = (
    strategy_backtest_module.calculate_strategy_score
)


def _average(values):
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _percent(numerator, denominator):
    return numerator / denominator * 100 if denominator else 0.0


def _performance(period_results):
    trades = [
        trade
        for period in period_results
        for trade in period["trades_history"]
    ]
    gross = sum(
        period["gross_profit_before_costs"]
        for period in period_results
    )
    fees = sum(period["total_fees"] for period in period_results)
    slippage = sum(period["total_slippage"] for period in period_results)
    net = sum(period["net_profit"] for period in period_results)
    return {
        "periods": len(period_results),
        "buy_signals": 0,
        "completed_trades": len(trades),
        "wins": sum(period["wins"] for period in period_results),
        "losses": sum(period["losses"] for period in period_results),
        "win_rate": _percent(
            sum(period["wins"] for period in period_results),
            len(trades),
        ),
        "gross_profit_loss": gross,
        "fees": fees,
        "slippage": slippage,
        "net_profit_loss": net,
        "net_return_percent": (
            net / (STARTING_CAPITAL * len(period_results)) * 100
            if period_results
            else 0.0
        ),
        "average_trade_profit_loss": (
            net / len(trades) if trades else 0.0
        ),
        "cost_share_of_abs_gross_percent": (
            (fees + slippage) / abs(gross) * 100 if gross else 0.0
        ),
        "maximum_drawdown": max(
            (period["max_drawdown"] for period in period_results),
            default=0.0,
        ),
    }


def _regime_performance(period_results):
    result = {}
    for regime in ("Bull", "Sideways", "Bear"):
        selected = [
            period
            for period in period_results
            if period["regime"] == regime
        ]
        performance = _performance(selected)
        performance["market_return_average"] = _average(
            period["market_return"] for period in selected
        )
        result[regime] = performance
    return result


def original_score(*args, **kwargs):
    """Call the untouched production score function."""
    return _ORIGINAL_SCORE_FUNCTION(
        *args,
        **kwargs,
    )


def rsi_floor_score(*args, **kwargs):
    """Experimental decision wrapper; score and conditions stay unchanged."""
    score, decision, reasons, conditions = original_score(*args, **kwargs)
    rsi = args[4] if len(args) > 4 else kwargs["rsi"]
    if rsi < RSI_ENTRY_FLOOR and decision == "BUY CANDIDATE":
        decision = "NO TRADE"
        reasons = list(reasons) + [
            "Experimental RSI >=60 entry gate"
        ]
    return score, decision, reasons, conditions


@contextmanager
def patched_candidate_score(candidate):
    if candidate == "control":
        yield
        return
    if candidate not in ("candidate_a", "candidate_b"):
        raise ValueError(f"Unknown candidate: {candidate}")
    original = strategy_backtest_module.calculate_strategy_score
    strategy_backtest_module.calculate_strategy_score = rsi_floor_score
    try:
        yield
    finally:
        strategy_backtest_module.calculate_strategy_score = original


class CandidateMultiPeriodBacktester(MultiPeriodBacktester):
    def __init__(self, candidate, starting_capital=STARTING_CAPITAL):
        super().__init__(starting_capital=starting_capital)
        self.candidate = candidate

    def _run_period(
        self,
        index,
        candles,
        period_label=None,
        source_label=None,
        source_kind=None,
        notifier=None,
    ):
        with patched_candidate_score(self.candidate):
            return super()._run_period(
                index,
                candles,
                period_label=period_label,
                source_label=source_label,
                source_kind=source_kind,
                notifier=notifier,
            )


def _run_period_group(candidate, selected, notifier):
    runner = CandidateMultiPeriodBacktester(candidate)
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


def _analyze_group(period_results, period_candles):
    calibration = StrategyCalibrationStudy().analyze(
        period_results,
        period_candles,
    )
    performance = _performance(period_results)
    performance["buy_signals"] = calibration["signal_count"]
    return {
        "performance": performance,
        "periods": [
            {
                "period": period["period"],
                "start_date": period["start_date"],
                "end_date": period["end_date"],
                "regime": period["regime"],
                "net_profit_loss": period["net_profit"],
                "return_percent": period["return_percent"],
                "trades": period["trades"],
            }
            for period in period_results
        ],
        "regime_performance": _regime_performance(period_results),
        "score_bands": calibration["score_bands"],
        "rsi_bands": calibration["rsi_bands"],
        "condition_combinations": calibration[
            "condition_combinations"
        ],
        "early_movement": calibration["early_movement"],
        "cost_break_even": calibration["cost_break_even"],
        "signals": calibration["signals"],
    }


def _compare_periods(control, candidate):
    comparisons = []
    for control_period, candidate_period in zip(
        control["periods"],
        candidate["periods"],
    ):
        delta = (
            candidate_period["net_profit_loss"] -
            control_period["net_profit_loss"]
        )
        comparisons.append({
            "period": control_period["period"],
            "regime": control_period["regime"],
            "control_net_profit_loss": control_period[
                "net_profit_loss"
            ],
            "candidate_net_profit_loss": candidate_period[
                "net_profit_loss"
            ],
            "net_profit_delta": delta,
            "return_delta": (
                candidate_period["return_percent"] -
                control_period["return_percent"]
            ),
            "trade_delta": (
                candidate_period["trades"] -
                control_period["trades"]
            ),
            "classification": (
                "improvement" if delta > 0
                else "regression" if delta < 0
                else "no meaningful difference"
            ),
        })
    return comparisons


def _compare_groups(control, candidate):
    control_performance = control["performance"]
    candidate_performance = candidate["performance"]
    net_delta = (
        candidate_performance["net_profit_loss"] -
        control_performance["net_profit_loss"]
    )
    return {
        "control_net_profit_loss": control_performance[
            "net_profit_loss"
        ],
        "candidate_net_profit_loss": candidate_performance[
            "net_profit_loss"
        ],
        "net_profit_delta": net_delta,
        "return_delta": (
            candidate_performance["net_return_percent"] -
            control_performance["net_return_percent"]
        ),
        "trade_delta": (
            candidate_performance["completed_trades"] -
            control_performance["completed_trades"]
        ),
        "cost_share_delta": (
            candidate_performance[
                "cost_share_of_abs_gross_percent"
            ] -
            control_performance[
                "cost_share_of_abs_gross_percent"
            ]
        ),
        "classification": (
            "improvement" if net_delta > 0
            else "regression" if net_delta < 0
            else "no meaningful difference"
        ),
    }


def _candidate_classification(candidate, research_comparison, validation_comparison):
    research_delta = research_comparison["net_profit_delta"]
    validation_delta = validation_comparison["net_profit_delta"]
    validation = candidate["validation"]
    validation_performance = validation["performance"]
    has_validation_sample = (
        validation_performance["buy_signals"] >=
        MIN_VALIDATION_SIGNALS_FOR_PROMOTION
    )
    validation_costs_reasonable = (
        validation_performance["cost_share_of_abs_gross_percent"] <
        MAX_VALIDATION_COST_SHARE_PERCENT
    )
    if research_delta <= 0 or validation_delta < 0:
        classification = "REJECTED"
        reason = (
            "The candidate did not improve net P/L in both chronological "
            "research and untouched validation results."
        )
    elif (
        research_delta > 0 and
        validation_delta >= 0 and
        has_validation_sample and
        validation_costs_reasonable
    ):
        classification = "VALIDATION CANDIDATE"
        reason = (
            "The candidate improved both groups with sufficient validation "
            "signals and without material validation cost dominance."
        )
    else:
        classification = "PROMISING"
        reason = (
            "The candidate improved or preserved net P/L, but validation "
            "sample size or cost burden requires more evidence."
        )
    if candidate["name"] == "Candidate B":
        reason += (
            " Candidate B is a diagnostic view of Candidate A, not a "
            "separate execution rule."
        )
    return {
        "classification": classification,
        "reason": reason,
    }


def evaluate_promotion_gates(candidate, evidence=None, *, now=None):
    """Return an auditable, fail-closed promotion decision.

    Evidence is intentionally caller-supplied: this function never treats a
    backtest or research result as paper observation evidence.
    """
    evidence = evidence if isinstance(evidence, dict) else {}
    research = candidate.get("research") if isinstance(candidate, dict) else None
    validation = candidate.get("validation") if isinstance(candidate, dict) else None
    research = research if isinstance(research, dict) else {}
    validation = validation if isinstance(validation, dict) else {}
    validation_performance = validation.get("performance") or {}
    stages = {}
    for stage in RESEARCH_LIFECYCLE_STAGES:
        item = evidence.get(stage.lower())
        if isinstance(item, dict):
            stages[stage] = dict(item)
        else:
            stages[stage] = {"status": "BLOCKED", "reason": "evidence not supplied"}
    # Only the outputs this study actually owns may be inferred.
    if not evidence.get("research"):
        stages["RESEARCH"] = {
            "status": "PASS" if research.get("periods") else "BLOCKED",
            "reason": "research periods available" if research.get("periods")
            else "research periods are missing",
            "source": "controlled candidate study",
        }
    if not evidence.get("candidate"):
        stages["CANDIDATE"] = {
            "status": "PASS" if candidate.get("name") else "BLOCKED",
            "reason": "candidate identity declared" if candidate.get("name")
            else "candidate identity is missing",
        }
    if not evidence.get("backtest"):
        stages["BACKTEST"] = {
            "status": "PASS" if candidate.get("research_period_results")
            and candidate.get("validation_period_results") else "BLOCKED",
            "reason": "chronological and untouched backtests available"
            if candidate.get("research_period_results")
            and candidate.get("validation_period_results")
            else "research and validation backtests are incomplete",
        }
    gates = []
    def gate(name, passed, reason):
        gates.append({"name": name, "status": "PASS" if passed else "BLOCKED",
                      "reason": reason})
    gate("sample", validation_performance.get("buy_signals", 0)
         >= MIN_VALIDATION_SIGNALS_FOR_PROMOTION,
         f"validation requires at least {MIN_VALIDATION_SIGNALS_FOR_PROMOTION} signals")
    gate("data_quality", evidence.get("data_quality") == "PASS"
         and evidence.get("freshness") not in {"STALE", "UNAVAILABLE"},
         "validated, fresh source-quality evidence is required")
    gate("robustness", evidence.get("robustness") == "PASS",
         "stress-test robustness evidence is required")
    cost = validation_performance.get("cost_share_of_abs_gross_percent")
    gate("costs", isinstance(cost, (int, float))
         and cost < MAX_VALIDATION_COST_SHARE_PERCENT,
         f"validation cost share must be below {MAX_VALIDATION_COST_SHARE_PERCENT:.0f}%")
    gate("risk", evidence.get("risk") == "PASS",
         "risk and drawdown evidence is required")
    paper = evidence.get("paper_observation")
    gate("paper_observation", paper == "PASS",
         "qualified genuine paper-observation evidence is required")
    for stage in RESEARCH_LIFECYCLE_STAGES:
        if stages[stage].get("status") != "PASS":
            gates.append({"name": f"stage:{stage.lower()}",
                          "status": "BLOCKED",
                          "reason": stages[stage].get("reason", "stage incomplete")})
    blocked = [item["reason"] for item in gates if item["status"] != "PASS"]
    return {
        "status": "PROMOTED" if not blocked else "BLOCKED",
        "lifecycle": stages,
        "gates": gates,
        "blocked_reasons": blocked,
        "provenance": {
            "candidate": candidate.get("name") if isinstance(candidate, dict) else None,
            "research_source": research.get("source"),
            "validation_source": validation.get("source"),
            "uncertainty": "No production promotion is possible without every gate.",
        },
    }


def _candidate_result(name, research_results, validation_results, research, validation):
    return {
        "name": name,
        "research": research,
        "validation": validation,
        "research_period_results": research_results,
        "validation_period_results": validation_results,
    }


def run_strategy_candidate_study(notifier=None):
    if notifier is None:
        from btc_cad_preflight import send_slack_notification

        notifier = send_slack_notification
    loader = YahooBTCADMarketData(data_range="10y")
    candles = loader.load()
    if not candles:
        raise RuntimeError(loader.last_error or "Yahoo BTC/CAD data unavailable")
    selected = select_score_study_periods(candles)
    if len(selected) != len(SCORE_STUDY_PERIODS):
        raise RuntimeError("Candidate study requires all fixed periods")
    research_periods, validation_periods = _split_periods(selected)
    research_candles = [period["candles"] for period in research_periods]
    validation_candles = [period["candles"] for period in validation_periods]

    # Develop and analyze every candidate on research periods first. The
    # validation periods are not run until all research analysis is complete.
    research_period_results = {
        candidate: _run_period_group(
            candidate,
            research_periods,
            notifier,
        )
        for candidate in ("control", "candidate_a", "candidate_b")
    }
    research = {
        candidate: _analyze_group(
            results,
            research_candles,
        )
        for candidate, results in research_period_results.items()
    }

    validation_period_results = {
        candidate: _run_period_group(
            candidate,
            validation_periods,
            notifier,
        )
        for candidate in ("control", "candidate_a", "candidate_b")
    }
    validation = {
        candidate: _analyze_group(
            results,
            validation_candles,
        )
        for candidate, results in validation_period_results.items()
    }

    named_candidates = {
        "candidate_a": "Candidate A",
        "candidate_b": "Candidate B",
    }
    candidates = {}
    comparisons = {}
    for key, name in named_candidates.items():
        candidate = _candidate_result(
            name,
            research_period_results[key],
            validation_period_results[key],
            research[key],
            validation[key],
        )
        research_comparison = _compare_groups(
            research["control"],
            research[key],
        )
        validation_comparison = _compare_groups(
            validation["control"],
            validation[key],
        )
        candidates[key] = candidate
        comparisons[key] = {
            "research": research_comparison,
            "validation": validation_comparison,
            "research_periods": _compare_periods(
                research["control"],
                research[key],
            ),
            "validation_periods": _compare_periods(
                validation["control"],
                validation[key],
            ),
        }
        candidates[key]["classification"] = _candidate_classification(
            candidate,
            research_comparison,
            validation_comparison,
        )
        candidates[key]["promotion"] = evaluate_promotion_gates(candidates[key])

    return {
        "source": "Yahoo Finance BTC/CAD aggregated daily data",
        "real_money_trading": False,
        "split": {
            "research_start": research_periods[0]["start_date"],
            "research_end": research_periods[-1]["end_date"],
            "research_periods": len(research_periods),
            "research_candles": len(research_periods) * 365,
            "validation_start": validation_periods[0]["start_date"],
            "validation_end": validation_periods[-1]["end_date"],
            "validation_periods": len(validation_periods),
            "validation_candles": len(validation_periods) * 365,
        },
        "control": {
            "research": research["control"],
            "validation": validation["control"],
        },
        "candidates": candidates,
        "comparisons": comparisons,
        "candidate_definitions": {
            "control": "Exact existing StrategyBacktester.",
            "candidate_a": (
                "Original score and conditions, with experimental RSI >=60 "
                "BUY decision gate."
            ),
            "candidate_b": (
                "Candidate A execution plus cost-awareness diagnostics; "
                "same execution as Candidate A."
            ),
        },
    }


def _print_metrics(label, group):
    performance = group["performance"]
    print(f"\n{label}")
    print(
        f"BUY signals={performance['buy_signals']}, "
        f"trades={performance['completed_trades']}, "
        f"wins/losses={performance['wins']}/{performance['losses']}, "
        f"win rate={performance['win_rate']:.2f}%"
    )
    print(
        f"gross=${performance['gross_profit_loss']:+.4f}, "
        f"fees=${performance['fees']:.4f}, "
        f"slippage=${performance['slippage']:.4f}, "
        f"net=${performance['net_profit_loss']:+.4f}, "
        f"return={performance['net_return_percent']:+.2f}%, "
        f"avg trade=${performance['average_trade_profit_loss']:+.4f}, "
        f"max DD={performance['maximum_drawdown']:.2f}%, "
        f"cost share={performance['cost_share_of_abs_gross_percent']:.2f}%"
    )
    for horizon, summary in group["early_movement"].items():
        print(
            f"  +{horizon} candles: avg={summary['average']:+.2f}%, "
            f"positive={summary['positive_percent']:.2f}%, "
            f"break-even={group['cost_break_even']['overall']['horizons'][horizon]['reached_break_even_percent']:.2f}%"
        )
    print("RSI bands (5-candle average):")
    for label, summary in group["rsi_bands"].items():
        print(
            f"  {label}: n={summary['signals']}, "
            f"avg={summary['forward_returns']['5']['average']:+.2f}%"
        )
    print("Score bands (5-candle average):")
    for label, summary in group["score_bands"].items():
        print(
            f"  {label}: n={summary['signals']}, "
            f"avg={summary['forward_returns']['5']['average']:+.2f}%"
        )
    print("Regimes:")
    for regime, summary in _regime_performance_from_group(group).items():
        if summary["periods"]:
            print(
                f"  {regime}: periods={summary['periods']}, "
                f"net=${summary['net_profit_loss']:+.4f}, "
                f"trades={summary['completed_trades']}"
            )


def _regime_performance_from_group(group):
    result = {}
    for regime in ("Bull", "Sideways", "Bear"):
        selected = [
            period
            for period in group["periods"]
            if period["regime"] == regime
        ]
        result[regime] = {
            "periods": len(selected),
            "net_profit_loss": sum(
                period["net_profit_loss"] for period in selected
            ),
            "completed_trades": sum(
                period["trades"] for period in selected
            ),
        }
    return result


def print_report(results):
    print("BTC/CAD CONTROLLED STRATEGY IMPROVEMENT — ANALYSIS ONLY")
    print("REAL-MONEY TRADING: DISABLED")
    split = results["split"]
    print(
        f"Research: {split['research_start']} to {split['research_end']} "
        f"({split['research_periods']} periods, {split['research_candles']} candles)"
    )
    print(
        f"Validation: {split['validation_start']} to "
        f"{split['validation_end']} ({split['validation_periods']} periods, "
        f"{split['validation_candles']} candles)"
    )
    for group_name, group in results["control"].items():
        _print_metrics(f"CONTROL · {group_name}", group)
    for candidate_key, candidate in results["candidates"].items():
        print(f"\n=== {candidate['name']} ===")
        print(results["candidate_definitions"][candidate_key])
        for group_name in ("research", "validation"):
            _print_metrics(
                f"{candidate['name']} · {group_name}",
                candidate[group_name],
            )
        comparison = results["comparisons"][candidate_key]
        for group_name in ("research", "validation"):
            group_comparison = comparison[group_name]
            print(
                f"{group_name} net delta vs control: "
                f"${group_comparison['net_profit_delta']:+.4f} "
                f"({group_comparison['classification']})"
            )
        print(
            f"CLASSIFICATION: {candidate['classification']['classification']} — "
            f"{candidate['classification']['reason']}"
        )


def main():
    results = run_strategy_candidate_study()
    print_report(results)
    return results


if __name__ == "__main__":
    main()