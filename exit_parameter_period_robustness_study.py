"""Per-period and regime robustness follow-up for Step 25's 6% targets."""

from collections import defaultdict

from exit_parameter_robustness_study import (
    ACTIVE_CONTROL_STATUS,
    CONTROL,
    RESEARCH_ONLY_STATUS,
    _metrics,
    _run_variant,
)
from out_of_sample_validation import _split_periods
from multi_period_backtest import MultiPeriodBacktester
from score_effectiveness_study import (
    SCORE_STUDY_PERIODS,
    select_score_study_periods,
)
from strategy_calibration_study import STARTING_CAPITAL
from yahoo_btc_cad_data import YahooBTCADMarketData
from config import (
    EXIT_CONTROL,
    EXIT_PROMOTION_MAX_COST_SHARE_PERCENT,
    EXIT_PROMOTION_MIN_UNTOUCHED_PERIODS,
)


VARIANTS = (
    ("control", CONTROL),
    ("stop_2.0_target_6.0", (2.0, 6.0)),
    ("stop_1.5_target_6.0", (1.5, 6.0)),
)
ADDITIONAL_PERIODS = (
    {
        "period": "Supplemental Period K",
        "start_date": "2014-09-17",
        "end_date": "2015-09-16",
        "regime": "Bear",
    },
    {
        "period": "Supplemental Period L",
        "start_date": "2015-09-17",
        "end_date": "2016-09-15",
        "regime": "Bull",
    },
    {
        "period": "Supplemental Period M",
        "start_date": "2016-09-16",
        "end_date": "2017-09-15",
        "regime": "Bull",
    },
)
MIN_TRADES_PER_PERIOD = 3
CONCENTRATION_THRESHOLD_PERCENT = 60.0
MIN_UNTOUCHED_PERIODS_FOR_PROMOTION = EXIT_PROMOTION_MIN_UNTOUCHED_PERIODS
MIN_PERIODS_FOR_BREADTH = MIN_UNTOUCHED_PERIODS_FOR_PROMOTION
MAX_COST_SHARE_DELTA_PERCENT = EXIT_PROMOTION_MAX_COST_SHARE_PERCENT


def _period_metric(pair, label, parameters):
    result, _ = pair
    metric = _metrics([pair])
    metric.update({
        "period": result["period"],
        "regime": result["regime"],
        "start_date": result["start_date"],
        "end_date": result["end_date"],
        "stop_loss": parameters[0],
        "take_profit": parameters[1],
        "insufficient_evidence": (
            metric["trades"] < MIN_TRADES_PER_PERIOD
        ),
        "variant": label,
    })
    return metric


def _add_delta_metrics(candidate_metrics, control_metrics, total_delta):
    delta = candidate_metrics["net"] - control_metrics["net"]
    candidate_metrics["net_delta_vs_control"] = delta
    candidate_metrics["return_delta_vs_control"] = (
        candidate_metrics["return_percent"]
        - control_metrics["return_percent"]
    )
    candidate_metrics["drawdown_delta_vs_control"] = (
        candidate_metrics["maximum_drawdown"]
        - control_metrics["maximum_drawdown"]
    )
    candidate_metrics["contribution_to_total_improvement_percent"] = (
        delta / total_delta * 100 if total_delta else None
    )


def _regime_summary(period_metrics, control_period_metrics):
    grouped = defaultdict(list)
    control_grouped = defaultdict(list)
    for item in period_metrics:
        grouped[item["regime"]].append(item)
    for item in control_period_metrics:
        control_grouped[item["regime"]].append(item)
    summaries = {}
    for regime, items in grouped.items():
        controls = control_grouped[regime]
        net = sum(item["net"] for item in items)
        control_net = sum(item["net"] for item in controls)
        trades = sum(item["trades"] for item in items)
        summaries[regime] = {
            "regime": regime,
            "periods": len(items),
            "trades": trades,
            "net": net,
            "net_delta_vs_control": net - control_net,
            "win_rate": (
                sum(item["win_rate"] * item["trades"] for item in items)
                / trades if trades else 0.0
            ),
            "maximum_drawdown": max(
                (item["maximum_drawdown"] for item in items), default=0.0
            ),
            "cost_share": (
                sum(item["fees"] + item["slippage"] for item in items)
                / abs(sum(item["gross"] for item in items)) * 100
                if sum(item["gross"] for item in items) else 0.0
            ),
            "average_duration": (
                sum(item["average_duration"] * item["trades"] for item in items)
                / trades if trades else 0.0
            ),
            "insufficient_evidence": (
                len(items) < MIN_PERIODS_FOR_BREADTH
                or trades < MIN_TRADES_PER_PERIOD
            ),
        }
    return summaries


def _breadth(candidate_periods, control_periods, total_delta):
    eligible = [
        item for item in candidate_periods
        if not item["insufficient_evidence"]
    ]
    positive = [
        item for item in eligible if item["net_delta_vs_control"] > 0
    ]
    contributions = [
        item["contribution_to_total_improvement_percent"]
        for item in eligible
        if item["contribution_to_total_improvement_percent"] is not None
        and item["net_delta_vs_control"] > 0
    ]
    largest_contribution = max(contributions, default=0.0)
    insufficient_evidence = len(eligible) < MIN_PERIODS_FOR_BREADTH
    concentrated = largest_contribution >= CONCENTRATION_THRESHOLD_PERCENT
    return {
        "eligible_periods": len(eligible),
        "positive_periods": len(positive),
        "positive_period_share_percent": (
            len(positive) / len(eligible) * 100 if eligible else 0.0
        ),
        "largest_positive_period_contribution_percent": largest_contribution,
        "concentrated_in_one_or_few_periods": concentrated,
        "insufficient_evidence": insufficient_evidence,
        "research_only": insufficient_evidence or concentrated,
        "promotion_status": (
            RESEARCH_ONLY_STATUS
            if insufficient_evidence or concentrated
            else "UNDECIDED_PENDING_FULL_GATE"
        ),
        "control_periods": len(control_periods),
        "total_delta": total_delta,
    }


def _promotion_gate(label, research, validation, additional):
    """Return a conservative promotion decision for one non-control variant."""
    reasons = []
    research_breadth = research["breadth"][label]
    untouched = (validation, additional)
    untouched_breadth = [analysis["breadth"][label] for analysis in untouched]
    research_candidate = research["aggregate"][label]
    research_control = research["aggregate"]["control"]
    research_net_delta = (
        research_candidate.get("net_delta_vs_control")
        if "net_delta_vs_control" in research_candidate
        else research_candidate["net"] - research_control["net"]
    )
    if research_net_delta <= 0:
        reasons.append("research uplift is not positive")
    if any(
        breadth["insufficient_evidence"]
        or breadth["eligible_periods"] < MIN_UNTOUCHED_PERIODS_FOR_PROMOTION
        for breadth in untouched_breadth
    ):
        reasons.append(
            "untouched evidence has fewer than "
            f"{MIN_UNTOUCHED_PERIODS_FOR_PROMOTION} eligible periods"
        )
    if any(
        breadth["positive_period_share_percent"] < 100.0
        or breadth["concentrated_in_one_or_few_periods"]
        for breadth in (research_breadth, *untouched_breadth)
    ):
        reasons.append(
            "positive results are not broad across periods or include "
            f"concentration at/above {CONCENTRATION_THRESHOLD_PERCENT:.0f}%"
        )
    for name, analysis in (("validation", validation), ("additional", additional)):
        candidate = analysis["aggregate"][label]
        control = analysis["aggregate"]["control"]
        drawdown_delta = candidate.get(
            "drawdown_delta_vs_control",
            candidate.get("maximum_drawdown", 0.0)
            - control.get("maximum_drawdown", 0.0),
        )
        cost_delta = candidate["cost_share"] - control["cost_share"]
        net_delta = candidate.get(
            "net_delta_vs_control",
            candidate["net"] - control["net"],
        )
        if drawdown_delta > 0:
            reasons.append(f"{name} drawdown is worse than the control")
        if cost_delta > MAX_COST_SHARE_DELTA_PERCENT:
            reasons.append(f"{name} cost share is worse than the control")
        if net_delta <= 0:
            reasons.append(f"{name} uplift is not positive")
    return {
        "status": "PROMOTION_ELIGIBLE" if not reasons else RESEARCH_ONLY_STATUS,
        "research_only": bool(reasons),
        "reasons": reasons or ["all promotion criteria passed"],
        "control": EXIT_CONTROL,
        "minimum_untouched_periods": MIN_UNTOUCHED_PERIODS_FOR_PROMOTION,
        "maximum_concentration_percent": CONCENTRATION_THRESHOLD_PERCENT,
    }


def _run_split(selected, notifier):
    runs = {}
    for label, parameters in VARIANTS:
        runs[label] = _run_variant(
            parameters[0],
            parameters[1],
            selected,
            notifier,
        )
    return runs


def _select_additional_periods(source_candles):
    runner = MultiPeriodBacktester()
    selected = []
    for specification in ADDITIONAL_PERIODS:
        candles = [
            candle
            for candle in source_candles
            if specification["start_date"]
            <= runner.format_date(candle["timestamp"])
            <= specification["end_date"]
        ]
        actual_dates = (
            runner.format_date(candles[0]["timestamp"]),
            runner.format_date(candles[-1]["timestamp"]),
        ) if candles else (None, None)
        if len(candles) != 365 or actual_dates != (
            specification["start_date"],
            specification["end_date"],
        ):
            raise RuntimeError(
                f"{specification['period']} no longer matches its "
                "recorded 365-candle date boundary"
            )
        regime, _ = runner.classify_regime(candles)
        if regime != specification["regime"]:
            raise RuntimeError(
                f"{specification['period']} was expected to be "
                f"{specification['regime']}, but is now {regime}"
            )
        selected.append({**specification, "candles": candles})
    return selected


def _analyze_split(selected, notifier):
    runs = _run_split(selected, notifier)
    period_metrics = {
        label: [
            _period_metric(pair, label, parameters)
            for pair in pairs
        ]
        for (label, parameters), pairs in zip(VARIANTS, runs.values())
    }
    aggregate = {
        label: _metrics(runs[label])
        for label, _ in VARIANTS
    }
    for label, parameters in VARIANTS:
        aggregate[label].update({
            "variant": label,
            "stop_loss": parameters[0],
            "take_profit": parameters[1],
        })
    control_periods = period_metrics["control"]
    control_aggregate = aggregate["control"]
    for label, _ in VARIANTS:
        total_delta = aggregate[label]["net"] - control_aggregate["net"]
        for item in period_metrics[label]:
            control = next(
                control_item for control_item in control_periods
                if control_item["period"] == item["period"]
            )
            _add_delta_metrics(item, control, total_delta)
    breadth = {}
    regimes = {}
    for label, _ in VARIANTS:
        total_delta = aggregate[label]["net"] - control_aggregate["net"]
        breadth[label] = _breadth(
            period_metrics[label],
            control_periods,
            total_delta,
        )
        regimes[label] = _regime_summary(
            period_metrics[label],
            control_periods,
        )
    return {
        "aggregate": aggregate,
        "periods": period_metrics,
        "regimes": regimes,
        "breadth": breadth,
    }


def _overall_outcome(research, validation, additional):
    candidates = tuple(label for label, _ in VARIANTS if label != "control")
    additional_is_broad = all(
        additional["breadth"][label]["positive_period_share_percent"] == 100.0
        and not additional["breadth"][label]["concentrated_in_one_or_few_periods"]
        and not additional["breadth"][label]["insufficient_evidence"]
        for label in candidates
    )
    any_candidate_improved = any(
        additional["aggregate"][label]["net"]
        > additional["aggregate"]["control"]["net"]
        for label in candidates
    )
    if additional_is_broad and all(
        not validation["breadth"][label]["insufficient_evidence"]
        for label in candidates
    ):
        return {
            "label": "STRONGER EVIDENCE",
            "color": "green",
            "reason": "Improvement is positive and broad across sufficient untouched evidence.",
        }
    if any_candidate_improved:
        return {
            "label": "INTERESTING BUT UNPROVEN",
            "color": "yellow",
            "reason": (
                "Improvement appears in additional periods, but concentration "
                "or sparse locked validation prevents a repeatability claim."
            ),
        }
    return {
        "label": "REJECTED",
        "color": "red",
        "reason": (
            "Improvement is concentrated or absent outside the original "
            "research evidence and does not persist in the additional periods."
        ),
    }


def run_exit_parameter_period_robustness_study(notifier=None):
    if notifier is None:
        notifier = lambda *_args: None
    candles = YahooBTCADMarketData(data_range="15y").load()
    if not candles:
        raise RuntimeError("Yahoo BTC/CAD data unavailable")
    selected = select_score_study_periods(candles)
    if len(selected) != len(SCORE_STUDY_PERIODS):
        raise RuntimeError("Period robustness study requires all fixed periods")
    research, validation = _split_periods(selected)
    additional = _select_additional_periods(candles)
    results = {
        "real_money_trading": False,
        "variants": VARIANTS,
        "research": _analyze_split(research, notifier),
        "validation": _analyze_split(validation, notifier),
        "additional": _analyze_split(additional, notifier),
        "note": (
            "These are independent executable paper backtests. Per-period "
            "and regime results are descriptive; sparse groups are inconclusive. "
            "Supplemental periods are additional historical evidence and do not "
            "alter the locked research or validation windows."
        ),
    }
    results["outcome"] = _overall_outcome(
        results["research"],
        results["validation"],
        results["additional"],
    )
    results["promotion"] = {
        label: _promotion_gate(
            label,
            results["research"],
            results["validation"],
            results["additional"],
        )
        for label, _ in VARIANTS
        if label != "control"
    }
    for label, _ in VARIANTS:
        status = (
            ACTIVE_CONTROL_STATUS if label == "control"
            else results["promotion"][label]["status"]
        )
        results["research"]["aggregate"][label]["promotion_status"] = status
        results["validation"]["aggregate"][label]["promotion_status"] = status
        results["additional"]["aggregate"][label]["promotion_status"] = status
    results["control"] = EXIT_CONTROL
    results["control_status"] = ACTIVE_CONTROL_STATUS
    results["promotion_policy"] = (
        "Exit candidates remain research-only unless positive, sufficiently "
        "sampled evidence persists across multiple untouched periods with "
        "no concentration at or above the declared threshold and no worse "
        "cost or drawdown than the active control."
    )
    return results


def _money(value):
    return f"${value:+.4f}"


def _print_period_table(label, analysis):
    print(f"\n=== {label} per-period comparison ===")
    print(
        "Variant | Period | Regime | Trades | Net | Net Δ | "
        "Win% | Max DD | Cost% | Avg dur | Contribution | Evidence"
    )
    for variant, items in analysis["periods"].items():
        for item in items:
            contribution = item["contribution_to_total_improvement_percent"]
            contribution_text = (
                "n/a" if contribution is None else f"{contribution:+.1f}%"
            )
            evidence = (
                "INSUFFICIENT" if item["insufficient_evidence"] else "usable"
            )
            print(
                f"{variant} | {item['period']} | {item['regime']} | "
                f"{item['trades']} | {_money(item['net'])} | "
                f"{_money(item['net_delta_vs_control'])} | "
                f"{item['win_rate']:.1f}% | "
                f"${item['maximum_drawdown']:.4f} | "
                f"{item['cost_share']:.1f}% | "
                f"{item['average_duration']:.2f} | "
                f"{contribution_text} | {evidence}"
            )


def _print_regime_table(label, analysis):
    print(f"\n=== {label} regime comparison ===")
    print(
        "Variant | Regime | Periods | Trades | Net | Net Δ | "
        "Win% | Max DD | Cost% | Avg dur | Evidence"
    )
    for variant, regimes in analysis["regimes"].items():
        for regime, item in regimes.items():
            evidence = (
                "INSUFFICIENT" if item["insufficient_evidence"] else "usable"
            )
            print(
                f"{variant} | {regime} | {item['periods']} | "
                f"{item['trades']} | {_money(item['net'])} | "
                f"{_money(item['net_delta_vs_control'])} | "
                f"{item['win_rate']:.1f}% | "
                f"${item['maximum_drawdown']:.4f} | "
                f"{item['cost_share']:.1f}% | "
                f"{item['average_duration']:.2f} | {evidence}"
            )


def print_report(results):
    print("BTC/CAD 6% TARGET PERIOD ROBUSTNESS STUDY — ANALYSIS ONLY")
    print("REAL-MONEY TRADING: DISABLED")
    print(results["note"])
    for label, analysis in (
        ("Research", results["research"]),
        ("Untouched validation", results["validation"]),
        ("Additional untouched history", results["additional"]),
    ):
        _print_period_table(label, analysis)
        _print_regime_table(label, analysis)
        print(f"\n=== {label} aggregate comparison ===")
        for variant, item in analysis["aggregate"].items():
            print(
                f"{variant}: trades={item['trades']}, net={_money(item['net'])}, "
                f"return={item['return_percent']:+.2f}%, "
                f"drawdown=${item['maximum_drawdown']:.4f}"
            )
        print(f"{label} breadth:", analysis["breadth"])
    print("\n=== Overall outcome ===")
    print(
        f"{results['outcome']['label']} ({results['outcome']['color']}): "
        f"{results['outcome']['reason']}"
    )
    print("\n=== Promotion gate ===")
    control = results.get("control", EXIT_CONTROL)
    control_status = results.get("control_status", ACTIVE_CONTROL_STATUS)
    print(f"Active production control: {control} ({control_status})")
    for label, decision in results.get("promotion", {}).items():
        print(
            f"{label}: {decision['status']} — "
            f"{'; '.join(decision['reasons'])}"
        )
    print("\n=== Interpretation boundary ===")
    print(
        "The 6% target remains a research candidate only. No exit setting, "
        "entry rule, risk control, or production behavior was changed."
    )


def main():
    results = run_exit_parameter_period_robustness_study()
    print_report(results)
    return results


if __name__ == "__main__":
    main()