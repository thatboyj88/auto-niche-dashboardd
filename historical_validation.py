"""Read-only comparison helpers for historical and genuine paper evidence."""

from __future__ import annotations

from collections import Counter
from typing import Any

from config import FEE_PERCENT, SLIPPAGE_PERCENT, STARTING_CAPITAL
from strategy_backtest import StrategyBacktester


CONSISTENCY_COST_CASES = (
    ("Baseline costs", FEE_PERCENT, SLIPPAGE_PERCENT),
    ("Zero costs", 0.0, 0.0),
    ("Double costs", FEE_PERCENT * 2, SLIPPAGE_PERCENT * 2),
)


def _decision_signature(result: dict[str, Any]) -> tuple[tuple[Any, ...], ...]:
    return tuple(
        (
            evaluation["timestamp"],
            evaluation["strategy_score"],
            evaluation["decision"],
            evaluation["long_term_trend"],
            evaluation["short_term_momentum"],
            evaluation["rsi_condition"],
            evaluation["volume"],
            evaluation["price_above_ema21"],
        )
        for evaluation in result["evaluation_history"]
    )


def measure_decision_consistency(
    period_candles: list[list[dict[str, Any]]],
) -> dict[str, Any]:
    """Replay fixed periods and boundary slices without mutating production state."""
    cases = []
    for period_index, candles in enumerate(period_candles):
        boundary_inputs = [
            ("Exact period", candles),
            ("Leading boundary removed", candles[1:]),
            ("Trailing boundary removed", candles[:-1]),
        ]
        for boundary_label, boundary_candles in boundary_inputs:
            if len(boundary_candles) < 201:
                continue
            for cost_label, fee, slippage in CONSISTENCY_COST_CASES:
                first = StrategyBacktester(
                    STARTING_CAPITAL,
                    fee_percent=fee,
                    slippage_percent=slippage,
                )
                second = StrategyBacktester(
                    STARTING_CAPITAL,
                    fee_percent=fee,
                    slippage_percent=slippage,
                )
                first.run(boundary_candles)
                second.run(boundary_candles)
                first_signature = _decision_signature(first.results())
                second_signature = _decision_signature(second.results())
                cases.append({
                    "period": f"Period {chr(ord('A') + period_index)}",
                    "boundary": boundary_label,
                    "cost_case": cost_label,
                    "evaluation_count": len(first_signature),
                    "repeatable": first_signature == second_signature,
                })

    repeatable = sum(case["repeatable"] for case in cases)
    return {
        "cases": cases,
        "case_count": len(cases),
        "repeatable_cases": repeatable,
        "non_repeatable_cases": len(cases) - repeatable,
        "repeatability_percent": (
            repeatable / len(cases) * 100 if cases else 0.0
        ),
        "conclusion": (
            "PASS: repeated decision outputs are identical across all "
            "tested cost and boundary cases."
            if cases and repeatable == len(cases)
            else
            "BLOCKED: no eligible consistency cases were available."
            if not cases
            else
            "FAIL: at least one repeated decision output changed."
        ),
    }


def summarize_genuine_paper_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize validated paper records without changing the observation store."""
    paper_records = [
        record
        for record in records
        if record.get("dataset") == "PAPER_OPERATIONAL"
    ]
    signals = [
        record.get("payload", {})
        for record in paper_records
        if record.get("record_type") == "SIGNAL"
    ]
    trades = [
        record.get("payload", {})
        for record in paper_records
        if record.get("record_type") == "TRADE"
    ]
    profits = [
        float(trade["profit_loss"])
        for trade in trades
        if isinstance(trade.get("profit_loss"), (int, float))
    ]
    wins = sum(profit > 0 for profit in profits)
    losses = sum(profit <= 0 for profit in profits)
    scores = [
        float(signal["strategy_score"])
        for signal in signals
        if isinstance(signal.get("strategy_score"), (int, float))
    ]
    drawdowns = [
        float(record["payload"]["max_drawdown_percent"])
        for record in paper_records
        if isinstance(
            record.get("payload", {}).get("max_drawdown_percent"),
            (int, float),
        )
    ]
    condition_groups: dict[str, list[dict[str, Any]]] = {}
    for trade in trades:
        condition = trade.get("market_condition")
        if condition in {"Bull", "Sideways", "Bear"}:
            condition_groups.setdefault(condition, []).append(trade)

    return {
        "records": len(paper_records),
        "signals": len(signals),
        "trades": len(trades),
        "wins": wins,
        "losses": losses,
        "win_rate": (wins / len(profits) * 100) if profits else 0.0,
        "profit": sum(profits),
        "fees": sum(
            float(trade["fees"])
            for trade in trades
            if isinstance(trade.get("fees"), (int, float))
        ),
        "slippage": sum(
            float(trade["slippage"])
            for trade in trades
            if isinstance(trade.get("slippage"), (int, float))
        ),
        "strategy_score_distribution": dict(
            sorted(Counter(scores).items(), key=lambda item: item[0])
        ),
        "max_drawdown": max(drawdowns) if drawdowns else None,
        "market_condition_performance": (
            {
                condition: {
                    "trades": len(group),
                    "wins": sum(
                        trade.get("profit_loss", 0) > 0
                        for trade in group
                    ),
                    "profit": sum(
                        trade.get("profit_loss", 0.0)
                        for trade in group
                    ),
                }
                for condition, group in sorted(condition_groups.items())
            }
            if condition_groups
            else None
        ),
    }


def summarize_historical_results(results: dict[str, Any]) -> dict[str, Any]:
    """Flatten multi-period results for a side-by-side evidence comparison."""
    periods = results.get("periods", [])
    aggregate = results.get("aggregate", {})
    trades = sum(period.get("trades", 0) for period in periods)
    wins = sum(period.get("wins", 0) for period in periods)
    evaluated_scores = [
        evaluation.get("strategy_score")
        for period in periods
        for evaluation in period.get("evaluation_history", [])
        if isinstance(evaluation.get("strategy_score"), (int, float))
    ]
    if not evaluated_scores:
        evaluated_scores = [
            trade.get("strategy_score")
            for period in periods
            for trade in period.get("trades_history", [])
            if isinstance(trade.get("strategy_score"), (int, float))
        ]
    return {
        "records": results.get("source_candles", 0),
        "periods": len(periods),
        "trades": trades,
        "wins": wins,
        "losses": max(0, trades - wins),
        "win_rate": (wins / trades * 100) if trades else 0.0,
        "profit": aggregate.get("total_profit", 0.0),
        "fees": aggregate.get("total_fees", 0.0),
        "slippage": aggregate.get("total_slippage", 0.0),
        "max_drawdown": aggregate.get("worst_drawdown", 0.0),
        "strategy_score_distribution": dict(
            sorted(
                Counter(evaluated_scores).items(),
                key=lambda item: item[0],
            )
        ),
        "market_condition_performance": {
            regime: {
                "periods": len(regime_periods),
                "trades": sum(
                    period.get("trades", 0) for period in regime_periods
                ),
                "profit": sum(
                    period.get("profit", 0.0) for period in regime_periods
                ),
            }
            for regime, regime_periods in results.get(
                "regime_summary", {}
            ).items()
        },
    }