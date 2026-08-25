"""Diagnostic-only Phase 2 execution economics study.

Five predeclared fee/slippage scenarios are compared using the unchanged
strategy and independent paper executions. No scenario is executable advice
and this module has no promotion pathway.
"""

from strategy_backtest import StrategyBacktester
from score_effectiveness_study import select_score_study_periods, SCORE_STUDY_PERIODS
from out_of_sample_validation import _split_periods
from yahoo_btc_cad_data import YahooBTCADMarketData


STARTING_CAPITAL = 25.00
SCENARIOS = (
    ("C0", 1.00, 1.00, "Actual current economics"),
    ("C1", 0.75, 0.75, "Mild hypothetical improvement"),
    ("C2", 0.50, 0.50, "Material hypothetical improvement"),
    ("C3", 0.25, 0.25, "Highly favorable hypothetical"),
    ("C4", 0.00, 0.00, "Gross-edge ceiling"),
)
MIN_VALIDATION_TRADES_FOR_CLASSIFICATION = 20
OUTCOME_GROSS_EDGE_AND_COST_SENSITIVE = "GROSS_EDGE_AND_COST_SENSITIVE"
OUTCOME_GROSS_EDGE_NOT_ROBUST = "GROSS_EDGE_NOT_ROBUST"
OUTCOME_INCONCLUSIVE_SMALL_SAMPLE = "INCONCLUSIVE_SMALL_SAMPLE"


def _periods(candles):
    selected = select_score_study_periods(candles)
    if len(selected) != len(SCORE_STUDY_PERIODS):
        raise RuntimeError("Pinned A-J study periods were not all available")
    return _split_periods(selected)


def _run(candles, fee_percent, slippage_percent):
    runner = StrategyBacktester(
        starting_capital=STARTING_CAPITAL,
        fee_percent=fee_percent,
        slippage_percent=slippage_percent,
    )
    runner.run(candles)
    return runner.results()


def _trade_diagnostics(trade, candles, fee_percent, slippage_percent):
    start = trade["entry_candle"]
    end = trade["exit_candle"] if trade["exit_candle"] is not None else start
    entry_market = trade["market_entry_price"]
    path = [candle["close"] for candle in candles[start:end + 1]]
    gross_moves = [(price / entry_market - 1) * 100 for price in path]
    mfe = max(gross_moves, default=0.0)
    mae = min(gross_moves, default=0.0)
    required = (2 * fee_percent + 2 * slippage_percent) * 100
    realized = trade["gross_profit_loss_before_costs"] / (
        trade["position_size"] * entry_market
    ) * 100
    return {
        "gross_profit_loss": trade["gross_profit_loss_before_costs"],
        "fees": trade["fees"],
        "slippage": trade["estimated_slippage"],
        "net_profit_loss": trade["net_profit_loss"],
        "mfe_percent": mfe,
        "mae_percent": mae,
        "required_break_even_percent": required,
        "realized_cleared_cost_hurdle": realized >= required,
        "mfe_cleared_cost_hurdle": mfe >= required,
        "duration": end - start,
        "exit_reason": trade["reason"],
    }


def _summarize(results, candles_by_period, fee_percent, slippage_percent):
    diagnostics = []
    for result, candles in zip(results, candles_by_period):
        diagnostics.extend(
            _trade_diagnostics(
                trade, candles, fee_percent, slippage_percent
            )
            for trade in result["trades_history"]
        )
    gross = sum(item["gross_profit_loss"] for item in diagnostics)
    costs = sum(item["fees"] + item["slippage"] for item in diagnostics)
    return {
        "trades": len(diagnostics),
        "gross_profit_loss": gross,
        "fees": sum(item["fees"] for item in diagnostics),
        "slippage": sum(item["slippage"] for item in diagnostics),
        "net_profit_loss": sum(item["net_profit_loss"] for item in diagnostics),
        "cost_share_percent": costs / abs(gross) * 100 if gross else 0.0,
        "average_mfe_percent": (
            sum(item["mfe_percent"] for item in diagnostics) / len(diagnostics)
            if diagnostics else 0.0
        ),
        "average_mae_percent": (
            sum(item["mae_percent"] for item in diagnostics) / len(diagnostics)
            if diagnostics else 0.0
        ),
        "realized_cleared_hurdle_percent": (
            sum(item["realized_cleared_cost_hurdle"] for item in diagnostics)
            / len(diagnostics) * 100 if diagnostics else 0.0
        ),
        "mfe_cleared_hurdle_percent": (
            sum(item["mfe_cleared_cost_hurdle"] for item in diagnostics)
            / len(diagnostics) * 100 if diagnostics else 0.0
        ),
        "average_duration": (
            sum(item["duration"] for item in diagnostics) / len(diagnostics)
            if diagnostics else 0.0
        ),
        "exit_reasons": {
            reason: sum(item["exit_reason"] == reason for item in diagnostics)
            for reason in sorted({item["exit_reason"] for item in diagnostics})
        },
        "trade_diagnostics": diagnostics,
    }


def run_execution_economics_study():
    candles = YahooBTCADMarketData(data_range="10y").load()
    research, validation = _periods(candles)
    output = {}
    for label, fee_factor, slippage_factor, interpretation in SCENARIOS:
        fee = 0.004 * fee_factor
        slippage = 0.001 * slippage_factor
        research_candles = [period["candles"] for period in research]
        validation_candles = [period["candles"] for period in validation]
        research_results = [
            _run(period, fee, slippage) for period in research_candles
        ]
        validation_results = [
            _run(period, fee, slippage) for period in validation_candles
        ]
        output[label] = {
            "interpretation": interpretation,
            "fees_percent_of_current": fee_factor * 100,
            "slippage_percent_of_current": slippage_factor * 100,
            "research": _summarize(
                research_results, research_candles, fee, slippage
            ),
            "validation": _summarize(
                validation_results, validation_candles, fee, slippage
            ),
        }
    c0 = output["C0"]
    validation_trades = c0["validation"]["trades"]
    if validation_trades < MIN_VALIDATION_TRADES_FOR_CLASSIFICATION:
        outcome = OUTCOME_INCONCLUSIVE_SMALL_SAMPLE
    elif c0["validation"]["gross_profit_loss"] > 0:
        outcome = OUTCOME_GROSS_EDGE_AND_COST_SENSITIVE
    else:
        outcome = OUTCOME_GROSS_EDGE_NOT_ROBUST
    return {
        "scenarios": output,
        "outcome": outcome,
        "promotion_status": "DIAGNOSTIC_ONLY_NO_PROMOTION",
        "production_control": "2.0% stop / 4.0% target",
        "live_trading": False,
    }


def print_report(result):
    print("PHASE 2 EXECUTION ECONOMICS STUDY — DIAGNOSTIC ONLY")
    print("No scenario is a forecast or proposed execution assumption.")
    print("Production control:", result["production_control"])
    for label, data in result["scenarios"].items():
        print(f"\n{label}: {data['interpretation']}")
        print("Research:", data["research"])
        print("Validation:", data["validation"])
    print("\nFinal classification:", result["outcome"])
    print("Promotion:", result["promotion_status"])


if __name__ == "__main__":
    print_report(run_execution_economics_study())