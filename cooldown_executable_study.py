"""Executable paper-backtest study for predetermined trade cooldowns."""

from contextlib import contextmanager

import multi_period_backtest as multi_period_backtest_module
import strategy_backtest as strategy_backtest_module
from multi_period_backtest import MultiPeriodBacktester
from out_of_sample_validation import _split_periods
from score_effectiveness_study import SCORE_STUDY_PERIODS, select_score_study_periods
from strategy_backtest import StrategyBacktester
from strategy_calibration_study import STARTING_CAPITAL
from yahoo_btc_cad_data import YahooBTCADMarketData


COOLDOWNS = (0, 1, 2, 3, 5, 10)


class CooldownStrategyBacktester(StrategyBacktester):
    """Run the original engine while suppressing new entries after exits."""

    def __init__(self, starting_capital, cooldown_candles):
        super().__init__(starting_capital=starting_capital)
        self.cooldown_candles = cooldown_candles
        self.cooldown_remaining = 0

    def close_position(self, *args, **kwargs):
        result = super().close_position(*args, **kwargs)
        self.cooldown_remaining = self.cooldown_candles
        return result

    def run(self, candles):
        original_score = strategy_backtest_module.calculate_strategy_score

        def cooldown_score(*args, **kwargs):
            score, decision, reasons, conditions = original_score(
                *args, **kwargs
            )
            if self.cooldown_remaining > 0:
                self.cooldown_remaining -= 1
                if decision == "BUY CANDIDATE":
                    decision = "NO TRADE"
                    reasons = list(reasons) + [
                        f"Cooldown active: {self.cooldown_candles} candles"
                    ]
            return score, decision, reasons, conditions

        strategy_backtest_module.calculate_strategy_score = cooldown_score
        try:
            return super().run(candles)
        finally:
            strategy_backtest_module.calculate_strategy_score = original_score


class CooldownMultiPeriodBacktester(MultiPeriodBacktester):
    def __init__(self, cooldown_candles, starting_capital=STARTING_CAPITAL):
        super().__init__(starting_capital=starting_capital)
        self.cooldown_candles = cooldown_candles

    def _run_period(self, *args, **kwargs):
        original_backtester = multi_period_backtest_module.StrategyBacktester

        def factory(starting_capital):
            return CooldownStrategyBacktester(
                starting_capital,
                self.cooldown_candles,
            )

        multi_period_backtest_module.StrategyBacktester = factory
        try:
            return super()._run_period(*args, **kwargs)
        finally:
            multi_period_backtest_module.StrategyBacktester = original_backtester


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
        "signals": sum(
            evaluation["decision"] == "BUY"
            for period in period_results
            for evaluation in period["evaluation_history"]
        ),
        "trades": len(trades),
        "gross": gross,
        "fees": fees,
        "slippage": slippage,
        "net": net,
        "return_percent": (
            net / (STARTING_CAPITAL * len(period_results)) * 100
            if period_results else 0.0
        ),
        "maximum_drawdown": max(
            (period["max_drawdown"] for period in period_results),
            default=0.0,
        ),
        "win_rate": (
            sum(period["wins"] for period in period_results)
            / len(trades) * 100
            if trades else 0.0
        ),
        "cost_share": (
            (fees + slippage) / abs(gross) * 100 if gross else 0.0
        ),
    }


def _run_period_group(cooldown, selected, notifier):
    runner = (
        MultiPeriodBacktester(starting_capital=STARTING_CAPITAL)
        if cooldown == 0
        else CooldownMultiPeriodBacktester(cooldown)
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


def run_cooldown_executable_study(notifier=None):
    if notifier is None:
        from btc_cad_preflight import send_slack_notification
        notifier = send_slack_notification
    loader = YahooBTCADMarketData(data_range="10y")
    candles = loader.load()
    if not candles:
        raise RuntimeError(loader.last_error or "Yahoo BTC/CAD data unavailable")
    selected = select_score_study_periods(candles)
    if len(selected) != len(SCORE_STUDY_PERIODS):
        raise RuntimeError("Cooldown study requires all fixed periods")
    research_periods, validation_periods = _split_periods(selected)
    research = {}
    validation = {}
    for cooldown in COOLDOWNS:
        research[cooldown] = _performance(
            _run_period_group(cooldown, research_periods, notifier)
        )
        validation[cooldown] = _performance(
            _run_period_group(cooldown, validation_periods, notifier)
        )
    for dataset in (research, validation):
        control = dataset[0]["net"]
        for cooldown in COOLDOWNS:
            dataset[cooldown]["net_delta_vs_control"] = (
                dataset[cooldown]["net"] - control
            )
    return {
        "real_money_trading": False,
        "cooldowns": COOLDOWNS,
        "research": research,
        "validation": validation,
        "note": (
            "Cooldowns are executable paper-backtest variants. The control "
            "uses the unchanged MultiPeriodBacktester; no production code "
            "or trading rule is changed."
        ),
    }


def _print_group(label, metrics):
    print(f"\n=== {label} ===")
    print(
        "Cooldown | Signals | Trades | Gross | Costs | Net | Return | "
        "Max DD | Win rate | Cost share | Δ vs control"
    )
    for cooldown in COOLDOWNS:
        item = metrics[cooldown]
        print(
            f"{cooldown:>8} | {item['signals']:<7} | {item['trades']:<6} | "
            f"${item['gross']:+.4f} | "
            f"${item['fees'] + item['slippage']:.4f} | "
            f"${item['net']:+.4f} | {item['return_percent']:+.2f}% | "
            f"${item['maximum_drawdown']:.4f} | {item['win_rate']:.2f}% | "
            f"{item['cost_share']:.2f}% | "
            f"${item['net_delta_vs_control']:+.4f}"
        )


def print_report(results):
    print("BTC/CAD EXECUTABLE COOLDOWN STUDY — ANALYSIS ONLY")
    print("REAL-MONEY TRADING: DISABLED")
    print(results["note"])
    _print_group("Research", results["research"])
    _print_group("Untouched validation", results["validation"])
    passing = [
        cooldown
        for cooldown in COOLDOWNS
        if cooldown > 0 and results["research"][cooldown]["net_delta_vs_control"] > 0
        and results["validation"][cooldown]["net_delta_vs_control"] > 0
    ]
    print("\n=== Validation gate ===")
    print(f"Cooldowns improving both splits: {passing if passing else 'none'}")
    print(
        "No cooldown is selected by peak backtest result; a frequency "
        "reduction is useful only if its improvement repeats in validation."
    )


def main():
    results = run_cooldown_executable_study()
    print_report(results)
    return results


if __name__ == "__main__":
    main()