"""Executable robustness study for isolated stop-loss/take-profit parameters."""

from statistics import median

import multi_period_backtest as multi_period_backtest_module
import strategy_backtest as strategy_backtest_module
from indicators import (
    calculate_average_volume,
    calculate_ema,
    calculate_rsi,
)
from multi_period_backtest import MultiPeriodBacktester
from out_of_sample_validation import _split_periods
from score_effectiveness_study import SCORE_STUDY_PERIODS, select_score_study_periods
from strategy_backtest import StrategyBacktester
from strategy_calibration_study import STARTING_CAPITAL
from trade_path_exit_timing_study import TradePathExitTimingStudy
from yahoo_btc_cad_data import YahooBTCADMarketData
from config import EXIT_CONTROL


STOP_LOSSES = (1.5, 2.0, 2.5, 3.0)
TAKE_PROFITS = (3.0, 4.0, 5.0, 6.0)
EXIT_GRID = tuple(
    (stop_loss, take_profit)
    for stop_loss in STOP_LOSSES
    for take_profit in TAKE_PROFITS
)
CONTROL = EXIT_CONTROL
RESEARCH_ONLY_STATUS = "RESEARCH_ONLY"
ACTIVE_CONTROL_STATUS = "ACTIVE_PRODUCTION_CONTROL"


class ExitParameterStrategyBacktester(StrategyBacktester):
    """Isolated copy of the control loop with only exit levels injected."""

    def __init__(self, starting_capital, stop_loss_percent, take_profit_percent):
        super().__init__(starting_capital=starting_capital)
        self.stop_loss_percent = stop_loss_percent / 100
        self.take_profit_percent = take_profit_percent / 100

    def run(self, candles):
        for i in range(len(candles)):
            if i < 200:
                continue
            candle = candles[i]
            current_day = candle["timestamp"] // 24
            if current_day != self.current_day:
                self.current_day = current_day
                self.trades_today = 0
                self.daily_starting_capital = self.capital
            historical = candles[:i + 1]
            prices = [item["close"] for item in historical]
            volumes = [item["volume"] for item in historical]
            current_price = prices[-1]
            current_volume = volumes[-1]
            ema_9 = calculate_ema(prices, 9)
            ema_21 = calculate_ema(prices, 21)
            ema_50 = calculate_ema(prices, 50)
            ema_200 = calculate_ema(prices, 200)
            rsi = calculate_rsi(prices)
            if rsi is not None:
                self.lowest_rsi = min(self.lowest_rsi, rsi)
                self.highest_rsi = max(self.highest_rsi, rsi)
            average_volume = calculate_average_volume(volumes)
            if None in (ema_9, ema_21, ema_50, ema_200, rsi, average_volume):
                continue
            self.evaluations += 1
            score, decision, reasons, conditions = (
                strategy_backtest_module.calculate_strategy_score(
                    ema_9, ema_21, ema_50, ema_200, rsi,
                    current_price, average_volume, current_volume
                )
            )
            strategy_decision = "BUY" if decision == "BUY CANDIDATE" else "NO TRADE"
            self.evaluation_history.append({
                "evaluation_number": self.evaluations,
                "candle": i,
                "timestamp": candle["timestamp"],
                "strategy_score": score,
                "decision": strategy_decision,
                "long_term_trend": conditions["long_term_trend"],
                "short_term_momentum": conditions["short_term_momentum"],
                "rsi_condition": conditions["rsi"],
                "volume": conditions["volume"],
                "price_above_ema21": conditions["price_above_ema21"],
                "rsi": rsi,
                "ema21": ema_21,
                "ema50": ema_50,
                "ema200": ema_200,
                "current_price": current_price,
            })
            if self.position > 0:
                stop_price = self.entry_price * (1 - self.stop_loss_percent)
                target_price = self.entry_price * (1 + self.take_profit_percent)
                if current_price <= stop_price:
                    self.close_position(current_price, "STOP LOSS", i, candle["timestamp"])
                    self._record_equity(current_price)
                    continue
                if current_price >= target_price:
                    self.close_position(current_price, "TAKE PROFIT", i, candle["timestamp"])
                    self._record_equity(current_price)
                    continue
            if self.position == 0:
                self.highest_score = max(self.highest_score, score)
                if score >= 80:
                    self.score_80_or_more += 1
                for condition, passed in conditions.items():
                    if passed:
                        self.condition_counts[condition] += 1
                if decision != "BUY CANDIDATE":
                    self._record_equity(current_price)
                    continue
                if self.trades_today >= 3:
                    self._record_equity(current_price)
                    continue
                daily_loss = self.daily_starting_capital - self.capital
                if daily_loss >= self.daily_starting_capital * 0.03:
                    self._record_equity(current_price)
                    continue
                position_value = self.capital * 0.40
                actual_entry_price = current_price * (1 + self.slippage_percent)
                entry_fee = position_value * self.fee_percent
                total_entry_cost = position_value + entry_fee
                if total_entry_cost > self.capital:
                    self._record_equity(current_price)
                    continue
                self.position = position_value / actual_entry_price
                self.entry_price = actual_entry_price
                self.entry_value = position_value
                self.entry_candle = i
                self.entry_timestamp = candle["timestamp"]
                self.entry_score = score
                self.entry_decision = strategy_decision
                self.entry_rsi = rsi
                self.entry_fee = entry_fee
                self.entry_slippage = position_value * self.slippage_percent
                self.capital -= total_entry_cost
                self.total_fees += entry_fee
                self.total_slippage += self.entry_slippage
                self.trades_today += 1
            self._record_equity(current_price)
        if self.position > 0:
            self.close_position(
                candles[-1]["close"], "END OF TEST", len(candles) - 1,
                candles[-1]["timestamp"]
            )
            if self.equity_curve:
                self.equity_curve[-1] = self.capital


class ExitParameterMultiPeriodBacktester(MultiPeriodBacktester):
    def __init__(self, stop_loss, take_profit):
        super().__init__(starting_capital=STARTING_CAPITAL)
        self.stop_loss = stop_loss
        self.take_profit = take_profit

    def _run_period(self, *args, **kwargs):
        original = multi_period_backtest_module.StrategyBacktester

        def factory(starting_capital):
            return ExitParameterStrategyBacktester(
                starting_capital, self.stop_loss, self.take_profit
            )

        multi_period_backtest_module.StrategyBacktester = factory
        try:
            return super()._run_period(*args, **kwargs)
        finally:
            multi_period_backtest_module.StrategyBacktester = original


def _metrics(period_pairs):
    path_study = TradePathExitTimingStudy()
    trades = []
    for result, candles in period_pairs:
        for trade in result["trades_history"]:
            path = path_study.analyze_trade(
                trade, candles, result["period"], result["regime"]
            )
            trades.append({
                "gross": trade["gross_profit_loss_before_costs"],
                "net": trade["net_profit_loss"],
                "fees": trade["fees"],
                "slippage": trade["estimated_slippage"],
                "duration": trade["exit_candle"] - trade["entry_candle"],
                "mfe": path["mfe_percent"],
                "mae": path["mae_percent"],
                "reason": trade["reason"],
            })
    gross = sum(item["gross"] for item in trades)
    net = sum(item["net"] for item in trades)
    fees = sum(item["fees"] for item in trades)
    slippage = sum(item["slippage"] for item in trades)
    durations = [item["duration"] for item in trades]
    return {
        "signals": sum(
            evaluation["decision"] == "BUY"
            for result, _ in period_pairs
            for evaluation in result["evaluation_history"]
        ),
        "trades": len(trades),
        "gross": gross,
        "fees": fees,
        "slippage": slippage,
        "net": net,
        "return_percent": net / (STARTING_CAPITAL * len(period_pairs)) * 100 if period_pairs else 0.0,
        "maximum_drawdown": max(
            (result["max_drawdown"] for result, _ in period_pairs), default=0.0
        ),
        "win_rate": sum(item["net"] > 0 for item in trades) / len(trades) * 100 if trades else 0.0,
        "net_per_trade": net / len(trades) if trades else 0.0,
        "cost_share": (fees + slippage) / abs(gross) * 100 if gross else 0.0,
        "average_duration": sum(durations) / len(durations) if durations else 0.0,
        "median_duration": median(durations) if durations else 0.0,
        "average_mfe": sum(item["mfe"] for item in trades) / len(trades) if trades else 0.0,
        "average_mae": sum(item["mae"] for item in trades) / len(trades) if trades else 0.0,
        "stop_losses": sum(item["reason"] == "STOP LOSS" for item in trades),
        "take_profits": sum(item["reason"] == "TAKE PROFIT" for item in trades),
        "end_of_test": sum(item["reason"] == "END OF TEST" for item in trades),
    }


def _run_variant(stop_loss, take_profit, selected, notifier):
    runner = ExitParameterMultiPeriodBacktester(stop_loss, take_profit)
    return [
        (
            runner._run_period(
                index, period["candles"], period_label=period["period"],
                source_label="Yahoo Finance BTC/CAD fixed ten-year study",
                source_kind="fixed-study", notifier=notifier
            ),
            period["candles"],
        )
        for index, period in enumerate(selected)
    ]


def run_exit_parameter_robustness_study(notifier=None):
    if notifier is None:
        notifier = lambda *_args: None
    candles = YahooBTCADMarketData(data_range="10y").load()
    if not candles:
        raise RuntimeError("Yahoo BTC/CAD data unavailable")
    selected = select_score_study_periods(candles)
    if len(selected) != len(SCORE_STUDY_PERIODS):
        raise RuntimeError("Exit study requires all fixed periods")
    research_periods, validation_periods = _split_periods(selected)
    research = {}
    validation = {}
    for stop_loss, take_profit in EXIT_GRID:
        key = f"stop_{stop_loss:.1f}_target_{take_profit:.1f}"
        research[key] = _metrics(_run_variant(stop_loss, take_profit, research_periods, notifier))
        validation[key] = _metrics(_run_variant(stop_loss, take_profit, validation_periods, notifier))
        research[key]["stop_loss"] = stop_loss
        research[key]["take_profit"] = take_profit
        validation[key]["stop_loss"] = stop_loss
        validation[key]["take_profit"] = take_profit
    for dataset in (research, validation):
        control = dataset["stop_2.0_target_4.0"]
        for item in dataset.values():
            item["net_delta_vs_control"] = item["net"] - control["net"]
            item["return_delta_vs_control"] = item["return_percent"] - control["return_percent"]
            item["drawdown_delta_vs_control"] = item["maximum_drawdown"] - control["maximum_drawdown"]
            item["promotion_status"] = (
                ACTIVE_CONTROL_STATUS
                if (item["stop_loss"], item["take_profit"]) == CONTROL
                else RESEARCH_ONLY_STATUS
            )
    return {
        "real_money_trading": False,
        "grid": EXIT_GRID,
        "control": CONTROL,
        "control_status": ACTIVE_CONTROL_STATUS,
        "promotion_policy": (
            "Candidates are research-only until the period robustness study "
            "passes its untouched-period, concentration, cost, and drawdown gate."
        ),
        "research": research,
        "validation": validation,
    }


def _print_group(label, data):
    print(f"\n=== {label} ===")
    print("Stop/Target | Signals | Trades | Gross | Fees | Slip | Net | Return | Max DD | Win% | Net/trade | Cost% | AvgDur | MedDur | MFE | MAE | Status | Exits")
    for key, item in data.items():
        print(
            f"{item['stop_loss']:.1f}/{item['take_profit']:.1f} | "
            f"{item['signals']} | {item['trades']} | ${item['gross']:+.4f} | "
            f"${item['fees']:.4f} | ${item['slippage']:.4f} | ${item['net']:+.4f} | "
            f"{item['return_percent']:+.2f}% | ${item['maximum_drawdown']:.4f} | "
            f"{item['win_rate']:.2f}% | ${item['net_per_trade']:+.4f} | "
            f"{item['cost_share']:.2f}% | {item['average_duration']:.2f} | "
            f"{item['median_duration']:.2f} | {item['average_mfe']:+.2f}% | "
            f"{item['average_mae']:+.2f}% | "
            f"{item['promotion_status']} | "
            f"SL={item['stop_losses']},TP={item['take_profits']},EOT={item['end_of_test']}"
        )


def print_report(results):
    print("BTC/CAD EXIT-PARAMETER ROBUSTNESS STUDY — ANALYSIS ONLY")
    print("REAL-MONEY TRADING: DISABLED")
    _print_group("Research", results["research"])
    _print_group("Untouched validation", results["validation"])
    control_key = "stop_2.0_target_4.0"
    research_best = max(results["research"], key=lambda key: results["research"][key]["net"])
    validation_best = max(results["validation"], key=lambda key: results["validation"][key]["net"])
    both = [
        key for key in results["research"]
        if key != control_key
        and results["research"][key]["net_delta_vs_control"] > 0
        and results["validation"][key]["net_delta_vs_control"] > 0
        and results["validation"][key]["drawdown_delta_vs_control"] <= 0
    ]
    print("\n=== Robustness summary ===")
    print(f"Best research candidate: {research_best}")
    print(f"Best validation candidate: {validation_best}")
    print(f"Candidates improving both with no higher validation drawdown: {both or 'none'}")
    print("Classification: PROMISING only when research, untouched validation, risk, and evidence criteria all hold.")
    print("Recommendation: keep the original 2.0% stop / 4.0% target unchanged unless a candidate satisfies the full robustness rules.")
    print("\n=== Interpretation boundary ===")
    print("All 16 variants are isolated executable paper backtests. No production strategy or execution behavior was changed.")


def main():
    results = run_exit_parameter_robustness_study()
    print_report(results)
    return results


if __name__ == "__main__":
    main()