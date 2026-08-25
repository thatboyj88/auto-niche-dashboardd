from datetime import datetime, timezone

from multi_period_backtest import MultiPeriodBacktester
from strategy_backtest import StrategyBacktester
from yahoo_btc_cad_data import YahooBTCADMarketData


STARTING_CAPITAL = 25.00
FEE_PERCENT = 0.004
SLIPPAGE_PERCENT = 0.001
EXISTING_STOP_PERCENT = 0.02
EXISTING_TARGET_PERCENT = 0.04
WIDER_STOP_PERCENT = 0.04
EXTENDED_HOLD_CANDLES = 20
TIME_EXIT_CANDLES = 5
LOSING_DIAGNOSTIC_HORIZON_CANDLES = 20

STUDY_PERIODS = (
    {
        "label": "Bull Period A",
        "data_range": "10y",
        "start_date": "2023-08-20",
        "end_date": "2024-08-18",
        "regime": "Bull",
    },
    {
        "label": "Bull Period B",
        "data_range": "10y",
        "start_date": "2024-08-19",
        "end_date": "2025-08-18",
        "regime": "Bull",
    },
    {
        "label": "Sideways Period",
        "data_range": "10y",
        "start_date": "2019-08-20",
        "end_date": "2020-08-18",
        "regime": "Sideways",
    },
)


class CounterfactualExitStudy:
    """
    Analyze alternate exits using entries from an unchanged backtest.

    This class never calls the strategy to generate new entries. Every
    scenario reuses the original completed-trade entry candle, entry price,
    and position size, so the output is diagnostic rather than a new strategy.

    Alternate exits can overlap the next original entry. Their totals are
    therefore sums of independent fixed-entry outcomes, not an executable
    portfolio simulation or a new equity curve.
    """

    SCENARIO_LABELS = {
        "existing": "A — Existing strategy",
        "wider_stop": "B — Wider 4% stop",
        "extended_holding": (
            "C — Fixed 20-candle hold (no early stop/target)"
        ),
        "time_exit": "D — 5-candle time exit",
        "mfe_mae": "E — MFE / MAE before original exit",
    }

    def __init__(
        self,
        fee_percent=FEE_PERCENT,
        slippage_percent=SLIPPAGE_PERCENT,
    ):
        self.fee_percent = fee_percent
        self.slippage_percent = slippage_percent

    def analyze_period(self, label, candles):
        original_backtester = StrategyBacktester(
            starting_capital=STARTING_CAPITAL,
            fee_percent=self.fee_percent,
            slippage_percent=self.slippage_percent,
        )
        original_backtester.run(candles)
        original_results = original_backtester.results()

        scenario_results = {}
        for scenario in (
            "existing",
            "wider_stop",
            "extended_holding",
            "time_exit",
        ):
            scenario_trades = [
                self._analyze_trade(
                    trade,
                    candles,
                    scenario,
                )
                for trade in original_results["trades_history"]
            ]
            scenario_results[scenario] = self._summarize_scenario(
                scenario,
                scenario_trades,
                original_results,
            )
        excursions = [
            self._measure_excursion(trade, candles)
            for trade in original_results["trades_history"]
        ]
        scenario_results["mfe_mae"] = self._summarize_excursions(
            excursions,
        )

        return {
            "label": label,
            "start_date": self._format_date(candles[0]["timestamp"]),
            "end_date": self._format_date(candles[-1]["timestamp"]),
            "candles": len(candles),
            "market_return": (
                (candles[-1]["close"] / candles[0]["close"]) - 1
            ) * 100,
            "original_results": original_results,
            "scenarios": scenario_results,
            "losing_trade_diagnostics": [
                self._diagnose_original_loss(trade, candles)
                for trade in original_results["trades_history"]
                if trade["net_profit_loss"] <= 0
            ],
        }

    def _analyze_trade(self, trade, candles, scenario):
        if scenario == "existing":
            return self._copy_existing_trade(trade)

        entry_candle = trade["entry_candle"]
        entry_price = trade["entry_price"]
        final_candle = len(candles) - 1
        exit_candle = None
        exit_reason = None

        if scenario == "wider_stop":
            stop_percent = WIDER_STOP_PERCENT
            target_percent = EXISTING_TARGET_PERCENT
            max_candle = final_candle
        elif scenario == "extended_holding":
            stop_percent = None
            target_percent = None
            max_candle = min(
                entry_candle + EXTENDED_HOLD_CANDLES,
                final_candle,
            )
        elif scenario == "time_exit":
            stop_percent = None
            target_percent = None
            max_candle = min(
                entry_candle + TIME_EXIT_CANDLES,
                final_candle,
            )
        else:
            raise ValueError(f"Unknown counterfactual scenario: {scenario}")

        stop_price = (
            entry_price * (1 - stop_percent)
            if stop_percent is not None
            else None
        )
        target_price = (
            entry_price * (1 + target_percent)
            if target_percent is not None
            else None
        )

        for index in range(entry_candle + 1, max_candle + 1):
            close = candles[index]["close"]
            if stop_price is not None and close <= stop_price:
                exit_candle = index
                exit_reason = "COUNTERFACTUAL STOP LOSS"
                break
            if target_price is not None and close >= target_price:
                exit_candle = index
                exit_reason = "COUNTERFACTUAL TAKE PROFIT"
                break

        if exit_candle is None:
            exit_candle = max_candle
            exit_reason = (
                "COUNTERFACTUAL TIME EXIT"
                if scenario == "time_exit"
                else "COUNTERFACTUAL EXTENDED HOLD EXIT"
            )

        return self._build_trade_result(
            trade,
            candles[exit_candle],
            exit_candle,
            exit_reason,
        )

    def _build_trade_result(
        self,
        original_trade,
        exit_candle,
        exit_index,
        exit_reason,
    ):
        entry_price = original_trade["entry_price"]
        position_size = original_trade["position_size"]
        entry_value = position_size * entry_price
        entry_fee = entry_value * self.fee_percent
        actual_exit_price = (
            exit_candle["close"] *
            (1 - self.slippage_percent)
        )
        gross_value = position_size * actual_exit_price
        exit_fee = gross_value * self.fee_percent
        market_entry_price = (
            entry_price / (1 + self.slippage_percent)
        )
        market_exit_price = (
            exit_candle["close"]
        )
        gross_before_costs = position_size * (
            market_exit_price - market_entry_price
        )
        gross_execution = gross_value - entry_value
        fees = entry_fee + exit_fee
        modeled_slippage = (
            entry_value * self.slippage_percent
        ) + (
            gross_value * self.slippage_percent
        )
        execution_price_impact = (
            gross_before_costs - gross_execution
        )
        net_profit_loss = gross_execution - fees

        return {
            "trade_number": original_trade["trade_number"],
            "entry_candle": original_trade["entry_candle"],
            "exit_candle": exit_index,
            "entry_timestamp": original_trade["entry_timestamp"],
            "exit_timestamp": exit_candle["timestamp"],
            "entry_price": entry_price,
            "exit_price": actual_exit_price,
            "position_size": position_size,
            "gross_profit_loss_before_costs": gross_before_costs,
            "gross_profit_loss": gross_execution,
            "fees": fees,
            "estimated_slippage": modeled_slippage,
            "execution_price_impact": execution_price_impact,
            "net_profit_loss": net_profit_loss,
            "reason": exit_reason,
            "strategy_score": original_trade["strategy_score"],
            "rsi_at_entry": original_trade["rsi_at_entry"],
        }

    def _copy_existing_trade(self, trade):
        market_entry_price = (
            trade["entry_price"] /
            (1 + self.slippage_percent)
        )
        market_exit_price = (
            trade["exit_price"] /
            (1 - self.slippage_percent)
        )
        gross_before_costs = trade["position_size"] * (
            market_exit_price - market_entry_price
        )
        entry_value = trade["position_size"] * trade["entry_price"]
        gross_value = trade["position_size"] * trade["exit_price"]

        return {
            "trade_number": trade["trade_number"],
            "entry_candle": trade["entry_candle"],
            "exit_candle": trade["exit_candle"],
            "entry_timestamp": trade["entry_timestamp"],
            "exit_timestamp": trade["exit_timestamp"],
            "entry_price": trade["entry_price"],
            "exit_price": trade["exit_price"],
            "position_size": trade["position_size"],
            "gross_profit_loss_before_costs": gross_before_costs,
            "gross_profit_loss": trade["gross_profit_loss"],
            "fees": trade["fees"],
            "estimated_slippage": (
                (entry_value * self.slippage_percent) +
                (gross_value * self.slippage_percent)
            ),
            "execution_price_impact": (
                gross_before_costs - trade["gross_profit_loss"]
            ),
            "net_profit_loss": trade["net_profit_loss"],
            "reason": trade["reason"],
            "strategy_score": trade["strategy_score"],
            "rsi_at_entry": trade["rsi_at_entry"],
        }

    def _summarize_scenario(
        self,
        scenario,
        trades,
        original_results,
    ):
        total_trades = len(trades)
        wins = sum(
            1
            for trade in trades
            if trade["net_profit_loss"] > 0
        )
        losses = total_trades - wins
        gross = sum(
            trade["gross_profit_loss_before_costs"]
            for trade in trades
        )
        fees = sum(trade["fees"] for trade in trades)
        slippage = sum(
            trade["estimated_slippage"]
            for trade in trades
        )
        execution_price_impact = sum(
            trade["execution_price_impact"]
            for trade in trades
        )
        net = sum(
            trade["net_profit_loss"]
            for trade in trades
        )

        return {
            "scenario": scenario,
            "label": self.SCENARIO_LABELS[scenario],
            "trades": total_trades,
            "wins": wins,
            "losses": losses,
            "win_rate": (
                (wins / total_trades) * 100
                if total_trades
                else 0.0
            ),
            "gross_profit_before_costs": gross,
            "total_fees": fees,
            "total_slippage": slippage,
            "total_execution_price_impact": execution_price_impact,
            "net_profit": net,
            "average_trade_pnl": (
                net / total_trades
                if total_trades
                else 0.0
            ),
            "max_drawdown": (
                original_results["max_drawdown"]
                if scenario == "existing"
                else None
            ),
            "result_scope": (
                "Original executable backtest"
                if scenario == "existing"
                else (
                    "Sum of independent fixed-entry outcomes; "
                    "not an executable portfolio simulation"
                )
            ),
            "trades_history": trades,
        }

    def _diagnose_original_loss(self, trade, candles):
        excursion = self._measure_excursion(trade, candles)
        entry_index = trade["entry_candle"]
        diagnostic_end_index = min(
            entry_index + LOSING_DIAGNOSTIC_HORIZON_CANDLES,
            len(candles) - 1,
        )
        diagnostic_observation = candles[
            entry_index + 1:diagnostic_end_index + 1
        ]
        if not diagnostic_observation:
            diagnostic_observation = [candles[entry_index]]

        return {
            "trade_number": trade["trade_number"],
            "entry_price": trade["entry_price"],
            "original_exit_price": trade["exit_price"],
            "original_exit_reason": trade["reason"],
            "lowest_price_reached": min(
                candle["low"]
                for candle in diagnostic_observation
            ),
            "highest_price_reached": max(
                candle["high"]
                for candle in diagnostic_observation
            ),
            "diagnostic_horizon_candles": (
                diagnostic_end_index - entry_index
            ),
            "mfe_percent": excursion["mfe_percent"],
            "mae_percent": excursion["mae_percent"],
            "price_after_3_candles": self._close_after(
                candles,
                entry_index,
                3,
            ),
            "price_after_5_candles": self._close_after(
                candles,
                entry_index,
                5,
            ),
            "price_after_10_candles": self._close_after(
                candles,
                entry_index,
                10,
            ),
            "price_after_20_candles": self._close_after(
                candles,
                entry_index,
                20,
            ),
            "strategy_score": trade["strategy_score"],
            "rsi_at_entry": trade["rsi_at_entry"],
            "duration_candles": (
                trade["exit_candle"] - entry_index
            ),
        }

    def _measure_excursion(self, trade, candles):
        entry_index = trade["entry_candle"]
        exit_index = trade["exit_candle"]
        observation = candles[entry_index + 1:exit_index + 1]
        if not observation:
            observation = [candles[entry_index]]

        entry_price = (
            trade["entry_price"] /
            (1 + self.slippage_percent)
        )
        lowest_price = min(
            candle["low"]
            for candle in observation
        )
        highest_price = max(
            candle["high"]
            for candle in observation
        )

        return {
            "trade_number": trade["trade_number"],
            "lowest_price_reached": lowest_price,
            "highest_price_reached": highest_price,
            "mfe_percent": (
                (highest_price / entry_price) - 1
            ) * 100,
            "mae_percent": (
                (lowest_price / entry_price) - 1
            ) * 100,
        }

    def _summarize_excursions(self, excursions):
        trade_count = len(excursions)
        return {
            "scenario": "mfe_mae",
            "label": self.SCENARIO_LABELS["mfe_mae"],
            "trades": trade_count,
            "average_mfe_percent": (
                sum(
                    trade["mfe_percent"]
                    for trade in excursions
                ) / trade_count
                if trade_count
                else 0.0
            ),
            "average_mae_percent": (
                sum(
                    trade["mae_percent"]
                    for trade in excursions
                ) / trade_count
                if trade_count
                else 0.0
            ),
            "trades_history": excursions,
        }

    @staticmethod
    def _close_after(candles, entry_index, offset):
        index = entry_index + offset
        if index >= len(candles):
            return None
        return candles[index]["close"]

    @staticmethod
    def _format_date(timestamp):
        return datetime.fromtimestamp(
            timestamp,
            tz=timezone.utc,
        ).strftime("%Y-%m-%d")


def load_study_periods():
    """
    Load the original Task #2 bull windows and the objective sideways window.
    """
    runner = MultiPeriodBacktester()
    source_candles = {
        data_range: YahooBTCADMarketData(
            data_range=data_range,
        ).load()
        for data_range in {
            specification["data_range"]
            for specification in STUDY_PERIODS
        }
    }
    return select_study_periods(source_candles, runner)


def select_study_periods(source_candles, runner=None):
    """Select and validate the fixed study windows from loaded source data."""
    runner = runner or MultiPeriodBacktester()
    selected_periods = {}

    for specification in STUDY_PERIODS:
        candles = [
            candle
            for candle in source_candles[specification["data_range"]]
            if specification["start_date"] <=
            runner.format_date(candle["timestamp"]) <=
            specification["end_date"]
        ]
        actual_dates = (
            runner.format_date(candles[0]["timestamp"]),
            runner.format_date(candles[-1]["timestamp"]),
        ) if candles else (None, None)
        if (
            len(candles) != 365 or
            actual_dates != (
                specification["start_date"],
                specification["end_date"],
            )
        ):
            raise RuntimeError(
                f"{specification['label']} no longer matches its "
                "recorded 365-candle date boundary"
            )

        regime, _ = runner.classify_regime(candles)
        if regime != specification["regime"]:
            raise RuntimeError(
                f"{specification['label']} was expected to be "
                f"{specification['regime']}, but is now {regime}"
            )
        selected_periods[specification["label"]] = candles

    return selected_periods


def print_report(study_results):
    print("COUNTERFACTUAL EXIT STUDY — ANALYSIS ONLY")
    print("REAL-MONEY TRADING: DISABLED")
    print("")

    for label, result in study_results.items():
        print("--------------------------------")
        print(label)
        print(
            f"Dates: {result['start_date']} to {result['end_date']}"
        )
        print(f"Candles: {result['candles']}")
        print(f"Market return: {result['market_return']:+.2f}%")
        print("")
        print("Scenario results:")

        for scenario in result["scenarios"].values():
            if scenario["scenario"] == "mfe_mae":
                print(
                    f"{scenario['label']}: "
                    f"average MFE {scenario['average_mfe_percent']:+.2f}%, "
                    f"average MAE {scenario['average_mae_percent']:+.2f}%. "
                    "No exit is selected, so P/L, costs, and drawdown are "
                    "not applicable."
                )
                for trade in scenario["trades_history"]:
                    print(
                        f"  Trade {trade['trade_number']}: "
                        f"MFE {trade['mfe_percent']:+.2f}%, "
                        f"MAE {trade['mae_percent']:+.2f}%"
                    )
                continue

            drawdown = (
                f"{scenario['max_drawdown']:.2f}%"
                if scenario["max_drawdown"] is not None
                else "N/A (fixed entries can overlap)"
            )
            print(
                f"{scenario['label']}: "
                f"gross ${scenario['gross_profit_before_costs']:+.4f}, "
                f"wins/losses {scenario['wins']}/{scenario['losses']}, "
                f"win rate {scenario['win_rate']:.2f}%, "
                f"drawdown {drawdown}, "
                f"average trade ${scenario['average_trade_pnl']:+.4f}, "
                f"fees ${scenario['total_fees']:.4f}, "
                f"modeled slippage ${scenario['total_slippage']:.4f}, "
                "execution-price impact "
                f"${scenario['total_execution_price_impact']:.4f}, "
                f"net ${scenario['net_profit']:+.4f}"
            )
            if scenario["scenario"] != "existing":
                print(f"  Scope: {scenario['result_scope']}.")

        print("")
        print("Original losing-trade diagnostics:")
        for loss in result["losing_trade_diagnostics"]:
            print(
                f"Trade {loss['trade_number']}: "
                f"entry ${loss['entry_price']:.2f}, "
                f"exit ${loss['original_exit_price']:.2f}, "
                f"reason {loss['original_exit_reason']}, "
                f"post-entry {loss['diagnostic_horizon_candles']}-candle "
                f"low ${loss['lowest_price_reached']:.2f}, "
                f"high ${loss['highest_price_reached']:.2f}, "
                f"MFE/MAE {loss['mfe_percent']:+.2f}%/"
                f"{loss['mae_percent']:+.2f}%, "
                f"after 3/5/10/20 candles "
                f"{loss['price_after_3_candles']}, "
                f"{loss['price_after_5_candles']}, "
                f"{loss['price_after_10_candles']}, "
                f"{loss['price_after_20_candles']}"
            )


def main():
    periods = load_study_periods()
    study = CounterfactualExitStudy()
    results = {
        label: study.analyze_period(label, candles)
        for label, candles in periods.items()
    }
    print_report(results)
    return results


if __name__ == "__main__":
    main()