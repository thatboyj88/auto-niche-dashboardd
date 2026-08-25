"""Analysis-only trade-path and original-exit timing study."""

from statistics import median

from config import STOP_LOSS_PERCENT, TAKE_PROFIT_PERCENT
from multi_period_backtest import MultiPeriodBacktester
from out_of_sample_validation import _split_periods
from score_effectiveness_study import SCORE_STUDY_PERIODS, select_score_study_periods
from strategy_calibration_study import (
    FEE_PERCENT,
    SLIPPAGE_PERCENT,
    STARTING_CAPITAL,
)
from yahoo_btc_cad_data import YahooBTCADMarketData


HORIZONS = (1, 2, 3, 5, 10, 20)
TARGETS = (("break_even", 1.005), ("two_percent", 2.0), ("four_percent", 4.0))
EXIT_TIMING_CATEGORIES = ("before_strongest", "near_strongest", "after_strongest")


def _average(values):
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else 0.0


def _median(values):
    values = [value for value in values if value is not None]
    return median(values) if values else 0.0


def _percent(numerator, denominator):
    return numerator / denominator * 100 if denominator else 0.0


def _summary(values):
    values = [value for value in values if value is not None]
    return {
        "count": len(values),
        "average": _average(values),
        "median": _median(values),
    }


class TradePathExitTimingStudy:
    """Describe full available paths around unchanged control exits."""

    def analyze_trade(self, trade, candles, period, regime):
        entry = trade["entry_candle"]
        exit_index = trade["exit_candle"]
        if not 0 <= entry <= exit_index < len(candles):
            raise ValueError("trade candle indexes are outside the candle set")
        entry_price = trade["market_entry_price"]
        target_price = entry_price * (1 + TAKE_PROFIT_PERCENT)
        stop_price = entry_price * (1 - STOP_LOSS_PERCENT)
        path = candles[entry + 1:]
        trade_path = candles[entry + 1:exit_index + 1]
        mfe_index, mfe_percent = self._extreme(
            path, entry_price, "high", entry + 1
        )
        mae_index, mae_percent = self._extreme(
            path, entry_price, "low", entry + 1
        )
        target_levels = dict(TARGETS)
        target_reach = {
            name: self._first_level(
                path,
                entry + 1,
                entry_price * (1 + level / 100),
            )
            for name, level in target_levels.items()
        }
        original_exit_before = {
            name: bool(item["reached"] and item["candle"] >= exit_index)
            for name, item in target_reach.items()
        }
        post_exit = candles[exit_index + 1:]
        if trade["reason"] == "STOP LOSS":
            post_exit_targets = {
                "recovered_entry": any(item["high"] >= entry_price for item in post_exit),
                "reached_two_percent": any(
                    item["high"] >= entry_price * 1.02 for item in post_exit
                ),
                "reached_four_percent": any(
                    item["high"] >= entry_price * 1.04 for item in post_exit
                ),
            }
        elif trade["reason"] == "TAKE PROFIT":
            post_exit_targets = {
                "fell_below_entry": any(
                    item["low"] < entry_price for item in post_exit
                ),
                "fell_below_original_target": any(
                    item["low"] < target_price for item in post_exit
                ),
                "reached_two_percent_beyond_target": any(
                    item["high"] >= entry_price * 1.06 for item in post_exit
                ),
                "reached_four_percent_beyond_target": any(
                    item["high"] >= entry_price * 1.08 for item in post_exit
                ),
            }
        else:
            post_exit_targets = {
                "fell_below_entry": None,
                "fell_below_original_target": None,
                "reached_two_percent_beyond_target": None,
                "reached_four_percent_beyond_target": None,
            }
        hypothetical = {
            name: self._hypothetical_exit(
                trade,
                item,
                entry_price * (1 + target_levels[name] / 100),
            )
            for name, item in target_reach.items()
            if item["reached"] and item["candle"] <= exit_index
        }
        return {
            "trade_number": trade["trade_number"],
            "period": period,
            "regime": regime,
            "entry_price": entry_price,
            "entry_score": trade["strategy_score"],
            "entry_rsi": trade["rsi_at_entry"],
            "original_exit": trade["market_exit_price"],
            "exit_reason": trade["reason"],
            "exit_price": trade["market_exit_price"],
            "entry_candle": entry,
            "exit_candle": exit_index,
            "candles_held": exit_index - entry,
            "net_profit_loss": trade["net_profit_loss"],
            "mfe_percent": mfe_percent,
            "mae_percent": mae_percent,
            "mfe_candle": mfe_index,
            "mae_candle": mae_index,
            "time_to_mfe": mfe_index - entry if mfe_index is not None else None,
            "time_to_mae": mae_index - entry if mae_index is not None else None,
            "early_movement_percent": {
                horizon: (
                    (candles[entry + horizon]["close"] / entry_price - 1) * 100
                    if entry + horizon < len(candles) else None
                )
                for horizon in HORIZONS
            },
            "target_reach": target_reach,
            "original_exit_before_target": original_exit_before,
            "target_before_stop": (
                trade["reason"] == "TAKE PROFIT"
            ),
            "stop_before_target": (
                trade["reason"] == "STOP LOSS"
            ),
            "post_exit_stop_first": post_exit_targets,
            "hypothetical_exits": hypothetical,
            "exit_timing": self._exit_timing(exit_index, mfe_index),
            "trade_path_candles": len(trade_path),
            "stop_price": stop_price,
            "target_price": target_price,
        }

    @staticmethod
    def _extreme(candles, entry_price, field, start_index):
        if not candles:
            return None, 0.0
        selector = max if field == "high" else min
        index, price = selector(
            enumerate(candles),
            key=lambda item: item[1][field],
        )
        return start_index + index, (price[field] / entry_price - 1) * 100

    @staticmethod
    def _first_level(candles, start_index, level):
        for offset, candle in enumerate(candles):
            if candle["high"] >= level:
                return {
                    "reached": True,
                    "candle": start_index + offset,
                    "candles_from_entry": offset + 1,
                    "price": level,
                }
        return {
            "reached": False,
            "candle": None,
            "candles_from_entry": None,
            "price": level,
        }

    @staticmethod
    def _hypothetical_exit(trade, target, target_price):
        position = trade["position_size"]
        entry_value = position * trade["market_entry_price"]
        exit_value = position * target_price
        gross = exit_value - entry_value
        fees = (entry_value + exit_value) * FEE_PERCENT
        slippage = (entry_value + exit_value) * SLIPPAGE_PERCENT
        hypothetical_net = gross - fees - slippage
        return {
            "target_price": target_price,
            "candle": target["candle"],
            "gross_profit_loss": gross,
            "fees": fees,
            "slippage": slippage,
            "hypothetical_net_profit_loss": hypothetical_net,
            "improvement_vs_original": hypothetical_net - trade["net_profit_loss"],
        }

    @staticmethod
    def _exit_timing(exit_index, mfe_index):
        if mfe_index is None:
            return "near_strongest"
        distance = mfe_index - exit_index
        if distance > 1:
            return "before_strongest"
        if distance < -1:
            return "after_strongest"
        return "near_strongest"

    def analyze_group(self, period_results, period_candles):
        if len(period_results) != len(period_candles):
            raise ValueError("period result and candle counts must match")
        periods = []
        for result, candles in zip(period_results, period_candles):
            trades = [
                self.analyze_trade(
                    trade, candles, result["period"], result["regime"]
                )
                for trade in result["trades_history"]
            ]
            periods.append({
                "period": result["period"],
                "start_date": result["start_date"],
                "end_date": result["end_date"],
                "regime": result["regime"],
                "trades": trades,
                "summary": self.summarize_trades(trades),
            })
        trades = [trade for period in periods for trade in period["trades"]]
        return {
            "period_count": len(periods),
            "trade_count": len(trades),
            "periods": periods,
            "summary": self.summarize_trades(trades),
        }

    def summarize_trades(self, trades):
        groups = {
            "winners": [trade for trade in trades if trade["net_profit_loss"] > 0],
            "losers": [trade for trade in trades if trade["net_profit_loss"] <= 0],
            "stop_first": [trade for trade in trades if trade["stop_before_target"]],
            "target_first": [trade for trade in trades if trade["target_before_stop"]],
            "end_of_test": [trade for trade in trades if trade["exit_reason"] == "END OF TEST"],
        }
        return {
            "trade_count": len(trades),
            "exit_reasons": {
                reason: sum(trade["exit_reason"] == reason for trade in trades)
                for reason in ("TAKE PROFIT", "STOP LOSS", "END OF TEST")
            },
            "mfe": _summary(trade["mfe_percent"] for trade in trades),
            "mae": _summary(trade["mae_percent"] for trade in trades),
            "duration": _summary(trade["candles_held"] for trade in trades),
            "time_to_mfe": _summary(trade["time_to_mfe"] for trade in trades),
            "time_to_mae": _summary(trade["time_to_mae"] for trade in trades),
            "early_movement": {
                horizon: _summary(
                    trade["early_movement_percent"][horizon] for trade in trades
                )
                for horizon in HORIZONS
            },
            "exit_timing": {
                category: sum(
                    trade["exit_timing"] == category for trade in trades
                )
                for category in EXIT_TIMING_CATEGORIES
            },
            "groups": {
                name: self._group_summary(group)
                for name, group in groups.items()
            },
            "stop_first_recovery": {
                name: sum(trade["post_exit_stop_first"][name] for trade in groups["stop_first"])
                for name in ("recovered_entry", "reached_two_percent", "reached_four_percent")
            },
            "target_first_aftermath": {
                name: sum(
                    trade["post_exit_stop_first"][name]
                    for trade in groups["target_first"]
                )
                for name in (
                    "fell_below_entry",
                    "fell_below_original_target",
                    "reached_two_percent_beyond_target",
                    "reached_four_percent_beyond_target",
                )
            },
            "counterfactual": self._counterfactual_summary(trades),
        }

    @staticmethod
    def _group_summary(trades):
        return {
            "count": len(trades),
            "mfe": _summary(trade["mfe_percent"] for trade in trades),
            "mae": _summary(trade["mae_percent"] for trade in trades),
            "duration": _summary(trade["candles_held"] for trade in trades),
            "time_to_mfe": _summary(trade["time_to_mfe"] for trade in trades),
            "time_to_mae": _summary(trade["time_to_mae"] for trade in trades),
            "entry_score": _summary(trade["entry_score"] for trade in trades),
            "entry_rsi": _summary(trade["entry_rsi"] for trade in trades),
            "early_movement": {
                horizon: _summary(
                    trade["early_movement_percent"][horizon] for trade in trades
                )
                for horizon in (1, 2, 3, 5)
            },
        }

    @staticmethod
    def _counterfactual_summary(trades):
        result = {}
        for name, _ in TARGETS:
            exits = [
                trade["hypothetical_exits"][name]
                for trade in trades
                if name in trade["hypothetical_exits"]
            ]
            result[name] = {
                "count": len(exits),
                "hypothetical_net_profit_loss": sum(
                    item["hypothetical_net_profit_loss"] for item in exits
                ),
                "improvement_vs_original": sum(
                    item["improvement_vs_original"] for item in exits
                ),
            }
        return result


def _run_period_group(selected, notifier):
    runner = MultiPeriodBacktester(starting_capital=STARTING_CAPITAL)
    return [
        runner._run_period(
            index, period["candles"], period_label=period["period"],
            source_label="Yahoo Finance BTC/CAD fixed ten-year study",
            source_kind="fixed-study", notifier=notifier,
        )
        for index, period in enumerate(selected)
    ]


def run_trade_path_exit_timing_study(notifier=None):
    if notifier is None:
        from btc_cad_preflight import send_slack_notification
        notifier = send_slack_notification
    loader = YahooBTCADMarketData(data_range="10y")
    candles = loader.load()
    if not candles:
        raise RuntimeError(loader.last_error or "Yahoo BTC/CAD data unavailable")
    selected = select_score_study_periods(candles)
    if len(selected) != len(SCORE_STUDY_PERIODS):
        raise RuntimeError("Trade path study requires all fixed periods")
    research, validation = _split_periods(selected)
    study = TradePathExitTimingStudy()
    research_results = _run_period_group(research, notifier)
    validation_results = _run_period_group(validation, notifier)
    return {
        "real_money_trading": False,
        "split": {
            "research_start": research[0]["start_date"],
            "research_end": research[-1]["end_date"],
            "validation_start": validation[0]["start_date"],
            "validation_end": validation[-1]["end_date"],
        },
        "research": study.analyze_group(
            research_results, [period["candles"] for period in research]
        ),
        "validation": study.analyze_group(
            validation_results, [period["candles"] for period in validation]
        ),
    }


def print_report(results):
    print("BTC/CAD TRADE PATH & EXIT TIMING STUDY — ANALYSIS ONLY")
    print("REAL-MONEY TRADING: DISABLED")
    for label, group in (("Research", results["research"]), ("Validation", results["validation"])):
        summary = group["summary"]
        print(f"\n=== {label} ===")
        print(
            f"trades={summary['trade_count']}, exits={summary['exit_reasons']}, "
            f"MFE avg={summary['mfe']['average']:+.2f}%, "
            f"MAE avg={summary['mae']['average']:+.2f}%"
        )
        print(
            "exit timing: "
            + ", ".join(
                f"{key}={value} ({_percent(value, summary['trade_count']):.2f}%)"
                for key, value in summary["exit_timing"].items()
            )
        )
        for name, item in summary["groups"].items():
            print(
                f"{name}: n={item['count']}, "
                f"MFE={item['mfe']['average']:+.2f}%, "
                f"MAE={item['mae']['average']:+.2f}%, "
                f"duration={item['duration']['average']:.2f}"
            )
        print(
            "STOP_FIRST recovery: "
            f"entry={summary['stop_first_recovery']['recovered_entry']}, "
            f"+2%={summary['stop_first_recovery']['reached_two_percent']}, "
            f"+4%={summary['stop_first_recovery']['reached_four_percent']}"
        )
        print(
            "TARGET_FIRST aftermath: "
            + ", ".join(
                f"{name}={count}"
                for name, count in summary["target_first_aftermath"].items()
            )
        )
        print(f"counterfactual diagnostics: {summary['counterfactual']}")
    print("\n=== Interpretation boundary ===")
    print(
        "Counterfactual exits are diagnostic only, require an observed candle "
        "to reach the level, and do not define an executable alternate strategy."
    )


def main():
    results = run_trade_path_exit_timing_study()
    print_report(results)
    return results


if __name__ == "__main__":
    main()