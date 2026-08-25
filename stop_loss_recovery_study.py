"""Analysis-only recovery study for original STOP LOSS trades."""

from statistics import median

from multi_period_backtest import MultiPeriodBacktester
from out_of_sample_validation import _split_periods
from score_effectiveness_study import (
    SCORE_STUDY_PERIODS,
    select_score_study_periods,
)
from strategy_calibration_study import STARTING_CAPITAL
from yahoo_btc_cad_data import YahooBTCADMarketData


RECOVERY_HORIZONS = (5, 10, 20, 40)
TARGETS = (("two_percent", 2.0), ("four_percent", 4.0))
EXIT_REASON = "STOP LOSS"
MIN_PATTERN_SAMPLE = 20


def _average(values):
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _median(values):
    values = list(values)
    return median(values) if values else 0.0


def _percent(numerator, denominator):
    return numerator / denominator * 100 if denominator else 0.0


def _summary(values):
    values = list(values)
    return {
        "count": len(values),
        "average": _average(values),
        "median": _median(values),
        "minimum": min(values) if values else None,
        "maximum": max(values) if values else None,
    }


class StopLossRecoveryStudy:
    """Measure post-stop recovery without changing the control backtest."""

    def analyze_trade(self, trade, candles, evaluation, period, regime):
        if trade["reason"] != EXIT_REASON:
            raise ValueError("analyze_trade requires a STOP LOSS trade")
        entry_index = trade["entry_candle"]
        exit_index = trade["exit_candle"]
        if not 0 <= entry_index <= exit_index < len(candles):
            raise ValueError("trade candle indexes are outside the candle set")

        entry_price = trade["market_entry_price"]
        exit_price = trade["market_exit_price"]
        after = candles[exit_index + 1:]
        loss_distance = max(entry_price - exit_price, 0.0)
        recovery_targets = {
            "recover_50_percent": exit_price + loss_distance * 0.5,
            "recover_100_percent": entry_price,
        }
        recovery_timing = {
            name: self._first_recovery(
                candles,
                exit_index,
                price,
            )
            for name, price in recovery_targets.items()
        }
        max_recovery = self._max_movement(after, entry_price)
        target_reach = {
            name: self._first_recovery(
                candles,
                exit_index,
                entry_price * (1 + threshold / 100),
            )
            for name, threshold in TARGETS
        }
        recovered_entry = recovery_timing["recover_100_percent"]["reached"]
        reached_two = target_reach["two_percent"]["reached"]
        reached_four = target_reach["four_percent"]["reached"]
        return {
            "trade_number": trade["trade_number"],
            "period": period,
            "regime": regime,
            "entry_candle": entry_index,
            "exit_candle": exit_index,
            "entry_timestamp": trade["entry_timestamp"],
            "exit_timestamp": trade["exit_timestamp"],
            "entry_price": entry_price,
            "exit_market_price": exit_price,
            "exit_reason": trade["reason"],
            "entry_rsi": evaluation["rsi"],
            "entry_score": evaluation["strategy_score"],
            "conditions": {
                name: bool(evaluation[field])
                for field, name in (
                    ("long_term_trend", "long_term_trend"),
                    ("short_term_momentum", "short_term_momentum"),
                    ("rsi_condition", "rsi_condition"),
                    ("volume", "volume"),
                    ("price_above_ema21", "price_above_ema21"),
                )
            },
            "max_adverse_after_exit_percent": (
                min(
                    (candle["low"] / entry_price - 1) * 100
                    for candle in after
                )
                if after else 0.0
            ),
            "max_favorable_recovery_percent": max_recovery,
            "recovery_timing": recovery_timing,
            "target_reach": target_reach,
            "recovered_entry": recovered_entry,
            "reached_two_percent": reached_two,
            "reached_four_percent": reached_four,
            "continued_loser": not recovered_entry,
            "recovering_trade": recovered_entry,
            "strong_false_stop_candidate": (
                recovered_entry and (reached_two or reached_four)
            ),
            "horizon_recovery": {
                horizon: self._horizon_summary(
                    after,
                    entry_price,
                    horizon,
                )
                for horizon in RECOVERY_HORIZONS
            },
        }

    @staticmethod
    def _first_recovery(candles, exit_index, target_price):
        for index in range(exit_index + 1, len(candles)):
            if candles[index]["high"] >= target_price:
                return {
                    "reached": True,
                    "candles_after_exit": index - exit_index,
                    "candle": index,
                    "timestamp": candles[index]["timestamp"],
                }
        return {
            "reached": False,
            "candles_after_exit": None,
            "candle": None,
            "timestamp": None,
        }

    @staticmethod
    def _max_movement(candles, entry_price):
        if not candles:
            return 0.0
        return (max(candle["high"] for candle in candles) / entry_price - 1) * 100

    @staticmethod
    def _horizon_summary(candles, entry_price, horizon):
        observed = candles[:horizon]
        return {
            "requested_candles": horizon,
            "observed_candles": len(observed),
            "period_end_limited": len(observed) < horizon,
            "max_recovery_percent": StopLossRecoveryStudy._max_movement(
                observed,
                entry_price,
            ),
            "recovered_entry": any(
                candle["high"] >= entry_price for candle in observed
            ),
            "reached_two_percent": any(
                candle["high"] >= entry_price * 1.02
                for candle in observed
            ),
            "reached_four_percent": any(
                candle["high"] >= entry_price * 1.04
                for candle in observed
            ),
        }

    def analyze_period(self, period_result, candles):
        evaluations = {
            evaluation["candle"]: evaluation
            for evaluation in period_result["evaluation_history"]
        }
        trades = []
        for trade in period_result["trades_history"]:
            if trade["reason"] != EXIT_REASON:
                continue
            evaluation = evaluations.get(trade["entry_candle"])
            if evaluation is None:
                raise ValueError(
                    "STOP LOSS trade is missing its entry evaluation"
                )
            trades.append(
                self.analyze_trade(
                    trade,
                    candles,
                    evaluation,
                    period_result["period"],
                    period_result["regime"],
                )
            )
        return {
            "period": period_result["period"],
            "start_date": period_result["start_date"],
            "end_date": period_result["end_date"],
            "regime": period_result["regime"],
            "candles": len(candles),
            "stop_loss_count": len(trades),
            "trades": trades,
            "summary": self.summarize_trades(trades),
        }

    def analyze_group(self, period_results, period_candles):
        if len(period_results) != len(period_candles):
            raise ValueError("period result and candle counts must match")
        periods = [
            self.analyze_period(result, candles)
            for result, candles in zip(period_results, period_candles)
        ]
        trades = [trade for period in periods for trade in period["trades"]]
        return {
            "period_count": len(periods),
            "stop_loss_count": len(trades),
            "periods": periods,
            "summary": self.summarize_trades(trades),
        }

    def summarize_trades(self, trades):
        groups = {
            "continued_losers": [
                trade for trade in trades if trade["continued_loser"]
            ],
            "recovering_trades": [
                trade for trade in trades if trade["recovering_trade"]
            ],
            "strong_false_stop_candidates": [
                trade for trade in trades
                if trade["strong_false_stop_candidate"]
            ],
        }
        return {
            "stop_loss_count": len(trades),
            "recovered_entry_count": sum(
                trade["recovered_entry"] for trade in trades
            ),
            "recovered_entry_percent": _percent(
                sum(trade["recovered_entry"] for trade in trades),
                len(trades),
            ),
            "reached_two_percent_count": sum(
                trade["reached_two_percent"] for trade in trades
            ),
            "reached_two_percent_percent": _percent(
                sum(trade["reached_two_percent"] for trade in trades),
                len(trades),
            ),
            "reached_four_percent_count": sum(
                trade["reached_four_percent"] for trade in trades
            ),
            "reached_four_percent_percent": _percent(
                sum(trade["reached_four_percent"] for trade in trades),
                len(trades),
            ),
            "max_recovery": _summary(
                trade["max_favorable_recovery_percent"] for trade in trades
            ),
            "max_adverse_after_exit": _summary(
                trade["max_adverse_after_exit_percent"] for trade in trades
            ),
            "recovery_timing": {
                name: _summary(
                    item["candles_after_exit"]
                    for trade in trades
                    for item in [trade["recovery_timing"][name]]
                    if item["reached"]
                )
                for name in ("recover_50_percent", "recover_100_percent")
            },
            "horizon_recovery": {
                horizon: {
                    "max_recovery_percent": _summary(
                        trade["horizon_recovery"][horizon][
                            "max_recovery_percent"
                        ]
                        for trade in trades
                    ),
                    "recovered_entry_count": sum(
                        trade["horizon_recovery"][horizon][
                            "recovered_entry"
                        ]
                        for trade in trades
                    ),
                    "reached_two_percent_count": sum(
                        trade["horizon_recovery"][horizon][
                            "reached_two_percent"
                        ]
                        for trade in trades
                    ),
                    "reached_four_percent_count": sum(
                        trade["horizon_recovery"][horizon][
                            "reached_four_percent"
                        ]
                        for trade in trades
                    ),
                }
                for horizon in RECOVERY_HORIZONS
            },
            "groups": {
                name: self._group_summary(selected)
                for name, selected in groups.items()
            },
        }

    @staticmethod
    def _group_summary(trades):
        return {
            "count": len(trades),
            "percent_of_stop_losses": None,
            "average_entry_rsi": _average(
                trade["entry_rsi"] for trade in trades
            ),
            "average_entry_score": _average(
                trade["entry_score"] for trade in trades
            ),
            "max_recovery_percent": _summary(
                trade["max_favorable_recovery_percent"] for trade in trades
            ),
            "max_adverse_after_exit_percent": _summary(
                trade["max_adverse_after_exit_percent"] for trade in trades
            ),
            "conditions": {
                condition: _percent(
                    sum(trade["conditions"][condition] for trade in trades),
                    len(trades),
                )
                for condition in (
                    "long_term_trend",
                    "short_term_momentum",
                    "rsi_condition",
                    "volume",
                    "price_above_ema21",
                )
            },
            "regimes": {
                regime: sum(trade["regime"] == regime for trade in trades)
                for regime in ("Bull", "Sideways", "Bear")
            },
        }


def _run_period_group(selected, notifier):
    runner = MultiPeriodBacktester(starting_capital=STARTING_CAPITAL)
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


def run_stop_loss_recovery_study(notifier=None):
    if notifier is None:
        from btc_cad_preflight import send_slack_notification
        notifier = send_slack_notification
    loader = YahooBTCADMarketData(data_range="10y")
    candles = loader.load()
    if not candles:
        raise RuntimeError(loader.last_error or "Yahoo BTC/CAD data unavailable")
    selected = select_score_study_periods(candles)
    if len(selected) != len(SCORE_STUDY_PERIODS):
        raise RuntimeError("Stop-loss recovery study requires all fixed periods")
    research, validation = _split_periods(selected)
    study = StopLossRecoveryStudy()
    research_results = _run_period_group(research, notifier)
    validation_results = _run_period_group(validation, notifier)
    return {
        "source": "Yahoo Finance BTC/CAD aggregated daily data",
        "real_money_trading": False,
        "split": {
            "research_start": research[0]["start_date"],
            "research_end": research[-1]["end_date"],
            "research_periods": len(research),
            "research_candles": len(research) * 365,
            "validation_start": validation[0]["start_date"],
            "validation_end": validation[-1]["end_date"],
            "validation_periods": len(validation),
            "validation_candles": len(validation) * 365,
        },
        "research": study.analyze_group(
            research_results,
            [period["candles"] for period in research],
        ),
        "validation": study.analyze_group(
            validation_results,
            [period["candles"] for period in validation],
        ),
    }


def _print_group(label, group):
    summary = group["summary"]
    print(f"\n=== {label} ===")
    print(
        f"STOP LOSS trades={summary['stop_loss_count']}, "
        f"recovered entry={summary['recovered_entry_count']} "
        f"({summary['recovered_entry_percent']:.2f}%), "
        f"+2%={summary['reached_two_percent_count']} "
        f"({summary['reached_two_percent_percent']:.2f}%), "
        f"+4%={summary['reached_four_percent_count']} "
        f"({summary['reached_four_percent_percent']:.2f}%)"
    )
    print(
        f"max recovery average={summary['max_recovery']['average']:+.2f}%, "
        f"median={summary['max_recovery']['median']:+.2f}%"
    )
    for name, timing in summary["recovery_timing"].items():
        print(
            f"{name}: reached={timing['count']}, "
            f"average candles={timing['average']:.2f}, "
            f"median={timing['median']:.2f}"
        )
    for group_name, group_summary in summary["groups"].items():
        print(
            f"{group_name}: n={group_summary['count']}, "
            f"score={group_summary['average_entry_score']:.2f}, "
            f"RSI={group_summary['average_entry_rsi']:.2f}, "
            f"recovery={group_summary['max_recovery_percent']['average']:+.2f}%"
        )
    for horizon in RECOVERY_HORIZONS:
        horizon_summary = summary["horizon_recovery"][horizon]
        print(
            f"{horizon}-candle recovery: "
            f"entry={horizon_summary['recovered_entry_count']}, "
            f"+2%={horizon_summary['reached_two_percent_count']}, "
            f"+4%={horizon_summary['reached_four_percent_count']}"
        )


def print_report(results):
    print("BTC/CAD STOP-LOSS RECOVERY & FALSE-STOP STUDY — ANALYSIS ONLY")
    print("REAL-MONEY TRADING: DISABLED")
    split = results["split"]
    print(
        f"Research: {split['research_start']} to {split['research_end']} "
        f"({split['research_periods']} periods, {split['research_candles']} candles)"
    )
    print(
        f"Validation: {split['validation_start']} to {split['validation_end']} "
        f"({split['validation_periods']} periods, {split['validation_candles']} candles)"
    )
    _print_group("Research", results["research"])
    _print_group("Untouched validation", results["validation"])
    print("\n=== Interpretation boundary ===")
    print(
        "This study describes post-stop price paths. It does not prove that "
        "a wider stop or alternate stop rule would improve executable net P/L, "
        "and it does not recommend changing the stop-loss rule."
    )


def main():
    results = run_stop_loss_recovery_study()
    print_report(results)
    return results


if __name__ == "__main__":
    main()