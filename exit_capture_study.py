"""Analysis-only study of favorable movement around original trade exits."""

from statistics import median

from multi_period_backtest import MultiPeriodBacktester
from out_of_sample_validation import _split_periods
from score_effectiveness_study import (
    SCORE_STUDY_PERIODS,
    select_score_study_periods,
)
from strategy_calibration_study import (
    FEE_PERCENT,
    SLIPPAGE_PERCENT,
    STARTING_CAPITAL,
    StrategyCalibrationStudy,
)
from yahoo_btc_cad_data import YahooBTCADMarketData


BREAK_EVEN_MOVE_PERCENT = StrategyCalibrationStudy.break_even_move_percent(
    fee_percent=FEE_PERCENT,
    slippage_percent=SLIPPAGE_PERCENT,
)
TARGETS = (
    ("break_even", BREAK_EVEN_MOVE_PERCENT),
    ("two_percent", 2.0),
    ("four_percent", 4.0),
)
EXIT_REASONS = ("STOP LOSS", "TAKE PROFIT", "END OF TEST")
PHASES = ("before_exit", "at_exit", "after_exit", "never")
CONFIDENCE_LEVEL = 0.95
WILSON_Z_95 = 1.959963984540054


def _average(values):
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _median(values):
    values = list(values)
    return median(values) if values else 0.0


def _percent(numerator, denominator):
    return numerator / denominator * 100 if denominator else 0.0


def _wilson_interval(successes, trials):
    """Return a pre-specified 95% Wilson interval in percentage points."""
    if trials < 0 or successes < 0 or successes > trials:
        raise ValueError("successes and trials must be a valid binomial count")
    if not trials:
        return {"lower": 0.0, "upper": 0.0}

    proportion = successes / trials
    z_squared = WILSON_Z_95 ** 2
    denominator = 1 + z_squared / trials
    center = (proportion + z_squared / (2 * trials)) / denominator
    margin = (
        WILSON_Z_95
        * (
            proportion * (1 - proportion) / trials
            + z_squared / (4 * trials ** 2)
        )
        ** 0.5
        / denominator
    )
    return {
        "lower": (center - margin) * 100,
        "upper": (center + margin) * 100,
    }


def _summary(values):
    values = list(values)
    return {
        "count": len(values),
        "average": _average(values),
        "median": _median(values),
        "minimum": min(values) if values else None,
        "maximum": max(values) if values else None,
    }


class ExitCaptureStudy:
    """Measure original trade-path movement without changing execution."""

    def analyze_trade(self, trade, candles):
        entry = trade["entry_candle"]
        exit_index = trade["exit_candle"]
        if not 0 <= entry <= exit_index < len(candles):
            raise ValueError("trade candle indexes are outside the candle set")

        entry_price = trade["market_entry_price"]
        before = candles[entry + 1:exit_index]
        at_exit = candles[exit_index:exit_index + 1]
        after = candles[exit_index + 1:]

        targets = {
            name: self._classify_target(
                candles,
                entry,
                exit_index,
                entry_price,
                threshold,
            )
            for name, threshold in TARGETS
        }
        next_candle = self._next_candle_summary(
            candles,
            exit_index,
            entry_price,
        )
        return {
            "trade_number": trade["trade_number"],
            "entry_candle": entry,
            "exit_candle": exit_index,
            "entry_timestamp": trade["entry_timestamp"],
            "exit_timestamp": trade["exit_timestamp"],
            "entry_price": trade["entry_price"],
            "exit_price": trade["exit_price"],
            "market_entry_price": entry_price,
            "exit_reason": trade["reason"],
            "net_profit_loss": trade["net_profit_loss"],
            "gross_profit_loss_before_costs": (
                trade["gross_profit_loss_before_costs"]
            ),
            "targets": targets,
            "cost_model": {
                "fee_percent": FEE_PERCENT,
                "slippage_percent": SLIPPAGE_PERCENT,
                "required_move_percent": BREAK_EVEN_MOVE_PERCENT,
            },
            "mfe_before_exit_percent": self._mfe_percent(
                before, entry_price
            ),
            "favorable_at_exit_percent": self._mfe_percent(
                at_exit, entry_price
            ),
            "mfe_during_trade_percent": self._mfe_percent(
                before + at_exit, entry_price
            ),
            "mfe_after_exit_percent": self._mfe_percent(
                after, entry_price
            ),
            "post_exit_candles": len(after),
            "next_candle": next_candle,
        }

    @staticmethod
    def _mfe_percent(candles, entry_price):
        if not candles:
            return 0.0
        return (max(candle["high"] for candle in candles) / entry_price - 1) * 100

    @staticmethod
    def _classify_target(
        candles,
        entry_candle,
        exit_candle,
        entry_price,
        threshold,
    ):
        target_price = entry_price * (1 + threshold / 100)
        before_indexes = range(entry_candle + 1, exit_candle)
        for index in before_indexes:
            if candles[index]["high"] >= target_price:
                return {
                    "threshold_percent": threshold,
                    "reached": True,
                    "phase": "before_exit",
                    "candle": index,
                    "candles_after_exit": None,
                    "timestamp": candles[index]["timestamp"],
                }
        if candles[exit_candle]["high"] >= target_price:
            return {
                "threshold_percent": threshold,
                "reached": True,
                "phase": "at_exit",
                "candle": exit_candle,
                "candles_after_exit": 0,
                "timestamp": candles[exit_candle]["timestamp"],
            }
        for index in range(exit_candle + 1, len(candles)):
            if candles[index]["high"] >= target_price:
                return {
                    "threshold_percent": threshold,
                    "reached": True,
                    "phase": "after_exit",
                    "candle": index,
                    "candles_after_exit": index - exit_candle,
                    "timestamp": candles[index]["timestamp"],
                }
        return {
            "threshold_percent": threshold,
            "reached": False,
            "phase": "never",
            "candle": None,
            "candles_after_exit": None,
            "timestamp": None,
        }

    @staticmethod
    def _next_candle_summary(candles, exit_candle, entry_price):
        index = exit_candle + 1
        if index >= len(candles):
            return {
                "available": False,
                "candles_after_exit": 0,
                "high_percent": None,
                "close_percent": None,
                "low_percent": None,
            }
        candle = candles[index]
        return {
            "available": True,
            "candles_after_exit": 1,
            "high_percent": (candle["high"] / entry_price - 1) * 100,
            "close_percent": (candle["close"] / entry_price - 1) * 100,
            "low_percent": (candle["low"] / entry_price - 1) * 100,
        }

    def analyze_period(self, period_result, candles):
        trades = [
            self.analyze_trade(trade, candles)
            for trade in period_result["trades_history"]
        ]
        return {
            "period": period_result["period"],
            "start_date": period_result["start_date"],
            "end_date": period_result["end_date"],
            "regime": period_result["regime"],
            "candles": len(candles),
            "trade_count": len(trades),
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
        trades = [
            trade for period in periods for trade in period["trades"]
        ]
        return {
            "period_count": len(periods),
            "trade_count": len(trades),
            "periods": periods,
            "summary": self.summarize_trades(trades),
        }

    def summarize_trades(self, trades):
        return {
            "cost_model": {
                "fee_percent": FEE_PERCENT,
                "slippage_percent": SLIPPAGE_PERCENT,
                "required_move_percent": BREAK_EVEN_MOVE_PERCENT,
            },
            "target_capture": {
                name: self._target_summary(trades, name)
                for name, _ in TARGETS
            },
            "mfe": {
                "before_exit_percent": _summary(
                    trade["mfe_before_exit_percent"] for trade in trades
                ),
                "during_trade_percent": _summary(
                    trade["mfe_during_trade_percent"] for trade in trades
                ),
                "after_exit_percent": _summary(
                    trade["mfe_after_exit_percent"] for trade in trades
                ),
            },
            "by_exit_reason": {
                reason: self._reason_summary(
                    [trade for trade in trades
                     if trade["exit_reason"] == reason]
                )
                for reason in EXIT_REASONS
            },
        }

    @staticmethod
    def _target_summary(trades, name):
        selected = [trade["targets"][name] for trade in trades]
        after_exit = [
            item for item in selected if item["phase"] == "after_exit"
        ]
        reached = sum(item["reached"] for item in selected)
        return {
            "threshold_percent": (
                selected[0]["threshold_percent"] if selected else
                dict(TARGETS)[name]
            ),
            "signals": len(selected),
            "reached": reached,
            "reached_percent": _percent(reached, len(selected)),
            "phases": {
                phase: sum(item["phase"] == phase for item in selected)
                for phase in PHASES
            },
            "phase_percent": {
                phase: _percent(
                    sum(item["phase"] == phase for item in selected),
                    len(selected),
                )
                for phase in PHASES
            },
            "after_exit_average_candles": _average(
                item["candles_after_exit"]
                for item in selected
                if item["phase"] == "after_exit"
            ),
            "after_exit_reached": sum(
                item["phase"] == "after_exit" for item in selected
            ),
            "after_exit_reached_percent": _percent(
                len(after_exit), len(selected)
            ),
            "after_exit_reached_confidence_interval_percent": {
                "method": "Wilson score interval",
                "confidence_level": CONFIDENCE_LEVEL,
                **_wilson_interval(len(after_exit), len(selected)),
            },
            "after_exit_average_candles": _average(
                item["candles_after_exit"] for item in after_exit
            ),
            "after_exit_candles": _summary(
                item["candles_after_exit"] for item in after_exit
            ),
        }

    def _reason_summary(self, trades):
        next_candles = [
            trade["next_candle"] for trade in trades
            if trade["next_candle"]["available"]
        ]
        return {
            "trades": len(trades),
            "net_profit_loss": sum(
                trade["net_profit_loss"] for trade in trades
            ),
            "mfe_before_exit_percent": _summary(
                trade["mfe_before_exit_percent"] for trade in trades
            ),
            "mfe_after_exit_percent": _summary(
                trade["mfe_after_exit_percent"] for trade in trades
            ),
            "next_candle": {
                "available": len(next_candles),
                "high_percent": _summary(
                    item["high_percent"] for item in next_candles
                ),
                "close_percent": _summary(
                    item["close_percent"] for item in next_candles
                ),
                "low_percent": _summary(
                    item["low_percent"] for item in next_candles
                ),
            },
            "targets_after_exit": {
                name: self._target_summary(trades, name)
                for name, _ in TARGETS
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


def run_exit_capture_study(notifier=None):
    if notifier is None:
        from btc_cad_preflight import send_slack_notification
        notifier = send_slack_notification
    loader = YahooBTCADMarketData(data_range="10y")
    candles = loader.load()
    if not candles:
        raise RuntimeError(loader.last_error or "Yahoo BTC/CAD data unavailable")
    selected = select_score_study_periods(candles)
    if len(selected) != len(SCORE_STUDY_PERIODS):
        raise RuntimeError("Exit-capture study requires all fixed periods")
    research, validation = _split_periods(selected)
    study = ExitCaptureStudy()
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
    print(f"\n=== {label} ===")
    print(f"completed trades={group['trade_count']}")
    cost_model = group["summary"]["cost_model"]
    print(
        "Required movement to cover unchanged costs: "
        f"{cost_model['required_move_percent']:.3f}% "
        f"(fee={cost_model['fee_percent']:.3f}%, "
        f"slippage={cost_model['slippage_percent']:.3f}% per side)"
    )
    for name, threshold in TARGETS:
        summary = group["summary"]["target_capture"][name]
        phases = ", ".join(
            f"{phase}={summary['phases'][phase]} "
            f"({summary['phase_percent'][phase]:.2f}%)"
            for phase in PHASES
        )
        confidence_interval = (
            summary["after_exit_reached_confidence_interval_percent"]
        )
        print(
            f"{name} ({threshold:.3f}%): {phases}; "
            f"post-exit={summary['after_exit_reached']} "
            f"({summary['after_exit_reached_percent']:.2f}%), "
            f"95% Wilson CI="
            f"[{confidence_interval['lower']:.2f}, "
            f"{confidence_interval['upper']:.2f}]%, "
            f"average candles after exit="
            f"{summary['after_exit_candles']['average']:.2f}"
        )
    mfe = group["summary"]["mfe"]
    print(
        "MFE average: "
        f"before exit={mfe['before_exit_percent']['average']:+.2f}%, "
        f"during trade={mfe['during_trade_percent']['average']:+.2f}%, "
        f"after exit={mfe['after_exit_percent']['average']:+.2f}%"
    )
    for reason, summary in group["summary"]["by_exit_reason"].items():
        if not summary["trades"]:
            continue
        next_high = summary["next_candle"]["high_percent"]["average"]
        print(
            f"{reason}: trades={summary['trades']}, "
            f"net=${summary['net_profit_loss']:+.4f}, "
            f"post-exit MFE={summary['mfe_after_exit_percent']['average']:+.2f}%, "
            f"next-candle high={next_high:+.2f}%"
        )
        for name, threshold in TARGETS:
            target = summary["targets_after_exit"][name]
            print(
                f"  {name} ({threshold:.3f}% required): "
                f"post-exit={target['after_exit_reached']} "
                f"({target['after_exit_reached_percent']:.2f}%), "
                f"95% Wilson CI="
                f"[{target['after_exit_reached_confidence_interval_percent']['lower']:.2f}, "
                f"{target['after_exit_reached_confidence_interval_percent']['upper']:.2f}]%"
            )
    print("By market period:")
    for period in group["periods"]:
        print(
            f"  {period['period']} ({period['start_date']} to "
            f"{period['end_date']}, regime={period['regime']}): "
            f"trades={period['trade_count']}"
        )
        for name, threshold in TARGETS:
            target = period["summary"]["target_capture"][name]
            confidence_interval = (
                target["after_exit_reached_confidence_interval_percent"]
            )
            print(
                f"    {name} ({threshold:.3f}% required): "
                f"post-exit={target['after_exit_reached']} "
                f"({target['after_exit_reached_percent']:.2f}%), "
                f"95% Wilson CI="
                f"[{confidence_interval['lower']:.2f}, "
                f"{confidence_interval['upper']:.2f}]%"
            )
        for reason in ("STOP LOSS", "TAKE PROFIT"):
            summary = period["summary"]["by_exit_reason"][reason]
            if not summary["trades"]:
                continue
            print(f"    {reason}: trades={summary['trades']}")
            for name, threshold in TARGETS:
                target = summary["targets_after_exit"][name]
                confidence_interval = (
                    target[
                        "after_exit_reached_confidence_interval_percent"
                    ]
                )
                print(
                    f"      {name} ({threshold:.3f}% required): "
                    f"post-exit={target['after_exit_reached']} "
                    f"({target['after_exit_reached_percent']:.2f}%), "
                    f"95% Wilson CI="
                    f"[{confidence_interval['lower']:.2f}, "
                    f"{confidence_interval['upper']:.2f}]%"
                )


def print_report(results):
    print("BTC/CAD EXIT-CAPTURE STUDY — ANALYSIS ONLY")
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
        "These are descriptive path measurements, not proof that alternate "
        "exits would improve net P/L. No strategy change is recommended "
        "automatically."
    )


def main():
    results = run_exit_capture_study()
    print_report(results)
    return results


if __name__ == "__main__":
    main()