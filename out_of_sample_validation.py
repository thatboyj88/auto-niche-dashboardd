from strategy_calibration_study import (
    FEE_PERCENT,
    FORWARD_HORIZONS,
    MIN_CALIBRATION_SIGNALS,
    RSI_BANDS,
    SCORE_BANDS,
    SLIPPAGE_PERCENT,
    STARTING_CAPITAL,
    StrategyCalibrationStudy,
)
from score_effectiveness_study import (
    SCORE_STUDY_PERIODS,
    select_score_study_periods,
)
from multi_period_backtest import MultiPeriodBacktester
from yahoo_btc_cad_data import YahooBTCADMarketData


VALIDATION_PERIOD_COUNT = 2
MAX_MEANINGFUL_COST_SHARE_PERCENT = 50.0
BREAK_EVEN_MOVE_PERCENT = (
    StrategyCalibrationStudy.break_even_move_percent(
        fee_percent=FEE_PERCENT,
        slippage_percent=SLIPPAGE_PERCENT,
    )
)


def _average(values):
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _percent(numerator, denominator):
    return numerator / denominator * 100 if denominator else 0.0


def _split_periods(selected):
    if len(selected) != len(SCORE_STUDY_PERIODS):
        raise RuntimeError(
            "Out-of-sample validation requires all fixed study periods"
        )
    split_index = len(selected) - VALIDATION_PERIOD_COUNT
    return selected[:split_index], selected[split_index:]


def _performance(period_results):
    completed_trades = sum(period["trades"] for period in period_results)
    gross = sum(
        period["gross_profit_before_costs"]
        for period in period_results
    )
    fees = sum(period["total_fees"] for period in period_results)
    slippage = sum(period["total_slippage"] for period in period_results)
    net = sum(period["net_profit"] for period in period_results)
    return {
        "periods": len(period_results),
        "candles": sum(period["candle_count"] for period in period_results),
        "buy_signals": sum(
            sum(
                evaluation["decision"] == "BUY"
                for evaluation in period["evaluation_history"]
            )
            for period in period_results
        ),
        "completed_trades": completed_trades,
        "gross_profit_loss": gross,
        "fees": fees,
        "slippage": slippage,
        "net_profit_loss": net,
        "net_return_percent": (
            net / (STARTING_CAPITAL * len(period_results)) * 100
            if period_results
            else 0.0
        ),
        "cost_share_of_abs_gross_percent": (
            (fees + slippage) / abs(gross) * 100 if gross else 0.0
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
        performance.update({
            "market_return_average": _average(
                period["market_return"] for period in selected
            ),
            "strategy_return_average": _average(
                period["return_percent"] for period in selected
            ),
            "win_rate": _percent(
                sum(period["wins"] for period in selected),
                sum(period["trades"] for period in selected),
            ),
            "drawdown_worst": max(
                (period["max_drawdown"] for period in selected),
                default=0.0,
            ),
        })
        result[regime] = performance
    return result


class OutOfSampleValidationStudy:
    """Validate Step 8 relationships without fitting on validation data."""

    def __init__(self):
        self.calibration = StrategyCalibrationStudy()

    def analyze_group(self, period_results, period_candles):
        calibration = self.calibration.analyze(
            period_results,
            period_candles,
        )
        return {
            "performance": _performance(period_results),
            "periods": [
                {
                    "period": period["period"],
                    "start_date": period["start_date"],
                    "end_date": period["end_date"],
                    "regime": period["regime"],
                    "candles": period["candle_count"],
                    "market_return": period["market_return"],
                    "strategy_return": period["return_percent"],
                }
                for period in period_results
            ],
            "regime_performance": _regime_performance(period_results),
            "signal_count": calibration["signal_count"],
            "completed_entry_count": calibration["completed_entry_count"],
            "score_bands": calibration["score_bands"],
            "rsi_bands": calibration["rsi_bands"],
            "condition_combinations": (
                calibration["condition_combinations"]
            ),
            "early_movement": calibration["early_movement"],
            "cost_break_even": calibration["cost_break_even"],
            "signals": calibration["signals"],
        }

    @staticmethod
    def _forward_summary(signals, predicate):
        selected = [signal for signal in signals if predicate(signal)]
        return {
            "signals": len(selected),
            "insufficient_evidence": (
                len(selected) < MIN_CALIBRATION_SIGNALS
            ),
            "horizons": {
                str(horizon): {
                    "average": _average(
                        signal["forward_returns"][horizon]
                        for signal in selected
                        if signal["forward_returns"][horizon] is not None
                    ),
                    "positive_percent": _percent(
                        sum(
                            signal["forward_returns"][horizon] > 0
                            for signal in selected
                            if signal["forward_returns"][horizon] is not None
                        ),
                        sum(
                            signal["forward_returns"][horizon] is not None
                            for signal in selected
                        ),
                    ),
                    "break_even_percent": _percent(
                        sum(
                            signal["forward_returns"][horizon]
                            >= BREAK_EVEN_MOVE_PERCENT
                            for signal in selected
                            if signal["forward_returns"][horizon] is not None
                        ),
                        sum(
                            signal["forward_returns"][horizon] is not None
                            for signal in selected
                        ),
                    ),
                }
                for horizon in FORWARD_HORIZONS
            },
        }

    def hypothesis_results(self, group):
        signals = group["signals"]
        if (
            not signals
            or group.get("performance", {}).get("periods") == 0
        ):
            return self._insufficient_hypothesis_results()

        rsi_lower = self._forward_summary(
            signals,
            lambda signal: 50 <= signal["entry_rsi"] < 60,
        )
        rsi_higher = self._forward_summary(
            signals,
            lambda signal: signal["entry_rsi"] >= 60,
        )
        score_lower = self._forward_summary(
            signals,
            lambda signal: 80 <= signal["score"] <= 84,
        )
        score_higher = self._forward_summary(
            signals,
            lambda signal: 85 <= signal["score"] <= 89,
        )
        five_day = group["cost_break_even"]["overall"]["horizons"]["5"]
        established_regimes = [
            (regime, summary)
            for regime, summary in group["regime_performance"].items()
            if summary["buy_signals"] >= MIN_CALIBRATION_SIGNALS
        ]
        regime_averages = [
            self._regime_signal_summary(signals, regime)
            for regime, _ in established_regimes
        ]

        return {
            "rsi_60_vs_50_59": self._compare_groups(
                rsi_higher,
                rsi_lower,
            ),
            "score_85_89_vs_80_84": self._compare_groups(
                score_higher,
                score_lower,
            ),
            "break_even_reach": {
                "status": (
                    "supported"
                    if five_day["reached_break_even_percent"] >= 50
                    else "not supported"
                ),
                "signals": five_day["signals"],
                "reached_percent": five_day[
                    "reached_break_even_percent"
                ],
                "threshold_percent": BREAK_EVEN_MOVE_PERCENT,
            },
            "net_after_costs": {
                **self._net_after_costs_hypothesis(
                    group["performance"]
                ),
            },
            "regime_consistency": self._regime_consistency(
                regime_averages
            ),
        }

    @staticmethod
    def _insufficient_hypothesis_results():
        return {
            name: {"status": "insufficient evidence"}
            for name in (
                "rsi_60_vs_50_59",
                "score_85_89_vs_80_84",
                "break_even_reach",
                "net_after_costs",
                "regime_consistency",
            )
        }

    @staticmethod
    def _net_after_costs_hypothesis(performance):
        return {
            "status": (
                "supported"
                if (
                    performance["net_profit_loss"] > 0 and
                    performance["cost_share_of_abs_gross_percent"] <
                    MAX_MEANINGFUL_COST_SHARE_PERCENT
                )
                else "not supported"
            ),
            "gross_profit_loss": performance["gross_profit_loss"],
            "net_profit_loss": performance["net_profit_loss"],
            "fees": performance["fees"],
            "slippage": performance["slippage"],
            "cost_share_of_abs_gross_percent": performance[
                "cost_share_of_abs_gross_percent"
            ],
        }

    @staticmethod
    def _compare_groups(higher, lower):
        enough = not (
            higher["insufficient_evidence"] or
            lower["insufficient_evidence"]
        )
        higher_average = higher["horizons"]["5"]["average"]
        lower_average = lower["horizons"]["5"]["average"]
        return {
            "status": (
                "supported"
                if enough and higher_average > lower_average
                else "not supported"
                if enough
                else "insufficient evidence"
            ),
            "higher": higher,
            "lower": lower,
            "difference_5_candle_average": higher_average - lower_average,
        }

    @staticmethod
    def _regime_signal_summary(signals, regime):
        selected = [
            signal for signal in signals if signal["regime"] == regime
        ]
        five_day = [
            signal["forward_returns"][5]
            for signal in selected
            if signal["forward_returns"][5] is not None
        ]
        return {
            "regime": regime,
            "signals": len(selected),
            "average_5_candle_return": _average(five_day),
            "positive_5_candle_percent": _percent(
                sum(value > 0 for value in five_day),
                len(five_day),
            ),
        }

    @staticmethod
    def _regime_consistency(regime_averages):
        if len(regime_averages) < 2:
            return {
                "status": "insufficient evidence",
                "regimes": regime_averages,
            }
        positive = [
            summary["average_5_candle_return"] > 0
            for summary in regime_averages
        ]
        return {
            "status": (
                "supported" if all(positive) else "not supported"
            ),
            "regimes": regime_averages,
        }

    def analyze_split(
        self,
        research_periods,
        research_candles,
        validation_periods,
        validation_candles,
    ):
        research = self.analyze_group(
            research_periods,
            research_candles,
        )
        validation = self.analyze_group(
            validation_periods,
            validation_candles,
        )
        return {
            "source": "Yahoo Finance BTC/CAD aggregated daily data",
            "split": {
                "research_periods": len(research_periods),
                "research_candles": sum(
                    len(candles) for candles in research_candles
                ),
                "research_start": (
                    research_periods[0]["start_date"]
                    if research_periods else None
                ),
                "research_end": (
                    research_periods[-1]["end_date"]
                    if research_periods else None
                ),
                "validation_periods": len(validation_periods),
                "validation_candles": sum(
                    len(candles) for candles in validation_candles
                ),
                "validation_start": (
                    validation_periods[0]["start_date"]
                    if validation_periods else None
                ),
                "validation_end": (
                    validation_periods[-1]["end_date"]
                    if validation_periods else None
                ),
            },
            "research": research,
            "validation": validation,
            "research_hypotheses": self.hypothesis_results(research),
            "validation_hypotheses": self.hypothesis_results(validation),
            "hypotheses": self._compare_hypotheses(
                self.hypothesis_results(research),
                self.hypothesis_results(validation),
            ),
        }

    @staticmethod
    def _compare_hypotheses(research, validation):
        names = (
            "rsi_60_vs_50_59",
            "score_85_89_vs_80_84",
            "break_even_reach",
            "net_after_costs",
            "regime_consistency",
        )
        return {
            name: {
                "research": research[name]["status"],
                "validation": validation[name]["status"],
                "survived": (
                    research[name]["status"] == "supported" and
                    validation[name]["status"] == "supported"
                ),
            }
            for name in names
        }


def _run_period_group(runner, selected, notifier):
    period_results = []
    for index, period in enumerate(selected):
        period_results.append(
            runner._run_period(
                index,
                period["candles"],
                period_label=period["period"],
                source_label="Yahoo Finance BTC/CAD fixed ten-year study",
                source_kind="fixed-study",
                notifier=notifier,
            )
        )
    return period_results


def run_out_of_sample_validation(notifier=None):
    if notifier is None:
        from btc_cad_preflight import send_slack_notification

        notifier = send_slack_notification

    loader = YahooBTCADMarketData(data_range="10y")
    candles = loader.load()
    if not candles:
        raise RuntimeError(loader.last_error or "Yahoo BTC/CAD data unavailable")
    selected = select_score_study_periods(candles)
    research, validation = _split_periods(selected)
    runner = MultiPeriodBacktester(starting_capital=STARTING_CAPITAL)

    # These are intentionally separate calls. Validation is not analyzed until
    # all research-period analysis has completed.
    research_results = _run_period_group(runner, research, notifier)
    research_analysis = StrategyCalibrationStudy().analyze(
        research_results,
        [period["candles"] for period in research],
    )
    validation_results = _run_period_group(runner, validation, notifier)
    validation_analysis = StrategyCalibrationStudy().analyze(
        validation_results,
        [period["candles"] for period in validation],
    )
    study = OutOfSampleValidationStudy()
    return study.analyze_split(
        research_results,
        [period["candles"] for period in research],
        validation_results,
        [period["candles"] for period in validation],
    )


def _print_group(label, group, hypotheses):
    performance = group["performance"]
    print(f"\n=== {label} ===")
    dates = (
        f"{group['periods'][0]['start_date']} to "
        f"{group['periods'][-1]['end_date']}"
        if group["periods"]
        else "unavailable (no period results)"
    )
    print(
        f"Periods={performance['periods']} | "
        f"Dates={dates} | "
        f"Candles={performance['candles']}"
    )
    if (
        not group["periods"]
        or group.get("signal_count", 0) == 0
    ):
        print(
            "Insufficient evidence: no study-period results or BUY signals "
            "are available for this group."
        )
    rsi_comparison = hypotheses["rsi_60_vs_50_59"]
    rsi_averages = (
        f"{rsi_comparison['lower']['horizons']['5']['average']:+.2f}% / "
        f"{rsi_comparison['higher']['horizons']['5']['average']:+.2f}%"
        if "lower" in rsi_comparison
        else "unavailable (insufficient evidence)"
    )
    print(
        f"BUY signals={group['signal_count']}, "
        f"completed trades={group['completed_entry_count']}, "
        f"gross=${performance['gross_profit_loss']:+.4f}, "
        f"fees=${performance['fees']:.4f}, "
        f"slippage=${performance['slippage']:.4f}, "
        f"net=${performance['net_profit_loss']:+.4f}"
    )
    print(
        "Costs as share of absolute gross P/L: "
        f"{performance['cost_share_of_abs_gross_percent']:.2f}%"
    )
    for horizon, summary in group["early_movement"].items():
        print(
            f"+{horizon} candles: avg={summary['average']:+.2f}%, "
            f"positive={summary['positive_percent']:.2f}%"
        )
    print(
        "RSI 50-59 / RSI >=60 5-candle averages: "
        f"{rsi_averages}"
    )
    print("RSI bands:")
    for label, summary in group["rsi_bands"].items():
        five_day = summary["forward_returns"]["5"]
        print(
            f"  {label}: signals={summary['signals']}, "
            f"avg={five_day['average']:+.2f}%, "
            f"positive={five_day['positive_percent']:.2f}%"
        )
    score_comparison = hypotheses["score_85_89_vs_80_84"]
    score_averages = (
        f"{score_comparison['lower']['horizons']['5']['average']:+.2f}% / "
        f"{score_comparison['higher']['horizons']['5']['average']:+.2f}%"
        if "lower" in score_comparison
        else "unavailable (insufficient evidence)"
    )
    print(
        "Score 80-84 / 85-89 5-candle averages: "
        f"{score_averages}"
    )
    print("Score bands:")
    for label, summary in group["score_bands"].items():
        five_day = summary["forward_returns"]["5"]
        print(
            f"  {label}: signals={summary['signals']}, "
            f"avg={five_day['average']:+.2f}%, "
            f"positive={five_day['positive_percent']:.2f}%"
        )
    print("Condition combinations:")
    for label, summary in group["condition_combinations"].items():
        five_day = summary["forward_returns"]["5"]
        print(
            f"  {label}: signals={summary['signals']}, "
            f"avg={five_day['average']:+.2f}%, "
            f"positive={five_day['positive_percent']:.2f}%"
        )
    print("Cost break-even:")
    for horizon, summary in group["cost_break_even"]["overall"]["horizons"].items():
        print(
            f"  +{horizon} candles: "
            f"{summary['reached_break_even_percent']:.2f}% reached "
            f"{BREAK_EVEN_MOVE_PERCENT:.3f}%"
        )
    print("Regimes:")
    for regime, summary in group["regime_performance"].items():
        if summary["periods"]:
            print(
                f"  {regime}: periods={summary['periods']}, "
                f"signals={summary['buy_signals']}, "
                f"net=${summary['net_profit_loss']:+.4f}, "
                f"win rate={summary['win_rate']:.2f}%"
            )


def print_report(results):
    print("BTC/CAD OUT-OF-SAMPLE VALIDATION — ANALYSIS ONLY")
    print("REAL-MONEY TRADING: DISABLED")
    split = results["split"]
    research_dates = (
        f"{split['research_start']} to {split['research_end']}"
        if split["research_start"] is not None
        else "unavailable"
    )
    validation_dates = (
        f"{split['validation_start']} to {split['validation_end']}"
        if split["validation_start"] is not None
        else "unavailable"
    )
    print(
        f"Research: {research_dates} "
        f"({split['research_periods']} periods, {split['research_candles']} candles)"
    )
    print(
        f"Validation: {validation_dates} "
        f"({split['validation_periods']} periods, "
        f"{split['validation_candles']} candles)"
    )
    _print_group("Research / calibration", results["research"], results["research_hypotheses"])
    _print_group("Out-of-sample validation", results["validation"], results["validation_hypotheses"])
    print("\n=== Hypothesis survival ===")
    for name, result in results["hypotheses"].items():
        print(
            f"{name}: research={result['research']}, "
            f"validation={result['validation']}, "
            f"survived={'YES' if result['survived'] else 'NO'}"
        )
    print(
        "\nObserved relationships are not proven predictive relationships. "
        "No statistical significance is claimed; small groups and regime "
        "coverage limitations remain explicit."
    )


def main():
    results = run_out_of_sample_validation()
    print_report(results)
    return results


if __name__ == "__main__":
    main()