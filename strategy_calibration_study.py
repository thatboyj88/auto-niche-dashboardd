from statistics import median

from multi_period_backtest import MultiPeriodBacktester
from score_effectiveness_study import (
    SCORE_STUDY_PERIODS,
    select_score_study_periods,
)
from yahoo_btc_cad_data import YahooBTCADMarketData


STARTING_CAPITAL = 25.00
FEE_PERCENT = 0.004
SLIPPAGE_PERCENT = 0.001
FORWARD_HORIZONS = (3, 5, 10, 20)
MIN_CALIBRATION_SIGNALS = 20
SCORE_BANDS = (
    ("80-84", 80, 84),
    ("85-89", 85, 89),
    ("90-94", 90, 94),
    ("95-100", 95, 100),
)
RSI_BANDS = (
    ("<50", None, 50),
    ("50-59", 50, 60),
    ("60-69", 60, 70),
    ("70-79", 70, 80),
    ("80+", 80, None),
)
CONDITIONS = (
    ("long_term_trend", "Long-term trend"),
    ("short_term_momentum", "Short-term momentum"),
    ("rsi_condition", "RSI"),
    ("volume", "Volume"),
    ("price_above_ema21", "Price above EMA21"),
)


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
        "positive_percent": _percent(
            sum(value > 0 for value in values),
            len(values),
        ),
    }


class StrategyCalibrationStudy:
    """Calibrate existing entry signals without changing execution."""

    @staticmethod
    def break_even_move_percent(
        fee_percent=FEE_PERCENT,
        slippage_percent=SLIPPAGE_PERCENT,
    ):
        """Underlying close movement required for one long trade to break even."""
        return (
            (
                (1 + slippage_percent) * (1 + fee_percent) /
                ((1 - slippage_percent) * (1 - fee_percent))
            ) - 1
        ) * 100

    def analyze(self, period_results, period_candles):
        if len(period_results) != len(period_candles):
            raise ValueError("period result and candle counts must match")

        signals = []
        for period_result, candles in zip(period_results, period_candles):
            trade_by_entry = {
                trade["entry_candle"]: trade
                for trade in period_result["trades_history"]
            }
            for evaluation in period_result["evaluation_history"]:
                if evaluation["decision"] != "BUY":
                    continue
                signals.append(
                    self._build_signal(
                        evaluation,
                        candles,
                        trade_by_entry.get(evaluation["candle"]),
                        period_result["regime"],
                        period_result["period"],
                    )
                )

        return {
            "source": "Yahoo Finance BTC/CAD aggregated daily data",
            "period_count": len(period_results),
            "signal_count": len(signals),
            "completed_entry_count": sum(
                signal["completed_trade"]
                for signal in signals
            ),
            "break_even_move_percent": self.break_even_move_percent(),
            "score_bands": self._band_summary(
                signals,
                lambda signal: signal["score"],
                SCORE_BANDS,
            ),
            "rsi_bands": self._band_summary(
                signals,
                lambda signal: signal["entry_rsi"],
                RSI_BANDS,
            ),
            "condition_combinations": self._condition_combinations(
                signals
            ),
            "early_movement": self._movement_summary(signals),
            "cost_break_even": self._cost_break_even(signals),
            "diagnosis": self._diagnosis(signals),
            "signals": signals,
        }

    def _build_signal(
        self,
        evaluation,
        candles,
        trade,
        regime,
        period,
    ):
        entry_candle = evaluation["candle"]
        entry_close = candles[entry_candle]["close"]
        forward_returns = {
            horizon: self._forward_return(
                candles,
                entry_candle,
                horizon,
                entry_close,
            )
            for horizon in FORWARD_HORIZONS
        }
        passed = tuple(
            label
            for field, label in CONDITIONS
            if evaluation[field]
        )
        return {
            "period": period,
            "regime": regime,
            "candle": entry_candle,
            "score": evaluation["strategy_score"],
            "entry_rsi": evaluation["rsi"],
            "passed_conditions": passed,
            "passed_condition_count": len(passed),
            "forward_returns": forward_returns,
            "completed_trade": trade is not None,
            "completed_trade_net_profit": (
                trade["net_profit_loss"] if trade else None
            ),
        }

    @staticmethod
    def _forward_return(candles, entry_candle, horizon, entry_close):
        target = entry_candle + horizon
        if target >= len(candles):
            return None
        return (candles[target]["close"] / entry_close - 1) * 100

    def _band_summary(self, signals, value_getter, bands):
        result = {}
        for label, minimum, maximum in bands:
            selected = [
                signal
                for signal in signals
                if self._in_band(
                    value_getter(signal),
                    minimum,
                    maximum,
                    label,
                )
            ]
            result[label] = self._summarize_signals(selected)
        return result

    @staticmethod
    def _in_band(value, minimum, maximum, label):
        if minimum is not None and value < minimum:
            return False
        if maximum is None:
            return True
        if label == "95-100":
            return value <= maximum
        return value < maximum

    def _condition_combinations(self, signals):
        combinations = {}
        for signal in signals:
            label = " + ".join(signal["passed_conditions"]) or "None"
            combinations.setdefault(label, []).append(signal)
        return {
            label: self._summarize_signals(selected)
            for label, selected in sorted(
                combinations.items(),
                key=lambda item: (-len(item[1]), item[0]),
            )
        }

    def _summarize_signals(self, signals):
        return {
            "signals": len(signals),
            "completed_entries": sum(
                signal["completed_trade"] for signal in signals
            ),
            "insufficient_evidence": (
                len(signals) < MIN_CALIBRATION_SIGNALS
            ),
            "forward_returns": {
                str(horizon): _summary(
                    signal["forward_returns"][horizon]
                    for signal in signals
                    if signal["forward_returns"][horizon] is not None
                )
                for horizon in FORWARD_HORIZONS
            },
        }

    def _movement_summary(self, signals):
        return {
            str(horizon): _summary(
                signal["forward_returns"][horizon]
                for signal in signals
                if signal["forward_returns"][horizon] is not None
            )
            for horizon in FORWARD_HORIZONS
        }

    def _cost_break_even(self, signals):
        threshold = self.break_even_move_percent()
        result = {
            "required_move_percent": threshold,
            "overall": {},
            "by_score_band": {},
            "by_rsi_band": {},
        }
        for label, minimum, maximum in SCORE_BANDS:
            selected = [
                signal
                for signal in signals
                if self._in_band(
                    signal["score"],
                    minimum,
                    maximum,
                    label,
                )
            ]
            result["by_score_band"][label] = (
                self._break_even_summary(selected, threshold)
            )
        for label, minimum, maximum in RSI_BANDS:
            selected = [
                signal
                for signal in signals
                if self._in_band(
                    signal["entry_rsi"],
                    minimum,
                    maximum,
                    label,
                )
            ]
            result["by_rsi_band"][label] = (
                self._break_even_summary(selected, threshold)
            )
        result["overall"] = self._break_even_summary(signals, threshold)
        return result

    @staticmethod
    def _break_even_summary(signals, threshold):
        result = {
            "signals": len(signals),
            "required_move_percent": threshold,
            "horizons": {},
        }
        for horizon in FORWARD_HORIZONS:
            movements = [
                signal["forward_returns"][horizon]
                for signal in signals
                if signal["forward_returns"][horizon] is not None
            ]
            result["horizons"][str(horizon)] = {
                "signals": len(movements),
                "reached_break_even": sum(
                    movement >= threshold for movement in movements
                ),
                "reached_break_even_percent": _percent(
                    sum(movement >= threshold for movement in movements),
                    len(movements),
                ),
                "average_movement": _average(movements),
            }
        return result

    def _diagnosis(self, signals):
        established_scores = [
            (label, summary)
            for label, summary in self._band_summary(
                signals,
                lambda signal: signal["score"],
                SCORE_BANDS,
            ).items()
            if summary["signals"] >= MIN_CALIBRATION_SIGNALS
        ]
        established_rsi = [
            (label, summary)
            for label, summary in self._band_summary(
                signals,
                lambda signal: signal["entry_rsi"],
                RSI_BANDS,
            ).items()
            if summary["signals"] >= MIN_CALIBRATION_SIGNALS
        ]
        score_averages = [
            summary["forward_returns"]["5"]["average"]
            for _, summary in established_scores
        ]
        rsi_averages = [
            summary["forward_returns"]["5"]["average"]
            for _, summary in established_rsi
        ]
        cost = self._cost_break_even(signals)["overall"]
        reached_5d = cost["horizons"]["5"][
            "reached_break_even_percent"
        ]

        evidence = []
        limitations = []
        if len(established_scores) >= 2:
            score_ordered = [
                summary["forward_returns"]["5"]["average"]
                for _, summary in established_scores
            ]
            if score_ordered == sorted(score_ordered):
                score_calibration = (
                    "supported"
                    if len(established_scores) >= 3
                    else "suggestive but incomplete"
                )
            else:
                score_calibration = "not supported"
            evidence.append(
                "5-candle average forward returns across established "
                f"score bands: {score_ordered}"
            )
        else:
            score_calibration = "insufficient evidence"
            limitations.append(
                "fewer than two score bands reached the minimum signal count"
            )

        if len(established_rsi) >= 2:
            rsi_calibration = (
                "supported"
                if max(rsi_averages) - min(rsi_averages) > 1.0
                else "weak separation"
            )
            evidence.append(
                "5-candle RSI-band return spread: "
                f"{max(rsi_averages) - min(rsi_averages):+.2f}%"
            )
        else:
            rsi_calibration = "insufficient evidence"
            limitations.append(
                "fewer than two RSI bands reached the minimum signal count"
            )

        cost_conclusion = (
            "costs materially reduce signal viability"
            if reached_5d < 50
            else "at least half of signals reached cost break-even by 5 candles"
        )
        evidence.append(
            f"{reached_5d:.2f}% of signals reached the "
            f"{cost['required_move_percent']:.3f}% cost break-even move "
            "within five candles"
        )

        if score_calibration == "not supported":
            conclusion = (
                "Entry quality is not demonstrated by monotonic score "
                "calibration; score bands do not consistently improve "
                "early forward movement."
            )
        elif score_calibration == "suggestive but incomplete":
            conclusion = (
                "Higher observed score bands had stronger early movement, "
                "but only two score bands met the sample threshold and no "
                "90+ BUY signals were observed. This is suggestive rather "
                "than conclusive entry-quality evidence."
            )
        elif score_calibration == "insufficient evidence":
            conclusion = (
                "Entry-quality calibration remains uncertain because "
                "the score-band sample is too sparse."
            )
        else:
            conclusion = (
                "Entry-quality evidence shows a monotonic score relationship "
                "in the established bands, subject to regime and cost checks."
            )

        return {
            "score_calibration": score_calibration,
            "rsi_calibration": rsi_calibration,
            "cost_conclusion": cost_conclusion,
            "conclusion": conclusion,
            "evidence": evidence,
            "limitations": limitations,
            "established_score_bands": len(established_scores),
            "established_rsi_bands": len(established_rsi),
        }


def run_strategy_calibration_study(notifier=None):
    if notifier is None:
        from btc_cad_preflight import send_slack_notification

        notifier = send_slack_notification

    loader = YahooBTCADMarketData(data_range="10y")
    candles = loader.load()
    if not candles:
        raise RuntimeError(loader.last_error or "Yahoo BTC/CAD data unavailable")

    selected = select_score_study_periods(candles)
    if len(selected) != len(SCORE_STUDY_PERIODS):
        raise RuntimeError("Calibration study did not select all ten periods")

    runner = MultiPeriodBacktester(starting_capital=STARTING_CAPITAL)
    backtest = runner.run(candles, notifier=notifier)
    expected = [
        (period["start_date"], period["end_date"], period["regime"])
        for period in selected
    ]
    actual = [
        (period["start_date"], period["end_date"], period["regime"])
        for period in backtest["periods"]
    ]
    if actual != expected:
        raise RuntimeError(
            "Calibration backtest periods do not match fixed boundaries"
        )

    return StrategyCalibrationStudy().analyze(
        backtest["periods"],
        [period["candles"] for period in selected],
    )


def _print_band_section(title, bands):
    print(f"\n=== {title} ===")
    for label, summary in bands.items():
        forward = summary["forward_returns"]["5"]
        evidence = "INSUFFICIENT EVIDENCE" if summary[
            "insufficient_evidence"
        ] else ""
        print(
            f"{label}: signals={summary['signals']}, "
            f"completed={summary['completed_entries']}, "
            f"5-candle avg={forward['average']:+.2f}%, "
            f"positive={forward['positive_percent']:.2f}% "
            f"{evidence}"
        )


def print_report(results):
    print("BTC/CAD SCORE / ENTRY CALIBRATION STUDY — ANALYSIS ONLY")
    print("REAL-MONEY TRADING: DISABLED")
    print(
        f"Ten independent periods; {results['signal_count']} BUY signals; "
        f"{results['completed_entry_count']} completed entries."
    )
    _print_band_section("Score bands", results["score_bands"])
    _print_band_section("RSI bands", results["rsi_bands"])

    print("\n=== Condition combinations ===")
    for label, summary in results["condition_combinations"].items():
        forward = summary["forward_returns"]["5"]
        print(
            f"{label}: signals={summary['signals']}, "
            f"5-candle avg={forward['average']:+.2f}%, "
            f"positive={forward['positive_percent']:.2f}%"
        )

    print("\n=== Early movement ===")
    for horizon, summary in results["early_movement"].items():
        print(
            f"+{horizon} candles: signals={summary['count']}, "
            f"avg={summary['average']:+.2f}%, "
            f"positive={summary['positive_percent']:.2f}%"
        )

    cost = results["cost_break_even"]
    print("\n=== Cost break-even ===")
    print(
        f"Required underlying move: "
        f"{cost['required_move_percent']:.3f}%"
    )
    for horizon, summary in cost["overall"]["horizons"].items():
        print(
            f"+{horizon} candles: "
            f"{summary['reached_break_even_percent']:.2f}% reached "
            f"break-even; average movement "
            f"{summary['average_movement']:+.2f}%"
        )
    for title, bands in (
        ("Score bands", cost["by_score_band"]),
        ("RSI bands", cost["by_rsi_band"]),
    ):
        print(f"  {title}, +5 candles:")
        for label, summary in bands.items():
            horizon = summary["horizons"]["5"]
            print(
                f"    {label}: signals={horizon['signals']}, "
                f"break-even={horizon['reached_break_even_percent']:.2f}%"
            )

    diagnosis = results["diagnosis"]
    print("\n=== Final entry-quality diagnosis ===")
    print(f"Score calibration: {diagnosis['score_calibration']}")
    print(f"RSI calibration: {diagnosis['rsi_calibration']}")
    print(f"Cost conclusion: {diagnosis['cost_conclusion']}")
    print(diagnosis["conclusion"])
    for evidence in diagnosis["evidence"]:
        print(f"- {evidence}")
    for limitation in diagnosis["limitations"]:
        print(f"- Limitation: {limitation}")


def main():
    results = run_strategy_calibration_study()
    print_report(results)
    return results


if __name__ == "__main__":
    main()