"""Analysis-only comparison of market state before original STOP LOSS exits."""

from statistics import median

from config import STOP_LOSS_PERCENT
from indicators import calculate_ema, calculate_rsi, calculate_average_volume
from multi_period_backtest import MultiPeriodBacktester
from out_of_sample_validation import _split_periods
from score_effectiveness_study import (
    SCORE_STUDY_PERIODS,
    select_score_study_periods,
)
from strategy_calibration_study import STARTING_CAPITAL
from yahoo_btc_cad_data import YahooBTCADMarketData


PRE_STOP_OFFSETS = (-3, -2, -1, 0)
RECOVERED_ENTRY = "RECOVERED_ENTRY"
CONTINUED_LOSS = "CONTINUED_LOSS"
TARGETS = (("two_percent", 2.0), ("four_percent", 4.0))


def _average(values):
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else 0.0


def _median(values):
    values = [value for value in values if value is not None]
    return median(values) if values else 0.0


def _percent(numerator, denominator):
    return numerator / denominator * 100 if denominator else 0.0


def _feature_summary(values):
    values = [value for value in values if value is not None]
    return {
        "count": len(values),
        "average": _average(values),
        "median": _median(values),
    }


class PreStopMarketStateStudy:
    """Compare pre-stop states without changing the control backtest."""

    def analyze_trade(self, trade, candles, evaluations, period, regime):
        if trade["reason"] != "STOP LOSS":
            raise ValueError("analyze_trade requires a STOP LOSS trade")
        entry = trade["entry_candle"]
        stop = trade["exit_candle"]
        if not 0 <= entry <= stop < len(candles):
            raise ValueError("trade candle indexes are outside the candle set")

        entry_price = trade["market_entry_price"]
        stop_price = entry_price * (1 - STOP_LOSS_PERCENT)
        recovery = self._recovery_classification(
            candles[stop + 1:],
            entry_price,
        )
        entry_evaluation = evaluations.get(entry)
        if entry_evaluation is None:
            raise ValueError("STOP LOSS trade is missing entry evaluation")

        states = {}
        for offset in PRE_STOP_OFFSETS:
            candle_index = stop + offset
            states[offset] = self._state_at(
                candle_index,
                candles,
                evaluations,
                entry_price,
                stop_price,
            )

        return {
            "trade_number": trade["trade_number"],
            "period": period,
            "regime": regime,
            "entry_candle": entry,
            "stop_candle": stop,
            "candles_held": stop - entry,
            "entry_score": entry_evaluation["strategy_score"],
            "entry_rsi": entry_evaluation["rsi"],
            "classification": recovery["classification"],
            "reached_two_percent": recovery["reached_two_percent"],
            "reached_four_percent": recovery["reached_four_percent"],
            "states": states,
        }

    @staticmethod
    def _recovery_classification(after, entry_price):
        recovered = any(candle["high"] >= entry_price for candle in after)
        reached_two = any(
            candle["high"] >= entry_price * 1.02 for candle in after
        )
        reached_four = any(
            candle["high"] >= entry_price * 1.04 for candle in after
        )
        return {
            "classification": RECOVERED_ENTRY if recovered else CONTINUED_LOSS,
            "reached_two_percent": reached_two,
            "reached_four_percent": reached_four,
        }

    @staticmethod
    def _relative_percent(numerator, denominator):
        if denominator in (None, 0):
            return None
        return (numerator / denominator - 1) * 100

    @classmethod
    def _state_at(
        cls,
        index,
        candles,
        evaluations,
        entry_price,
        stop_price,
    ):
        if index < 0 or index >= len(candles):
            return None
        candle = candles[index]
        evaluation = evaluations.get(index)
        prices = [item["close"] for item in candles[:index + 1]]
        volumes = [item["volume"] for item in candles[:index + 1]]
        ema21 = calculate_ema(prices, 21)
        ema50 = calculate_ema(prices, 50)
        ema200 = calculate_ema(prices, 200)
        rsi = calculate_rsi(prices)
        average_volume = calculate_average_volume(volumes)
        previous_close = candles[index - 1]["close"] if index else None
        previous_volume = candles[index - 1]["volume"] if index else None

        if evaluation is not None:
            rsi = evaluation["rsi"]
            ema21 = evaluation["ema21"]
            ema50 = evaluation["ema50"]
            ema200 = evaluation["ema200"]
        condition_state = {
            "short_term_momentum": (
                bool(evaluation["short_term_momentum"])
                if evaluation else None
            ),
            "long_term_trend": (
                bool(evaluation["long_term_trend"])
                if evaluation else None
            ),
        }
        return {
            "candle": index,
            "timestamp": candle["timestamp"],
            "rsi": rsi,
            "rsi_change_1": (
                rsi - evaluations[index - 1]["rsi"]
                if rsi is not None and index - 1 in evaluations
                else None
            ),
            "rsi_change_3": (
                rsi - evaluations[index - 3]["rsi"]
                if rsi is not None and index - 3 in evaluations
                else None
            ),
            "price_vs_ema21_percent": cls._relative_percent(
                candle["close"], ema21
            ),
            "ema21_vs_ema50_percent": cls._relative_percent(ema21, ema50),
            "ema50_vs_ema200_percent": cls._relative_percent(ema50, ema200),
            "distance_from_entry_percent": cls._relative_percent(
                candle["close"], entry_price
            ),
            "distance_from_stop_percent": cls._relative_percent(
                candle["close"], stop_price
            ),
            "price_change_1_percent": (
                cls._relative_percent(candle["close"], previous_close)
                if previous_close else None
            ),
            "price_change_3_percent": (
                cls._relative_percent(
                    candle["close"], candles[index - 3]["close"]
                )
                if index >= 3 else None
            ),
            "price_change_5_percent": (
                cls._relative_percent(
                    candle["close"], candles[index - 5]["close"]
                )
                if index >= 5 else None
            ),
            "volume": candle["volume"],
            "volume_change_1_percent": (
                cls._relative_percent(candle["volume"], previous_volume)
                if previous_volume else None
            ),
            "volume_vs_average_percent": (
                cls._relative_percent(candle["volume"], average_volume)
                if average_volume else None
            ),
            "candle_body_percent": cls._relative_percent(
                candle["close"], candle["open"]
            ),
            "candle_range_percent": (
                (candle["high"] - candle["low"]) / candle["open"] * 100
                if candle["open"] else None
            ),
            "high_excursion_percent": cls._relative_percent(
                candle["high"], candle["close"]
            ),
            "low_excursion_percent": cls._relative_percent(
                candle["low"], candle["close"]
            ),
            **condition_state,
        }

    def analyze_period(self, period_result, candles):
        evaluations = {
            item["candle"]: item
            for item in period_result["evaluation_history"]
        }
        trades = [
            self.analyze_trade(
                trade,
                candles,
                evaluations,
                period_result["period"],
                period_result["regime"],
            )
            for trade in period_result["trades_history"]
            if trade["reason"] == "STOP LOSS"
        ]
        return {
            "period": period_result["period"],
            "start_date": period_result["start_date"],
            "end_date": period_result["end_date"],
            "regime": period_result["regime"],
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
        variables = (
            "rsi",
            "rsi_change_1",
            "rsi_change_3",
            "price_vs_ema21_percent",
            "ema21_vs_ema50_percent",
            "ema50_vs_ema200_percent",
            "distance_from_entry_percent",
            "distance_from_stop_percent",
            "price_change_1_percent",
            "price_change_3_percent",
            "price_change_5_percent",
            "volume",
            "volume_change_1_percent",
            "volume_vs_average_percent",
            "candle_body_percent",
            "candle_range_percent",
            "high_excursion_percent",
            "low_excursion_percent",
            "short_term_momentum",
            "long_term_trend",
            "entry_score",
            "entry_rsi",
            "candles_held",
        )
        comparisons = {}
        for offset in PRE_STOP_OFFSETS:
            comparisons[offset] = {}
            for variable in variables:
                if variable in ("entry_score", "entry_rsi", "candles_held"):
                    values = {
                        RECOVERED_ENTRY: [
                            trade[variable] for trade in trades
                            if trade["classification"] == RECOVERED_ENTRY
                        ],
                        CONTINUED_LOSS: [
                            trade[variable] for trade in trades
                            if trade["classification"] == CONTINUED_LOSS
                        ],
                    }
                else:
                    values = {
                        group: [
                            trade["states"][offset][variable]
                            for trade in trades
                            if trade["classification"] == group
                            and trade["states"][offset] is not None
                        ]
                        for group in (RECOVERED_ENTRY, CONTINUED_LOSS)
                    }
                recovering = _feature_summary(values[RECOVERED_ENTRY])
                continued = _feature_summary(values[CONTINUED_LOSS])
                comparisons[offset][variable] = {
                    "recovering": recovering,
                    "continued_loss": continued,
                    "difference_recovering_minus_continued": (
                        recovering["average"] - continued["average"]
                        if recovering["count"] and continued["count"]
                        else None
                    ),
                }
        return {
            "stop_loss_count": len(trades),
            "recovered_entry_count": sum(
                trade["classification"] == RECOVERED_ENTRY for trade in trades
            ),
            "continued_loss_count": sum(
                trade["classification"] == CONTINUED_LOSS for trade in trades
            ),
            "recovered_entry_percent": _percent(
                sum(trade["classification"] == RECOVERED_ENTRY for trade in trades),
                len(trades),
            ),
            "reached_two_percent_count": sum(
                trade["reached_two_percent"] for trade in trades
            ),
            "reached_four_percent_count": sum(
                trade["reached_four_percent"] for trade in trades
            ),
            "comparisons": comparisons,
            "pattern_assessment": self._pattern_assessment(comparisons),
        }

    @staticmethod
    def _pattern_assessment(comparisons):
        assessment = {}
        for variable in comparisons[0]:
            differences = [
                comparisons[offset][variable][
                    "difference_recovering_minus_continued"
                ]
                for offset in PRE_STOP_OFFSETS
            ]
            differences = [value for value in differences if value is not None]
            assessment[variable] = {
                "available_offsets": len(differences),
                "direction": (
                    "recovering_higher" if all(value > 0 for value in differences)
                    else "recovering_lower" if all(value < 0 for value in differences)
                    else "mixed_or_unavailable"
                ),
            }
        return assessment


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


def run_pre_stop_market_state_study(notifier=None):
    if notifier is None:
        from btc_cad_preflight import send_slack_notification
        notifier = send_slack_notification
    loader = YahooBTCADMarketData(data_range="10y")
    candles = loader.load()
    if not candles:
        raise RuntimeError(loader.last_error or "Yahoo BTC/CAD data unavailable")
    selected = select_score_study_periods(candles)
    if len(selected) != len(SCORE_STUDY_PERIODS):
        raise RuntimeError("Pre-stop study requires all fixed periods")
    research, validation = _split_periods(selected)
    study = PreStopMarketStateStudy()
    research_results = _run_period_group(research, notifier)
    validation_results = _run_period_group(validation, notifier)
    return {
        "real_money_trading": False,
        "research": study.analyze_group(
            research_results,
            [period["candles"] for period in research],
        ),
        "validation": study.analyze_group(
            validation_results,
            [period["candles"] for period in validation],
        ),
        "split": {
            "research_start": research[0]["start_date"],
            "research_end": research[-1]["end_date"],
            "validation_start": validation[0]["start_date"],
            "validation_end": validation[-1]["end_date"],
        },
    }


def print_report(results):
    print("BTC/CAD PRE-STOP MARKET-STATE STUDY — ANALYSIS ONLY")
    print("REAL-MONEY TRADING: DISABLED")
    for label, group in (
        ("Research", results["research"]),
        ("Untouched validation", results["validation"]),
    ):
        summary = group["summary"]
        print(f"\n=== {label} ===")
        print(
            f"STOP LOSS trades={summary['stop_loss_count']}, "
            f"recovered={summary['recovered_entry_count']} "
            f"({summary['recovered_entry_percent']:.2f}%), "
            f"continued={summary['continued_loss_count']}"
        )
        for offset in PRE_STOP_OFFSETS:
            strongest = sorted(
                (
                    (name, item["difference_recovering_minus_continued"])
                    for name, item in summary["comparisons"][offset].items()
                    if item["difference_recovering_minus_continued"] is not None
                ),
                key=lambda item: abs(item[1]),
                reverse=True,
            )[:5]
            print(
                f"{offset:+d} candle state strongest differences: "
                + ", ".join(f"{name}={difference:+.3f}" for name, difference in strongest)
                if strongest else f"{offset:+d} candle state: no comparison group"
            )
        print(
            "Targets after stop: "
            f"+2%={summary['reached_two_percent_count']}, "
            f"+4%={summary['reached_four_percent_count']}"
        )
    print("\n=== Interpretation boundary ===")
    print(
        "Findings are descriptive. Validation is not used to tune rules, "
        "and no stop-loss candidate or production change is recommended."
    )


def main():
    results = run_pre_stop_market_state_study()
    print_report(results)
    return results


if __name__ == "__main__":
    main()