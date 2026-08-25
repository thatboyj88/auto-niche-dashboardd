from statistics import median

from multi_period_backtest import MultiPeriodBacktester
from out_of_sample_validation import _split_periods
from score_effectiveness_study import (
    SCORE_STUDY_PERIODS,
    select_score_study_periods,
)
from strategy_calibration_study import (
    FEE_PERCENT,
    RSI_BANDS,
    SCORE_BANDS,
    SLIPPAGE_PERCENT,
    STARTING_CAPITAL,
    StrategyCalibrationStudy,
)
from yahoo_btc_cad_data import YahooBTCADMarketData


FORWARD_HORIZONS = (1, 3, 5, 10, 20)
BREAK_EVEN_MOVE_PERCENT = StrategyCalibrationStudy.break_even_move_percent(
    fee_percent=FEE_PERCENT,
    slippage_percent=SLIPPAGE_PERCENT,
)
TARGETS = (
    ("break_even", BREAK_EVEN_MOVE_PERCENT),
    ("two_percent", 2.0),
    ("take_profit", 4.0),
)
SIGNAL_QUALITY_BREAK_EVEN_THRESHOLD = 50.0
MATERIAL_COST_SHARE_THRESHOLD = 50.0
MIN_REGIME_SIGNALS = 20
MAX_REASONABLE_AVG_CANDLES_BETWEEN_TRADES = 5.0


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


class TradeEconomicsStudy:
    """Measure signal opportunity and trade economics without tuning."""

    @staticmethod
    def _target_reach(candles, entry_candle, entry_close, target_percent):
        target_price = entry_close * (1 + target_percent / 100)
        for offset, candle in enumerate(
            candles[entry_candle + 1:entry_candle + 21],
            start=1,
        ):
            if candle["high"] >= target_price:
                return {
                    "reached": True,
                    "candles": offset,
                    "date": candle["timestamp"],
                }
        return {
            "reached": False,
            "candles": None,
            "date": None,
        }

    def _build_signal(self, evaluation, candles, trade, regime, period):
        entry_candle = evaluation["candle"]
        entry_close = candles[entry_candle]["close"]
        observation = candles[entry_candle + 1:entry_candle + 21]
        forward_returns = {}
        for horizon in FORWARD_HORIZONS:
            target = entry_candle + horizon
            forward_returns[horizon] = (
                (candles[target]["close"] / entry_close - 1) * 100
                if target < len(candles)
                else None
            )
        favorable = [
            (candle["high"] / entry_close - 1) * 100
            for candle in observation
        ]
        adverse = [
            (candle["low"] / entry_close - 1) * 100
            for candle in observation
        ]
        targets = {
            name: self._target_reach(
                candles,
                entry_candle,
                entry_close,
                threshold,
            )
            for name, threshold in TARGETS
        }
        if not targets["break_even"]["reached"]:
            movement_category = "Never reached break-even"
        elif not targets["two_percent"]["reached"]:
            movement_category = "Reached break-even, not 2%"
        elif not targets["take_profit"]["reached"]:
            movement_category = "Reached 2%, not 4%"
        else:
            movement_category = "Reached 4%"
        return {
            "period": period,
            "regime": regime,
            "candle": entry_candle,
            "timestamp": evaluation["timestamp"],
            "price": entry_close,
            "score": evaluation["strategy_score"],
            "rsi": evaluation["rsi"],
            "conditions": {
                key: bool(evaluation[key])
                for key in (
                    "long_term_trend",
                    "short_term_momentum",
                    "rsi_condition",
                    "volume",
                    "price_above_ema21",
                )
            },
            "forward_returns": forward_returns,
            "mfe_percent": max(favorable, default=0.0),
            "mae_percent": min(adverse, default=0.0),
            "targets": targets,
            "movement_category": movement_category,
            "completed_trade": trade is not None,
            "trade": trade,
        }

    def analyze_group(self, period_results, period_candles):
        if len(period_results) != len(period_candles):
            raise ValueError("period result and candle counts must match")
        signals = []
        for period_result, candles in zip(period_results, period_candles):
            trades = {
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
                        trades.get(evaluation["candle"]),
                        period_result["regime"],
                        period_result["period"],
                    )
                )
        return {
            "signals": signals,
            "signal_count": len(signals),
            "completed_trade_count": sum(
                signal["completed_trade"] for signal in signals
            ),
            "signal_economics": self._signal_economics(signals),
            "trade_economics": self._trade_economics(period_results),
            "frequency": self._frequency(period_results, signals),
            "score_bands": self._band_economics(
                signals,
                lambda signal: signal["score"],
                SCORE_BANDS,
            ),
            "rsi_bands": self._band_economics(
                signals,
                lambda signal: signal["rsi"],
                RSI_BANDS,
            ),
            "winner_loser": self._winner_loser(signals),
            "movement_categories": self._movement_categories(signals),
            "by_regime": self._by_regime(signals),
            "cost_break_even_percent": self._cost_break_even_percent(
                signals
            ),
        }

    def _signal_economics(self, signals):
        return {
            "forward_returns": {
                str(horizon): _summary(
                    signal["forward_returns"][horizon]
                    for signal in signals
                    if signal["forward_returns"][horizon] is not None
                )
                for horizon in FORWARD_HORIZONS
            },
            "mfe_percent": _summary(
                signal["mfe_percent"] for signal in signals
            ),
            "mae_percent": _summary(
                signal["mae_percent"] for signal in signals
            ),
            "targets": {
                name: {
                    "reached": sum(
                        signal["targets"][name]["reached"]
                        for signal in signals
                    ),
                    "reached_percent": _percent(
                        sum(
                            signal["targets"][name]["reached"]
                            for signal in signals
                        ),
                        len(signals),
                    ),
                    "average_candles_when_reached": _average(
                        signal["targets"][name]["candles"]
                        for signal in signals
                        if signal["targets"][name]["reached"]
                    ),
                }
                for name, _ in TARGETS
            },
        }

    @staticmethod
    def _trade_economics(period_results):
        trades = [
            trade
            for period in period_results
            for trade in period["trades_history"]
        ]
        winners = [trade for trade in trades if trade["net_profit_loss"] > 0]
        losers = [trade for trade in trades if trade["net_profit_loss"] <= 0]
        gross = sum(
            trade["gross_profit_loss_before_costs"]
            for trade in trades
        )
        fees = sum(trade["fees"] for trade in trades)
        slippage = sum(trade["estimated_slippage"] for trade in trades)
        net = sum(trade["net_profit_loss"] for trade in trades)

        def trade_summary(selected):
            return {
                "count": len(selected),
                "gross_per_trade": (
                    _average(
                        trade["gross_profit_loss_before_costs"]
                        for trade in selected
                    )
                ),
                "net_per_trade": _average(
                    trade["net_profit_loss"] for trade in selected
                ),
                "fees": sum(trade["fees"] for trade in selected),
                "slippage": sum(
                    trade["estimated_slippage"] for trade in selected
                ),
            }

        return {
            "completed_trades": len(trades),
            "fees": fees,
            "slippage": slippage,
            "total_costs": fees + slippage,
            "cost_per_completed_trade": _average(
                trade["fees"] + trade["estimated_slippage"]
                for trade in trades
            ),
            "gross_profit_loss": gross,
            "net_profit_loss": net,
            "gross_per_trade": _average(
                trade["gross_profit_loss_before_costs"]
                for trade in trades
            ),
            "net_per_trade": _average(
                trade["net_profit_loss"] for trade in trades
            ),
            "gross_per_winning_trade": _average(
                trade["gross_profit_loss_before_costs"]
                for trade in winners
            ),
            "net_per_winning_trade": _average(
                trade["net_profit_loss"] for trade in winners
            ),
            "gross_per_losing_trade": _average(
                trade["gross_profit_loss_before_costs"]
                for trade in losers
            ),
            "net_per_losing_trade": _average(
                trade["net_profit_loss"] for trade in losers
            ),
            "cost_share_percent": (
                (fees + slippage) / abs(gross) * 100
                if gross
                else 0.0
            ),
            "winners": trade_summary(winners),
            "losers": trade_summary(losers),
        }

    @staticmethod
    def _frequency(period_results, signals):
        trade_entries = []
        durations = []
        turnover = 0.0
        for period in period_results:
            entries = sorted(
                trade["entry_candle"]
                for trade in period["trades_history"]
            )
            trade_entries.extend(entries)
            durations.extend(
                trade["exit_candle"] - trade["entry_candle"]
                for trade in period["trades_history"]
            )
            turnover += sum(
                trade["position_size"] * trade["market_entry_price"]
                for trade in period["trades_history"]
            )
        gaps = []
        for period in period_results:
            entries = sorted(
                trade["entry_candle"]
                for trade in period["trades_history"]
            )
            gaps.extend(
                later - earlier
                for earlier, later in zip(entries, entries[1:])
            )
        period_count = len(period_results)
        return {
            "buy_signals_per_period": _average(
                [sum(signal["period"] == period["period"] for signal in signals)
                 for period in period_results]
            ),
            "completed_trades_per_period": _average(
                [len(period["trades_history"]) for period in period_results]
            ),
            "average_candles_between_trades": _average(gaps),
            "average_days_between_trades": _average(gaps),
            "average_trade_duration_candles": _average(durations),
            "average_trade_duration_days": _average(durations),
            "trades_per_year": (
                len(trade_entries) / period_count if period_count else 0.0
            ),
            "turnover": turnover,
            "turnover_relative_to_starting_capital": (
                turnover / (STARTING_CAPITAL * period_count)
                if period_count
                else 0.0
            ),
        }

    def _band_economics(self, signals, value_getter, bands):
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
            result[label] = self._selected_economics(selected)
        return result

    @staticmethod
    def _in_band(value, minimum, maximum, label):
        if minimum is not None and value < minimum:
            return False
        if maximum is None:
            return True
        if label in ("80-84", "85-89", "90-94", "95-100"):
            return value <= maximum
        return value < maximum

    def _selected_economics(self, signals):
        trades = [
            signal["trade"] for signal in signals
            if signal["trade"] is not None
        ]
        gross = sum(
            trade["gross_profit_loss_before_costs"] for trade in trades
        )
        net = sum(trade["net_profit_loss"] for trade in trades)
        fees = sum(trade["fees"] for trade in trades)
        slippage = sum(trade["estimated_slippage"] for trade in trades)
        return {
            "signals": len(signals),
            "completed_trades": len(trades),
            "forward_returns": {
                str(horizon): _summary(
                    signal["forward_returns"][horizon]
                    for signal in signals
                    if signal["forward_returns"][horizon] is not None
                )
                for horizon in FORWARD_HORIZONS
            },
            "break_even_rate": _percent(
                sum(
                    signal["targets"]["break_even"]["reached"]
                    for signal in signals
                ),
                len(signals),
            ),
            "two_percent_rate": _percent(
                sum(
                    signal["targets"]["two_percent"]["reached"]
                    for signal in signals
                ),
                len(signals),
            ),
            "take_profit_rate": _percent(
                sum(
                    signal["targets"]["take_profit"]["reached"]
                    for signal in signals
                ),
                len(signals),
            ),
            "mfe_percent": _summary(
                signal["mfe_percent"] for signal in signals
            ),
            "mae_percent": _summary(
                signal["mae_percent"] for signal in signals
            ),
            "gross_profit_loss": gross,
            "net_profit_loss": net,
            "cost_share_percent": (
                (fees + slippage) / abs(gross) * 100
                if gross
                else 0.0
            ),
        }

    @staticmethod
    def _winner_loser(signals):
        completed = [signal for signal in signals if signal["trade"]]
        result = {}
        for label, selected in (
            ("winners", [
                signal for signal in completed
                if signal["trade"]["net_profit_loss"] > 0
            ]),
            ("losers", [
                signal for signal in completed
                if signal["trade"]["net_profit_loss"] <= 0
            ]),
        ):
            trades = [signal["trade"] for signal in selected]
            result[label] = {
                "count": len(selected),
                "entry_score": _summary(
                    signal["score"] for signal in selected
                ),
                "entry_rsi": _summary(
                    signal["rsi"] for signal in selected
                ),
                "duration_candles": _summary(
                    trade["exit_candle"] - trade["entry_candle"]
                    for trade in trades
                ),
                "mfe_percent": _summary(
                    signal["mfe_percent"] for signal in selected
                ),
                "mae_percent": _summary(
                    signal["mae_percent"] for signal in selected
                ),
                "forward_5_candle": _summary(
                    signal["forward_returns"][5]
                    for signal in selected
                    if signal["forward_returns"][5] is not None
                ),
                "gross_profit_loss": sum(
                    trade["gross_profit_loss_before_costs"]
                    for trade in trades
                ),
                "net_profit_loss": sum(
                    trade["net_profit_loss"] for trade in trades
                ),
                "exit_reasons": {
                    reason: sum(
                        trade["reason"] == reason for trade in trades
                    )
                    for reason in (
                        "STOP LOSS",
                        "TAKE PROFIT",
                        "END OF TEST",
                    )
                },
                "cost_burden": sum(
                    trade["fees"] + trade["estimated_slippage"]
                    for trade in trades
                ),
            }
        return result

    @staticmethod
    def _movement_categories(signals):
        categories = (
            "Never reached break-even",
            "Reached break-even, not 2%",
            "Reached 2%, not 4%",
            "Reached 4%",
        )
        return {
            category: {
                "signals": sum(
                    signal["movement_category"] == category
                    for signal in signals
                ),
                "percent": _percent(
                    sum(
                        signal["movement_category"] == category
                        for signal in signals
                    ),
                    len(signals),
                ),
            }
            for category in categories
        }

    def _by_regime(self, signals):
        return {
            regime: self._selected_economics([
                signal for signal in signals
                if signal["regime"] == regime
            ])
            for regime in ("Bull", "Sideways", "Bear")
        }

    @staticmethod
    def _cost_break_even_percent(signals):
        return _percent(
            sum(
                signal["targets"]["break_even"]["reached"]
                for signal in signals
            ),
            len(signals),
        )

    def diagnose(self, analysis):
        signal_economics = analysis["signal_economics"]
        trade_economics = analysis["trade_economics"]
        frequency = analysis["frequency"]
        findings = []
        evidence = []
        five_day_break_even = analysis["cost_break_even_percent"]
        if five_day_break_even < SIGNAL_QUALITY_BREAK_EVEN_THRESHOLD:
            findings.append("SIGNAL QUALITY")
            evidence.append(
                f"only {five_day_break_even:.2f}% of BUY signals ever "
                f"reached the {BREAK_EVEN_MOVE_PERCENT:.3f}% break-even "
                "movement within the 20-candle observation window"
            )
        if (
            trade_economics["cost_share_percent"] >=
            MATERIAL_COST_SHARE_THRESHOLD
        ):
            findings.append("COST BURDEN")
            evidence.append(
                f"fees plus slippage consumed "
                f"{trade_economics['cost_share_percent']:.2f}% of "
                "absolute gross completed-trade P/L"
            )
        if (
            frequency["average_candles_between_trades"] and
            frequency["average_candles_between_trades"] <
            MAX_REASONABLE_AVG_CANDLES_BETWEEN_TRADES and
            trade_economics["net_per_trade"] <= 0
        ):
            findings.append("TRADE FREQUENCY")
            evidence.append(
                f"average trade spacing was "
                f"{frequency['average_candles_between_trades']:.2f} "
                "candles while average net trade P/L was non-positive"
            )
        take_profit_rate = signal_economics["targets"]["take_profit"][
            "reached_percent"
        ]
        if take_profit_rate >= 25.0 and trade_economics["net_per_trade"] <= 0:
            findings.append("EXIT CAPTURE")
            evidence.append(
                f"{take_profit_rate:.2f}% of signals reached +4% while "
                "average completed-trade net P/L was non-positive"
            )
        regime_signal_counts = {
            regime: summary["signals"]
            for regime, summary in analysis["by_regime"].items()
        }
        active = [
            regime for regime, count in regime_signal_counts.items()
            if count >= MIN_REGIME_SIGNALS
        ]
        inactive = [
            regime for regime, count in regime_signal_counts.items()
            if count == 0
        ]
        if active and inactive:
            findings.append("REGIME DEPENDENCE")
            evidence.append(
                "signals met the minimum sample threshold in "
                + ", ".join(active)
                + " but none were recorded in "
                + ", ".join(inactive)
            )
        if not findings:
            findings.append("NO DOMINANT WEAKNESS IDENTIFIED")
            evidence.append(
                "no diagnostic threshold was crossed by the measured "
                "economic categories"
            )
        return {
            "dominant_weaknesses": findings,
            "evidence": evidence,
            "unsupported": [
                "No strategy changes are supported by this diagnostic alone.",
                "No predictive or statistically significant claim is made.",
            ],
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


def run_trade_economics_study(notifier=None):
    if notifier is None:
        from btc_cad_preflight import send_slack_notification

        notifier = send_slack_notification
    loader = YahooBTCADMarketData(data_range="10y")
    candles = loader.load()
    if not candles:
        raise RuntimeError(loader.last_error or "Yahoo BTC/CAD data unavailable")
    selected = select_score_study_periods(candles)
    if len(selected) != len(SCORE_STUDY_PERIODS):
        raise RuntimeError("Trade economics study requires all fixed periods")
    research, validation = _split_periods(selected)
    study = TradeEconomicsStudy()

    research_results = _run_period_group(research, notifier)
    research_analysis = study.analyze_group(
        research_results,
        [period["candles"] for period in research],
    )
    validation_results = _run_period_group(validation, notifier)
    validation_analysis = study.analyze_group(
        validation_results,
        [period["candles"] for period in validation],
    )
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
        "research": research_analysis,
        "validation": validation_analysis,
        "research_diagnosis": study.diagnose(research_analysis),
        "validation_diagnosis": study.diagnose(validation_analysis),
    }


def _print_analysis(label, analysis, diagnosis):
    print(f"\n=== {label} ===")
    print(
        f"BUY signals={analysis['signal_count']}, "
        f"completed trades={analysis['completed_trade_count']}"
    )
    trade = analysis["trade_economics"]
    print(
        f"gross=${trade['gross_profit_loss']:+.4f}, "
        f"fees=${trade['fees']:.4f}, "
        f"slippage=${trade['slippage']:.4f}, "
        f"net=${trade['net_profit_loss']:+.4f}, "
        f"net/trade=${trade['net_per_trade']:+.4f}, "
        f"cost share={trade['cost_share_percent']:.2f}%"
    )
    frequency = analysis["frequency"]
    print(
        f"signals/period={frequency['buy_signals_per_period']:.2f}, "
        f"trades/period={frequency['completed_trades_per_period']:.2f}, "
        f"avg spacing={frequency['average_candles_between_trades']:.2f} "
        f"candles, avg duration={frequency['average_trade_duration_candles']:.2f} "
        f"candles, trades/year={frequency['trades_per_year']:.2f}, "
        f"turnover/start={frequency['turnover_relative_to_starting_capital']:.2f}x"
    )
    signal = analysis["signal_economics"]
    for horizon in FORWARD_HORIZONS:
        summary = signal["forward_returns"][str(horizon)]
        print(
            f"+{horizon} candles: avg={summary['average']:+.2f}%, "
            f"positive={_percent(sum(value > 0 for value in [
                item['forward_returns'][horizon]
                for item in analysis['signals']
                if item['forward_returns'][horizon] is not None
            ]), summary['count']):.2f}%"
        )
    print("Target reach:")
    for name, target in TARGETS:
        target_summary = signal["targets"][name]
        print(
            f"  {name} ({target:.3f}%): "
            f"{target_summary['reached']} "
            f"({target_summary['reached_percent']:.2f}%), "
            f"avg candles={target_summary['average_candles_when_reached']:.2f}"
        )
    print("Movement categories:")
    for category, summary in analysis["movement_categories"].items():
        print(
            f"  {category}: {summary['signals']} "
            f"({summary['percent']:.2f}%)"
        )
    print("Dominant measured weaknesses: " + ", ".join(
        diagnosis["dominant_weaknesses"]
    ))
    for evidence in diagnosis["evidence"]:
        print(f"- {evidence}")


def print_report(results):
    print("BTC/CAD TRADE ECONOMICS & OPPORTUNITY QUALITY — ANALYSIS ONLY")
    print("REAL-MONEY TRADING: DISABLED")
    split = results["split"]
    print(
        f"Research: {split['research_start']} to {split['research_end']} "
        f"({split['research_periods']} periods, {split['research_candles']} candles)"
    )
    print(
        f"Validation: {split['validation_start']} to "
        f"{split['validation_end']} "
        f"({split['validation_periods']} periods, "
        f"{split['validation_candles']} candles)"
    )
    _print_analysis(
        "Research",
        results["research"],
        results["research_diagnosis"],
    )
    _print_analysis(
        "Out-of-sample validation",
        results["validation"],
        results["validation_diagnosis"],
    )
    print("\n=== Research → validation consistency ===")
    for label, getter in (
        (
            "5-candle break-even rate",
            lambda analysis: analysis["cost_break_even_percent"],
        ),
        (
            "average net P/L per completed trade",
            lambda analysis: analysis["trade_economics"]["net_per_trade"],
        ),
        (
            "completed trades",
            lambda analysis: analysis["completed_trade_count"],
        ),
    ):
        research_value = getter(results["research"])
        validation_value = getter(results["validation"])
        print(f"{label}: research={research_value} → validation={validation_value}")
    print("\n=== Winner vs loser economics (research / validation) ===")
    for group_name in ("research", "validation"):
        print(f"{group_name}:")
        for outcome, summary in results[group_name]["winner_loser"].items():
            print(
                f"  {outcome}: n={summary['count']}, "
                f"score={summary['entry_score']['average']:.2f}, "
                f"RSI={summary['entry_rsi']['average']:.2f}, "
                f"MFE={summary['mfe_percent']['average']:+.2f}%, "
                f"MAE={summary['mae_percent']['average']:+.2f}%, "
                f"net=${summary['net_profit_loss']:+.4f}"
            )
    print("\n=== Final verdict ===")
    print(
        "The findings are diagnostic observations, not proof of predictive "
        "relationships. No strategy change is recommended automatically."
    )


def main():
    results = run_trade_economics_study()
    print_report(results)
    return results


if __name__ == "__main__":
    main()