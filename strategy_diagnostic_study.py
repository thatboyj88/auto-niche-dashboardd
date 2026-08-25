from statistics import median

from multi_period_backtest import MultiPeriodBacktester
from score_effectiveness_study import (
    SCORE_STUDY_PERIODS,
    select_score_study_periods,
)
from yahoo_btc_cad_data import YahooBTCADMarketData
from historical_validation import measure_decision_consistency


STARTING_CAPITAL = 25.00
FORWARD_HORIZONS = (3, 5, 10, 20)
SCORE_BUCKETS = (
    ("80-84", 80, 84),
    ("85-89", 85, 89),
    ("90-94", 90, 94),
    ("95-100", 95, 100),
)
CONDITIONS = (
    ("long_term_trend", "Long-term trend"),
    ("short_term_momentum", "Short-term momentum"),
    ("rsi_condition", "RSI"),
    ("volume", "Volume"),
    ("price_above_ema21", "Price above EMA21"),
)
EXIT_REASONS = ("STOP LOSS", "TAKE PROFIT", "END OF TEST")
MIN_SCORE_BUCKET_TRADES = 20
MATERIAL_COST_SHARE_PERCENT = 50.0
MATERIAL_STOP_LOSS_PERCENT = 60.0


def _average(values):
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _median(values):
    values = list(values)
    return median(values) if values else 0.0


def _percent(numerator, denominator):
    return numerator / denominator * 100 if denominator else 0.0


def _value_summary(values):
    values = list(values)
    return {
        "count": len(values),
        "average": _average(values),
        "median": _median(values),
        "minimum": min(values) if values else None,
        "maximum": max(values) if values else None,
    }


class StrategyDiagnosticStudy:
    """Analyze unchanged paper-backtest results without placing trades."""

    def analyze_period(self, period_result, candles):
        evaluations = {
            evaluation["candle"]: evaluation
            for evaluation in period_result["evaluation_history"]
        }
        trades = [
            self._enrich_trade(trade, evaluations, candles)
            for trade in period_result["trades_history"]
        ]
        return {
            "period": period_result["period"],
            "start_date": period_result["start_date"],
            "end_date": period_result["end_date"],
            "regime": period_result["regime"],
            "market_return": period_result["market_return"],
            "strategy_return": period_result["return_percent"],
            "candles": period_result["candle_count"],
            "trades": trades,
            "score_effectiveness": self._score_effectiveness(trades),
            "condition_effectiveness": (
                self._condition_effectiveness(trades)
            ),
            "entry_timing": self._entry_timing(trades),
            "exit_behavior": self._exit_behavior(trades),
            "mfe_mae": self._mfe_mae(trades),
        }

    def analyze(self, period_results, period_candles):
        if len(period_results) != len(period_candles):
            raise ValueError(
                "period result and candle counts must match"
            )

        periods = [
            self.analyze_period(result, candles)
            for result, candles in zip(period_results, period_candles)
        ]
        all_trades = [
            trade
            for period in periods
            for trade in period["trades"]
        ]
        return {
            "source": "Yahoo Finance BTC/CAD aggregated daily data",
            "period_count": len(periods),
            "periods": periods,
            "by_regime": self._regime_performance(period_results),
            "score_effectiveness": self._score_effectiveness(all_trades),
            "condition_effectiveness": (
                self._condition_effectiveness(all_trades)
            ),
            "entry_timing": self._entry_timing(all_trades),
            "exit_behavior": self._exit_behavior(all_trades),
            "mfe_mae": self._mfe_mae(all_trades),
            "cost_sensitivity": self._cost_sensitivity(period_results),
            "diagnosis": self._diagnosis(
                period_results,
                all_trades,
            ),
        }

    def _enrich_trade(self, trade, evaluations, candles):
        evaluation = evaluations.get(trade["entry_candle"])
        if evaluation is None:
            raise ValueError(
                "Completed trade is missing its entry evaluation: "
                f"candle {trade['entry_candle']}"
            )

        entry_candle = trade["entry_candle"]
        entry_close = candles[entry_candle]["close"]
        forward_movements = {
            horizon: self._forward_return(
                candles,
                entry_candle,
                horizon,
                entry_close,
            )
            for horizon in FORWARD_HORIZONS
        }
        ema_relationships = {
            "price_vs_ema21_percent": self._relative_percent(
                evaluation["current_price"],
                evaluation["ema21"],
            ),
            "ema21_vs_ema50_percent": self._relative_percent(
                evaluation["ema21"],
                evaluation["ema50"],
            ),
            "ema50_vs_ema200_percent": self._relative_percent(
                evaluation["ema50"],
                evaluation["ema200"],
            ),
        }
        observation = candles[entry_candle + 1:trade["exit_candle"] + 1]
        if not observation:
            observation = [candles[entry_candle]]
        market_entry_price = trade["market_entry_price"]
        maximum_price = max(candle["high"] for candle in observation)
        minimum_price = min(candle["low"] for candle in observation)

        return {
            "trade_number": trade["trade_number"],
            "entry_candle": entry_candle,
            "exit_candle": trade["exit_candle"],
            "duration_candles": (
                trade["exit_candle"] - entry_candle
            ),
            "score": trade["strategy_score"],
            "entry_rsi": trade["rsi_at_entry"],
            "conditions": {
                field: bool(evaluation[field])
                for field, _ in CONDITIONS
            },
            "ema_relationships": ema_relationships,
            "forward_price_movement": forward_movements,
            "gross_profit_loss": (
                trade["gross_profit_loss_before_costs"]
            ),
            "fees": trade["fees"],
            "slippage": trade["estimated_slippage"],
            "net_profit_loss": trade["net_profit_loss"],
            "exit_reason": trade["reason"],
            "mfe_percent": (
                (maximum_price / market_entry_price) - 1
            ) * 100,
            "mae_percent": (
                (minimum_price / market_entry_price) - 1
            ) * 100,
            "is_win": trade["net_profit_loss"] > 0,
        }

    @staticmethod
    def _relative_percent(numerator, denominator):
        if denominator == 0:
            return 0.0
        return (numerator / denominator - 1) * 100

    @staticmethod
    def _forward_return(candles, entry_candle, horizon, entry_close):
        target = entry_candle + horizon
        if target >= len(candles):
            return None
        return (candles[target]["close"] / entry_close - 1) * 100

    def _score_effectiveness(self, trades):
        buckets = {}
        for label, minimum, maximum in SCORE_BUCKETS:
            selected = [
                trade
                for trade in trades
                if minimum <= trade["score"] <= maximum
            ]
            wins = sum(trade["is_win"] for trade in selected)
            net = sum(trade["net_profit_loss"] for trade in selected)
            buckets[label] = {
                "trade_count": len(selected),
                "wins": wins,
                "losses": len(selected) - wins,
                "win_rate": _percent(wins, len(selected)),
                "gross_profit_loss": sum(
                    trade["gross_profit_loss"] for trade in selected
                ),
                "net_profit_loss": net,
                "average_net_profit_loss": (
                    net / len(selected) if selected else 0.0
                ),
            }
        return buckets

    def _condition_effectiveness(self, trades):
        result = {}
        for field, label in CONDITIONS:
            result[label] = {}
            for group_name, selected in self._trade_groups(trades).items():
                passed = sum(
                    trade["conditions"][field] for trade in selected
                )
                result[label][group_name] = {
                    "trades": len(selected),
                    "passed": passed,
                    "pass_rate": _percent(passed, len(selected)),
                }
        return result

    @staticmethod
    def _trade_groups(trades):
        return {
            "all": trades,
            "wins": [trade for trade in trades if trade["is_win"]],
            "losses": [trade for trade in trades if not trade["is_win"]],
        }

    def _entry_timing(self, trades):
        groups = self._trade_groups(trades)
        result = {}
        for group_name, selected in groups.items():
            result[group_name] = {
                "trades": len(selected),
                "entry_rsi": _value_summary(
                    trade["entry_rsi"] for trade in selected
                ),
                "duration_candles": _value_summary(
                    trade["duration_candles"] for trade in selected
                ),
                "ema_relationships": {
                    name: _value_summary(
                        trade["ema_relationships"][name]
                        for trade in selected
                    )
                    for name in (
                        "price_vs_ema21_percent",
                        "ema21_vs_ema50_percent",
                        "ema50_vs_ema200_percent",
                    )
                },
                "forward_price_movement": {
                    str(horizon): _value_summary(
                        trade["forward_price_movement"][horizon]
                        for trade in selected
                        if trade["forward_price_movement"][horizon]
                        is not None
                    )
                    for horizon in FORWARD_HORIZONS
                },
            }
        return result

    def _exit_behavior(self, trades):
        result = {}
        for reason in EXIT_REASONS:
            selected = [
                trade
                for trade in trades
                if trade["exit_reason"] == reason
            ]
            result[reason] = self._trade_cost_summary(
                selected,
                len(trades),
            )
        return result

    @staticmethod
    def _trade_cost_summary(trades, total_trades):
        return {
            "frequency": len(trades),
            "frequency_percent": _percent(len(trades), total_trades),
            "gross_profit_loss": sum(
                trade["gross_profit_loss"] for trade in trades
            ),
            "fees": sum(trade["fees"] for trade in trades),
            "slippage": sum(trade["slippage"] for trade in trades),
            "net_profit_loss": sum(
                trade["net_profit_loss"] for trade in trades
            ),
            "average_duration_candles": _average(
                [trade["duration_candles"] for trade in trades]
            ),
            "median_duration_candles": _median(
                [trade["duration_candles"] for trade in trades]
            ),
        }

    def _mfe_mae(self, trades):
        result = {}
        for group_name, selected in self._trade_groups(trades).items():
            result[group_name] = {
                "trades": len(selected),
                "mfe_percent": _value_summary(
                    trade["mfe_percent"] for trade in selected
                ),
                "mae_percent": _value_summary(
                    trade["mae_percent"] for trade in selected
                ),
            }
        return result

    @staticmethod
    def _regime_performance(period_results):
        result = {}
        for regime in ("Bull", "Sideways", "Bear"):
            selected = [
                period
                for period in period_results
                if period["regime"] == regime
            ]
            trades = sum(period["trades"] for period in selected)
            wins = sum(period["wins"] for period in selected)
            result[regime] = {
                "periods": len(selected),
                "market_return": _average(
                    period["market_return"] for period in selected
                ),
                "strategy_return": _average(
                    period["return_percent"] for period in selected
                ),
                "gross_profit_loss": sum(
                    period["gross_profit_before_costs"]
                    for period in selected
                ),
                "fees": sum(period["total_fees"] for period in selected),
                "slippage": sum(
                    period["total_slippage"] for period in selected
                ),
                "net_profit_loss": sum(
                    period["net_profit"] for period in selected
                ),
                "trades": trades,
                "wins": wins,
                "losses": trades - wins,
                "win_rate": _percent(wins, trades),
                "maximum_drawdown": max(
                    (period["max_drawdown"] for period in selected),
                    default=0.0,
                ),
            }
        return result

    @staticmethod
    def _cost_sensitivity(period_results):
        result = {}
        for label, selected in [
            ("Overall", period_results),
            *[
                (
                    regime,
                    [
                        period
                        for period in period_results
                        if period["regime"] == regime
                    ],
                )
                for regime in ("Bull", "Sideways", "Bear")
            ],
        ]:
            gross = sum(
                period["gross_profit_before_costs"]
                for period in selected
            )
            fees = sum(period["total_fees"] for period in selected)
            slippage = sum(
                period["total_slippage"] for period in selected
            )
            result[label] = {
                "gross_profit_loss": gross,
                "fees": fees,
                "slippage": slippage,
                "total_costs": fees + slippage,
                "net_profit_loss": sum(
                    period["net_profit"] for period in selected
                ),
                "costs_as_percent_of_abs_gross": (
                    (fees + slippage) / abs(gross) * 100
                    if gross
                    else 0.0
                ),
            }
        return result

    @staticmethod
    def _diagnosis(period_results, trades):
        total_trades = len(trades)
        gross = sum(trade["gross_profit_loss"] for trade in trades)
        net = sum(trade["net_profit_loss"] for trade in trades)
        costs = sum(
            trade["fees"] + trade["slippage"] for trade in trades
        )
        winners = [trade for trade in trades if trade["is_win"]]
        losers = [trade for trade in trades if not trade["is_win"]]
        zero_trade_periods = sum(
            period["trades"] == 0 for period in period_results
        )
        labels = []
        evidence = []
        insufficient_evidence = []
        score_summary = StrategyDiagnosticStudy()._score_effectiveness(
            trades
        )
        established_score_buckets = [
            (label, summary)
            for label, summary in score_summary.items()
            if summary["trade_count"] >= MIN_SCORE_BUCKET_TRADES
        ]
        negative_established_buckets = [
            label
            for label, summary in established_score_buckets
            if summary["average_net_profit_loss"] <= 0
        ]
        cost_share = (
            costs / abs(gross) * 100
            if gross
            else 0.0
        )
        exit_summary = StrategyDiagnosticStudy()._exit_behavior(trades)
        stop_loss_frequency = exit_summary["STOP LOSS"][
            "frequency_percent"
        ]
        loser_mfe = _average(
            trade["mfe_percent"] for trade in losers
        )

        if (
            established_score_buckets and
            len(negative_established_buckets) == len(
                established_score_buckets
            )
        ):
            labels.append("entry quality")
            evidence.append(
                "all entry-score buckets with at least "
                f"{MIN_SCORE_BUCKET_TRADES} completed trades had "
                "non-positive average net P/L"
            )
        elif not established_score_buckets:
            insufficient_evidence.append(
                "Entry quality: no score bucket has at least "
                f"{MIN_SCORE_BUCKET_TRADES} completed trades"
            )

        if (
            stop_loss_frequency >= MATERIAL_STOP_LOSS_PERCENT and
            loser_mfe > 0
        ):
            labels.append("exit behavior")
            evidence.append(
                f"stop losses were {stop_loss_frequency:.2f}% of exits "
                f"while losing trades first reached average MFE "
                f"{loser_mfe:+.2f}%"
            )
        else:
            insufficient_evidence.append(
                "Exit behavior: stop-loss frequency and losing-trade MFE "
                "do not meet the predeclared dominance threshold"
            )

        if cost_share >= MATERIAL_COST_SHARE_PERCENT:
            labels.append("trading costs")
            evidence.append(
                f"fees plus slippage consumed {cost_share:.2f}% of "
                f"${gross:.4f} gross P/L"
            )
        if zero_trade_periods:
            labels.append("insufficient opportunity")
            evidence.append(
                f"{zero_trade_periods} of {len(period_results)} periods "
                "produced no completed trades"
            )
        zero_trade_regimes = {
            period["regime"]
            for period in period_results
            if period["trades"] == 0
        }
        active_regimes = {
            period["regime"]
            for period in period_results
            if period["trades"] > 0
        }
        if zero_trade_regimes and active_regimes:
            labels.append("regime dependence")
            evidence.append(
                "completed trades occurred in some regimes but not in "
                + ", ".join(sorted(zero_trade_regimes))
            )
        if not labels:
            labels.append("no dominant weakness identified")
            evidence.append(
                "the measured diagnostics do not cross a defined "
                "dominance threshold"
            )

        return {
            "primary_findings": labels,
            "evidence": evidence,
            "gross_profit_loss": gross,
            "net_profit_loss": net,
            "total_costs": costs,
            "completed_trades": total_trades,
            "insufficient_evidence": insufficient_evidence,
        }


def _validate_fixed_periods(candles):
    selected = select_score_study_periods(candles)
    if len(selected) != len(SCORE_STUDY_PERIODS):
        raise RuntimeError("Diagnostic study did not select all ten periods")
    return selected


def run_strategy_diagnostic_study(notifier=None):
    if notifier is None:
        from btc_cad_preflight import send_slack_notification

        notifier = send_slack_notification

    loader = YahooBTCADMarketData(data_range="10y")
    candles = loader.load()
    if not candles:
        raise RuntimeError(loader.last_error or "Yahoo BTC/CAD data unavailable")

    selected = _validate_fixed_periods(candles)
    runner = MultiPeriodBacktester(starting_capital=STARTING_CAPITAL)
    backtest = runner.run(candles, notifier=notifier)
    expected_boundaries = [
        (period["start_date"], period["end_date"], period["regime"])
        for period in selected
    ]
    actual_boundaries = [
        (period["start_date"], period["end_date"], period["regime"])
        for period in backtest["periods"]
    ]
    if actual_boundaries != expected_boundaries:
        raise RuntimeError(
            "Gated backtest periods do not match the fixed diagnostic "
            "study boundaries"
        )
    period_candles = [period["candles"] for period in selected]
    study = StrategyDiagnosticStudy()
    results = study.analyze(backtest["periods"], period_candles)
    results["unused_candles"] = backtest["unused_candles"]
    results["decision_consistency"] = measure_decision_consistency(
        period_candles
    )
    return results


def _money(value):
    return f"${value:+.4f}"


def _print_timing(timing):
    print(
        f"  trades={timing['trades']}, "
        f"RSI avg={timing['entry_rsi']['average']:.2f}, "
        f"duration avg={timing['duration_candles']['average']:.2f} candles"
    )
    for horizon in FORWARD_HORIZONS:
        stats = timing["forward_price_movement"][str(horizon)]
        print(
            f"    +{horizon} candles: n={stats['count']}, "
            f"avg={stats['average']:+.2f}%"
        )
    for relationship, stats in timing["ema_relationships"].items():
        print(
            f"    {relationship}: n={stats['count']}, "
            f"avg={stats['average']:+.2f}%, "
            f"median={stats['median']:+.2f}%"
        )


def print_report(results):
    print("BTC/CAD STRATEGY DIAGNOSTIC STUDY — ANALYSIS ONLY")
    print("REAL-MONEY TRADING: DISABLED")
    print(
        "All results use ten independent, preflight-validated 365-candle "
        "Yahoo BTC/CAD periods and the unchanged StrategyBacktester."
    )
    print("\n=== Period coverage ===")
    for period in results["periods"]:
        print(
            f"{period['period']}: {period['start_date']} to "
            f"{period['end_date']} | {period['regime']} | "
            f"market {period['market_return']:+.2f}% | "
            f"strategy {period['strategy_return']:+.2f}%"
        )

    print("\n=== Regime-level performance ===")
    for regime, summary in results["by_regime"].items():
        print(
            f"{regime}: periods={summary['periods']}, "
            f"market avg={summary['market_return']:+.2f}%, "
            f"strategy avg={summary['strategy_return']:+.2f}%, "
            f"gross={_money(summary['gross_profit_loss'])}, "
            f"fees={_money(summary['fees'])}, "
            f"slippage={_money(summary['slippage'])}, "
            f"net={_money(summary['net_profit_loss'])}, "
            f"trades={summary['trades']}, "
            f"win rate={summary['win_rate']:.2f}%, "
            f"max DD={summary['maximum_drawdown']:.2f}%"
        )

    print("\n=== Score effectiveness ===")
    for bucket, summary in results["score_effectiveness"].items():
        print(
            f"{bucket}: trades={summary['trade_count']}, "
            f"wins/losses={summary['wins']}/{summary['losses']}, "
            f"win rate={summary['win_rate']:.2f}%, "
            f"gross={_money(summary['gross_profit_loss'])}, "
            f"net={_money(summary['net_profit_loss'])}, "
            f"avg net={_money(summary['average_net_profit_loss'])}"
        )

    print("\n=== Condition effectiveness ===")
    for condition, groups in results["condition_effectiveness"].items():
        print(
            f"{condition}: "
            + ", ".join(
                f"{group} {data['passed']}/{data['trades']} "
                f"({data['pass_rate']:.2f}%)"
                for group, data in groups.items()
            )
        )

    print("\n=== Entry timing ===")
    for group, timing in results["entry_timing"].items():
        print(f"{group}:")
        _print_timing(timing)

    print("\n=== Exit behavior ===")
    for reason, summary in results["exit_behavior"].items():
        print(
            f"{reason}: frequency={summary['frequency']} "
            f"({summary['frequency_percent']:.2f}% of completed trades), "
            f"gross={_money(summary['gross_profit_loss'])}, "
            f"fees={_money(summary['fees'])}, "
            f"slippage={_money(summary['slippage'])}, "
            f"net={_money(summary['net_profit_loss'])}, "
            f"avg duration={summary['average_duration_candles']:.2f}"
        )

    print("\n=== MFE / MAE ===")
    for group, summary in results["mfe_mae"].items():
        print(
            f"{group}: trades={summary['trades']}, "
            f"MFE avg={summary['mfe_percent']['average']:+.2f}%, "
            f"MAE avg={summary['mae_percent']['average']:+.2f}%"
        )

    print("\n=== Cost sensitivity ===")
    for group, summary in results["cost_sensitivity"].items():
        print(
            f"{group}: gross={_money(summary['gross_profit_loss'])}, "
            f"fees={_money(summary['fees'])}, "
            f"slippage={_money(summary['slippage'])}, "
            f"net={_money(summary['net_profit_loss'])}, "
            f"total costs={_money(summary['total_costs'])}"
        )

    print("\n=== Final diagnosis ===")
    diagnosis = results["diagnosis"]
    print("Dominant measured findings: " + ", ".join(
        diagnosis["primary_findings"]
    ))
    for evidence in diagnosis["evidence"]:
        print(f"- {evidence}")
    for note in diagnosis["insufficient_evidence"]:
        print(f"- Insufficient evidence: {note}")

    consistency = results["decision_consistency"]
    print("\n=== Decision consistency ===")
    print(
        f"{consistency['repeatable_cases']}/"
        f"{consistency['case_count']} cases repeat exactly "
        f"({consistency['repeatability_percent']:.2f}%)."
    )
    print(consistency["conclusion"])


def main():
    results = run_strategy_diagnostic_study()
    print_report(results)
    return results


if __name__ == "__main__":
    main()