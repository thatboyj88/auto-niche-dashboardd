"""Analysis-only screen for fewer-but-better trade filters.

The screen keeps the original completed control trades and applies fixed,
predeclared selection rules. Research-derived movement profiles are used for
the projected-movement fields; validation is never used to construct them.
This is a diagnostic candidate screen, not an executable alternate strategy.
"""

from statistics import median

from cost_viability_study import CostViabilityStudy
from multi_period_backtest import MultiPeriodBacktester
from out_of_sample_validation import _split_periods
from score_effectiveness_study import SCORE_STUDY_PERIODS, select_score_study_periods
from strategy_calibration_study import STARTING_CAPITAL
from yahoo_btc_cad_data import YahooBTCADMarketData


BREAK_EVEN_PERCENT = 1.005
PROFILE_HORIZON = 5
SCORE_BANDS = ((0, 79), (80, 84), (85, 89), (90, 94), (95, 100))
RSI_BANDS = ((0, 54.99), (55, 59.99), (60, 64.99), (65, 100))


def _average(values):
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _median(values):
    values = list(values)
    return median(values) if values else 0.0


def _percent(numerator, denominator):
    return numerator / denominator * 100 if denominator else 0.0


class TradeFilterCandidateStudy:
    """Compare fixed filters against the exact original completed trades."""

    CANDIDATES = (
        ("minimum_expected_movement_1_5", "Minimum expected movement >=1.5%"),
        ("minimum_historical_mfe_2", "Minimum historical MFE >=2%"),
        ("minimum_score_85", "Minimum entry score >=85"),
        ("minimum_rsi_60", "Minimum entry RSI >=60"),
        (
            "minimum_break_even_distance_0_5",
            "Expected movement at least 0.5% beyond break-even",
        ),
        (
            "minimum_reward_cost_ratio_1_5",
            "Projected reward/cost ratio >=1.5",
        ),
        ("cooldown_3", "At least 3 candles between selected trades"),
    )

    def _bucket(self, score, rsi):
        score_bucket = next(
            (band for band in SCORE_BANDS if band[0] <= score <= band[1]),
            SCORE_BANDS[-1],
        )
        rsi_bucket = next(
            (band for band in RSI_BANDS if band[0] <= rsi <= band[1]),
            RSI_BANDS[-1],
        )
        return score_bucket, rsi_bucket

    def build_research_profile(self, period_results, period_candles):
        buckets = {}
        for result, candles in zip(period_results, period_candles):
            evaluations = {
                item["candle"]: item for item in result["evaluation_history"]
            }
            for trade in result["trades_history"]:
                entry = trade["entry_candle"]
                evaluation = evaluations.get(entry)
                if evaluation is None:
                    continue
                forward = candles[entry + 1:entry + 1 + PROFILE_HORIZON]
                if not forward:
                    continue
                entry_price = trade["market_entry_price"]
                key = self._bucket(
                    evaluation["strategy_score"],
                    evaluation["rsi"],
                )
                bucket = buckets.setdefault(key, {"mfe": [], "movement": []})
                bucket["mfe"].append(
                    (max(item["high"] for item in forward) / entry_price - 1)
                    * 100
                )
                bucket["movement"].append(
                    (forward[-1]["close"] / entry_price - 1) * 100
                )
        return {
            key: {
                "historical_mfe_percent": _average(value["mfe"]),
                "expected_movement_percent": _average(value["movement"]),
                "sample_count": len(value["mfe"]),
            }
            for key, value in buckets.items()
        }

    def describe_trades(self, period_results, period_candles, profile):
        described = []
        for result, candles in zip(period_results, period_candles):
            evaluations = {
                item["candle"]: item for item in result["evaluation_history"]
            }
            for trade in result["trades_history"]:
                evaluation = evaluations.get(trade["entry_candle"])
                if evaluation is None:
                    raise ValueError("trade is missing its entry evaluation")
                key = self._bucket(
                    evaluation["strategy_score"],
                    evaluation["rsi"],
                )
                profile_row = profile.get(
                    key,
                    {
                        "historical_mfe_percent": 0.0,
                        "expected_movement_percent": 0.0,
                        "sample_count": 0,
                    },
                )
                costs = trade["fees"] + trade["estimated_slippage"]
                notional = trade["position_size"] * trade["market_entry_price"]
                described.append({
                    "trade": trade,
                    "period": result["period"],
                    "entry_candle": trade["entry_candle"],
                    "entry_score": evaluation["strategy_score"],
                    "entry_rsi": evaluation["rsi"],
                    "expected_movement_percent": profile_row[
                        "expected_movement_percent"
                    ],
                    "historical_mfe_percent": profile_row[
                        "historical_mfe_percent"
                    ],
                    "break_even_distance_percent": (
                        profile_row["expected_movement_percent"]
                        - BREAK_EVEN_PERCENT
                    ),
                    "projected_reward_cost_ratio": (
                        profile_row["expected_movement_percent"]
                        / (costs / notional * 100)
                        if costs and notional else 0.0
                    ),
                    "current_net": (
                        trade["gross_profit_loss_before_costs"] - costs
                    ),
                })
        return described

    @staticmethod
    def keep(candidate_key, item, previous_selected):
        if candidate_key == "minimum_expected_movement_1_5":
            return item["expected_movement_percent"] >= 1.5
        if candidate_key == "minimum_historical_mfe_2":
            return item["historical_mfe_percent"] >= 2.0
        if candidate_key == "minimum_score_85":
            return item["entry_score"] >= 85
        if candidate_key == "minimum_rsi_60":
            return item["entry_rsi"] >= 60
        if candidate_key == "minimum_break_even_distance_0_5":
            return item["break_even_distance_percent"] >= 0.5
        if candidate_key == "minimum_reward_cost_ratio_1_5":
            return item["projected_reward_cost_ratio"] >= 1.5
        if candidate_key == "cooldown_3":
            return (
                not previous_selected
                or item["entry_candle"] - previous_selected[-1]["entry_candle"] >= 3
            )
        raise ValueError(f"Unknown filter candidate: {candidate_key}")

    def select(self, candidate_key, trades):
        selected = []
        for item in sorted(trades, key=lambda row: (row["period"], row["entry_candle"])):
            if self.keep(candidate_key, item, selected):
                selected.append(item)
        return selected

    def performance(self, trades):
        gross = sum(item["trade"]["gross_profit_loss_before_costs"] for item in trades)
        fees = sum(item["trade"]["fees"] for item in trades)
        slippage = sum(item["trade"]["estimated_slippage"] for item in trades)
        net = gross - fees - slippage
        return {
            "trades": len(trades),
            "gross": gross,
            "costs": fees + slippage,
            "net": net,
            "profitable_trades": sum(
                item["current_net"] > 0 for item in trades
            ),
            "average_net_per_trade": net / len(trades) if trades else 0.0,
        }

    def screen(self, control_trades):
        control = self.performance(control_trades)
        results = {}
        for key, label in self.CANDIDATES:
            selected = self.select(key, control_trades)
            performance = self.performance(selected)
            results[key] = {
                "label": label,
                "performance": performance,
                "trade_reduction": control["trades"] - performance["trades"],
                "research_net_delta": None,
                "validation_net_delta": None,
                "beats_validation_control": None,
                "status": "NOT_YET_GATED",
            }
        return control, results

    def screen_research_validation(self, research_trades, validation_trades):
        research_control, research = self.screen(research_trades)
        validation_control, validation = self.screen(validation_trades)
        for key in research:
            research[key]["research_net_delta"] = (
                research[key]["performance"]["net"] - research_control["net"]
            )
            validation[key]["validation_net_delta"] = (
                validation[key]["performance"]["net"] - validation_control["net"]
            )
            validation[key]["beats_validation_control"] = (
                validation[key]["validation_net_delta"] > 0
            )
            research[key]["status"] = (
                "PASSES_VALIDATION_GATE"
                if (
                    research[key]["research_net_delta"] > 0
                    and validation[key]["validation_net_delta"] > 0
                )
                else "REJECTED"
            )
            validation[key]["status"] = research[key]["status"]
        return {
            "research_control": research_control,
            "validation_control": validation_control,
            "research": research,
            "validation": validation,
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


def run_trade_filter_candidate_study(notifier=None):
    if notifier is None:
        from btc_cad_preflight import send_slack_notification
        notifier = send_slack_notification
    loader = YahooBTCADMarketData(data_range="10y")
    candles = loader.load()
    if not candles:
        raise RuntimeError(loader.last_error or "Yahoo BTC/CAD data unavailable")
    selected = select_score_study_periods(candles)
    if len(selected) != len(SCORE_STUDY_PERIODS):
        raise RuntimeError("Trade filter study requires all fixed periods")
    research_periods, validation_periods = _split_periods(selected)
    research_results = _run_period_group(research_periods, notifier)
    validation_results = _run_period_group(validation_periods, notifier)
    study = TradeFilterCandidateStudy()
    research_candles = [period["candles"] for period in research_periods]
    validation_candles = [period["candles"] for period in validation_periods]
    profile = study.build_research_profile(research_results, research_candles)
    research_trades = study.describe_trades(
        research_results, research_candles, profile
    )
    validation_trades = study.describe_trades(
        validation_results, validation_candles, profile
    )
    screened = study.screen_research_validation(
        research_trades,
        validation_trades,
    )
    return {
        "real_money_trading": False,
        "filter_definitions": dict(TradeFilterCandidateStudy.CANDIDATES),
        "research_profile_sample_count": sum(
            row["sample_count"] for row in profile.values()
        ),
        **screened,
        "note": (
            "Diagnostic selection of unchanged completed control trades; "
            "not an executable alternate backtest."
        ),
    }


def _print_table(title, control, candidates, delta_key):
    print(f"\n=== {title} ===")
    print("Test | Trades | Gross | Costs | Net | Validation gate")
    print(
        f"Original control | {control['trades']} | "
        f"${control['gross']:+.4f} | ${control['costs']:.4f} | "
        f"${control['net']:+.4f} | baseline"
    )
    for key, result in candidates.items():
        performance = result["performance"]
        delta = result[delta_key]
        print(
            f"{result['label']} | {performance['trades']} | "
            f"${performance['gross']:+.4f} | ${performance['costs']:.4f} | "
            f"${performance['net']:+.4f} | "
            f"{result['status']} (delta ${delta:+.4f})"
        )


def print_report(results):
    print("BTC/CAD TRADE-FILTER CANDIDATE SCREEN — ANALYSIS ONLY")
    print("REAL-MONEY TRADING: DISABLED")
    _print_table(
        "Research",
        results["research_control"],
        results["research"],
        "research_net_delta",
    )
    _print_table(
        "Untouched validation",
        results["validation_control"],
        results["validation"],
        "validation_net_delta",
    )
    print("\n=== Interpretation boundary ===")
    print(
        "All filters are diagnostic candidates applied to unchanged completed "
        "control trades. A candidate is rejected unless it beats control net "
        "P/L in both research and untouched validation. No filter is promoted."
    )


def main():
    results = run_trade_filter_candidate_study()
    print_report(results)
    return results


if __name__ == "__main__":
    main()