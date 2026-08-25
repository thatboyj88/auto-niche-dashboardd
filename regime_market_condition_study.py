"""Analysis-only regime and entry-condition study for control trades."""

from multi_period_backtest import MultiPeriodBacktester
from out_of_sample_validation import _split_periods
from score_effectiveness_study import SCORE_STUDY_PERIODS, select_score_study_periods
from strategy_calibration_study import STARTING_CAPITAL
from trade_path_exit_timing_study import TradePathExitTimingStudy
from yahoo_btc_cad_data import YahooBTCADMarketData


MINIMUM_GROUP_TRADES = 3
FINE_REGIMES = (
    "Strong Bull",
    "Weak Bull",
    "Neutral/Sideways",
    "Weak Bear",
    "Strong Bear",
)
BROAD_REGIMES = ("Bull", "Sideways", "Bear")


def classify_entry_environment(evaluation):
    """Classify using only indicator values calculated at the entry candle."""
    price = evaluation["current_price"]
    ema21 = evaluation["ema21"]
    ema50 = evaluation["ema50"]
    ema200 = evaluation["ema200"]
    bullish_alignment = price > ema21 > ema50 > ema200
    bearish_alignment = price < ema21 < ema50 < ema200
    if bullish_alignment and evaluation["long_term_trend"] and evaluation[
        "short_term_momentum"
    ]:
        fine = "Strong Bull"
    elif bullish_alignment or (price > ema21 and ema21 > ema50):
        fine = "Weak Bull"
    elif bearish_alignment and not evaluation["short_term_momentum"]:
        fine = "Strong Bear"
    elif bearish_alignment or (price < ema21 and ema21 < ema50):
        fine = "Weak Bear"
    else:
        fine = "Neutral/Sideways"
    broad = (
        "Bull" if "Bull" in fine
        else "Bear" if "Bear" in fine
        else "Sideways"
    )
    return broad, fine


def _average(values):
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else 0.0


def _percent(numerator, denominator):
    return numerator / denominator * 100 if denominator else 0.0


def _summary(items):
    gross = sum(item["gross_profit_loss"] for item in items)
    fees = sum(item["fees"] for item in items)
    slippage = sum(item["slippage"] for item in items)
    net = sum(item["net_profit_loss"] for item in items)
    return {
        "trade_count": len(items),
        "gross_profit_loss": gross,
        "net_profit_loss": net,
        "fees": fees,
        "slippage": slippage,
        "costs": fees + slippage,
        "net_per_trade": net / len(items) if items else 0.0,
        "win_rate": _percent(
            sum(item["net_profit_loss"] > 0 for item in items),
            len(items),
        ),
        "cost_share": (
            (fees + slippage) / abs(gross) * 100 if gross else 0.0
        ),
        "average_score": _average(item["entry_score"] for item in items),
        "average_rsi": _average(item["entry_rsi"] for item in items),
        "average_mfe_percent": _average(
            item["mfe_percent"] for item in items
        ),
        "average_mae_percent": _average(
            item["mae_percent"] for item in items
        ),
        "average_duration": _average(item["trade_duration"] for item in items),
        "exit_reasons": {
            reason: sum(item["exit_reason"] == reason for item in items)
            for reason in ("STOP LOSS", "TAKE PROFIT", "END OF TEST")
        },
        "insufficient_evidence": len(items) < MINIMUM_GROUP_TRADES,
    }


class RegimeMarketConditionStudy:
    def analyze_period(self, period_result, candles):
        evaluations = {
            evaluation["candle"]: evaluation
            for evaluation in period_result["evaluation_history"]
        }
        path_study = TradePathExitTimingStudy()
        items = []
        for trade in period_result["trades_history"]:
            evaluation = evaluations.get(trade["entry_candle"])
            if evaluation is None:
                raise ValueError("trade is missing its entry evaluation")
            broad, fine = classify_entry_environment(evaluation)
            path = path_study.analyze_trade(
                trade,
                candles,
                period_result["period"],
                period_result["regime"],
            )
            items.append({
                "trade_number": trade["trade_number"],
                "period": period_result["period"],
                "market_regime": period_result["regime"],
                "entry_broad_regime": broad,
                "entry_fine_regime": fine,
                "entry_candle": trade["entry_candle"],
                "exit_candle": trade["exit_candle"],
                "trade_duration": (
                    trade["exit_candle"] - trade["entry_candle"]
                ),
                "entry_score": evaluation["strategy_score"],
                "entry_rsi": evaluation["rsi"],
                "price_vs_ema21": (
                    evaluation["current_price"] / evaluation["ema21"] - 1
                ) * 100,
                "ema21_vs_ema50": (
                    evaluation["ema21"] / evaluation["ema50"] - 1
                ) * 100,
                "ema50_vs_ema200": (
                    evaluation["ema50"] / evaluation["ema200"] - 1
                ) * 100,
                "short_term_momentum": evaluation["short_term_momentum"],
                "long_term_trend": evaluation["long_term_trend"],
                "volume_condition": evaluation["volume"],
                "gross_profit_loss": trade["gross_profit_loss_before_costs"],
                "net_profit_loss": trade["net_profit_loss"],
                "fees": trade["fees"],
                "slippage": trade["estimated_slippage"],
                "win": trade["net_profit_loss"] > 0,
                "mfe_percent": path["mfe_percent"],
                "mae_percent": path["mae_percent"],
                "exit_reason": trade["reason"],
            })
        return items

    def analyze_group(self, period_pairs):
        trades = [
            item
            for period_result, candles in period_pairs
            for item in self.analyze_period(period_result, candles)
        ]
        return {
            "trade_count": len(trades),
            "trades": trades,
            "broad_regimes": {
                regime: _summary([
                    item for item in trades
                    if item["entry_broad_regime"] == regime
                ])
                for regime in BROAD_REGIMES
            },
            "fine_regimes": {
                regime: _summary([
                    item for item in trades
                    if item["entry_fine_regime"] == regime
                ])
                for regime in FINE_REGIMES
            },
        }


def _run_period_group(selected, notifier):
    runner = MultiPeriodBacktester(starting_capital=STARTING_CAPITAL)
    pairs = []
    for index, period in enumerate(selected):
        result = runner._run_period(
            index,
            period["candles"],
            period_label=period["period"],
            source_label="Yahoo Finance BTC/CAD fixed ten-year study",
            source_kind="fixed-study",
            notifier=notifier,
        )
        pairs.append((result, period["candles"]))
    return pairs


def _hypothesis_results(research, validation):
    robust_positive = []
    for regime in BROAD_REGIMES:
        research_group = research["broad_regimes"][regime]
        validation_group = validation["broad_regimes"][regime]
        if (
            not research_group["insufficient_evidence"]
            and not validation_group["insufficient_evidence"]
            and research_group["net_per_trade"] > 0
            and validation_group["net_per_trade"] > 0
        ):
            robust_positive.append(regime)
    return {
        "broad_regimes_positive_in_both_splits": robust_positive,
        "edge_survives_across_broad_regimes": len(robust_positive) >= 2,
        "insufficient_evidence_regimes": [
            regime for regime in BROAD_REGIMES
            if (
                research["broad_regimes"][regime]["insufficient_evidence"]
                or validation["broad_regimes"][regime]["insufficient_evidence"]
            )
        ],
    }


def run_regime_market_condition_study(notifier=None):
    if notifier is None:
        from btc_cad_preflight import send_slack_notification
        notifier = send_slack_notification
    loader = YahooBTCADMarketData(data_range="10y")
    candles = loader.load()
    if not candles:
        raise RuntimeError(loader.last_error or "Yahoo BTC/CAD data unavailable")
    selected = select_score_study_periods(candles)
    if len(selected) != len(SCORE_STUDY_PERIODS):
        raise RuntimeError("Regime study requires all fixed periods")
    research_periods, validation_periods = _split_periods(selected)
    study = RegimeMarketConditionStudy()
    research = study.analyze_group(
        _run_period_group(research_periods, notifier)
    )
    validation = study.analyze_group(
        _run_period_group(validation_periods, notifier)
    )
    return {
        "real_money_trading": False,
        "classification_rules": {
            "strong_bull": (
                "price > EMA21 > EMA50 > EMA200, long-term trend and "
                "short-term momentum are true"
            ),
            "weak_bull": (
                "bullish EMA alignment, or price > EMA21 > EMA50, without "
                "strong-bull conditions"
            ),
            "strong_bear": (
                "price < EMA21 < EMA50 < EMA200 and short-term momentum is false"
            ),
            "weak_bear": (
                "bearish EMA alignment without strong-bear conditions"
            ),
            "neutral": "All remaining entry-time configurations.",
        },
        "research": research,
        "validation": validation,
        "hypotheses": _hypothesis_results(research, validation),
    }


def _print_regime_table(title, regimes):
    print(f"\n{title}")
    print(
        "Regime | Trades | Gross | Costs | Net | Net/trade | Win rate | "
        "Score | RSI | MFE | MAE | Duration | Evidence"
    )
    for regime, item in regimes.items():
        evidence = (
            "INSUFFICIENT" if item["insufficient_evidence"] else "usable"
        )
        print(
            f"{regime} | {item['trade_count']} | "
            f"${item['gross_profit_loss']:+.4f} | ${item['costs']:.4f} | "
            f"${item['net_profit_loss']:+.4f} | "
            f"${item['net_per_trade']:+.4f} | {item['win_rate']:.2f}% | "
            f"{item['average_score']:.1f} | {item['average_rsi']:.1f} | "
            f"{item['average_mfe_percent']:+.2f}% | "
            f"{item['average_mae_percent']:+.2f}% | "
            f"{item['average_duration']:.1f} | {evidence}"
        )


def _print_group(label, group):
    print(f"\n=== {label} ===")
    _print_regime_table("Broad entry environments", group["broad_regimes"])
    _print_regime_table("Fine entry environments", group["fine_regimes"])


def print_report(results):
    print("BTC/CAD REGIME & MARKET-CONDITION ROBUSTNESS STUDY — ANALYSIS ONLY")
    print("REAL-MONEY TRADING: DISABLED")
    print(
        "Regimes use only entry-candle indicators; no future prices or "
        "future returns are used for classification."
    )
    _print_group("Research", results["research"])
    _print_group("Untouched validation", results["validation"])
    hypotheses = results["hypotheses"]
    print("\n=== Hypothesis results ===")
    print(
        "Broad regimes with positive net/trade in both splits: "
        f"{hypotheses['broad_regimes_positive_in_both_splits'] or 'none'}"
    )
    print(
        "Strategy edge survives across at least two broad regimes: "
        f"{hypotheses['edge_survives_across_broad_regimes']}"
    )
    print(
        "Broad regimes with insufficient evidence: "
        f"{hypotheses['insufficient_evidence_regimes'] or 'none'}"
    )
    print("\n=== Interpretation boundary ===")
    print(
        "This is descriptive analysis of unchanged control trades. It does "
        "not change strategy.py, entries, exits, sizing, or risk behavior."
    )


def main():
    results = run_regime_market_condition_study()
    print_report(results)
    return results


if __name__ == "__main__":
    main()