"""Analysis-only robustness study for predeclared RSI entry thresholds."""

from multi_period_backtest import MultiPeriodBacktester
from out_of_sample_validation import _split_periods
from score_effectiveness_study import SCORE_STUDY_PERIODS, select_score_study_periods
from strategy_calibration_study import STARTING_CAPITAL
from yahoo_btc_cad_data import YahooBTCADMarketData


RSI_THRESHOLDS = (55, 58, 60, 62, 65, 68)


def _percent(numerator, denominator):
    return numerator / denominator * 100 if denominator else 0.0


def _performance(trades, signal_count, period_results):
    gross = sum(trade["gross_profit_loss_before_costs"] for trade in trades)
    fees = sum(trade["fees"] for trade in trades)
    slippage = sum(trade["estimated_slippage"] for trade in trades)
    net = gross - fees - slippage
    equity = STARTING_CAPITAL
    peak = equity
    max_drawdown = 0.0
    selected_by_period = {
        id(period): [] for period in period_results
    }
    period_lookup = {
        (period["period"], trade["trade_number"]): trade
        for period in period_results
        for trade in period["trades_history"]
    }
    for trade in trades:
        selected_by_period.setdefault(
            trade["_period_result_id"], []
        ).append(trade)
    for period in period_results:
        equity = STARTING_CAPITAL
        peak = equity
        for trade in selected_by_period.get(id(period), []):
            equity += trade["net_profit_loss"]
            peak = max(peak, equity)
            max_drawdown = max(max_drawdown, peak - equity)
    return {
        "signals": signal_count,
        "trades": len(trades),
        "gross": gross,
        "fees": fees,
        "slippage": slippage,
        "costs": fees + slippage,
        "net": net,
        "net_per_trade": net / len(trades) if trades else 0.0,
        "maximum_drawdown": max_drawdown,
        "win_rate": _percent(
            sum(trade["net_profit_loss"] > 0 for trade in trades),
            len(trades),
        ),
        "cost_share": (fees + slippage) / abs(gross) * 100 if gross else 0.0,
    }


class RSICandidateRobustnessStudy:
    """Compare fixed RSI thresholds while retaining the original control."""

    def _annotate(self, period_results):
        trades = []
        signal_rows = []
        for period in period_results:
            evaluations = {
                item["candle"]: item for item in period["evaluation_history"]
            }
            signal_rows.extend(
                (period, item)
                for item in period["evaluation_history"]
                if item["decision"] == "BUY"
            )
            for trade in period["trades_history"]:
                evaluation = evaluations.get(trade["entry_candle"])
                if evaluation is None:
                    raise ValueError("trade is missing its entry evaluation")
                annotated = dict(trade)
                annotated["_entry_rsi"] = evaluation["rsi"]
                annotated["_period_result_id"] = id(period)
                trades.append(annotated)
        return trades, signal_rows

    def analyze_group(self, period_results):
        trades, signal_rows = self._annotate(period_results)
        control = _performance(
            trades,
            len(signal_rows),
            period_results,
        )
        thresholds = {}
        for threshold in RSI_THRESHOLDS:
            selected_trades = [
                trade for trade in trades if trade["_entry_rsi"] >= threshold
            ]
            selected_signals = [
                item for _, item in signal_rows if item["rsi"] >= threshold
            ]
            performance = _performance(
                selected_trades,
                len(selected_signals),
                period_results,
            )
            thresholds[threshold] = {
                "threshold": threshold,
                "performance": performance,
                "improvement_vs_control": (
                    performance["net"] - control["net"]
                ),
                "trade_delta_vs_control": (
                    performance["trades"] - control["trades"]
                ),
            }
        return {
            "control": control,
            "thresholds": thresholds,
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


def run_rsi_candidate_robustness_study(notifier=None):
    if notifier is None:
        from btc_cad_preflight import send_slack_notification
        notifier = send_slack_notification
    loader = YahooBTCADMarketData(data_range="10y")
    candles = loader.load()
    if not candles:
        raise RuntimeError(loader.last_error or "Yahoo BTC/CAD data unavailable")
    selected = select_score_study_periods(candles)
    if len(selected) != len(SCORE_STUDY_PERIODS):
        raise RuntimeError("RSI robustness study requires all fixed periods")
    research_periods, validation_periods = _split_periods(selected)
    return {
        "real_money_trading": False,
        "research": RSICandidateRobustnessStudy().analyze_group(
            _run_period_group(research_periods, notifier)
        ),
        "validation": RSICandidateRobustnessStudy().analyze_group(
            _run_period_group(validation_periods, notifier)
        ),
        "thresholds": RSI_THRESHOLDS,
        "note": (
            "Thresholds are diagnostic filters over unchanged control trades; "
            "this is not the executable candidate backtest."
        ),
    }


def _print_group(label, group):
    print(f"\n=== {label} ===")
    control = group["control"]
    print(
        "Control | "
        f"signals={control['signals']} | trades={control['trades']} | "
        f"gross=${control['gross']:+.4f} | fees=${control['fees']:.4f} | "
        f"slippage=${control['slippage']:.4f} | net=${control['net']:+.4f}"
    )
    print(
        "RSI floor | Signals | Trades | Gross | Fees | Slippage | Net | "
        "Net/trade | Max DD | Win rate | Cost share | Δ vs control"
    )
    for threshold, result in group["thresholds"].items():
        item = result["performance"]
        print(
            f">={threshold:<2} | {item['signals']:<7} | {item['trades']:<6} | "
            f"${item['gross']:+.4f} | ${item['fees']:.4f} | "
            f"${item['slippage']:.4f} | ${item['net']:+.4f} | "
            f"${item['net_per_trade']:+.4f} | ${item['maximum_drawdown']:.4f} | "
            f"{item['win_rate']:.2f}% | {item['cost_share']:.2f}% | "
            f"${result['improvement_vs_control']:+.4f}"
        )


def print_report(results):
    print("BTC/CAD RSI CANDIDATE ROBUSTNESS STUDY — ANALYSIS ONLY")
    print("REAL-MONEY TRADING: DISABLED")
    _print_group("Research", results["research"])
    _print_group("Untouched validation", results["validation"])
    print("\n=== Threshold stability ===")
    validation = results["validation"]
    passing = [
        threshold for threshold, result in validation["thresholds"].items()
        if result["improvement_vs_control"] > 0
    ]
    print(
        f"Thresholds beating validation control: "
        f"{passing if passing else 'none'}"
    )
    print(
        "No threshold is selected from the highest result. Stability requires "
        "a consistent neighborhood and a later executable paper backtest."
    )
    print("\n=== Interpretation boundary ===")
    print(
        "This study does not change entries or exits. It tests threshold "
        "robustness using unchanged control trades; an executable RSI "
        "candidate backtest is a separate next experiment."
    )


def main():
    results = run_rsi_candidate_robustness_study()
    print_report(results)
    return results


if __name__ == "__main__":
    main()