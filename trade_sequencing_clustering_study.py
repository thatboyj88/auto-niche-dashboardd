"""Analysis-only study of signal/trade sequencing in unchanged control trades."""

from multi_period_backtest import MultiPeriodBacktester
from out_of_sample_validation import _split_periods
from score_effectiveness_study import SCORE_STUDY_PERIODS, select_score_study_periods
from strategy_calibration_study import STARTING_CAPITAL
from trade_path_exit_timing_study import TradePathExitTimingStudy
from yahoo_btc_cad_data import YahooBTCADMarketData


SEQUENCE_WINDOWS = (3, 5, 10, 20)
CLUSTER_WINDOW = 5
ISOLATED_WINDOW = 20


def _average(values):
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else 0.0


def _percent(numerator, denominator):
    return numerator / denominator * 100 if denominator else 0.0


def _summarize(items):
    gross = sum(item["gross_profit_loss"] for item in items)
    net = sum(item["net_profit_loss"] for item in items)
    fees = sum(item["fees"] for item in items)
    slippage = sum(item["slippage"] for item in items)
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
        "average_mfe_percent": _average(
            item["mfe_percent"] for item in items
        ),
        "average_mae_percent": _average(
            item["mae_percent"] for item in items
        ),
        "exit_reasons": {
            reason: sum(item["exit_reason"] == reason for item in items)
            for reason in ("STOP LOSS", "TAKE PROFIT", "END OF TEST")
        },
    }


class TradeSequencingClusteringStudy:
    """Attach prior signal/trade context to every completed control trade."""

    def analyze_period(self, period_result, candles):
        evaluations = period_result["evaluation_history"]
        signals = [
            evaluation["candle"]
            for evaluation in evaluations
            if evaluation["decision"] == "BUY"
        ]
        trades = period_result["trades_history"]
        path_study = TradePathExitTimingStudy()
        annotated = []
        for index, trade in enumerate(trades):
            entry = trade["entry_candle"]
            previous_signals = [
                candle for candle in signals if candle < entry
            ]
            previous_trade = trades[index - 1] if index else None
            previous_trade_gap = (
                entry - previous_trade["exit_candle"]
                if previous_trade else None
            )
            signal_counts = {
                window: sum(
                    entry - window <= candle < entry
                    for candle in signals
                )
                for window in SEQUENCE_WINDOWS
            }
            trade_counts = {
                window: sum(
                    entry - window <= other["entry_candle"] < entry
                    for other in trades
                )
                for window in SEQUENCE_WINDOWS
            }
            previous_trade_signals = [
                candle for candle in signals
                if trade["entry_candle"] < candle <= trade["exit_candle"]
            ]
            path = path_study.analyze_trade(
                trade,
                candles,
                period_result["period"],
                period_result["regime"],
            )
            previous_outcome = (
                "WIN" if previous_trade and previous_trade["net_profit_loss"] > 0
                else "LOSS" if previous_trade
                else None
            )
            item = {
                "trade_number": trade["trade_number"],
                "entry_candle": entry,
                "exit_candle": trade["exit_candle"],
                "candles_since_previous_buy_signal": (
                    entry - previous_signals[-1]
                    if previous_signals else None
                ),
                "candles_since_previous_completed_trade": previous_trade_gap,
                "signal_while_previous_trade_open": bool(previous_trade_signals),
                "signals_while_previous_trade_open": len(previous_trade_signals),
                "previous_trade_outcome": previous_outcome,
                "immediately_after_loss": (
                    previous_outcome == "LOSS"
                    and previous_trade_gap <= CLUSTER_WINDOW
                ),
                "immediately_after_win": (
                    previous_outcome == "WIN"
                    and previous_trade_gap <= CLUSTER_WINDOW
                ),
                "signals_previous": signal_counts,
                "completed_trades_previous": trade_counts,
                "gross_profit_loss": trade["gross_profit_loss_before_costs"],
                "net_profit_loss": trade["net_profit_loss"],
                "fees": trade["fees"],
                "slippage": trade["estimated_slippage"],
                "mfe_percent": path["mfe_percent"],
                "mae_percent": path["mae_percent"],
                "exit_reason": trade["reason"],
            }
            item["isolated"] = (
                signal_counts[ISOLATED_WINDOW] == 0
                and trade_counts[ISOLATED_WINDOW] == 0
            )
            item["clustered"] = (
                signal_counts[CLUSTER_WINDOW] > 0
                or trade_counts[CLUSTER_WINDOW] > 0
                or item["signal_while_previous_trade_open"]
            )
            annotated.append(item)
        return annotated

    def analyze_group(self, period_pairs):
        trades = [
            item
            for period_result, candles in period_pairs
            for item in self.analyze_period(period_result, candles)
        ]
        groups = {
            "all": trades,
            "isolated": [item for item in trades if item["isolated"]],
            "clustered": [item for item in trades if item["clustered"]],
            "post_loss": [
                item for item in trades if item["immediately_after_loss"]
            ],
            "post_win": [
                item for item in trades if item["immediately_after_win"]
            ],
        }
        return {
            "trade_count": len(trades),
            "trades": trades,
            "groups": {
                name: _summarize(items) for name, items in groups.items()
            },
            "sequence_averages": {
                "candles_since_previous_buy_signal": _average(
                    item["candles_since_previous_buy_signal"]
                    for item in trades
                ),
                "candles_since_previous_completed_trade": _average(
                    item["candles_since_previous_completed_trade"]
                    for item in trades
                ),
                **{
                    f"signals_previous_{window}": _average(
                        item["signals_previous"][window] for item in trades
                    )
                    for window in SEQUENCE_WINDOWS
                },
                **{
                    f"completed_trades_previous_{window}": _average(
                        item["completed_trades_previous"][window]
                        for item in trades
                    )
                    for window in SEQUENCE_WINDOWS
                },
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


def _hypothesis_results(group):
    groups = group["groups"]
    clustered_lower = (
        groups["clustered"]["net_per_trade"]
        < groups["isolated"]["net_per_trade"]
    )
    post_loss_lower = (
        groups["post_loss"]["net_per_trade"]
        < groups["post_win"]["net_per_trade"]
    )
    return {
        "clustered_trades_underperform_isolated": clustered_lower,
        "post_loss_trades_underperform_post_win": post_loss_lower,
        "clustered_trade_count": groups["clustered"]["trade_count"],
        "isolated_trade_count": groups["isolated"]["trade_count"],
        "post_loss_trade_count": groups["post_loss"]["trade_count"],
        "post_win_trade_count": groups["post_win"]["trade_count"],
    }


def run_trade_sequencing_clustering_study(notifier=None):
    if notifier is None:
        from btc_cad_preflight import send_slack_notification
        notifier = send_slack_notification
    loader = YahooBTCADMarketData(data_range="10y")
    candles = loader.load()
    if not candles:
        raise RuntimeError(loader.last_error or "Yahoo BTC/CAD data unavailable")
    selected = select_score_study_periods(candles)
    if len(selected) != len(SCORE_STUDY_PERIODS):
        raise RuntimeError("Sequencing study requires all fixed periods")
    research_periods, validation_periods = _split_periods(selected)
    study = TradeSequencingClusteringStudy()
    research = study.analyze_group(
        _run_period_group(research_periods, notifier)
    )
    validation = study.analyze_group(
        _run_period_group(validation_periods, notifier)
    )
    research_hypotheses = _hypothesis_results(research)
    validation_hypotheses = _hypothesis_results(validation)
    return {
        "real_money_trading": False,
        "research": research,
        "validation": validation,
        "hypotheses": {
            "research": research_hypotheses,
            "validation": validation_hypotheses,
            "clustered_underperformance_survives": (
                research_hypotheses["clustered_trades_underperform_isolated"]
                and validation_hypotheses[
                    "clustered_trades_underperform_isolated"
                ]
            ),
            "post_loss_underperformance_survives": (
                research_hypotheses["post_loss_trades_underperform_post_win"]
                and validation_hypotheses[
                    "post_loss_trades_underperform_post_win"
                ]
            ),
        },
        "definitions": {
            "isolated": "No signal or completed trade in the prior 20 candles.",
            "clustered": (
                "A signal or completed trade in the prior 5 candles, or "
                "a signal while the previous trade was open."
            ),
            "post_loss": "Previous trade was a loss and exited within 5 candles.",
            "post_win": "Previous trade was a win and exited within 5 candles.",
        },
    }


def _print_group(label, group):
    print(f"\n=== {label} ===")
    print(
        "Group | Trades | Gross | Costs | Net | Net/trade | Win rate | "
        "MFE | MAE | Cost share | Exits"
    )
    for name in ("all", "isolated", "clustered", "post_loss", "post_win"):
        item = group["groups"][name]
        exits = ", ".join(
            f"{reason}={count}"
            for reason, count in item["exit_reasons"].items()
            if count
        ) or "none"
        print(
            f"{name} | {item['trade_count']} | "
            f"${item['gross_profit_loss']:+.4f} | "
            f"${item['costs']:.4f} | ${item['net_profit_loss']:+.4f} | "
            f"${item['net_per_trade']:+.4f} | {item['win_rate']:.2f}% | "
            f"{item['average_mfe_percent']:+.2f}% | "
            f"{item['average_mae_percent']:+.2f}% | "
            f"{item['cost_share']:.2f}% | {exits}"
        )
    print("Sequence averages:", group["sequence_averages"])


def print_report(results):
    print("BTC/CAD TRADE SEQUENCING & SIGNAL CLUSTERING STUDY — ANALYSIS ONLY")
    print("REAL-MONEY TRADING: DISABLED")
    _print_group("Research", results["research"])
    _print_group("Untouched validation", results["validation"])
    hypotheses = results["hypotheses"]
    print("\n=== Hypothesis results ===")
    print(
        "Clustered trades underperform isolated trades in research: "
        f"{hypotheses['research']['clustered_trades_underperform_isolated']}"
    )
    print(
        "Clustered trades underperform isolated trades in validation: "
        f"{hypotheses['validation']['clustered_trades_underperform_isolated']}"
    )
    print(
        "Clustered underperformance survives validation: "
        f"{hypotheses['clustered_underperformance_survives']}"
    )
    print(
        "Post-loss trades underperform post-win trades in research: "
        f"{hypotheses['research']['post_loss_trades_underperform_post_win']}"
    )
    print(
        "Post-loss trades underperform post-win trades in validation: "
        f"{hypotheses['validation']['post_loss_trades_underperform_post_win']}"
    )
    print(
        "Post-loss underperformance survives validation: "
        f"{hypotheses['post_loss_underperformance_survives']}"
    )
    print("\n=== Interpretation boundary ===")
    print(
        "This is descriptive analysis of unchanged completed control trades. "
        "It does not change signals, RSI, cooldowns, exits, or production behavior."
    )


def main():
    results = run_trade_sequencing_clustering_study()
    print_report(results)
    return results


if __name__ == "__main__":
    main()