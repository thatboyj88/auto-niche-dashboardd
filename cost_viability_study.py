"""Analysis-only study of per-trade economic edge and cost sensitivity."""

from statistics import median

from multi_period_backtest import MultiPeriodBacktester
from out_of_sample_validation import _split_periods
from score_effectiveness_study import SCORE_STUDY_PERIODS, select_score_study_periods
from strategy_calibration_study import STARTING_CAPITAL
from yahoo_btc_cad_data import YahooBTCADMarketData


COST_MODELS = (
    ("current", 1.00, "current model"),
    ("twenty_five_percent_lower", 0.75, "25% lower"),
    ("fifty_percent_lower", 0.50, "50% lower"),
    ("seventy_five_percent_lower", 0.25, "75% lower"),
    ("zero", 0.00, "zero costs"),
)
TOP_SHARES = (0.05, 0.10, 0.20)


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


class CostViabilityStudy:
    """Reprice unchanged completed trades under hypothetical cost factors."""

    def analyze_trade(self, trade, period, regime):
        gross = trade["gross_profit_loss_before_costs"]
        costs = trade["fees"] + trade["estimated_slippage"]
        position_notional = (
            trade["position_size"] * trade["market_entry_price"]
        )
        if position_notional <= 0:
            raise ValueError("trade must have positive entry notional")
        return {
            "trade_number": trade["trade_number"],
            "period": period,
            "regime": regime,
            "reason": trade["reason"],
            "entry_candle": trade["entry_candle"],
            "entry_price": trade["market_entry_price"],
            "position_size": trade["position_size"],
            "entry_notional": position_notional,
            "gross_profit_loss": gross,
            "gross_return_percent": gross / position_notional * 100,
            "fees": trade["fees"],
            "slippage": trade["estimated_slippage"],
            "current_costs": costs,
            "required_move_percent": costs / position_notional * 100,
            "current_net_profit_loss": gross - costs,
        }

    def reprice_trade(self, trade, cost_factor):
        net = trade["gross_profit_loss"] - trade["current_costs"] * cost_factor
        return {
            "net_profit_loss": net,
            "profitable": net > 0,
            "costs": trade["current_costs"] * cost_factor,
        }

    def analyze_trades(self, trades):
        cost_models = {}
        for key, factor, label in COST_MODELS:
            repriced = [self.reprice_trade(trade, factor) for trade in trades]
            net_values = [item["net_profit_loss"] for item in repriced]
            cost_models[key] = {
                "label": label,
                "cost_factor": factor,
                "trade_count": len(trades),
                "profitable_trade_count": sum(
                    item["profitable"] for item in repriced
                ),
                "profitable_trade_percent": _percent(
                    sum(item["profitable"] for item in repriced),
                    len(trades),
                ),
                "total_costs": sum(item["costs"] for item in repriced),
                "total_net_profit_loss": sum(net_values),
                "average_net_per_trade": _average(net_values),
                "median_net_per_trade": _median(net_values),
            }
        gross_values = [trade["gross_profit_loss"] for trade in trades]
        positive_gross = [value for value in gross_values if value > 0]
        positive_trades = [trade for trade in trades if trade["gross_profit_loss"] > 0]
        positive_gross_total = sum(positive_gross)
        sorted_positive = sorted(
            positive_trades,
            key=lambda trade: trade["gross_profit_loss"],
            reverse=True,
        )
        concentration = {}
        for share in TOP_SHARES:
            count = max(1, int(len(sorted_positive) * share)) if sorted_positive else 0
            contribution = sum(
                trade["gross_profit_loss"] for trade in sorted_positive[:count]
            )
            concentration[str(int(share * 100))] = {
                "top_trade_count": count,
                "positive_gross_profit": contribution,
                "share_of_positive_gross_percent": _percent(
                    contribution,
                    positive_gross_total,
                ),
            }
        return {
            "trade_count": len(trades),
            "gross_profit_loss": sum(gross_values),
            "positive_gross_profit_loss": positive_gross_total,
            "negative_gross_profit_loss": sum(
                value for value in gross_values if value <= 0
            ),
            "average_gross_per_trade": _average(gross_values),
            "median_gross_per_trade": _median(gross_values),
            "positive_gross_trade_count": len(positive_gross),
            "positive_gross_trade_percent": _percent(
                len(positive_gross), len(trades)
            ),
            "required_move_percent": _summary(
                trade["required_move_percent"] for trade in trades
            ),
            "current_costs": sum(trade["current_costs"] for trade in trades),
            "fees": sum(trade["fees"] for trade in trades),
            "slippage": sum(trade["slippage"] for trade in trades),
            "cost_models": cost_models,
            "gross_profit_concentration": concentration,
            "by_exit_reason": {
                reason: self._reason_summary(
                    [trade for trade in trades if trade["reason"] == reason]
                )
                for reason in ("TAKE PROFIT", "STOP LOSS", "END OF TEST")
            },
        }

    def _reason_summary(self, trades):
        return {
            "trade_count": len(trades),
            "gross_profit_loss": sum(
                trade["gross_profit_loss"] for trade in trades
            ),
            "current_net_profit_loss": sum(
                trade["current_net_profit_loss"] for trade in trades
            ),
            "average_gross_per_trade": _average(
                trade["gross_profit_loss"] for trade in trades
            ),
            "positive_gross_trade_count": sum(
                trade["gross_profit_loss"] > 0 for trade in trades
            ),
        }

    def analyze_group(self, period_results):
        trades = [
            self.analyze_trade(
                trade,
                period["period"],
                period["regime"],
            )
            for period in period_results
            for trade in period["trades_history"]
        ]
        return self.analyze_trades(trades)


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


def run_cost_viability_study(notifier=None):
    if notifier is None:
        from btc_cad_preflight import send_slack_notification
        notifier = send_slack_notification
    loader = YahooBTCADMarketData(data_range="10y")
    candles = loader.load()
    if not candles:
        raise RuntimeError(loader.last_error or "Yahoo BTC/CAD data unavailable")
    selected = select_score_study_periods(candles)
    if len(selected) != len(SCORE_STUDY_PERIODS):
        raise RuntimeError("Cost viability study requires all fixed periods")
    research, validation = _split_periods(selected)
    research_results = _run_period_group(research, notifier)
    validation_results = _run_period_group(validation, notifier)
    study = CostViabilityStudy()
    return {
        "source": "Yahoo Finance BTC/CAD aggregated daily data",
        "real_money_trading": False,
        "split": {
            "research_start": research[0]["start_date"],
            "research_end": research[-1]["end_date"],
            "research_periods": len(research),
            "validation_start": validation[0]["start_date"],
            "validation_end": validation[-1]["end_date"],
            "validation_periods": len(validation),
        },
        "research": study.analyze_group(research_results),
        "validation": study.analyze_group(validation_results),
    }


def _print_group(label, group):
    print(f"\n=== {label} ===")
    print(
        f"trades={group['trade_count']}, "
        f"gross=${group['gross_profit_loss']:+.4f}, "
        f"positive gross trades={group['positive_gross_trade_count']} "
        f"({group['positive_gross_trade_percent']:.2f}%), "
        f"fees=${group['fees']:.4f}, slippage=${group['slippage']:.4f}"
    )
    required = group["required_move_percent"]
    print(
        f"required move for profitability: average={required['average']:.3f}%, "
        f"median={required['median']:.3f}%"
    )
    print("Cost sensitivity:")
    for key, model in group["cost_models"].items():
        print(
            f"  {model['label']}: net=${model['total_net_profit_loss']:+.4f}, "
            f"profitable trades={model['profitable_trade_count']}/"
            f"{model['trade_count']} "
            f"({model['profitable_trade_percent']:.2f}%)"
        )
    print("Gross-profit concentration:")
    for share, item in group["gross_profit_concentration"].items():
        print(
            f"  top {share}%: {item['top_trade_count']} trades, "
            f"{item['share_of_positive_gross_percent']:.2f}% of positive gross"
        )


def print_report(results):
    print("BTC/CAD COST VIABILITY & GROSS EDGE STUDY — ANALYSIS ONLY")
    print("REAL-MONEY TRADING: DISABLED")
    split = results["split"]
    print(
        f"Research: {split['research_start']} to {split['research_end']} "
        f"({split['research_periods']} periods)"
    )
    print(
        f"Validation: {split['validation_start']} to {split['validation_end']} "
        f"({split['validation_periods']} periods)"
    )
    _print_group("Research", results["research"])
    _print_group("Untouched validation", results["validation"])
    print("\n=== Interpretation boundary ===")
    print(
        "This reprices the unchanged completed trades under hypothetical cost "
        "factors. It does not change trade frequency, entries, exits, sizing, "
        "or execution, and validation is not used for tuning."
    )


def main():
    results = run_cost_viability_study()
    print_report(results)
    return results


if __name__ == "__main__":
    main()