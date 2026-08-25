"""Cost-adjusted, analysis-only hypothetical exit economics."""

import math
from statistics import median

from exit_capture_study import (
    BREAK_EVEN_MOVE_PERCENT,
    TARGETS,
    ExitCaptureStudy,
    _run_period_group,
)
from out_of_sample_validation import _split_periods
from score_effectiveness_study import (
    SCORE_STUDY_PERIODS,
    select_score_study_periods,
)
from strategy_calibration_study import (
    FEE_PERCENT,
    SLIPPAGE_PERCENT,
    STARTING_CAPITAL,
)
from yahoo_btc_cad_data import YahooBTCADMarketData


CONFIDENCE_LEVEL = 0.95
Z_SCORE_95 = 1.959963984540054


def _average(values):
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _summary(values):
    values = list(values)
    return {
        "count": len(values),
        "average": _average(values),
        "median": median(values) if values else 0.0,
        "minimum": min(values) if values else None,
        "maximum": max(values) if values else None,
    }


def _percent(numerator, denominator):
    return numerator / denominator * 100 if denominator else 0.0


def _wilson_interval(successes, trials, z=Z_SCORE_95):
    """Return a two-sided Wilson interval as percentages."""
    if not trials:
        return {
            "confidence_level": CONFIDENCE_LEVEL,
            "lower_percent": None,
            "upper_percent": None,
        }
    proportion = successes / trials
    denominator = 1 + z * z / trials
    center = (
        proportion + z * z / (2 * trials)
    ) / denominator
    margin = (
        z * math.sqrt(
            proportion * (1 - proportion) / trials
            + z * z / (4 * trials * trials)
        ) / denominator
    )
    return {
        "confidence_level": CONFIDENCE_LEVEL,
        "lower_percent": max(0.0, (center - margin) * 100),
        "upper_percent": min(100.0, (center + margin) * 100),
    }


class ExitEconomicsStudy:
    """Compare original exits with cost-adjusted post-exit target exits."""

    def __init__(
        self,
        fee_percent=FEE_PERCENT,
        slippage_percent=SLIPPAGE_PERCENT,
    ):
        self.fee_percent = fee_percent
        self.slippage_percent = slippage_percent
        self.capture = ExitCaptureStudy()

    def analyze_trade(self, trade, candles):
        capture = self.capture.analyze_trade(trade, candles)
        hypothetical_exits = {
            name: self._hypothetical_exit(
                trade,
                capture["targets"][name],
                capture["market_entry_price"],
            )
            for name, _ in TARGETS
        }
        return {
            **capture,
            "hypothetical_exits": hypothetical_exits,
        }

    def _hypothetical_exit(
        self,
        trade,
        target,
        market_entry_price,
    ):
        if target["phase"] != "after_exit":
            return None

        target_market_price = market_entry_price * (
            1 + target["threshold_percent"] / 100
        )
        position_size = trade["position_size"]
        entry_value = position_size * trade["entry_price"]
        entry_fee = entry_value * self.fee_percent
        execution_exit_price = target_market_price * (
            1 - self.slippage_percent
        )
        exit_value = position_size * execution_exit_price
        exit_fee = exit_value * self.fee_percent
        fees = entry_fee + exit_fee
        slippage = (
            entry_value * self.slippage_percent
            + exit_value * self.slippage_percent
        )
        gross_profit_loss = exit_value - entry_value
        net_profit_loss = gross_profit_loss - fees
        return {
            "threshold_percent": target["threshold_percent"],
            "exit_candle": target["candle"],
            "candles_after_original_exit": target["candles_after_exit"],
            "market_exit_price": target_market_price,
            "execution_exit_price": execution_exit_price,
            "gross_profit_loss": gross_profit_loss,
            "fees": fees,
            "slippage": slippage,
            "net_profit_loss": net_profit_loss,
            "net_improvement_vs_original": (
                net_profit_loss - trade["net_profit_loss"]
            ),
            "profitable_improvement": (
                net_profit_loss - trade["net_profit_loss"] > 0
            ),
        }

    def analyze_group(self, period_results, period_candles):
        if len(period_results) != len(period_candles):
            raise ValueError("period result and candle counts must match")
        periods = [
            self._analyze_period(result, candles)
            for result, candles in zip(period_results, period_candles)
        ]
        trades = [
            trade for period in periods for trade in period["trades"]
        ]
        return {
            "period_count": len(periods),
            "trade_count": len(trades),
            "periods": periods,
            "summary": self._group_summary(trades),
        }

    def _analyze_period(self, period_result, candles):
        trades = [
            self.analyze_trade(trade, candles)
            for trade in period_result["trades_history"]
        ]
        return {
            "period": period_result["period"],
            "start_date": period_result["start_date"],
            "end_date": period_result["end_date"],
            "regime": period_result["regime"],
            "candles": len(candles),
            "trade_count": len(trades),
            "trades": trades,
            "summary": self._group_summary(trades),
        }

    def _group_summary(self, trades):
        return {
            "targets": {
                name: self._target_summary(trades, name)
                for name, _ in TARGETS
            },
            "by_exit_reason": {
                reason: self._reason_summary(
                    [trade for trade in trades
                     if trade["exit_reason"] == reason]
                )
                for reason in ("STOP LOSS", "TAKE PROFIT", "END OF TEST")
            },
        }

    @staticmethod
    def _target_summary(trades, name):
        opportunities = [
            trade["hypothetical_exits"][name]
            for trade in trades
            if trade["hypothetical_exits"][name] is not None
        ]
        improvements = [
            item["net_improvement_vs_original"]
            for item in opportunities
        ]
        positive = sum(
            item["profitable_improvement"] for item in opportunities
        )
        total = len(trades)
        gross = sum(item["gross_profit_loss"] for item in opportunities)
        fees = sum(item["fees"] for item in opportunities)
        slippage = sum(item["slippage"] for item in opportunities)
        return {
            "threshold_percent": dict(TARGETS)[name],
            "all_completed_trades": total,
            "post_exit_opportunities": len(opportunities),
            "post_exit_opportunity_percent": _percent(
                len(opportunities), total
            ),
            "post_exit_opportunity_interval": _wilson_interval(
                len(opportunities), total
            ),
            "additional_candles": _summary(
                item["candles_after_original_exit"]
                for item in opportunities
            ),
            "original_net_profit_loss": sum(
                trade["net_profit_loss"] for trade in trades
                if trade["hypothetical_exits"][name] is not None
            ),
            "hypothetical_gross_profit_loss": gross,
            "hypothetical_fees": fees,
            "hypothetical_slippage": slippage,
            "hypothetical_net_profit_loss": sum(
                item["net_profit_loss"] for item in opportunities
            ),
            "net_improvement": _summary(improvements),
            "total_net_improvement": sum(improvements),
            "profitable_improvement_count": positive,
            "profitable_improvement_percent": _percent(
                positive, len(opportunities)
            ),
            "profitable_improvement_interval": _wilson_interval(
                positive, len(opportunities)
            ),
            "cost_share_of_hypothetical_gross_percent": (
                (fees + slippage) / abs(gross) * 100 if gross else 0.0
            ),
        }

    @staticmethod
    def _reason_summary(trades):
        return {
            "trades": len(trades),
            "targets": {
                name: ExitEconomicsStudy._target_summary(trades, name)
                for name, _ in TARGETS
            },
        }


def run_exit_economics_study(notifier=None):
    if notifier is None:
        from btc_cad_preflight import send_slack_notification
        notifier = send_slack_notification
    loader = YahooBTCADMarketData(data_range="10y")
    candles = loader.load()
    if not candles:
        raise RuntimeError(loader.last_error or "Yahoo BTC/CAD data unavailable")
    selected = select_score_study_periods(candles)
    if len(selected) != len(SCORE_STUDY_PERIODS):
        raise RuntimeError("Exit economics study requires all fixed periods")
    research, validation = _split_periods(selected)
    research_results = _run_period_group(research, notifier)
    validation_results = _run_period_group(validation, notifier)
    study = ExitEconomicsStudy()
    return {
        "source": "Yahoo Finance BTC/CAD aggregated daily data",
        "real_money_trading": False,
        "cost_model": {
            "fee_percent": FEE_PERCENT,
            "slippage_percent": SLIPPAGE_PERCENT,
            "break_even_move_percent": BREAK_EVEN_MOVE_PERCENT,
        },
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
        "research": study.analyze_group(
            research_results,
            [period["candles"] for period in research],
        ),
        "validation": study.analyze_group(
            validation_results,
            [period["candles"] for period in validation],
        ),
    }


def _print_group(label, group):
    print(f"\n=== {label} ===")
    print(f"completed trades={group['trade_count']}")
    for name, threshold in TARGETS:
        summary = group["summary"]["targets"][name]
        interval = summary["profitable_improvement_interval"]
        print(
            f"{name} ({threshold:.3f}%): "
            f"post-exit opportunities={summary['post_exit_opportunities']} "
            f"({summary['post_exit_opportunity_percent']:.2f}%), "
            f"additional candles avg="
            f"{summary['additional_candles']['average']:.2f}, "
            f"hypothetical fees=${summary['hypothetical_fees']:.4f}, "
            f"slippage=${summary['hypothetical_slippage']:.4f}"
        )
        print(
            f"  hypothetical net="
            f"${summary['hypothetical_net_profit_loss']:+.4f}, "
            f"improvement avg="
            f"${summary['net_improvement']['average']:+.4f}, "
            f"total="
            f"${summary['total_net_improvement']:+.4f}, "
            f"positive={summary['profitable_improvement_percent']:.2f}% "
            f"(95% Wilson interval "
            f"{interval['lower_percent']:.2f}%–{interval['upper_percent']:.2f}%)"
        )
    for reason, reason_summary in group["summary"]["by_exit_reason"].items():
        if not reason_summary["trades"]:
            continue
        print(f"{reason}: trades={reason_summary['trades']}")
        for name, _ in TARGETS:
            summary = reason_summary["targets"][name]
            print(
                f"  {name}: opportunities={summary['post_exit_opportunities']}, "
                f"net improvement avg="
                f"${summary['net_improvement']['average']:+.4f}"
            )


def print_report(results):
    print("BTC/CAD EXIT ECONOMICS — STEP 13 ANALYSIS ONLY")
    print("REAL-MONEY TRADING: DISABLED")
    split = results["split"]
    print(
        f"Research: {split['research_start']} to {split['research_end']} "
        f"({split['research_periods']} periods, {split['research_candles']} candles)"
    )
    print(
        f"Validation: {split['validation_start']} to {split['validation_end']} "
        f"({split['validation_periods']} periods, {split['validation_candles']} candles)"
    )
    for label, key in (
        ("Research", "research"),
        ("Untouched validation", "validation"),
    ):
        _print_group(label, results[key])
    print("\n=== Interpretation boundary ===")
    print(
        "Hypothetical exits use the first post-exit candle reaching the "
        "target price and apply the unchanged fee/slippage model. They are "
        "diagnostic opportunity estimates, not executable trades or proof "
        "that alternate exits would improve the strategy."
    )


def main():
    results = run_exit_economics_study()
    print_report(results)
    return results


if __name__ == "__main__":
    main()