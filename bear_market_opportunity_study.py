"""Analysis-only study of opportunities the control skipped in bear conditions."""

from config import STOP_LOSS_PERCENT, TAKE_PROFIT_PERCENT
from multi_period_backtest import MultiPeriodBacktester
from out_of_sample_validation import _split_periods
from regime_market_condition_study import classify_entry_environment
from score_effectiveness_study import SCORE_STUDY_PERIODS, select_score_study_periods
from strategy_calibration_study import (
    FEE_PERCENT,
    SLIPPAGE_PERCENT,
    STARTING_CAPITAL,
)
from yahoo_btc_cad_data import YahooBTCADMarketData


FORWARD_HORIZONS = (1, 3, 5, 10, 20)
POSITION_FRACTION = 0.40


def _summary(items):
    available = {
        horizon: [
            item for item in items if item["forward"][horizon]["available"]
        ]
        for horizon in FORWARD_HORIZONS
    }
    return {
        "opportunities": len(items),
        "profitable_at_horizon": {
            horizon: sum(
                item["forward"][horizon]["net_profit_loss"] > 0
                for item in available[horizon]
            )
            for horizon in FORWARD_HORIZONS
        },
        "average_close_return": {
            horizon: (
                sum(
                    item["forward"][horizon]["close_return_percent"]
                    for item in available[horizon]
                )
                / len(available[horizon])
                if available[horizon] else 0.0
            )
            for horizon in FORWARD_HORIZONS
        },
        "average_mfe": {
            horizon: (
                sum(
                    item["forward"][horizon]["mfe_percent"]
                    for item in available[horizon]
                )
                / len(available[horizon])
                if available[horizon] else 0.0
            )
            for horizon in FORWARD_HORIZONS
        },
        "cost_covering_opportunities": {
            horizon: sum(
                item["forward"][horizon]["mfe_percent"]
                >= item["required_move_percent"]
                for item in available[horizon]
            )
            for horizon in FORWARD_HORIZONS
        },
        "exit_simulation": {
            horizon: {
                "count": sum(
                    item["forward"][horizon]["exit_simulation"]["net_profit_loss"] > 0
                    for item in available[horizon]
                ),
                "net_profit_loss": sum(
                    item["forward"][horizon]["exit_simulation"]["net_profit_loss"]
                    for item in available[horizon]
                ),
            }
            for horizon in FORWARD_HORIZONS
        },
    }


class BearMarketOpportunityStudy:
    def analyze_period(self, period_result, candles):
        opportunities = []
        for evaluation in period_result["evaluation_history"]:
            if evaluation["decision"] == "BUY":
                continue
            broad, fine = classify_entry_environment(evaluation)
            if broad != "Bear":
                continue
            entry = evaluation["candle"]
            entry_price = evaluation["current_price"]
            required_move = (
                (1 + SLIPPAGE_PERCENT) / (1 - SLIPPAGE_PERCENT)
                * (1 + FEE_PERCENT) / (1 - FEE_PERCENT)
                - 1
            ) * 100
            forward = {}
            for horizon in FORWARD_HORIZONS:
                path = candles[entry + 1:entry + horizon + 1]
                close_candle = candles[entry + horizon] if entry + horizon < len(candles) else None
                if not path or close_candle is None:
                    forward[horizon] = {
                        "available": False,
                        "close_return_percent": None,
                        "mfe_percent": None,
                        "mae_percent": None,
                        "net_profit_loss": 0.0,
                        "exit_simulation": {
                            "reason": "INSUFFICIENT HORIZON",
                            "net_profit_loss": 0.0,
                        },
                    }
                    continue
                mfe = (max(candle["high"] for candle in path) / entry_price - 1) * 100
                mae = (min(candle["low"] for candle in path) / entry_price - 1) * 100
                close_return = (close_candle["close"] / entry_price - 1) * 100
                simulation = self._simulate_exit(path, entry_price)
                forward[horizon] = {
                    "available": True,
                    "close_return_percent": close_return,
                    "mfe_percent": mfe,
                    "mae_percent": mae,
                    "net_profit_loss": simulation["net_profit_loss"],
                    "exit_simulation": simulation,
                }
            opportunities.append({
                "candle": entry,
                "timestamp": evaluation["timestamp"],
                "fine_regime": fine,
                "score": evaluation["strategy_score"],
                "rsi": evaluation["rsi"],
                "price_vs_ema21": (
                    evaluation["current_price"] / evaluation["ema21"] - 1
                ) * 100,
                "required_move_percent": required_move,
                "forward": forward,
            })
        return opportunities

    @staticmethod
    def _simulate_exit(path, entry_price):
        position_value = STARTING_CAPITAL * POSITION_FRACTION
        entry_price_actual = entry_price * (1 + SLIPPAGE_PERCENT)
        position = position_value / entry_price_actual
        entry_fee = position_value * FEE_PERCENT
        entry_slippage = position_value * SLIPPAGE_PERCENT
        reason = "HORIZON"
        exit_price = path[-1]["close"]
        for candle in path:
            if candle["close"] <= entry_price * (1 - STOP_LOSS_PERCENT):
                exit_price = candle["close"]
                reason = "STOP LOSS"
                break
            if candle["close"] >= entry_price * (1 + TAKE_PROFIT_PERCENT):
                exit_price = candle["close"]
                reason = "TAKE PROFIT"
                break
        exit_value = position * exit_price * (1 - SLIPPAGE_PERCENT)
        exit_fee = exit_value * FEE_PERCENT
        exit_slippage = exit_value * SLIPPAGE_PERCENT
        net = (
            exit_value - exit_fee - position_value - entry_fee
            - entry_slippage - exit_slippage
        )
        return {
            "reason": reason,
            "net_profit_loss": net,
            "fees": entry_fee + exit_fee,
            "slippage": entry_slippage + exit_slippage,
        }

    def analyze_group(self, period_pairs):
        items = [
            item
            for result, candles in period_pairs
            for item in self.analyze_period(result, candles)
        ]
        return {
            "opportunities": items,
            "all": _summary(items),
            "by_fine_regime": {
                regime: _summary([
                    item for item in items if item["fine_regime"] == regime
                ])
                for regime in ("Weak Bear", "Strong Bear")
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


def run_bear_market_opportunity_study(notifier=None):
    if notifier is None:
        from btc_cad_preflight import send_slack_notification
        notifier = send_slack_notification
    loader = YahooBTCADMarketData(data_range="10y")
    candles = loader.load()
    if not candles:
        raise RuntimeError(loader.last_error or "Yahoo BTC/CAD data unavailable")
    selected = select_score_study_periods(candles)
    if len(selected) != len(SCORE_STUDY_PERIODS):
        raise RuntimeError("Bear opportunity study requires all fixed periods")
    research_periods, validation_periods = _split_periods(selected)
    study = BearMarketOpportunityStudy()
    return {
        "real_money_trading": False,
        "research": study.analyze_group(
            _run_period_group(research_periods, notifier)
        ),
        "validation": study.analyze_group(
            _run_period_group(validation_periods, notifier)
        ),
        "note": (
            "These are counterfactual skipped-entry diagnostics, not executed "
            "trades and not a proposed bear-market strategy."
        ),
    }


def _print_group(label, group):
    print(f"\n=== {label} ===")
    print(f"Bear no-trade opportunities: {group['all']['opportunities']}")
    print("Horizon | Profitable | Cost-covering | Avg close | Avg MFE | Sim net")
    for horizon in FORWARD_HORIZONS:
        print(
            f"{horizon} | "
            f"{group['all']['profitable_at_horizon'][horizon]} | "
            f"{group['all']['cost_covering_opportunities'][horizon]} | "
            f"{group['all']['average_close_return'][horizon]:+.2f}% | "
            f"{group['all']['average_mfe'][horizon]:+.2f}% | "
            f"${group['all']['exit_simulation'][horizon]['net_profit_loss']:+.4f}"
        )
    for regime, summary in group["by_fine_regime"].items():
        print(f"{regime}: {summary['opportunities']} opportunities")


def print_report(results):
    print("BTC/CAD BEAR-MARKET OPPORTUNITY STUDY — ANALYSIS ONLY")
    print("REAL-MONEY TRADING: DISABLED")
    print(results["note"])
    _print_group("Research", results["research"])
    _print_group("Untouched validation", results["validation"])
    print("\n=== Interpretation boundary ===")
    print(
        "Counterfactual skipped entries do not prove an executable bear-market "
        "edge. No bear filter, entry rule, or production behavior changed."
    )


def main():
    results = run_bear_market_opportunity_study()
    print_report(results)
    return results


if __name__ == "__main__":
    main()