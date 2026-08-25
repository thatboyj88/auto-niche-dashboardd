from datetime import datetime, timezone

from strategy_backtest import StrategyBacktester


PERIOD_CANDLES = 365
BULL_RETURN_PERCENT = 15.0
BEAR_RETURN_PERCENT = -15.0


class MultiPeriodBacktester:
    """
    Runs independent paper backtests over fixed historical daily periods.

    Regime labels are descriptive only. They are calculated after a period is
    selected and are never supplied to, or used by, the trading strategy.
    """

    def __init__(self, starting_capital=25.00):
        if starting_capital <= 0:
            raise ValueError("starting_capital must be positive")

        self.starting_capital = starting_capital

    def run(self, candles, notifier=None):
        periods = self.build_periods(candles)
        period_results = [
            self._run_period(
                index,
                period_candles,
                notifier=notifier,
            )
            for index, period_candles in enumerate(periods)
        ]

        result = {
            "periods": period_results,
            "aggregate": self._aggregate(period_results),
            "regime_summary": self._regime_summary(period_results),
            "period_candles": PERIOD_CANDLES,
            "source_candles": len(candles),
            "unused_candles": len(candles) % PERIOD_CANDLES,
        }
        result["sources"] = [{
            "label": "Historical daily data",
            "kind": "historical",
            "candle_count": len(candles),
            "period_count": len(periods),
            "unused_candles": len(candles) % PERIOD_CANDLES,
        }]
        return result

    def run_sources(self, sources, notifier=None):
        """
        Run each named source independently, then combine presentation data.

        Source boundaries are never joined into a trading period. This keeps
        an anchored sample from changing rolling-period dates or compounding
        one paper test into another.
        """
        period_results = []
        source_results = []

        for source in sources:
            candles = source["candles"]
            label = source["label"]
            kind = source["kind"]
            periods = self.build_periods(candles)
            source_periods = [
                self._run_period(
                    index,
                    period_candles,
                    period_label=(
                        f"{label} · Period "
                        f"{chr(ord('A') + index)}"
                    ),
                    source_label=label,
                    source_kind=kind,
                    notifier=notifier,
                )
                for index, period_candles in enumerate(periods)
            ]
            period_results.extend(source_periods)
            source_results.append({
                "label": label,
                "kind": kind,
                "candle_count": len(candles),
                "period_count": len(periods),
                "unused_candles": len(candles) % PERIOD_CANDLES,
            })

        result = {
            "periods": period_results,
            "aggregate": self._aggregate(period_results),
            "regime_summary": self._regime_summary(period_results),
            "period_candles": PERIOD_CANDLES,
            "source_candles": sum(
                source["candle_count"] for source in source_results
            ),
            "unused_candles": sum(
                source["unused_candles"] for source in source_results
            ),
            "sources": source_results,
        }
        return result

    @staticmethod
    def build_periods(candles):
        """
        Return complete, non-overlapping oldest-to-newest 365-candle periods.
        Any incomplete trailing period is intentionally excluded.
        """
        complete_candle_count = (
            len(candles) // PERIOD_CANDLES
        ) * PERIOD_CANDLES

        return [
            candles[start:start + PERIOD_CANDLES]
            for start in range(0, complete_candle_count, PERIOD_CANDLES)
        ]

    @staticmethod
    def classify_regime(candles):
        if len(candles) < 2:
            raise ValueError("at least two candles are required")

        opening_price = candles[0]["close"]
        closing_price = candles[-1]["close"]
        market_return = (
            (closing_price - opening_price) /
            opening_price
        ) * 100

        threshold_tolerance = 1e-9

        if market_return >= (
            BULL_RETURN_PERCENT - threshold_tolerance
        ):
            regime = "Bull"
        elif market_return <= (
            BEAR_RETURN_PERCENT + threshold_tolerance
        ):
            regime = "Bear"
        else:
            regime = "Sideways"

        return regime, market_return

    def _run_period(
        self,
        index,
        candles,
        period_label=None,
        source_label=None,
        source_kind=None,
        notifier=None,
    ):
        from btc_cad_preflight import (
            BTCADPreflightError,
            validate_period,
        )

        resolved_period_label = (
            period_label or f"Period {chr(ord('A') + index)}"
        )
        try:
            preflight = validate_period(
                candles,
                period=resolved_period_label,
                source=source_label,
            )
        except Exception as error:
            preflight = {
                "ok": False,
                "candle_count": (
                    len(candles) if isinstance(candles, list) else 0
                ),
                "start_date": None,
                "end_date": None,
                "period": resolved_period_label,
                "source": source_label,
                "failure": (
                    "BTC/CAD preflight could not validate the period: "
                    f"{error}"
                ),
            }
        # Presentation callers, including the Streamlit dashboard, must not
        # emit external notifications as a side effect of rendering. The
        # dedicated preflight and diagnostic workflows inject the Slack
        # notifier explicitly when an alert is required.
        notifier = notifier or (lambda _result: None)

        try:
            notifier(preflight)
        except Exception as error:
            raise BTCADPreflightError(
                "BTC/CAD preflight notification failed before "
                f"{resolved_period_label} could be backtested: {error}"
            ) from error

        if not preflight["ok"]:
            raise BTCADPreflightError(
                "BTC/CAD preflight failed before "
                f"{resolved_period_label} could be backtested: "
                f"{preflight['failure']}"
            )

        regime = preflight["regime"]
        market_return = preflight["market_return"]
        backtester = StrategyBacktester(
            starting_capital=self.starting_capital
        )
        backtester.run(candles)
        results = backtester.results()
        return_percent = (
            results["profit"] /
            results["starting_capital"]
        ) * 100
        gross_profit_before_costs = sum(
            trade["gross_profit_loss_before_costs"]
            for trade in results["trades_history"]
        )

        result = {
            "period": resolved_period_label,
            "candle_count": len(candles),
            "start_timestamp": candles[0]["timestamp"],
            "end_timestamp": candles[-1]["timestamp"],
            "start_date": self.format_date(candles[0]["timestamp"]),
            "end_date": self.format_date(candles[-1]["timestamp"]),
            "regime": regime,
            "market_return": market_return,
            "preflight": dict(preflight),
            "starting_capital": results["starting_capital"],
            "ending_capital": results["ending_capital"],
            "profit": results["profit"],
            "return_percent": return_percent,
            "trades": results["trades"],
            "wins": results["wins"],
            "losses": results["losses"],
            "win_rate": results["win_rate"],
            "gross_profit_before_costs": gross_profit_before_costs,
            "total_fees": results["total_fees"],
            "total_slippage": results["total_slippage"],
            "net_profit": results["profit"],
            "max_drawdown": results["max_drawdown"],
            "evaluations": results["evaluations"],
            "highest_score": results["highest_score"],
            "score_80_or_more": results["score_80_or_more"],
            "condition_counts": dict(results["condition_counts"]),
            "trades_history": list(results["trades_history"]),
            "evaluation_history": list(results["evaluation_history"]),
        }
        if source_label is not None:
            result["source_label"] = source_label
        if source_kind is not None:
            result["source_kind"] = source_kind
        return result

    @staticmethod
    def format_date(timestamp):
        return datetime.fromtimestamp(
            timestamp,
            tz=timezone.utc,
        ).strftime("%Y-%m-%d")

    @staticmethod
    def _aggregate(periods):
        if not periods:
            return {
                "total_profit": 0.0,
                "total_return": 0.0,
                "average_return": 0.0,
                "average_trades": 0.0,
                "average_win_rate": 0.0,
                "total_gross_profit_before_costs": 0.0,
                "total_fees": 0.0,
                "total_slippage": 0.0,
                "worst_drawdown": 0.0,
                "best_period": None,
                "worst_period": None,
            }

        return {
            "total_profit": sum(period["profit"] for period in periods),
            "total_return": sum(
                period["return_percent"]
                for period in periods
            ),
            "average_return": sum(
                period["return_percent"]
                for period in periods
            ) / len(periods),
            "average_trades": sum(
                period["trades"]
                for period in periods
            ) / len(periods),
            "average_win_rate": sum(
                period["win_rate"]
                for period in periods
            ) / len(periods),
            "total_gross_profit_before_costs": sum(
                period["gross_profit_before_costs"]
                for period in periods
            ),
            "total_fees": sum(
                period["total_fees"]
                for period in periods
            ),
            "total_slippage": sum(
                period["total_slippage"]
                for period in periods
            ),
            "worst_drawdown": max(
                period["max_drawdown"]
                for period in periods
            ),
            "best_period": max(
                periods,
                key=lambda period: period["return_percent"],
            ),
            "worst_period": min(
                periods,
                key=lambda period: period["return_percent"],
            ),
        }

    @staticmethod
    def _regime_summary(periods):
        """
        Group completed, independently run periods by descriptive regime.

        This summary is presentation metadata only. It is not supplied to the
        strategy or used to alter any strategy, risk, fee, or slippage setting.
        """
        return {
            regime: [
                period
                for period in periods
                if period["regime"] == regime
            ]
            for regime in ("Bull", "Bear", "Sideways")
        }
