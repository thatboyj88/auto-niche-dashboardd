import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, patch

from strategy_diagnostic_study import (
    SCORE_BUCKETS,
    StrategyDiagnosticStudy,
    run_strategy_diagnostic_study,
)
from score_effectiveness_study import SCORE_STUDY_PERIODS
from strategy_backtest import StrategyBacktester


def make_candles(count=30):
    return [
        {
            "timestamp": index * 86400,
            "open": 100.0 + index,
            "high": 101.0 + index,
            "low": 99.0 + index,
            "close": 100.0 + index,
            "volume": 1000.0,
        }
        for index in range(count)
    ]


def make_evaluation(candle, score, passed=True):
    return {
        "candle": candle,
        "current_price": 100.0 + candle,
        "ema21": 99.0 + candle,
        "ema50": 98.0 + candle,
        "ema200": 97.0 + candle,
        "rsi": 55.0,
        "long_term_trend": passed,
        "short_term_momentum": not passed,
        "rsi_condition": passed,
        "volume": passed,
        "price_above_ema21": passed,
        "strategy_score": score,
    }


def make_trade(
    number,
    entry_candle,
    exit_candle,
    score,
    net_profit_loss,
    reason,
):
    return {
        "trade_number": number,
        "entry_candle": entry_candle,
        "exit_candle": exit_candle,
        "market_entry_price": 100.0 + entry_candle,
        "gross_profit_loss_before_costs": net_profit_loss + 0.10,
        "fees": 0.05,
        "estimated_slippage": 0.05,
        "net_profit_loss": net_profit_loss,
        "strategy_score": score,
        "rsi_at_entry": 55.0,
        "reason": reason,
    }


def make_period_result(trades, evaluations):
    gross = sum(
        trade["gross_profit_loss_before_costs"]
        for trade in trades
    )
    fees = sum(trade["fees"] for trade in trades)
    slippage = sum(trade["estimated_slippage"] for trade in trades)
    net = sum(trade["net_profit_loss"] for trade in trades)
    wins = sum(trade["net_profit_loss"] > 0 for trade in trades)
    return {
        "period": "Period A",
        "start_date": "2020-01-01",
        "end_date": "2020-12-30",
        "regime": "Bull",
        "market_return": 20.0,
        "return_percent": 4.0,
        "candle_count": 365,
        "trades": len(trades),
        "wins": wins,
        "gross_profit_before_costs": gross,
        "total_fees": fees,
        "total_slippage": slippage,
        "net_profit": net,
        "profit": net,
        "max_drawdown": 2.0,
        "evaluation_history": evaluations,
        "trades_history": trades,
    }


class StrategyDiagnosticStudyTests(unittest.TestCase):
    def setUp(self):
        self.study = StrategyDiagnosticStudy()
        self.candles = make_candles()
        self.evaluations = [
            make_evaluation(5, 80, True),
            make_evaluation(10, 90, False),
            make_evaluation(15, 95, True),
        ]
        self.trades = [
            make_trade(1, 5, 8, 80, 1.0, "TAKE PROFIT"),
            make_trade(2, 10, 12, 90, -0.5, "STOP LOSS"),
            make_trade(3, 15, 20, 95, 0.25, "END OF TEST"),
        ]

    def test_analyze_period_covers_timing_exit_and_excursion_data(self):
        result = self.study.analyze_period(
            make_period_result(self.trades, self.evaluations),
            self.candles,
        )

        self.assertEqual(len(result["trades"]), 3)
        self.assertEqual(
            result["entry_timing"]["all"]["forward_price_movement"]["3"][
                "count"
            ],
            3,
        )
        self.assertEqual(
            result["exit_behavior"]["TAKE PROFIT"]["frequency"],
            1,
        )
        self.assertAlmostEqual(
            result["exit_behavior"]["TAKE PROFIT"]["frequency_percent"],
            100.0 / 3,
        )
        self.assertGreater(
            result["mfe_mae"]["wins"]["mfe_percent"]["average"],
            0,
        )

    def test_score_buckets_report_completed_trade_outcomes(self):
        result = self.study._score_effectiveness(self.study.analyze_period(
            make_period_result(self.trades, self.evaluations),
            self.candles,
        )["trades"])

        self.assertEqual(
            [result[label]["trade_count"] for label, _, _ in SCORE_BUCKETS],
            [1, 0, 1, 1],
        )
        self.assertEqual(result["80-84"]["wins"], 1)
        self.assertEqual(result["90-94"]["losses"], 1)
        self.assertAlmostEqual(result["95-100"]["average_net_profit_loss"], 0.25)

    def test_condition_effectiveness_compares_wins_and_losses(self):
        analyzed = self.study.analyze_period(
            make_period_result(self.trades, self.evaluations),
            self.candles,
        )
        condition = analyzed["condition_effectiveness"]["Long-term trend"]

        self.assertEqual(condition["wins"]["trades"], 2)
        self.assertEqual(condition["wins"]["passed"], 2)
        self.assertEqual(condition["losses"]["trades"], 1)
        self.assertEqual(condition["losses"]["passed"], 0)

    def test_regime_and_cost_summaries_are_independent(self):
        period_result = make_period_result(self.trades, self.evaluations)
        analyzed = self.study.analyze(
            [period_result],
            [self.candles],
        )

        regime = analyzed["by_regime"]["Bull"]
        self.assertEqual(regime["periods"], 1)
        self.assertEqual(regime["trades"], 3)
        self.assertAlmostEqual(regime["net_profit_loss"], 0.75)
        self.assertAlmostEqual(
            analyzed["cost_sensitivity"]["Overall"]["total_costs"],
            0.30,
        )

    def test_missing_entry_evaluation_fails_loudly(self):
        period_result = make_period_result(
            [make_trade(1, 25, 26, 80, 1.0, "TAKE PROFIT")],
            self.evaluations,
        )

        with self.assertRaisesRegex(
            ValueError,
            "missing its entry evaluation",
        ):
            self.study.analyze_period(period_result, self.candles)

    def test_diagnosis_uses_category_specific_evidence(self):
        costly_trades = [
            {
                **trade,
                "fees": 0.30,
                "estimated_slippage": 0.20,
            }
            for trade in self.trades
        ]
        analyzed = self.study.analyze(
            [make_period_result(costly_trades, self.evaluations)],
            [self.candles],
        )

        self.assertIn("trading costs", analyzed["diagnosis"]["primary_findings"])
        self.assertTrue(any(
            note.startswith("Exit behavior:")
            for note in analyzed["diagnosis"]["insufficient_evidence"]
        ))

    def test_ten_period_runner_uses_gated_backtests_and_preserves_returns(self):
        source_candles = []
        for specification in SCORE_STUDY_PERIODS:
            start = datetime.fromisoformat(
                f"{specification['start_date']}T00:00:00+00:00"
            )
            ending_price = {
                "Bull": 120.0,
                "Sideways": 105.0,
                "Bear": 80.0,
            }[specification["regime"]]
            for index in range(365):
                progress = index / 364
                close = 100.0 + (
                    (ending_price - 100.0) * progress
                )
                source_candles.append({
                    "timestamp": int(
                        (start + timedelta(days=index)).timestamp()
                    ),
                    "open": close,
                    "high": close * 1.01,
                    "low": close * 0.99,
                    "close": close,
                    "volume": 1000.0,
                })

        class FakeYahooData:
            last_error = None

            def __init__(self, *_args, **_kwargs):
                pass

            def load(self):
                return source_candles

        notifier = Mock()
        with patch(
            "strategy_diagnostic_study.YahooBTCADMarketData",
            FakeYahooData,
        ):
            result = run_strategy_diagnostic_study(notifier=notifier)

        self.assertEqual(result["period_count"], 10)
        self.assertEqual(notifier.call_count, 10)
        self.assertTrue(all(
            call.args[0]["ok"]
            for call in notifier.call_args_list
        ))

        expected_returns = []
        for start_index in range(0, len(source_candles), 365):
            backtester = StrategyBacktester(25.00)
            backtester.run(source_candles[start_index:start_index + 365])
            expected_returns.append(
                backtester.results()["profit"] / 25.00 * 100
            )
        self.assertEqual(
            [period["strategy_return"] for period in result["periods"]],
            expected_returns,
        )


if __name__ == "__main__":
    unittest.main()