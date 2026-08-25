import unittest
from datetime import datetime, timedelta, timezone

from counterfactual_exit_study import (
    STUDY_PERIODS,
    CounterfactualExitStudy,
    select_study_periods,
)


def make_candle(timestamp, close):
    return {
        "timestamp": timestamp,
        "open": close,
        "high": close * 1.01,
        "low": close * 0.99,
        "close": close,
        "volume": 1000.0,
    }


def make_trade():
    return {
        "trade_number": 1,
        "entry_candle": 0,
        "exit_candle": 1,
        "entry_timestamp": 0,
        "exit_timestamp": 86400,
        "entry_price": 100.10,
        "exit_price": 97.90,
        "position_size": 0.1,
        "gross_profit_loss_before_costs": -0.1,
        "gross_profit_loss": -0.22,
        "fees": 0.08,
        "estimated_slippage": 0.02,
        "net_profit_loss": -0.30,
        "reason": "STOP LOSS",
        "strategy_score": 80,
        "rsi_at_entry": 55.0,
        "market_entry_price": 100.0,
    }


class CounterfactualExitStudyTests(unittest.TestCase):
    def setUp(self):
        self.study = CounterfactualExitStudy()
        self.candles = [
            make_candle(0, 100.0),
            make_candle(86400, 97.0),
            make_candle(172800, 95.0),
            make_candle(259200, 104.5),
            make_candle(345600, 103.0),
            make_candle(432000, 102.0),
        ]

    def test_time_exit_keeps_entry_and_uses_fixed_horizon(self):
        result = self.study._analyze_trade(
            make_trade(),
            self.candles,
            "time_exit",
        )

        self.assertEqual(result["entry_candle"], 0)
        self.assertEqual(result["exit_candle"], 5)
        self.assertEqual(
            result["reason"],
            "COUNTERFACTUAL TIME EXIT",
        )

    def test_extended_holding_ignores_earlier_original_stop(self):
        result = self.study._analyze_trade(
            make_trade(),
            self.candles,
            "extended_holding",
        )

        self.assertEqual(result["exit_candle"], 5)
        self.assertEqual(
            result["reason"],
            "COUNTERFACTUAL EXTENDED HOLD EXIT",
        )

    def test_wider_stop_uses_the_same_entry_with_new_exit_rule(self):
        result = self.study._analyze_trade(
            make_trade(),
            self.candles,
            "wider_stop",
        )

        self.assertEqual(result["entry_price"], 100.10)
        self.assertEqual(result["exit_candle"], 2)
        self.assertEqual(
            result["reason"],
            "COUNTERFACTUAL STOP LOSS",
        )
        self.assertAlmostEqual(
            result["gross_profit_loss_before_costs"] -
            result["execution_price_impact"] -
            result["fees"],
            result["net_profit_loss"],
        )

    def test_excursion_summary_covers_each_original_trade(self):
        result = self.study._measure_excursion(
            make_trade(),
            self.candles,
        )
        summary = self.study._summarize_excursions([result])

        self.assertEqual(summary["trades"], 1)
        self.assertEqual(summary["trades_history"], [result])
        self.assertEqual(
            summary["average_mfe_percent"],
            result["mfe_percent"],
        )
        self.assertEqual(
            summary["average_mae_percent"],
            result["mae_percent"],
        )

    def test_losing_trade_diagnostics_include_post_exit_movement(self):
        diagnostic = self.study._diagnose_original_loss(
            make_trade(),
            self.candles,
        )

        self.assertAlmostEqual(
            diagnostic["lowest_price_reached"],
            self.candles[2]["low"],
        )
        self.assertAlmostEqual(
            diagnostic["highest_price_reached"],
            self.candles[3]["high"],
        )
        self.assertEqual(
            diagnostic["diagnostic_horizon_candles"],
            5,
        )
        self.assertEqual(
            diagnostic["price_after_3_candles"],
            self.candles[3]["close"],
        )
        self.assertEqual(
            diagnostic["price_after_5_candles"],
            self.candles[5]["close"],
        )

    def test_fixed_study_windows_use_the_ten_year_source(self):
        self.assertEqual(
            {specification["data_range"] for specification in STUDY_PERIODS},
            {"10y"},
        )

    def test_fixed_study_window_selector_checks_dates_and_regimes(self):
        source_candles = []
        for specification, ending_price in zip(
            STUDY_PERIODS,
            (130.0, 130.0, 105.0),
        ):
            start = datetime.fromisoformat(
                f"{specification['start_date']}T00:00:00+00:00"
            )
            for index in range(365):
                progress = index / 364
                close = 100.0 + (
                    (ending_price - 100.0) * progress
                )
                source_candles.append(
                    make_candle(
                        int(
                            (
                                start +
                                timedelta(days=index)
                            ).timestamp()
                        ),
                        close,
                    )
                )

        selected = select_study_periods(
            {"10y": source_candles},
        )

        self.assertEqual(
            list(selected),
            ["Bull Period A", "Bull Period B", "Sideways Period"],
        )
        self.assertTrue(all(
            len(period) == 365
            for period in selected.values()
        ))


if __name__ == "__main__":
    unittest.main()