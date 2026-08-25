import unittest

from regime_market_condition_study import (
    BROAD_REGIMES,
    FINE_REGIMES,
    classify_entry_environment,
    _summary,
)


def evaluation(price, ema21, ema50, ema200, long_term, momentum):
    return {
        "current_price": price,
        "ema21": ema21,
        "ema50": ema50,
        "ema200": ema200,
        "long_term_trend": long_term,
        "short_term_momentum": momentum,
    }


class RegimeMarketConditionStudyTests(unittest.TestCase):
    def test_entry_classification_uses_fixed_non_future_features(self):
        self.assertEqual(
            classify_entry_environment(
                evaluation(120, 110, 100, 90, True, True)
            ),
            ("Bull", "Strong Bull"),
        )
        self.assertEqual(
            classify_entry_environment(
                evaluation(80, 90, 100, 110, False, False)
            ),
            ("Bear", "Strong Bear"),
        )
        self.assertEqual(
            classify_entry_environment(
                evaluation(100, 100, 100, 100, False, False)
            ),
            ("Sideways", "Neutral/Sideways"),
        )

    def test_summary_marks_small_groups_insufficient(self):
        result = _summary([{
            "gross_profit_loss": 1.0,
            "net_profit_loss": 0.5,
            "fees": 0.3,
            "slippage": 0.2,
            "entry_score": 80,
            "entry_rsi": 60,
            "mfe_percent": 2.0,
            "mae_percent": -1.0,
            "trade_duration": 3,
            "exit_reason": "TAKE PROFIT",
        }])

        self.assertTrue(result["insufficient_evidence"])
        self.assertEqual(result["trade_count"], 1)
        self.assertEqual(result["exit_reasons"]["TAKE PROFIT"], 1)

    def test_regime_lists_are_fixed(self):
        self.assertEqual(BROAD_REGIMES, ("Bull", "Sideways", "Bear"))
        self.assertEqual(
            FINE_REGIMES,
            (
                "Strong Bull",
                "Weak Bull",
                "Neutral/Sideways",
                "Weak Bear",
                "Strong Bear",
            ),
        )


if __name__ == "__main__":
    unittest.main()