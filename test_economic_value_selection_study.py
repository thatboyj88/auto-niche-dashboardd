import unittest

from economic_value_selection_study import (
    BREAK_EVEN_PERCENT,
    EconomicValueModel,
    MIN_VALIDATION_TRADES_FOR_PROMOTION,
    _features,
)


class EconomicValueSelectionStudyTests(unittest.TestCase):
    def test_feature_vector_is_fixed_entry_time_shape(self):
        evaluation = {
            "strategy_score": 80,
            "rsi": 55,
            "current_price": 100,
            "ema21": 99,
            "ema50": 98,
            "ema200": 95,
            "long_term_trend": True,
            "short_term_momentum": True,
            "volume": True,
            "price_above_ema21": True,
        }
        self.assertEqual(len(_features(evaluation)), 9)
        self.assertAlmostEqual(BREAK_EVEN_PERCENT, 1.0)

    def test_model_uses_fixed_positive_expected_net_cutoff(self):
        rows = [
            ((1, 1, 1, 1, 1, 1, 1, 1, 1), 1.0),
            ((0, 0, 0, 0, 0, 0, 0, 0, 0), -1.0),
        ] * 20
        model = EconomicValueModel().fit(rows)
        self.assertTrue(model.accepts((1, 1, 1, 1, 1, 1, 1, 1, 1)))
        self.assertFalse(model.accepts((0, 0, 0, 0, 0, 0, 0, 0, 0)))
        self.assertEqual(MIN_VALIDATION_TRADES_FOR_PROMOTION, 20)


if __name__ == "__main__":
    unittest.main()