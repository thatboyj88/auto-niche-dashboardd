import unittest

from strategy_calibration_study import (
    RSI_BANDS,
    SCORE_BANDS,
    StrategyCalibrationStudy,
)


def make_signal(score, rsi, movements, passed_conditions):
    return {
        "score": score,
        "entry_rsi": rsi,
        "forward_returns": {
            horizon: movements.get(horizon)
            for horizon in (3, 5, 10, 20)
        },
        "passed_conditions": tuple(passed_conditions),
        "completed_trade": False,
        "completed_trade_net_profit": None,
    }


class StrategyCalibrationStudyTests(unittest.TestCase):
    def setUp(self):
        self.study = StrategyCalibrationStudy()
        self.signals = [
            make_signal(
                80,
                55,
                {3: 0.5, 5: 1.0, 10: 2.0, 20: 3.0},
                ("RSI", "Volume"),
            ),
            make_signal(
                85,
                65,
                {3: -0.5, 5: -1.0, 10: 1.0, 20: 2.0},
                ("RSI",),
            ),
            make_signal(
                95,
                75,
                {3: 2.0, 5: 3.0, 10: 4.0, 20: 5.0},
                ("RSI", "Volume"),
            ),
        ]

    def test_score_and_rsi_band_boundaries_are_inclusive_on_lower_edge(self):
        score = self.study._band_summary(
            self.signals,
            lambda signal: signal["score"],
            SCORE_BANDS,
        )
        rsi = self.study._band_summary(
            self.signals,
            lambda signal: signal["entry_rsi"],
            RSI_BANDS,
        )

        self.assertEqual(score["80-84"]["signals"], 1)
        self.assertEqual(score["85-89"]["signals"], 1)
        self.assertEqual(score["95-100"]["signals"], 1)
        self.assertEqual(rsi["50-59"]["signals"], 1)
        self.assertEqual(rsi["60-69"]["signals"], 1)
        self.assertEqual(rsi["70-79"]["signals"], 1)

    def test_score_100_belongs_to_the_95_to_100_band(self):
        bands = self.study._band_summary(
            [
                make_signal(
                    100,
                    70,
                    {3: 1.0, 5: 1.0, 10: 1.0, 20: 1.0},
                    ("RSI",),
                ),
            ],
            lambda signal: signal["score"],
            SCORE_BANDS,
        )

        self.assertEqual(bands["95-100"]["signals"], 1)

    def test_condition_combinations_are_grouped_and_early_movement_summarized(self):
        combinations = self.study._condition_combinations(self.signals)
        movement = self.study._movement_summary(self.signals)

        self.assertEqual(combinations["RSI + Volume"]["signals"], 2)
        self.assertAlmostEqual(
            movement["5"]["average"],
            1.0,
        )
        self.assertAlmostEqual(
            movement["5"]["positive_percent"],
            66.66666666666667,
        )

    def test_cost_break_even_uses_fee_and_slippage_formula(self):
        threshold = self.study.break_even_move_percent()
        self.assertGreater(threshold, 1.0)
        result = self.study._cost_break_even(self.signals)

        self.assertAlmostEqual(
            result["required_move_percent"],
            threshold,
        )
        self.assertEqual(
            result["overall"]["horizons"]["5"]["reached_break_even"],
            1,
        )

    def test_diagnosis_marks_sparse_bands_as_insufficient(self):
        diagnosis = self.study._diagnosis(self.signals)

        self.assertEqual(
            diagnosis["score_calibration"],
            "insufficient evidence",
        )
        self.assertTrue(diagnosis["limitations"])

    def test_two_ordered_score_bands_are_suggestive_not_conclusive(self):
        signals = [
            make_signal(
                80,
                55,
                {3: 0.0, 5: 0.5, 10: 0.0, 20: 0.0},
                ("RSI",),
            )
            for _ in range(20)
        ] + [
            make_signal(
                85,
                65,
                {3: 0.0, 5: 2.0, 10: 0.0, 20: 0.0},
                ("RSI", "Volume"),
            )
            for _ in range(20)
        ]

        diagnosis = self.study._diagnosis(signals)

        self.assertEqual(
            diagnosis["score_calibration"],
            "suggestive but incomplete",
        )


if __name__ == "__main__":
    unittest.main()