import unittest
from datetime import datetime, timedelta

from score_effectiveness_study import (
    SCORE_STUDY_PERIODS,
    ScoreEffectivenessStudy,
    select_score_study_periods,
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


def make_evaluation(number, candle, score):
    return {
        "evaluation_number": number,
        "candle": candle,
        "timestamp": candle * 86400,
        "strategy_score": score,
        "current_price": 100.0,
    }


class ScoreEffectivenessStudyTests(unittest.TestCase):
    def setUp(self):
        self.study = ScoreEffectivenessStudy()
        self.candles = [
            make_candle(index * 86400, 100.0 + index)
            for index in range(25)
        ]

    def test_score_buckets_include_their_boundary_scores(self):
        evaluations = [
            make_evaluation(1, 0, 59),
            make_evaluation(2, 0, 60),
            make_evaluation(3, 0, 70),
            make_evaluation(4, 0, 80),
            make_evaluation(5, 0, 90),
        ]

        result = self.study.analyze_evaluations(
            evaluations,
            self.candles,
        )

        self.assertEqual(result["valid_evaluations"], 5)
        self.assertEqual(
            [
                result["buckets"][bucket]["evaluations"]
                for bucket in ("0-59", "60-69", "70-79", "80-89", "90-100")
            ],
            [1, 1, 1, 1, 1],
        )
        self.assertTrue(
            result["buckets"]["90-100"]["insufficient_evidence"],
        )

    def test_forward_returns_are_calculated_from_evaluation_close(self):
        result = self.study.analyze_evaluations(
            [make_evaluation(1, 0, 80)],
            self.candles,
        )
        bucket = result["buckets"]["80-89"]

        self.assertAlmostEqual(
            bucket["horizons"][1]["average_return"],
            1.0,
        )
        self.assertAlmostEqual(
            bucket["horizons"][5]["median_return"],
            5.0,
        )
        self.assertAlmostEqual(
            bucket["horizons"][20]["positive_percent"],
            100.0,
        )
        self.assertGreater(
            bucket["maximum_favorable_movement"],
            20.0,
        )

    def test_incomplete_twenty_candle_lookahead_is_excluded(self):
        result = self.study.analyze_evaluations(
            [
                make_evaluation(1, 4, 80),
                make_evaluation(2, 5, 80),
            ],
            self.candles,
        )

        self.assertEqual(result["evaluations_recorded"], 2)
        self.assertEqual(result["valid_evaluations"], 1)
        self.assertEqual(
            result["buckets"]["80-89"]["evaluations"],
            1,
        )

    def test_single_period_regime_is_marked_as_exploratory(self):
        period_analysis = self.study.analyze_evaluations(
            [make_evaluation(1, 0, 80)],
            self.candles,
        )
        aggregate = self.study.aggregate_periods([
            {
                "period": "Period D",
                "regime": "Sideways",
                **period_analysis,
            }
        ])

        self.assertTrue(aggregate["insufficient_period_coverage"])
        self.assertTrue(
            aggregate["buckets"]["80-89"]["insufficient_evidence"],
        )

    def test_selector_preserves_all_fixed_regime_periods(self):
        source_candles = []
        for specification in SCORE_STUDY_PERIODS:
            ending_price = {
                "Bull": 130.0,
                "Bear": 80.0,
                "Sideways": 105.0,
            }[specification["regime"]]
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
                        int((start + timedelta(days=index)).timestamp()),
                        close,
                    )
                )

        selected = select_score_study_periods(source_candles)

        self.assertEqual(len(selected), len(SCORE_STUDY_PERIODS))
        self.assertEqual(
            [period["regime"] for period in selected],
            [specification["regime"] for specification in SCORE_STUDY_PERIODS],
        )


if __name__ == "__main__":
    unittest.main()