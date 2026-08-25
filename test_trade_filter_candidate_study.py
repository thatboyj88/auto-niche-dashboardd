import unittest

from trade_filter_candidate_study import TradeFilterCandidateStudy


def item(
    entry_candle,
    score=80,
    rsi=55,
    expected=1.0,
    historical_mfe=1.0,
    ratio=1.0,
    gross=1.0,
):
    return {
        "period": "Period A",
        "entry_candle": entry_candle,
        "entry_score": score,
        "entry_rsi": rsi,
        "expected_movement_percent": expected,
        "historical_mfe_percent": historical_mfe,
        "break_even_distance_percent": expected - 1.005,
        "projected_reward_cost_ratio": ratio,
        "current_net": gross - 0.25,
        "trade": {
            "gross_profit_loss_before_costs": gross,
            "fees": 0.20,
            "estimated_slippage": 0.05,
        },
    }


class TradeFilterCandidateStudyTests(unittest.TestCase):
    def setUp(self):
        self.study = TradeFilterCandidateStudy()

    def test_each_filter_applies_its_declared_measure(self):
        high = item(
            1,
            score=90,
            rsi=65,
            expected=2.0,
            historical_mfe=3.0,
            ratio=2.0,
        )
        low = item(2)

        self.assertEqual(
            len(self.study.select("minimum_score_85", [high, low])),
            1,
        )
        self.assertEqual(
            len(self.study.select("minimum_rsi_60", [high, low])),
            1,
        )
        self.assertEqual(
            len(self.study.select("minimum_reward_cost_ratio_1_5", [high, low])),
            1,
        )

    def test_cooldown_keeps_fewer_trades(self):
        trades = [item(1), item(2), item(5)]

        selected = self.study.select("cooldown_3", trades)

        self.assertEqual([trade["entry_candle"] for trade in selected], [1, 5])

    def test_screen_requires_positive_validation_delta(self):
        research = [item(1, score=90, gross=1.0), item(4, score=80, gross=-0.1)]
        validation = [item(1, score=80, gross=0.5)]

        result = self.study.screen_research_validation(research, validation)

        self.assertIn("minimum_score_85", result["research"])
        self.assertFalse(
            result["validation"]["minimum_score_85"]["beats_validation_control"]
        )
        self.assertEqual(
            result["research"]["minimum_score_85"]["status"],
            "REJECTED",
        )

    def test_empty_screen_is_safe(self):
        control, candidates = self.study.screen([])

        self.assertEqual(control["trades"], 0)
        self.assertEqual(candidates["cooldown_3"]["performance"]["trades"], 0)


if __name__ == "__main__":
    unittest.main()