import unittest

from rsi_candidate_robustness_study import (
    RSI_THRESHOLDS,
    RSICandidateRobustnessStudy,
)


def trade(number, entry_candle, rsi, gross, net):
    return {
        "trade_number": number,
        "entry_candle": entry_candle,
        "fees": 0.20,
        "estimated_slippage": 0.05,
        "gross_profit_loss_before_costs": gross,
        "net_profit_loss": net,
    }, {
        "candle": entry_candle,
        "decision": "BUY",
        "rsi": rsi,
    }


def period(trades, evaluations):
    return {
        "period": "Period A",
        "trades_history": trades,
        "evaluation_history": evaluations,
    }


class RSICandidateRobustnessStudyTests(unittest.TestCase):
    def test_all_predeclared_thresholds_are_reported(self):
        first_trade, first_eval = trade(1, 1, 56, 1.0, 0.75)
        second_trade, second_eval = trade(2, 5, 66, 1.0, 0.75)
        result = RSICandidateRobustnessStudy().analyze_group([
            period(
                [first_trade, second_trade],
                [first_eval, second_eval],
            )
        ])

        self.assertEqual(tuple(result["thresholds"]), RSI_THRESHOLDS)
        self.assertEqual(result["control"]["trades"], 2)
        self.assertEqual(result["thresholds"][60]["performance"]["trades"], 1)
        self.assertEqual(result["thresholds"][68]["performance"]["trades"], 0)

    def test_thresholds_keep_gross_and_reprice_net(self):
        selected_trade, selected_eval = trade(1, 1, 62, 1.0, 0.75)
        result = RSICandidateRobustnessStudy().analyze_group([
            period([selected_trade], [selected_eval])
        ])
        performance = result["thresholds"][60]["performance"]

        self.assertAlmostEqual(performance["gross"], 1.0)
        self.assertAlmostEqual(performance["fees"], 0.20)
        self.assertAlmostEqual(performance["slippage"], 0.05)
        self.assertAlmostEqual(performance["net"], 0.75)
        self.assertAlmostEqual(performance["win_rate"], 100.0)

    def test_empty_threshold_selection_is_safe(self):
        selected_trade, selected_eval = trade(1, 1, 55, 1.0, 0.75)
        result = RSICandidateRobustnessStudy().analyze_group([
            period([selected_trade], [selected_eval])
        ])
        performance = result["thresholds"][68]["performance"]

        self.assertEqual(performance["trades"], 0)
        self.assertEqual(performance["net"], 0.0)
        self.assertEqual(performance["maximum_drawdown"], 0.0)


if __name__ == "__main__":
    unittest.main()