import unittest

from rsi_candidate_executable_study import _performance


def period(net, gross, fees, slippage, trades, wins, decision_count=1):
    return {
        "period": "Period A",
        "regime": "Bull",
        "gross_profit_before_costs": gross,
        "total_fees": fees,
        "total_slippage": slippage,
        "net_profit": net,
        "max_drawdown": 0.5,
        "trades": trades,
        "wins": wins,
        "trades_history": [{} for _ in range(trades)],
        "evaluation_history": [
            {"decision": "BUY"} for _ in range(decision_count)
        ],
        "return_percent": net / 25 * 100,
    }


class RSIExecutableStudyTests(unittest.TestCase):
    def test_performance_aggregates_independent_periods(self):
        result = _performance([
            period(1.0, 2.0, 0.7, 0.3, 2, 1),
            period(-0.2, 0.5, 0.2, 0.1, 1, 0, decision_count=2),
        ])

        self.assertEqual(result["signals"], 3)
        self.assertEqual(result["trades"], 3)
        self.assertAlmostEqual(result["gross"], 2.5)
        self.assertAlmostEqual(result["fees"], 0.9)
        self.assertAlmostEqual(result["slippage"], 0.4)
        self.assertAlmostEqual(result["net"], 0.8)
        self.assertAlmostEqual(result["maximum_drawdown"], 0.5)
        self.assertAlmostEqual(result["win_rate"], 33.333333, places=4)

    def test_empty_performance_is_safe(self):
        result = _performance([])

        self.assertEqual(result["signals"], 0)
        self.assertEqual(result["trades"], 0)
        self.assertEqual(result["net"], 0.0)
        self.assertEqual(result["maximum_drawdown"], 0.0)


if __name__ == "__main__":
    unittest.main()