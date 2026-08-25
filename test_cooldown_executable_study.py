import unittest

from cooldown_executable_study import COOLDOWNS, _performance


def period(net, gross, fees, slippage, trades, wins):
    return {
        "gross_profit_before_costs": gross,
        "total_fees": fees,
        "total_slippage": slippage,
        "net_profit": net,
        "max_drawdown": 0.4,
        "trades_history": [{} for _ in range(trades)],
        "wins": wins,
        "evaluation_history": [{"decision": "BUY"} for _ in range(trades)],
    }


class CooldownExecutableStudyTests(unittest.TestCase):
    def test_predeclared_cooldowns_are_fixed(self):
        self.assertEqual(COOLDOWNS, (0, 1, 2, 3, 5, 10))

    def test_performance_aggregates_costs_and_drawdown(self):
        result = _performance([period(0.8, 1.2, 0.3, 0.1, 2, 1)])

        self.assertEqual(result["signals"], 2)
        self.assertEqual(result["trades"], 2)
        self.assertAlmostEqual(result["gross"], 1.2)
        self.assertAlmostEqual(result["fees"], 0.3)
        self.assertAlmostEqual(result["slippage"], 0.1)
        self.assertAlmostEqual(result["net"], 0.8)
        self.assertAlmostEqual(result["maximum_drawdown"], 0.4)
        self.assertAlmostEqual(result["win_rate"], 50.0)

    def test_empty_performance_is_safe(self):
        result = _performance([])

        self.assertEqual(result["trades"], 0)
        self.assertEqual(result["net"], 0.0)
        self.assertEqual(result["maximum_drawdown"], 0.0)


if __name__ == "__main__":
    unittest.main()