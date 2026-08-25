import unittest

from exit_parameter_robustness_study import (
    ACTIVE_CONTROL_STATUS,
    CONTROL,
    EXIT_GRID,
    STOP_LOSSES,
    TAKE_PROFITS,
)
from config import STOP_LOSS_PERCENT, TAKE_PROFIT_PERCENT


class ExitParameterRobustnessStudyTests(unittest.TestCase):
    def test_grid_is_exactly_predeclared_sixteen_variants(self):
        self.assertEqual(STOP_LOSSES, (1.5, 2.0, 2.5, 3.0))
        self.assertEqual(TAKE_PROFITS, (3.0, 4.0, 5.0, 6.0))
        self.assertEqual(len(EXIT_GRID), 16)
        self.assertEqual(CONTROL, (2.0, 4.0))
        self.assertIn((2.0, 4.0), EXIT_GRID)

    def test_grid_has_no_duplicate_pairs(self):
        self.assertEqual(len(set(EXIT_GRID)), 16)

    def test_control_matches_active_production_configuration(self):
        self.assertEqual(CONTROL, (STOP_LOSS_PERCENT * 100, TAKE_PROFIT_PERCENT * 100))
        self.assertEqual(ACTIVE_CONTROL_STATUS, "ACTIVE_PRODUCTION_CONTROL")


if __name__ == "__main__":
    unittest.main()