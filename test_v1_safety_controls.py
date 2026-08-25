import unittest
from unittest.mock import patch

from config import (
    FEE_PERCENT,
    MAX_DAILY_LOSS_PERCENT,
    MAX_POSITION_PERCENT,
    MAX_TRADES_PER_DAY,
    MIN_STRATEGY_SCORE,
    STARTING_CAPITAL,
    SLIPPAGE_PERCENT,
    STOP_LOSS_PERCENT,
    TAKE_PROFIT_PERCENT,
)
from dashboard import STARTING_CAPITAL as DASHBOARD_STARTING_CAPITAL
from live_market_backtest import STARTING_CAPITAL as LIVE_STARTING_CAPITAL
from risk_manager import risk_check
from strategy_backtest import StrategyBacktester


def make_candle(timestamp, close=100.0, volume=1000.0):
    return {
        "timestamp": timestamp,
        "open": close,
        "high": close * 1.01,
        "low": close * 0.99,
        "close": close,
        "volume": volume,
    }


class V1SafetyControlTests(unittest.TestCase):
    def test_operational_paths_share_canonical_starting_capital(self):
        self.assertEqual(DASHBOARD_STARTING_CAPITAL, STARTING_CAPITAL)
        self.assertEqual(LIVE_STARTING_CAPITAL, STARTING_CAPITAL)

    def test_backtester_uses_epoch_day_buckets(self):
        candles = [
            make_candle(1704067200 + (index * 3600))
            for index in range(202)
        ]
        backtester = StrategyBacktester(STARTING_CAPITAL)
        backtester.run(candles)
        self.assertEqual(
            backtester.current_day,
            candles[-1]["timestamp"] // 86400,
        )

    def test_risk_manager_uses_canonical_score_threshold(self):
        with patch("risk_manager.MIN_STRATEGY_SCORE", MIN_STRATEGY_SCORE + 1):
            allowed, reason = risk_check(
                capital=STARTING_CAPITAL,
                daily_loss=0.0,
                trades_today=0,
                strategy_score=MIN_STRATEGY_SCORE,
                entry_price=100.0,
            )
        self.assertFalse(allowed)
        self.assertEqual(reason, "Strategy score is below minimum.")

    def test_configured_exit_and_position_controls_are_the_v1_values(self):
        self.assertEqual(STOP_LOSS_PERCENT, 0.02)
        self.assertEqual(TAKE_PROFIT_PERCENT, 0.04)
        self.assertEqual(MAX_POSITION_PERCENT, 0.40)
        self.assertEqual(MAX_DAILY_LOSS_PERCENT, 0.03)
        self.assertEqual(MAX_TRADES_PER_DAY, 3)

    def test_execution_costs_are_frozen_in_the_canonical_policy(self):
        self.assertEqual(FEE_PERCENT, 0.004)
        self.assertEqual(SLIPPAGE_PERCENT, 0.001)
        backtester = StrategyBacktester(STARTING_CAPITAL)
        self.assertEqual(backtester.fee_percent, FEE_PERCENT)
        self.assertEqual(backtester.slippage_percent, SLIPPAGE_PERCENT)

    def test_daily_loss_uses_start_of_day_basis_when_supplied(self):
        allowed, reason = risk_check(
            capital=24.5,
            daily_loss=0.75,
            trades_today=0,
            strategy_score=MIN_STRATEGY_SCORE,
            entry_price=100.0,
            daily_starting_capital=STARTING_CAPITAL,
        )
        self.assertFalse(allowed)
        self.assertEqual(reason, "Daily loss limit reached.")