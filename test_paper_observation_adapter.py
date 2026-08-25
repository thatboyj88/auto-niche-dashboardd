import os
import tempfile
import unittest

from observation_store import PAPER_OPERATIONAL, SIGNAL, TRADE, ObservationStore
from paper_observation_adapter import (
    PaperObservationAdapter,
    PaperObservationValidationError,
)


class PaperObservationAdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.adapter = PaperObservationAdapter(
            ObservationStore(os.path.join(self.temp_dir.name, "observations.jsonl"))
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_records_genuine_signal_with_operational_dataset(self):
        record = self.adapter.record_signal(
            signal_id="signal-1",
            observed_at="2026-08-21T10:00:00+00:00",
            symbol="BTC/CAD",
            strategy_score=82.5,
            entry_eligible=True,
            market_data_timestamp="2026-08-21T09:59:00+00:00",
            data_health="healthy",
        )

        self.assertEqual(record["dataset"], PAPER_OPERATIONAL)
        self.assertEqual(record["record_type"], SIGNAL)
        self.assertEqual(record["payload"]["market_condition"], "UNAVAILABLE")
        self.assertIsNone(record["payload"]["max_drawdown_percent"])
        self.assertEqual(len(self.adapter.store.read_records()), 1)

    def test_records_optional_risk_and_condition_context(self):
        record = self.adapter.record_signal(
            signal_id="signal-context",
            observed_at="2026-08-21T10:00:00+00:00",
            symbol="BTC/CAD",
            strategy_score=82.5,
            entry_eligible=True,
            market_data_timestamp="2026-08-21T09:59:00+00:00",
            data_health="healthy",
            market_condition="Bull",
            market_condition_detail="Strong Bull",
            drawdown_percent=1.25,
            max_drawdown_percent=2.5,
        )

        self.assertEqual(record["payload"]["market_condition"], "Bull")
        self.assertEqual(record["payload"]["market_condition_detail"], "Strong Bull")
        self.assertEqual(record["payload"]["max_drawdown_percent"], 2.5)

    def test_records_completed_trade_without_changing_account_state(self):
        record = self.adapter.record_trade(
            trade_id="trade-1",
            signal_id="signal-1",
            entry_at="2026-08-21T10:00:00+00:00",
            exit_at="2026-08-21T12:00:00+00:00",
            entry_price=90000,
            exit_price=91800,
            profit_loss=0.42,
            fees=0.08,
            slippage=0.01,
            exit_reason="take_profit",
        )

        self.assertEqual(record["dataset"], PAPER_OPERATIONAL)
        self.assertEqual(record["record_type"], TRADE)
        self.assertEqual(record["payload"]["exit_reason"], "take_profit")

    def test_duplicate_signal_and_trade_events_are_idempotent(self):
        kwargs = {
            "signal_id": "signal-1",
            "observed_at": "2026-08-21T10:00:00+00:00",
            "symbol": "BTC/CAD",
            "strategy_score": 82.5,
            "entry_eligible": False,
            "market_data_timestamp": "2026-08-21T09:59:00+00:00",
            "data_health": "healthy",
        }
        first = self.adapter.record_signal(**kwargs)
        second = self.adapter.record_signal(**kwargs)

        self.assertEqual(first, second)
        self.assertEqual(len(self.adapter.store.read_records()), 1)

    def test_invalid_event_data_fails_closed(self):
        with self.assertRaises(PaperObservationValidationError):
            self.adapter.record_signal(
                signal_id="signal-1",
                observed_at="not-a-timestamp",
                symbol="BTC/CAD",
                strategy_score=82.5,
                entry_eligible=True,
                market_data_timestamp="2026-08-21T09:59:00+00:00",
                data_health="healthy",
            )
        with self.assertRaises(PaperObservationValidationError):
            self.adapter.record_trade(
                trade_id="trade-1",
                signal_id="signal-1",
                entry_at="2026-08-21T10:00:00+00:00",
                exit_at="2026-08-21T12:00:00+00:00",
                entry_price=float("nan"),
                exit_price=91800,
                profit_loss=0.42,
                fees=0.08,
                slippage=0.01,
                exit_reason="take_profit",
            )
        with self.assertRaises(PaperObservationValidationError):
            self.adapter.record_trade(
                trade_id="trade-2",
                signal_id="signal-1",
                entry_at="2026-08-21T12:00:00+00:00",
                exit_at="2026-08-21T10:00:00+00:00",
                entry_price=90000,
                exit_price=89000,
                profit_loss=-0.1,
                fees=0.08,
                slippage=0.01,
                exit_reason="stop_loss",
            )

    def test_adapter_does_not_create_records_until_called(self):
        self.assertEqual(self.adapter.store.read_records(), [])