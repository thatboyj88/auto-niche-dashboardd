import os
import json
import tempfile
import unittest
from unittest.mock import patch

from dashboard import load_live_observation_status
from config import (
    LIVE_TRADING,
    PAPER_TRADING,
    STARTING_CAPITAL,
    STOP_LOSS_PERCENT,
    TAKE_PROFIT_PERCENT,
)
from incremental_paper_engine import (
    FEE_PERCENT,
    SLIPPAGE_PERCENT,
    IncrementalPaperEngine,
    IncrementalPaperEngineError,
)
from observation_store import ObservationStore
from paper_observation_adapter import PaperObservationAdapter


def candle(timestamp, close, volume=100):
    return {
        "timestamp": timestamp,
        "open": close,
        "high": close * 1.001,
        "low": close * 0.999,
        "close": close,
        "volume": volume,
    }


class IncrementalPaperEngineTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        store = ObservationStore(os.path.join(self.temp_dir.name, "obs.jsonl"))
        adapter = PaperObservationAdapter(store)
        self.engine = IncrementalPaperEngine(
            adapter=adapter,
            state_path=os.path.join(self.temp_dir.name, "state.json"),
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_initialization_arms_without_backfilling_observations(self):
        candles = [candle(index * 3600, 100) for index in range(1, 205)]

        status = self.engine.initialize(candles)

        self.assertEqual(status["status"], "ARMED")
        self.assertEqual(status["genuine_signals"], 0)
        self.assertEqual(self.engine.adapter.store.read_records(), [])

    def test_invalid_restored_open_position_fails_closed(self):
        state_path = os.path.join(self.temp_dir.name, "invalid-state.json")
        with open(state_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "status": "RUNNING",
                    "capital": 20.0,
                    "position": 0.001,
                    "entry_price": 0.0,
                },
                handle,
            )
        with self.assertRaises(IncrementalPaperEngineError):
            IncrementalPaperEngine(
                adapter=self.engine.adapter,
                state_path=state_path,
            )

    def test_unhealthy_market_data_skips_cycle(self):
        candles = [candle(index * 3600, 100) for index in range(1, 205)]
        self.engine.initialize(candles)
        candles.append(candle(205 * 3600, 100))

        events = self.engine.process(candles, data_health="DEGRADED")

        self.assertEqual(events, [])
        self.assertEqual(self.engine.status()["status"], "WAITING_FOR_HEALTHY_DATA")
        self.assertEqual(self.engine.adapter.store.read_records(), [])

    def test_restart_restores_paper_state(self):
        candles = [candle(index * 3600, 100) for index in range(1, 205)]
        self.engine.initialize(candles)
        restarted = IncrementalPaperEngine(
            adapter=self.engine.adapter,
            state_path=self.engine.state_path,
        )

        self.assertEqual(restarted.status()["started_at"], self.engine.status()["started_at"])
        self.assertEqual(restarted.status()["cash"], STARTING_CAPITAL)
        self.assertEqual(restarted.status()["position"], 0.0)

    def test_invalid_or_live_configuration_fails_closed(self):
        self.assertTrue(PAPER_TRADING)
        self.assertFalse(LIVE_TRADING)
        with self.assertRaises(IncrementalPaperEngineError):
            IncrementalPaperEngine(
                adapter=self.engine.adapter,
                state_path=self.engine.state_path,
                starting_capital=100,
            )

    def test_frozen_execution_constants_are_unchanged(self):
        self.assertEqual(STARTING_CAPITAL, 25.00)
        self.assertEqual(STOP_LOSS_PERCENT, 0.02)
        self.assertEqual(TAKE_PROFIT_PERCENT, 0.04)
        self.assertEqual(FEE_PERCENT, 0.004)
        self.assertEqual(SLIPPAGE_PERCENT, 0.001)

    @patch(
        "incremental_paper_engine.calculate_strategy_score",
        return_value=(
            80,
            "BUY CANDIDATE",
            ["test-only isolated strategy result"],
            {
                "long_term_trend": True,
                "short_term_momentum": True,
                "rsi": True,
                "volume": True,
                "price_above_ema21": True,
            },
        ),
    )
    def test_isolated_engine_step_records_signal_and_completed_trade(self, _score):
        candles = [candle(index * 3600, 100) for index in range(1, 205)]
        self.engine.initialize(candles)
        entry_candle = candle(205 * 3600, 100)
        target_candle = candle(206 * 3600, 105)

        entry_events = self.engine.process(
            candles + [entry_candle],
            data_health="HEALTHY",
        )
        trade_events = self.engine.process(
            candles + [entry_candle, target_candle],
            data_health="HEALTHY",
        )

        self.assertEqual([event["type"] for event in entry_events], ["SIGNAL"])
        self.assertEqual(
            [event["type"] for event in trade_events],
            ["SIGNAL", "TRADE"],
        )
        self.assertEqual(self.engine.status()["genuine_signals"], 2)
        self.assertEqual(self.engine.status()["genuine_completed_trades"], 1)
        self.assertEqual(
            len(self.engine.adapter.store.read_records(dataset="PAPER_OPERATIONAL")),
            3,
        )

    @patch(
        "incremental_paper_engine.calculate_strategy_score",
        return_value=(
            80,
            "HOLD",
            ["test-only isolated strategy result"],
            {},
        ),
    )
    def test_runner_crash_recovery_keeps_signal_and_engine_state_paired(self, _score):
        candles = [candle(index * 3600, 100) for index in range(1, 205)]
        self.engine.initialize(candles)

        def crash(point):
            if point == "after_evidence":
                raise RuntimeError("injected process crash")

        self.engine.adapter.store._test_transaction_failpoint = crash
        next_candle = candle(205 * 3600, 100)
        with self.assertRaisesRegex(RuntimeError, "injected process crash"):
            self.engine.process(candles + [next_candle], data_health="HEALTHY")

        restarted = IncrementalPaperEngine(
            adapter=PaperObservationAdapter(ObservationStore(
                os.path.join(self.temp_dir.name, "obs.jsonl")
            )),
            state_path=self.engine.state_path,
        )
        self.assertEqual(restarted.status()["genuine_signals"], 1)
        self.assertEqual(restarted.status()["last_processed_timestamp"], 205 * 3600)
        self.assertEqual(
            len(restarted.adapter.store.read_records(dataset="PAPER_OPERATIONAL")),
            1,
        )

    def test_engine_state_fsync_failure_does_not_create_partial_state(self):
        with patch("incremental_paper_engine.os.fsync", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                self.engine._save_state()

        self.assertFalse(os.path.exists(self.engine.state_path))
        self.assertFalse(os.path.exists(str(self.engine.state_path) + ".tmp"))
        restarted = IncrementalPaperEngine(
            adapter=self.engine.adapter, state_path=self.engine.state_path
        )
        self.assertEqual(restarted.status()["status"], "STOPPED")

    def test_engine_state_write_failure_does_not_create_partial_state(self):
        with patch(
            "incremental_paper_engine.json.dump",
            side_effect=OSError("disk full"),
        ):
            with self.assertRaises(OSError):
                self.engine._save_state()

        self.assertFalse(os.path.exists(self.engine.state_path))
        self.assertFalse(os.path.exists(str(self.engine.state_path) + ".tmp"))
        restarted = IncrementalPaperEngine(
            adapter=self.engine.adapter, state_path=self.engine.state_path
        )
        self.assertEqual(restarted.status()["status"], "STOPPED")

    def test_engine_state_replace_failure_preserves_previous_state(self):
        self.engine.initialize([candle(index * 3600, 100) for index in range(1, 205)])
        with open(self.engine.state_path, encoding="utf-8") as handle:
            committed = json.load(handle)

        self.engine.state["status"] = "RUNNING"
        with patch(
            "incremental_paper_engine.os.replace",
            side_effect=OSError("disk full"),
        ):
            with self.assertRaises(OSError):
                self.engine._save_state()

        with open(self.engine.state_path, encoding="utf-8") as handle:
            self.assertEqual(json.load(handle), committed)
        self.assertFalse(os.path.exists(str(self.engine.state_path) + ".tmp"))
        restarted = IncrementalPaperEngine(
            adapter=self.engine.adapter, state_path=self.engine.state_path
        )
        self.assertEqual(restarted.status()["status"], "ARMED")

    @patch(
        "incremental_paper_engine.calculate_strategy_score",
        return_value=(
            80,
            "HOLD",
            ["test-only isolated strategy result"],
            {},
        ),
    )
    def test_storage_outage_survives_dashboard_reload_until_durable_transition(
        self, _score
    ):
        """Operators must see the outage until a later commit actually succeeds."""
        data_dir = self.temp_dir.name
        controller_path = os.path.join(data_dir, "observation_controller.json")
        observation_path = os.path.join(data_dir, "obs.jsonl")
        with open(controller_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "status": "RUNNING",
                    "started_at": "2026-08-23T00:00:00+00:00",
                    "last_cycle_at": "2026-08-23T00:00:00+00:00",
                    "last_data_health": "HEALTHY",
                    "cycles": 1,
                    "healthy_cycles": 1,
                    "unhealthy_cycles": 0,
                },
                handle,
            )

        candles = [candle(index * 3600, 100) for index in range(1, 205)]
        self.engine.initialize(candles)
        next_candle = candle(205 * 3600, 100)
        write_json_fsync = ObservationStore._write_json_fsync
        with patch.object(
            ObservationStore,
            "_write_json_fsync",
            side_effect=[OSError("disk full"), write_json_fsync],
        ):
            with self.assertRaisesRegex(
                IncrementalPaperEngineError,
                "evidence storage is unavailable",
            ):
                self.engine.process(
                    candles + [next_candle],
                    data_health="HEALTHY",
                )

        env = {
            "OBSERVATION_DATA_DIR": data_dir,
            "OBSERVATION_CONTROLLER_STATE_PATH": controller_path,
            "PAPER_ENGINE_STATE_PATH": str(self.engine.state_path),
            "OBSERVATION_STORE_PATH": observation_path,
        }
        with patch.dict(os.environ, env, clear=False):
            outage_status = load_live_observation_status()

        self.assertEqual(
            outage_status["paper_storage"]["status"],
            "UNAVAILABLE",
        )
        self.assertEqual(
            outage_status["paper_storage"]["error_code"],
            "PAPER_EVIDENCE_STORAGE_UNAVAILABLE",
        )
        self.assertEqual(
            outage_status["paper_storage"]["last_error"],
            "disk full",
        )

        restarted = IncrementalPaperEngine(
            adapter=PaperObservationAdapter(ObservationStore(observation_path)),
            state_path=self.engine.state_path,
        )
        self.assertEqual(
            restarted.status()["persistence_health"]["status"],
            "UNAVAILABLE",
        )

        with patch.dict(os.environ, env, clear=False):
            reloaded_outage_status = load_live_observation_status()
        self.assertEqual(
            reloaded_outage_status["paper_storage"]["error_code"],
            "PAPER_EVIDENCE_STORAGE_UNAVAILABLE",
        )

        restarted.process(
            candles + [next_candle],
            data_health="HEALTHY",
        )

        with patch.dict(os.environ, env, clear=False):
            recovered_status = load_live_observation_status()
        self.assertEqual(recovered_status["paper_storage"]["status"], "HEALTHY")
        self.assertIsNone(recovered_status["paper_storage"]["error_code"])
        self.assertIsNone(recovered_status["paper_storage"]["last_error"])
