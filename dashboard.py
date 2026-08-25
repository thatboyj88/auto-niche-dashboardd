import json
import os
import tempfile
from datetime import datetime, timedelta, timezone
from html import escape
from pathlib import Path

import fcntl
import streamlit as st
import streamlit.components.v1 as components

from config import (
    LIVE_TRADING,
    MAX_DAILY_LOSS_PERCENT,
    MAX_POSITION_PERCENT,
    MAX_TRADES_PER_DAY,
    PAPER_TRADING,
    STARTING_CAPITAL,
    STOP_LOSS_PERCENT,
    TAKE_PROFIT_PERCENT,
)
from ai_operations_assistant import (
    UNKNOWN,
    answer_question,
    build_assistant_context,
    format_failure_category,
    format_failure_category_counts,
    get_provider_health,
)
from research_providers import provider_catalog, research_readiness

from generate_test_data import generate_candles
from kraken_live_data import KrakenMarketData
from multi_period_backtest import (
    BEAR_RETURN_PERCENT,
    BULL_RETURN_PERCENT,
    MultiPeriodBacktester,
)
from strategy_backtest import StrategyBacktester
from yahoo_btc_cad_data import YahooBTCADMarketData
from investment_decision import (
    fetch_public_option_quote_candidates,
    review_defined_risk_option_candidates,
)
from observation_controller import (
    ObservationControlError,
    ObservationCriteria,
    apply_paper_control,
)
from observation_store import ObservationStore

KOVA_VOICE_COMPONENT = components.declare_component(
    "kova_voice_assistant",
    path=str(Path(__file__).resolve().parent / "kova_voice_component"),
)

OPTIONS_QUOTE_CACHE_TTL_SECONDS = float(
    os.getenv("OPTIONS_QUOTE_CACHE_TTL_SECONDS", "60")
)
APPEARANCE_OPTIONS = ("Light", "Dark", "System Default")
DASHBOARD_PREFERENCES_PATH = Path(
    os.getenv("KOVA_DASHBOARD_PREFERENCES_PATH", ".data/dashboard_preferences.json")
)


@st.cache_data(ttl=OPTIONS_QUOTE_CACHE_TTL_SECONDS, show_spinner=False)
def fetch_cached_public_option_quote_candidates(symbol):
    """Reuse a short-lived provider snapshot without weakening quote validation."""
    return fetch_public_option_quote_candidates(symbol)


def _option_snapshot_age(snapshot):
    fetched_at = snapshot.get("fetched_at")
    if not fetched_at:
        return "Unknown"
    try:
        fetched = datetime.fromisoformat(fetched_at.replace("Z", "+00:00"))
        if fetched.tzinfo is None:
            fetched = fetched.replace(tzinfo=timezone.utc)
        age_seconds = max(
            0, int((datetime.now(timezone.utc) - fetched).total_seconds())
        )
    except (AttributeError, TypeError, ValueError):
        return "Unknown"
    if age_seconds < 60:
        return f"{age_seconds}s ago"
    return f"{age_seconds // 60}m ago"


CONDITION_LABELS = {
    "long_term_trend": "Long-term trend",
    "short_term_momentum": "Short-term momentum",
    "rsi": "RSI",
    "volume": "Volume",
    "price_above_ema21": "Price above EMA21",
}

ANALYSIS_CONDITION_LABELS = {
    "long_term_trend": "Long-term trend",
    "short_term_momentum": "Short-term momentum",
    "rsi_condition": "RSI",
    "volume": "Volume",
    "price_above_ema21": "Price above EMA21",
}


def run_strategy_backtest():
    candles = generate_candles(1000)
    backtester = StrategyBacktester(STARTING_CAPITAL)
    backtester.run(candles)
    return backtester.results()


def format_market_timestamp(timestamp):
    return datetime.fromtimestamp(
        timestamp,
        tz=timezone.utc
    ).strftime("%Y-%m-%d %H:%M UTC")


def load_kraken_market_data():
    market_data = KrakenMarketData(interval=60)
    candles = market_data.load()
    return market_data, candles


def _read_observation_json(path):
    path = Path(path)
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _authenticated_user_key():
    """Return a stable account identifier, or None for anonymous sessions."""
    try:
        user = st.user
        if not bool(getattr(user, "is_logged_in", False)):
            return None
        for attribute in ("sub", "email"):
            value = getattr(user, attribute, None)
            if value:
                return f"{attribute}:{value}"
    except (AttributeError, RuntimeError, TypeError):
        return None
    return None


def _load_saved_dashboard_appearance():
    user_key = _authenticated_user_key()
    if not user_key:
        return None
    path = DASHBOARD_PREFERENCES_PATH
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
            preferences = json.load(handle)
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(preferences, dict):
        return None
    users = preferences.get("users")
    if not isinstance(users, dict):
        return None
    user_preferences = users.get(user_key)
    if not isinstance(user_preferences, dict):
        return None
    appearance = user_preferences.get("appearance")
    return appearance if appearance in APPEARANCE_OPTIONS else None


def _save_dashboard_appearance_preference():
    user_key = _authenticated_user_key()
    appearance = st.session_state.get("dashboard_appearance")
    if not user_key or appearance not in APPEARANCE_OPTIONS:
        return

    path = DASHBOARD_PREFERENCES_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(f"{path.name}.lock")
    try:
        with lock_path.open("a+", encoding="utf-8") as lock_handle:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            if path.exists():
                with path.open("r", encoding="utf-8") as handle:
                    preferences = json.load(handle)
            else:
                preferences = {}
            if not isinstance(preferences, dict):
                preferences = {}
            users = preferences.setdefault("users", {})
            if not isinstance(users, dict):
                users = {}
                preferences["users"] = users
            users[user_key] = {"appearance": appearance}

            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=path.parent,
                prefix=f".{path.name}.",
                delete=False,
            ) as temporary:
                json.dump(preferences, temporary, separators=(",", ":"))
                temporary.flush()
                os.fsync(temporary.fileno())
                temporary_path = Path(temporary.name)
            os.replace(temporary_path, path)
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return


def _read_observation_controller(path):
    """Read controller state while preserving restore failures for operators."""
    path = Path(path)
    if not path.exists():
        return {}, None
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}, "observation controller state cannot be restored"
    if not isinstance(value, dict):
        return {}, "observation controller state cannot be restored"
    return value, None


def load_live_observation_status():
    """Read-only snapshot of the running V2 observation process."""
    criteria = {
        "min_completed_trades": int(
            os.getenv("OBSERVATION_MIN_COMPLETED_TRADES", "20")
        ),
        "min_observation_days": int(
            os.getenv("OBSERVATION_MIN_DAYS", "7")
        ),
        "max_observation_days": int(
            os.getenv("OBSERVATION_MAX_DAYS", "14")
        ),
        "min_healthy_ratio": float(
            os.getenv("OBSERVATION_MIN_HEALTHY_RATIO", "0.95")
        ),
    }
    data_dir = Path(os.getenv("OBSERVATION_DATA_DIR", ".data"))
    controller_state_path = Path(
        os.getenv(
            "OBSERVATION_CONTROLLER_STATE_PATH",
            str(data_dir / "observation_controller.json"),
        )
    )
    controller, controller_restore_error = _read_observation_controller(
        controller_state_path
    )
    engine = _read_observation_json(
        os.getenv(
            "PAPER_ENGINE_STATE_PATH",
            str(data_dir / "paper_engine_state.json"),
        )
    )
    observation_path = Path(
        os.getenv(
            "OBSERVATION_STORE_PATH",
            str(data_dir / "observations.jsonl"),
        )
    )
    paper_signals = 0
    paper_trades = 0
    store_error = None
    if observation_path.exists():
        try:
            with observation_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    if record.get("dataset") != "PAPER_OPERATIONAL":
                        continue
                    if record.get("record_type") == "SIGNAL":
                        paper_signals += 1
                    elif record.get("record_type") == "TRADE":
                        paper_trades += 1
        except (OSError, json.JSONDecodeError, AttributeError):
            store_error = "Observation store unavailable or malformed."

    if (
        not controller
        and controller_restore_error is None
        and not engine
        and not observation_path.exists()
    ):
        return {
            "available": False,
            "status": "NOT_STARTED",
            "criteria": criteria,
            "store_error": None,
            "paper_storage": {
                "status": "UNKNOWN",
                "error_code": None,
                "last_error": None,
                "operation": None,
            },
        }
    status = (
        "BLOCKED_RESTORE"
        if controller_restore_error
        else controller.get("status", "UNKNOWN")
    )
    started_at = controller.get("started_at")
    last_cycle_at = controller.get("last_cycle_at")
    stale_after = os.getenv("OBSERVATION_STALE_AFTER_SECONDS")
    cycle_is_stale = False
    if status == "RUNNING" and stale_after and last_cycle_at:
        try:
            last_cycle = datetime.fromisoformat(
                last_cycle_at.replace("Z", "+00:00")
            )
            cycle_is_stale = (
                datetime.now(timezone.utc) - last_cycle
            ).total_seconds() > float(stale_after)
        except (TypeError, ValueError):
            cycle_is_stale = True
    observation_days = 0.0
    deadline = None
    if started_at:
        try:
            started = datetime.fromisoformat(
                started_at.replace("Z", "+00:00")
            )
            end = datetime.fromisoformat(
                (last_cycle_at or started_at).replace("Z", "+00:00")
            )
            observation_days = max(
                0.0, (end - started).total_seconds() / 86400
            )
            deadline = started + timedelta(
                days=criteria["max_observation_days"]
            )
        except (TypeError, ValueError):
            pass
    return {
        "available": True,
        "status": status,
        "runner_status": (
            "STALE"
            if cycle_is_stale
            else
            "RUNNING"
            if status in {"NOT_STARTED", "RUNNING"}
            else "BLOCKED_RESTORE"
            if status == "BLOCKED_RESTORE"
            else "STOPPED"
        ),
        "started_at": started_at,
        "last_cycle_at": last_cycle_at,
        "cycle_is_stale": cycle_is_stale,
        "observation_days": observation_days,
        "deadline": deadline.isoformat() if deadline else None,
        "criteria": criteria,
        "last_data_health": controller.get("last_data_health", "UNKNOWN"),
        "cycles": controller.get("cycles", 0),
        "healthy_cycles": controller.get("healthy_cycles", 0),
        "unhealthy_cycles": controller.get("unhealthy_cycles", 0),
        "healthy_ratio": (
            controller.get("healthy_cycles", 0)
            / controller.get("cycles", 1)
            if controller.get("cycles", 0)
            else 0
        ),
        "signals": paper_signals,
        "trades": paper_trades,
        "engine_signals": int(engine.get("genuine_signals", 0)),
        "engine_trades": int(engine.get("genuine_completed_trades", 0)),
        "evidence_reconciled": (
            paper_signals == int(engine.get("genuine_signals", 0))
            and paper_trades == int(engine.get("genuine_completed_trades", 0))
        ),
        "cash": engine.get("capital", STARTING_CAPITAL),
        "position": engine.get("position", 0),
        "entry_price": engine.get("entry_price"),
        "daily_starting_capital": engine.get(
            "daily_starting_capital", engine.get("capital", STARTING_CAPITAL)
        ),
        "trades_today": engine.get("trades_today", 0),
        "last_signal": engine.get("last_signal"),
        "last_completed_trade": engine.get("last_completed_trade"),
        "paper_storage": engine.get(
            "persistence_health",
            {
                "status": "UNKNOWN",
                "error_code": None,
                "last_error": None,
                "operation": None,
            },
        ),
        "store_error": store_error,
        "controller_state_path": str(controller_state_path),
        "controller_restore_error": controller_restore_error,
    }


def _paper_control_paths():
    data_dir = Path(os.getenv("OBSERVATION_DATA_DIR", ".data"))
    return (
        Path(os.getenv(
            "OBSERVATION_CONTROLLER_STATE_PATH",
            str(data_dir / "observation_controller.json"),
        )),
        Path(os.getenv(
            "OBSERVATION_RUNNER_LOCK_PATH",
            str(data_dir / "paper_observation_runner.lock"),
        )),
        Path(os.getenv(
            "OBSERVATION_STORE_PATH",
            str(data_dir / "observations.jsonl"),
        )),
    )


def _paper_control_criteria():
    return ObservationCriteria(
        min_completed_trades=int(os.getenv("OBSERVATION_MIN_COMPLETED_TRADES", "20")),
        min_observation_days=int(os.getenv("OBSERVATION_MIN_DAYS", "7")),
        max_observation_days=int(os.getenv("OBSERVATION_MAX_DAYS", "14")),
        min_healthy_ratio=float(os.getenv("OBSERVATION_MIN_HEALTHY_RATIO", "0.95")),
    )


def _paper_risk_governor(_action):
    """Final gate for controls: paper mode and durable evidence only."""
    if not PAPER_TRADING or LIVE_TRADING:
        return False
    snapshot = load_live_observation_status()
    if snapshot.get("store_error"):
        return False
    storage = snapshot.get("paper_storage") or {}
    if storage.get("status") == "UNAVAILABLE":
        return False
    return snapshot.get("evidence_reconciled", True)


def _apply_authenticated_paper_control(action):
    """Apply a confirmed dashboard control without exposing a runner handle."""
    controller_path, lock_path, store_path = _paper_control_paths()
    stale_after = os.getenv("OBSERVATION_STALE_AFTER_SECONDS")
    return apply_paper_control(
        action,
        authenticated=_authenticated_user_key() is not None,
        confirmed=bool(st.session_state.get(f"confirm_paper_{action.lower()}")),
        risk_governor=_paper_risk_governor,
        criteria=_paper_control_criteria(),
        state_path=controller_path,
        lock_path=lock_path,
        observation_store=ObservationStore(store_path),
        stale_after_seconds=float(stale_after) if stale_after else None,
    )


@st.fragment(run_every="30s")
def render_mc_live_observation_status(compact=False):
    """Read-only V2 status panel with a compact Overview mode."""
    snapshot = load_live_observation_status()
    render_mc_section(
        "GENUINE PAPER OBSERVATION",
        "Observation progress",
    )
    if not snapshot["available"]:
        st.info("V2 observation runner has not started.")
        return
    if snapshot.get("store_error"):
        st.error(snapshot["store_error"])
    blocked_restore = snapshot.get("status") == "BLOCKED_RESTORE"
    if blocked_restore:
        st.error(
            "Observation restore is blocked. Review the controller state file "
            "before restarting the observation runner."
        )
        st.caption("Controller state path")
        st.code(snapshot["controller_state_path"])
        st.caption(
            snapshot.get("controller_restore_error")
            or "The controller state could not be restored."
        )
    if not blocked_restore and not snapshot.get("evidence_reconciled", True):
        st.error(
            "Persisted paper evidence does not reconcile with engine state; "
            "completion is blocked."
        )
    if snapshot.get("cycle_is_stale"):
        st.warning(
            "Observation heartbeat is stale. The dashboard cannot restart the "
            "runner; review the runner workflow before any recovery action."
        )
    if compact:
        render_mc_status(
            "Observation",
            (
                "red"
                if blocked_restore
                else "green"
                if snapshot["status"] in {"RUNNING", "COMPLETED"}
                else "gray"
            ),
            snapshot["status"],
        )
        if blocked_restore:
            return
        render_mc_metrics(
            [
                (
                    "Completed paper trades",
                    f"{snapshot['trades']} / "
                    f"{snapshot['criteria']['min_completed_trades']}",
                ),
                ("Observation days", f"{snapshot['observation_days']:.2f}"),
                ("Healthy-data ratio", f"{snapshot['healthy_ratio'] * 100:.1f}%"),
            ],
            columns=3,
        )
        return
    state = snapshot["status"]
    state_color = (
        "green"
        if state in {"RUNNING", "COMPLETED"}
        else "red"
        if blocked_restore or state.startswith("STOPPED")
        else "gray"
    )
    render_mc_status(
        "Observation runner",
        state_color,
        snapshot.get("runner_status", "UNKNOWN"),
    )
    render_mc_status(
        "Observation controller",
        state_color,
        state,
    )
    if blocked_restore:
        return
    render_mc_status(
        "Market-data health",
        "green" if snapshot["last_data_health"] == "HEALTHY" else "red",
        snapshot["last_data_health"],
    )
    render_mc_metrics(
        [
            ("Genuine signals", str(snapshot["signals"])),
            ("Completed paper trades", str(snapshot["trades"])),
            ("Paper cash", f"${snapshot['cash']:,.2f}"),
            ("Position", f"{snapshot['position']:.8f}"),
            ("Healthy-data ratio", f"{snapshot['healthy_ratio'] * 100:.1f}%"),
            ("Observation cycles", str(snapshot["cycles"])),
            ("Observation days", f"{snapshot['observation_days']:.2f}"),
            (
                "Trade progress",
                f"{snapshot['trades']} / "
                f"{snapshot['criteria']['min_completed_trades']}",
            ),
            (
                "Minimum days",
                f"{snapshot['criteria']['min_observation_days']} days",
            ),
            (
                "Maximum deadline",
                snapshot.get("deadline") or "N/A",
            ),
            (
                "Healthy target",
                f"{snapshot['criteria']['min_healthy_ratio'] * 100:.0f}%",
            ),
            ("Started", snapshot.get("started_at") or "N/A"),
            ("Last cycle", snapshot.get("last_cycle_at") or "N/A"),
        ],
        columns=2,
    )


def run_live_market_backtest(candles):
    if not candles:
        return None

    backtester = StrategyBacktester(
        starting_capital=STARTING_CAPITAL
    )
    backtester.run(candles)
    return backtester.results()


def load_historical_btc_cad_data():
    market_data = YahooBTCADMarketData()
    rolling_candles = market_data.load()
    if not rolling_candles:
        return market_data, []

    rolling_results = MultiPeriodBacktester(
        starting_capital=STARTING_CAPITAL
    ).run(rolling_candles)
    has_sideways_period = bool(
        rolling_results["regime_summary"]["Sideways"]
    )
    sources = [{
        "candles": rolling_candles,
        "label": market_data.ROLLING_SOURCE_LABEL.replace(
            "10-year",
            market_data.data_range,
        ),
        "kind": "rolling",
    }]

    if not has_sideways_period:
        anchored_candles = market_data.load_anchored_sample()
        if anchored_candles:
            sources.append({
                "candles": anchored_candles,
                "label": market_data.ANCHORED_SOURCE_LABEL,
                "kind": "anchored",
            })

    return market_data, sources


def run_historical_market_backtest(candles):
    if not candles:
        return None

    backtester = MultiPeriodBacktester(
        starting_capital=STARTING_CAPITAL
    )
    if isinstance(candles[0], dict) and "timestamp" in candles[0]:
        return backtester.run(candles)
    return backtester.run_sources(candles)


def render_live_market_data(market_data, candles):
    st.header("LIVE MARKET DATA — KRAKEN")
    st.caption(
        "Display-only public Kraken XBT/CAD data. "
        "These candles are not used by the paper strategy backtest."
    )

    if not candles:
        st.error(
            "Kraken market data is unavailable right now. "
            f"{market_data.last_error or 'No candles were returned.'}"
        )
        return

    health = getattr(market_data, "health", {})
    latest_candle = candles[-1]
    previous_candle = candles[-2] if len(candles) > 1 else None
    previous_price = (
        previous_candle["close"]
        if previous_candle is not None
        else None
    )

    if previous_price is not None:
        price_change = latest_candle["close"] - previous_price
        price_change_percent = (
            price_change /
            previous_price
        ) * 100 if previous_price else 0
    else:
        price_change = None
        price_change_percent = None

    with st.container(border=True):
        render_metric("Market", market_data.pair_name or "XBT/CAD")
        render_metric("Exchange", "Kraken")
        render_metric("Timeframe", "60 minutes")
        render_metric("Candles Loaded", str(market_data.count()))
        render_metric(
            "Data Health",
            health.get("status", "UNKNOWN"),
        )
        render_metric(
            "Data Age",
            (
                f"{health['data_age_seconds']:.0f}s"
                if health.get("data_age_seconds") is not None
                else "UNKNOWN"
            ),
        )
        render_metric(
            "Latest Candle Timestamp",
            format_market_timestamp(latest_candle["timestamp"]),
        )
        render_metric(
            "Latest BTC/CAD Price",
            f"${latest_candle['close']:.2f}",
        )
        render_metric(
            "Previous Candle Price",
            (
                f"${previous_price:.2f}"
                if previous_price is not None
                else "N/A"
            ),
        )
        render_metric(
            "Price Change",
            (
                f"${price_change:+.2f}"
                if price_change is not None
                else "N/A"
            ),
        )
        render_metric(
            "Price Change Percentage",
            (
                f"{price_change_percent:+.3f}%"
                if price_change_percent is not None
                else "N/A"
            ),
        )
        render_metric(
            "First Candle Timestamp",
            format_market_timestamp(candles[0]["timestamp"]),
        )
        render_metric(
            "Last Candle Timestamp",
            format_market_timestamp(latest_candle["timestamp"]),
        )

    st.line_chart(
        {
            "XBT/CAD Close": [
                candle["close"]
                for candle in candles
            ]
        },
        height=220,
    )


def render_real_market_backtest(results, market_data):
    st.divider()
    st.header("REAL BTC/CAD PAPER BACKTEST")
    st.success("PAPER SIMULATION — REAL KRAKEN DATA")
    st.caption(
        f"Committed {market_data.pair_name or 'XBT/CAD'} candles from "
        "Kraken, simulated with the existing StrategyBacktester. "
        "This data is not connected to live trading."
    )

    if results is None:
        st.warning(
            "The real-market paper backtest is unavailable because "
            "Kraken candles could not be loaded."
        )
        return

    return_percent = (
        results["profit"] /
        results["starting_capital"]
    ) * 100

    with st.container(border=True):
        render_metric(
            "Starting Capital",
            f"${results['starting_capital']:.2f}",
        )
        render_metric(
            "Ending Capital",
            f"${results['ending_capital']:.4f}",
        )
        render_metric(
            "Profit",
            f"${results['profit']:.4f}",
        )
        render_metric(
            "Return",
            f"{return_percent:.2f}%",
        )

    st.subheader("Trade Results")
    with st.container(border=True):
        render_metric("Trades", str(results["trades"]))
        render_metric("Wins", str(results["wins"]))
        render_metric("Losses", str(results["losses"]))
        render_metric(
            "Win Rate",
            f"{results['win_rate']:.2f}%",
        )

    st.subheader("Risk & Costs")
    with st.container(border=True):
        render_metric(
            "Maximum Drawdown",
            f"{results['max_drawdown']:.2f}%",
        )
        render_metric(
            "Fees",
            f"${results['total_fees']:.4f}",
        )
        render_metric(
            "Estimated Slippage",
            f"${results['total_slippage']:.4f}",
        )

    st.subheader("Strategy Results")
    with st.container(border=True):
        render_metric(
            "Strategy Evaluations",
            str(results["evaluations"]),
        )
        render_metric(
            "Highest Strategy Score",
            f"{results['highest_score']}/100",
        )
        render_metric(
            "Scores >=80",
            str(results["score_80_or_more"]),
        )
        render_metric(
            "Lowest RSI",
            f"{results['lowest_rsi']:.2f}",
        )
        render_metric(
            "Highest RSI",
            f"{results['highest_rsi']:.2f}",
        )

    st.subheader("Strategy-Condition Counters")
    st.caption(
        f"Conditions passed during {results['evaluations']} "
        "real-market strategy evaluations."
    )

    for key, label in CONDITION_LABELS.items():
        render_metric(
            label,
            f"{results['condition_counts'][key]}/{results['evaluations']}",
        )


def render_historical_market_backtest(results, market_data):
    st.divider()
    st.header("HISTORICAL BTC/CAD REGIME BACKTEST")
    st.success("PAPER SIMULATION — YAHOO FINANCE DAILY DATA")
    st.caption(
        "Public aggregated BTC/CAD daily OHLCV data from Yahoo Finance. "
        "The rolling public-history window is evaluated first. If it no "
        "longer contains a completed Sideways period, the dashboard "
        "re-fetches a separate, date-anchored completed sample. "
        "It is not Kraken-specific data and is not connected to live trading."
    )

    if results is None:
        st.warning(
            "The historical multi-period backtest is unavailable because "
            "Yahoo Finance data could not be loaded. "
            f"{market_data.last_error or ''}"
        )
        return

    st.caption(
        f"Method: each complete, independent {results['period_candles']}-"
        "candle period starts at $25.00. Regime labels use only that "
        f"period's BTC/CAD return: Bull ≥ {BULL_RETURN_PERCENT:.0f}%, "
        f"Bear ≤ {BEAR_RETURN_PERCENT:.0f}%, otherwise Sideways. "
        "The labels do not affect strategy decisions."
    )

    with st.container(border=True):
        render_metric("Source", "Yahoo Finance (aggregated)")
        render_metric("Market", "BTC/CAD")
        render_metric("Timeframe", "Daily")
        render_metric("Historical Candles Loaded", str(market_data.count()))
        render_metric(
            "Completed Test Periods",
            str(len(results["periods"])),
        )

    if not results["periods"]:
        st.warning(
            "No complete 365-candle historical periods are available yet."
        )
        return

    st.subheader("Period Results")
    st.dataframe(
        [
            {
                "Period": period["period"],
                "Source": period.get(
                    "source_label",
                    "Historical daily data",
                ),
                "Dates": (
                    f"{period['start_date']} → {period['end_date']}"
                ),
                "Candles": period["candle_count"],
                "Regime": period["regime"],
                "Market Return": f"{period['market_return']:+.2f}%",
                "Starting": f"${period['starting_capital']:.2f}",
                "Ending": f"${period['ending_capital']:.4f}",
                "Gross P/L": (
                    f"${period['gross_profit_before_costs']:+.4f}"
                ),
                "Fees": f"${period['total_fees']:.4f}",
                "Slippage": f"${period['total_slippage']:.4f}",
                "Net P/L": f"${period['net_profit']:+.4f}",
                "Return": f"{period['return_percent']:+.2f}%",
                "Trades": period["trades"],
                "Win Rate": f"{period['win_rate']:.2f}%",
                "Drawdown": f"{period['max_drawdown']:.2f}%",
            }
            for period in results["periods"]
        ],
        hide_index=True,
        width="stretch",
    )

    st.subheader("Data Sources")
    for source in results.get("sources", []):
        source_type = (
            "rolling-window"
            if source["kind"] == "rolling"
            else "anchored supplemental"
        )
        st.caption(
            f"{source_type}: {source['label']} · "
            f"{source['candle_count']} candles · "
            f"{source['period_count']} complete independent $25 periods"
        )
    if (
        market_data.last_anchored_error
        and not any(
            source["kind"] == "anchored"
            for source in results.get("sources", [])
        )
    ):
        st.warning(
            "The rolling window had no completed Sideways period, and the "
            "anchored supplemental sample could not be loaded: "
            f"{market_data.last_anchored_error}"
        )

    regime_summary = results["regime_summary"]
    observed_regimes = {
        regime
        for regime, periods in regime_summary.items()
        if periods
    }
    sideways_periods = regime_summary["Sideways"]

    st.subheader("Regime Coverage")
    if sideways_periods:
        documented_periods = "; ".join(
            (
                f"{period['period']} "
                f"({period['start_date']} → {period['end_date']}, "
                f"{period['market_return']:+.2f}% market return)"
            )
            for period in sideways_periods
        )
        st.success(
            "Sideways evidence included: "
            f"{documented_periods}. Each period started independently at "
            "$25.00; source labels distinguish rolling from anchored data."
        )
    else:
        st.info(
            "No completed Sideways period was found in the available "
            "Yahoo Finance sources. Do not treat sideways conditions as "
            "tested yet."
        )

    missing_regimes = [
        regime
        for regime in ("Bull", "Bear", "Sideways")
        if regime not in observed_regimes
    ]
    if missing_regimes:
        st.info(
            "No completed historical period met the predeclared "
            f"{', '.join(missing_regimes)} classification in this dataset. "
            "Do not treat that market environment as tested yet."
        )

    st.subheader("Period Diagnostics")
    for period in results["periods"]:
        with st.expander(
            f"{period['period']} · {period['regime']} · "
            f"{period['start_date']} to {period['end_date']}"
        ):
            st.caption(
                "Cost breakdown separates market-price gross P/L from "
                "modeled fees, estimated slippage, and net paper P/L."
            )
            render_metric(
                "Gross Trading P/L Before Fees/Slippage",
                f"${period['gross_profit_before_costs']:+.4f}",
            )
            render_metric(
                "Total Fees",
                f"${period['total_fees']:.4f}",
            )
            render_metric(
                "Estimated Slippage",
                f"${period['total_slippage']:.4f}",
            )
            render_metric(
                "Net P/L",
                f"${period['net_profit']:+.4f}",
            )

            st.subheader("Strategy Activity")
            render_metric(
                "Strategy Evaluations",
                str(period["evaluations"]),
            )
            render_metric(
                "Highest Strategy Score",
                f"{period['highest_score']}/100",
            )
            render_metric(
                "Scores >=80",
                str(period["score_80_or_more"]),
            )

            st.subheader("Strategy-Condition Counters")
            for key, label in CONDITION_LABELS.items():
                render_metric(
                    label,
                    (
                        f"{period['condition_counts'][key]}/"
                        f"{period['evaluations']}"
                    ),
                )

            st.subheader("Completed Trade Audit")
            if not period["trades_history"]:
                st.info(
                    "No trades were placed in this independent period."
                )

            for trade in reversed(period["trades_history"]):
                with st.container(border=True):
                    duration_candles = (
                        trade["exit_candle"] -
                        trade["entry_candle"]
                    )
                    duration_days = (
                        trade["exit_timestamp"] -
                        trade["entry_timestamp"]
                    ) / 86400
                    st.caption(
                        f"Trade {trade['trade_number']} · "
                        f"{duration_candles} candles / "
                        f"{duration_days:.1f} days"
                    )
                    render_metric(
                        "Entry Price",
                        f"${trade['entry_price']:.4f}",
                    )
                    render_metric(
                        "Exit Price",
                        f"${trade['exit_price']:.4f}",
                    )
                    render_metric(
                        "Entry Score",
                        f"{trade['strategy_score']}/100",
                    )
                    render_metric(
                        "RSI at Entry",
                        f"{trade['rsi_at_entry']:.2f}",
                    )
                    render_metric(
                        "Exit Reason",
                        trade["reason"],
                    )
                    render_metric(
                        "Gross P/L Before Fees/Slippage",
                        (
                            "$"
                            f"{trade['gross_profit_loss_before_costs']:+.4f}"
                        ),
                    )
                    render_metric(
                        "Fees",
                        f"${trade['fees']:.4f}",
                    )
                    render_metric(
                        "Estimated Slippage",
                        f"${trade['estimated_slippage']:.4f}",
                    )
                    render_metric(
                        "Net P/L",
                        f"${trade['net_profit_loss']:+.4f}",
                    )

    observed_regimes = {
        period["regime"]
        for period in results["periods"]
    }
    missing_regimes = [
        regime
        for regime in ("Bull", "Bear", "Sideways")
        if regime not in observed_regimes
    ]
    if missing_regimes:
        st.info(
            "No completed historical period met the predeclared "
            f"{', '.join(missing_regimes)} classification in this dataset. "
            "Do not treat that market environment as tested yet."
        )

    aggregate = results["aggregate"]
    best_period = aggregate["best_period"]
    worst_period = aggregate["worst_period"]

    st.subheader("Robustness Summary")
    st.caption(
        "Total return is the sum of independent $25 period returns; it is "
        "not a compounded account-equity result."
    )

    with st.container(border=True):
        render_metric(
            "Total Return",
            f"{aggregate['total_return']:+.2f}%",
        )
        render_metric(
            "Average Return",
            f"{aggregate['average_return']:+.2f}%",
        )
        render_metric(
            "Best Period",
            (
                f"{best_period['period']} "
                f"({best_period['return_percent']:+.2f}%)"
            ),
        )
        render_metric(
            "Worst Period",
            (
                f"{worst_period['period']} "
                f"({worst_period['return_percent']:+.2f}%)"
            ),
        )
        render_metric(
            "Average Trades",
            f"{aggregate['average_trades']:.2f}",
        )
        render_metric(
            "Average Win Rate",
            f"{aggregate['average_win_rate']:.2f}%",
        )
        render_metric(
            "Total Gross P/L Before Fees/Slippage",
            f"${aggregate['total_gross_profit_before_costs']:+.4f}",
        )
        render_metric(
            "Total Fees",
            f"${aggregate['total_fees']:.4f}",
        )
        render_metric(
            "Total Estimated Slippage",
            f"${aggregate['total_slippage']:.4f}",
        )
        render_metric(
            "Worst Drawdown",
            f"{aggregate['worst_drawdown']:.2f}%",
        )
def render_metric(label, value, help_text=None):
    tooltip = (
        f' title="{escape(help_text)}"'
        if help_text
        else ""
    )
    st.markdown(
        f'<div class="mc-data-metric"{tooltip}><div class="mc-data-label">'
        f'{escape(str(label))}</div><div class="mc-data-value">'
        f'{escape(str(value))}</div></div>',
        unsafe_allow_html=True,
    )


def inject_mission_control_theme(appearance="Dark"):
    """Apply the dashboard palette without changing the trading surface."""
    light_theme = """
        :root {
            --mc-bg: #f4f7fb;
            --mc-panel: #ffffff;
            --mc-panel-2: #eef3fa;
            --mc-line: rgba(23, 40, 72, .16);
            --mc-text: #16233d;
            --mc-muted: #52627d;
        }
        .stApp {
            background:
                radial-gradient(circle at 0% 8%, rgba(117, 81, 255, .10) 0%, transparent 26rem),
                radial-gradient(circle at 100% 4%, rgba(1, 190, 254, .10) 0%, transparent 29rem),
                var(--mc-bg);
            color: var(--mc-text);
        }
        [data-testid="stHeader"]::before { color: #16233d; }
        [data-testid="stHeader"]::after { color: #52627d; }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #ffffff, #edf3fb);
            border-right-color: rgba(80, 103, 145, .22);
        }
        [data-testid="stSidebar"] [data-testid="stRadio"] label { color: #52627d; }
        [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
            background: rgba(117, 81, 255, .12);
            color: #39258f;
        }
        [data-testid="stSidebar"] [data-testid="stRadio"] label::before {
            background: rgba(53, 78, 120, .08);
            color: #39258f;
        }
        h1, h2, h3, h4, h5, h6, p, label, [data-testid="stMarkdownContainer"] {
            color: inherit;
        }
        [data-testid="stCaptionContainer"], .mc-subtitle, .mc-chart-note,
        .mc-kpi-detail, .mc-vision-card-subtitle, .orbit-brand-subtitle,
        .orbit-shell-meta, .orbit-activity-copy small { color: #52627d; }
        [data-testid="stVerticalBlockBorderWrapper"], [data-testid="stExpander"],
        .mc-kpi-card, .mc-chart-card, .mc-section-shell, .mc-panel, .orbit-panel {
            background: rgba(255, 255, 255, .88);
            border-color: rgba(80, 103, 145, .20);
            box-shadow: 0 10px 26px rgba(42, 66, 105, .08);
        }
        [data-testid="stMetric"], .mc-data-metric { color: #16233d; }
        [data-testid="stMetricLabel"], .mc-data-label { color: #52627d; }
        [data-testid="stMetricValue"], .mc-data-value, .mc-kpi-value,
        .mc-title, .mc-navbar-title, .mc-vision-card-title, .orbit-panel-title,
        .orbit-big-value { color: #16233d; }
        [data-testid="stRadio"] label { color: #52627d; }
        [data-testid="stRadio"] label:has(input:checked) {
            background: rgba(117, 81, 255, .12);
            border-color: rgba(117, 81, 255, .28);
            color: #39258f;
        }
        [data-testid="stRadio"] label p { color: inherit; }
        [data-testid="stTextInput"] input, [data-testid="stTextArea"] textarea,
        [data-testid="stNumberInput"] input, [data-testid="stSelectbox"] > div {
            background: #ffffff;
            border-color: rgba(80, 103, 145, .28);
            color: #16233d;
        }
        [data-testid="stDataFrame"] { border-color: rgba(80, 103, 145, .20); }
        .mc-chart-grid { stroke: rgba(80, 103, 145, .22); }
        .mc-chart-axis { fill: #52627d; }
        .mc-callout, .mc-condition, .orbit-activity-row { border-color: rgba(80, 103, 145, .20); }
        .orbit-overview { color: #16233d; }
        .orbit-mission {
            background: linear-gradient(135deg, rgba(235, 241, 255, .96), rgba(255, 255, 255, .98));
            border-color: rgba(117, 81, 255, .25);
        }
        .orbit-activity-copy { color: #263957; }
        .orbit-progress { background: rgba(80, 103, 145, .15); }
    """
    system_theme = f"""
        @media (prefers-color-scheme: light) {{
            {light_theme}
        }}
    """
    theme_overrides = light_theme if appearance == "Light" else (
        system_theme if appearance == "System Default" else ""
    )
    st.markdown(
        """
        <link rel="manifest" href="/app/static/manifest.json">
        <meta name="theme-color" content="#030c1d">
        <meta name="mobile-web-app-capable" content="yes">
        <meta name="apple-mobile-web-app-capable" content="yes">
        <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
        <meta name="apple-mobile-web-app-title" content="Kova">
        <link rel="icon" type="image/png" sizes="96x96"
              href="/app/static/kova/favicon.png">
        <link rel="apple-touch-icon"
              href="/app/static/kova/apple-touch-icon.png">
        <style>
        :root {
            --mc-bg: #030c1d;
            --mc-panel: #0f1535;
            --mc-panel-2: #141b41;
            --mc-line: rgba(226, 232, 255, .13);
            --mc-magenta: #e9aefa;
            --mc-violet: #7551ff;
            --mc-cyan: #01befe;
            --mc-rose: #ff3d71;
            --mc-text: #ffffff;
            --mc-muted: #a0aec0;
            --mc-green: #01b574;
            --mc-amber: #ffb547;
        }
        html, body, [class*="css"] {
            font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
            -webkit-text-size-adjust: 100%;
        }
        html, body { overflow-x: hidden; }
        .stApp {
            background:
                radial-gradient(circle at 0% 8%, rgba(72, 82, 185, .28) 0%, transparent 26rem),
                radial-gradient(circle at 100% 4%, rgba(106, 65, 193, .2) 0%, transparent 29rem),
                radial-gradient(circle at 65% 100%, rgba(1, 190, 254, .08) 0%, transparent 34rem),
                var(--mc-bg);
        }
        [data-testid="stHeader"] {
            background: transparent;
            border-bottom: 0;
            height: 78px;
            position: relative;
        }
        [data-testid="stHeader"]::before {
            color: #ffffff;
            content: "K O V A";
            font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
            font-size: clamp(1.35rem, 2.8vw, 1.85rem);
            font-weight: 800;
            left: 50%;
            letter-spacing: .2em;
            line-height: 1;
            pointer-events: none;
            position: absolute;
            top: 23px;
            transform: translateX(-50%);
            white-space: nowrap;
        }
        [data-testid="stHeader"]::after {
            color: rgba(198, 208, 235, .72);
            content: "Knowledge-Oriented Virtual Assistant";
            font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
            font-size: .58rem;
            font-weight: 300;
            left: 50%;
            letter-spacing: .12em;
            line-height: 1;
            pointer-events: none;
            position: absolute;
            text-align: center;
            top: 53px;
            transform: translateX(-50%);
            white-space: nowrap;
        }
        [data-testid="stMainMenuButton"] {
            display: none !important;
        }
        [data-testid="stSidebarCollapseButton"],
        [data-testid="stSidebarCollapseButton"] button,
        [data-testid="stSidebarOpenButton"],
        button[data-testid="stBaseButton-headerNoPadding"] {
            display: none !important;
        }
        [data-testid="stSidebar"] {
            background:
                linear-gradient(180deg, rgba(15, 21, 53, .98), rgba(3, 12, 29, .98));
            border-right: 1px solid rgba(157, 167, 255, .18);
        }
        [data-testid="stSidebar"] > div:first-child {
            padding: 1.4rem .75rem;
        }
        [data-testid="stSidebar"] [data-testid="stRadio"] > div {
            display: flex;
            flex-direction: column;
            gap: .22rem;
            overflow: visible;
            padding: .25rem 0;
            white-space: normal;
        }
        [data-testid="stSidebar"] [data-testid="stRadio"] label {
            background: transparent;
            border-color: transparent;
            border-radius: 10px;
            color: #a3aed0;
            align-items: center;
            display: flex;
            font-size: .76rem;
            font-weight: 600;
            letter-spacing: .01em;
            min-height: 41px;
            padding: .62rem .72rem;
            transition: background .15s ease, color .15s ease, transform .15s ease;
        }
        [data-testid="stSidebar"] [data-testid="stRadio"] label::before {
            align-items: center;
            background: rgba(255, 255, 255, .055);
            border-radius: 8px;
            color: #ffffff;
            display: inline-flex;
            font-size: .8rem;
            height: 25px;
            justify-content: center;
            margin-right: .65rem;
            width: 25px;
        }
        [data-testid="stSidebar"] [data-testid="stRadio"] label:nth-of-type(1)::before { content: "⌂"; }
        [data-testid="stSidebar"] [data-testid="stRadio"] label:nth-of-type(2)::before { content: "◉"; }
        [data-testid="stSidebar"] [data-testid="stRadio"] label:nth-of-type(3)::before { content: "◈"; }
        [data-testid="stSidebar"] [data-testid="stRadio"] label:nth-of-type(4)::before { content: "▣"; }
        [data-testid="stSidebar"] [data-testid="stRadio"] label:nth-of-type(5)::before { content: "⌁"; }
        [data-testid="stSidebar"] [data-testid="stRadio"] label:nth-of-type(6)::before { content: "◌"; }
        [data-testid="stSidebar"] [data-testid="stRadio"] label:nth-of-type(7)::before { content: "◒"; }
        [data-testid="stSidebar"] [data-testid="stRadio"] label:nth-of-type(8)::before { content: "⌕"; }
        [data-testid="stSidebar"] [data-testid="stRadio"] label:nth-of-type(9)::before { content: "▤"; }
        [data-testid="stSidebar"] [data-testid="stRadio"] label:nth-of-type(10)::before { content: "⚙"; }
        [data-testid="stSidebar"] [data-testid="stRadio"] label:nth-of-type(11) {
            border-top: 1px solid rgba(226, 232, 255, .13);
            margin-top: .9rem;
            padding-top: .92rem;
        }
        [data-testid="stSidebar"] [data-testid="stRadio"] label:nth-of-type(11)::before { content: "⚙"; }
        [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
            background: rgba(117, 81, 255, .12);
            color: white;
            transform: translateX(2px);
        }
        [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
            background: linear-gradient(90deg, rgba(117, 81, 255, .92), rgba(117, 81, 255, .65));
            border-color: rgba(177, 159, 255, .35);
            box-shadow: 0 9px 18px rgba(67, 24, 255, .26);
            color: white;
        }
        [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked)::before {
            background: rgba(255, 255, 255, .17);
        }
        [data-testid="stSidebar"] [data-testid="stRadio"] label p {
            margin: 0;
        }
        .block-container {
            max-width: 1440px;
            padding: 1.35rem 1.5rem 3.4rem;
        }
        h2, h3, [data-testid="stHeading"] {
            color: var(--mc-text) !important;
            letter-spacing: -.025em;
        }
        [data-testid="stCaptionContainer"] {
            color: var(--mc-muted);
        }
        [data-testid="stMetric"] {
            background: linear-gradient(145deg, rgba(29, 35, 56, .94), rgba(17, 21, 35, .94));
            border: 1px solid rgba(99, 110, 151, .28);
            border-radius: 14px;
            box-shadow: inset 0 1px rgba(255, 255, 255, .025), 0 10px 25px rgba(0, 0, 0, .16);
            padding: .76rem .9rem;
            min-height: 82px;
        }
        [data-testid="stMetricLabel"] {
            color: var(--mc-muted);
            font-size: .72rem;
            text-transform: uppercase;
            letter-spacing: .08em;
        }
        [data-testid="stMetricValue"] {
            color: var(--mc-text);
            font-size: 1.18rem;
            letter-spacing: -.03em;
            line-height: 1.2;
            overflow-wrap: anywhere;
        }
        .mc-data-metric {
            background: linear-gradient(145deg, rgba(31, 39, 84, .82), rgba(15, 21, 53, .94));
            border: 1px solid var(--mc-line);
            border-radius: 12px;
            box-shadow: inset 0 1px rgba(255, 255, 255, .03);
            margin: .23rem 0;
            min-height: 76px;
            padding: .76rem .8rem;
        }
        .mc-data-label {
            color: var(--mc-muted);
            font-size: .67rem;
            font-weight: 700;
            letter-spacing: .08em;
            text-transform: uppercase;
        }
        .mc-data-value {
            color: var(--mc-text);
            font-size: 1.03rem;
            font-weight: 700;
            letter-spacing: -.025em;
            margin-top: .36rem;
            overflow-wrap: anywhere;
        }
        [data-testid="stVerticalBlockBorderWrapper"] {
            background:
                linear-gradient(150deg, rgba(31, 39, 84, .90), rgba(15, 21, 53, .96));
            border: 1px solid var(--mc-line) !important;
            border-radius: 15px !important;
            box-shadow: 0 12px 28px rgba(0, 0, 0, .18), inset 0 1px rgba(255, 255, 255, .035);
        }
        [data-testid="stExpander"] {
            background: var(--mc-panel);
            border: 1px solid rgba(99, 110, 151, .28);
            border-radius: 16px;
            margin-bottom: .55rem;
        }
        [data-testid="stDataFrame"] {
            border: 1px solid rgba(99, 110, 151, .28);
            border-radius: 14px;
            overflow: hidden;
        }
        [data-testid="stRadio"] > div {
            gap: .35rem;
            max-width: 100%;
            overflow-x: auto;
            padding: .1rem 0 .45rem;
            scrollbar-width: thin;
            white-space: nowrap;
        }
        [data-testid="stRadio"] label {
            background: rgba(21, 26, 44, .82);
            border: 1px solid rgba(99, 110, 151, .3);
            border-radius: 9px;
            color: #a5adc4;
            font-size: .68rem;
            font-weight: 700;
            letter-spacing: .04em;
            padding: .34rem .58rem;
        }
        [data-testid="stRadio"] label:has(input:checked) {
            background: linear-gradient(90deg, rgba(112, 87, 235, .88), rgba(75, 156, 255, .88));
            border-color: rgba(147, 121, 255, .85);
            color: white;
        }
        [data-testid="stRadio"] label p {
            margin: 0;
        }
        .mc-topbar {
            align-items: center;
            backdrop-filter: blur(18px);
            background: linear-gradient(135deg, rgba(31, 39, 84, .78), rgba(15, 21, 53, .82));
            border: 1px solid rgba(255, 255, 255, .12);
            border-radius: 15px;
            display: flex;
            gap: 1rem;
            justify-content: space-between;
            margin-bottom: 1.15rem;
            padding: .8rem 1rem;
        }
        .mc-assistant-widget-title {
            align-items: center;
            color: var(--mc-text);
            display: flex;
            font-size: .78rem;
            font-weight: 800;
            gap: .42rem;
            letter-spacing: .03em;
            white-space: nowrap;
        }
        .mc-assistant-widget-icon {
            border-radius: 7px;
            display: inline-block;
            height: 21px;
            object-fit: cover;
            width: 21px;
        }
        .mc-assistant-widget-subtitle {
            color: var(--mc-muted);
            font-size: .58rem;
            font-weight: 700;
            letter-spacing: .08em;
            margin: .3rem 0 .55rem 1.65rem;
            white-space: nowrap;
        }
        .mc-brand {
            align-items: center;
            color: var(--mc-text);
            display: flex;
            font-size: .95rem;
            font-weight: 800;
            gap: .55rem;
            letter-spacing: -.02em;
            white-space: nowrap;
        }
        .mc-brand-mark {
            align-items: center;
            background: linear-gradient(135deg, #7551ff, #01befe);
            border-radius: 9px;
            color: white;
            display: inline-flex;
            font-size: .85rem;
            height: 26px;
            justify-content: center;
            width: 26px;
        }
        .mc-brand-orb {
            border-radius: 9px;
            display: inline-block;
            height: 28px;
            object-fit: cover;
            width: 28px;
        }
        .mc-nav {
            color: #7f89a6;
            display: flex;
            font-size: .7rem;
            gap: 1.1rem;
            letter-spacing: .02em;
        }
        .mc-nav-active {
            background: linear-gradient(90deg, rgba(117, 81, 255, .9), rgba(1, 190, 254, .86));
            border-radius: 7px;
            color: white;
            font-weight: 700;
            padding: .35rem .62rem;
        }
        .mc-top-status {
            color: var(--mc-green);
            font-size: .7rem;
            font-weight: 700;
            white-space: nowrap;
        }
        .mc-breadcrumb {
            color: var(--mc-muted);
            font-size: .7rem;
            font-weight: 600;
            letter-spacing: .01em;
            margin-bottom: .22rem;
        }
        .mc-breadcrumb strong { color: var(--mc-text); font-weight: 700; }
        .mc-navbar-title {
            color: var(--mc-text);
            font-size: 1rem;
            font-weight: 700;
            letter-spacing: -.02em;
        }
        .mc-navbar-actions { align-items: center; display: flex; gap: .55rem; }
        .mc-navbar-pill {
            align-items: center;
            background: rgba(255, 255, 255, .055);
            border: 1px solid rgba(255, 255, 255, .10);
            border-radius: 10px;
            color: var(--mc-text);
            display: inline-flex;
            font-size: .68rem;
            font-weight: 700;
            gap: .35rem;
            padding: .42rem .62rem;
            white-space: nowrap;
        }
        .mc-navbar-avatar {
            border-radius: 9px;
            display: inline-block;
            height: 29px;
            object-fit: cover;
            width: 29px;
        }
        .mc-kicker {
            color: var(--mc-cyan);
            font-size: .72rem;
            font-weight: 700;
            letter-spacing: .16em;
            margin: .25rem 0 .35rem;
            text-transform: uppercase;
        }
        .mc-title {
            color: var(--mc-text);
            font-size: clamp(1.9rem, 5vw, 3.35rem);
            font-weight: 800;
            letter-spacing: -.04em;
            line-height: 1;
            margin: 0 0 .35rem;
        }
        .mc-subtitle {
            color: var(--mc-muted);
            font-size: .9rem;
            margin-bottom: 1rem;
        }
        .mc-decision {
            background:
                radial-gradient(circle at 85% 5%, rgba(117, 81, 255, .28), transparent 11rem),
                linear-gradient(135deg, #202b6c, #111a42);
            border: 1px solid rgba(157, 167, 255, .35);
            border-radius: 17px;
            box-shadow: 0 0 32px rgba(159, 83, 240, .13), inset 0 1px rgba(255, 255, 255, .04);
            padding: 1.15rem;
        }
        .mc-decision-label {
            color: var(--mc-muted);
            font-size: .72rem;
            letter-spacing: .14em;
            text-transform: uppercase;
        }
        .mc-decision-value {
            color: #e89cff;
            font-size: 2.4rem;
            font-weight: 800;
            line-height: 1.1;
            margin: .25rem 0 .75rem;
        }
        .mc-score-row {
            align-items: center;
            color: var(--mc-muted);
            display: flex;
            font-size: .8rem;
            gap: .6rem;
            justify-content: space-between;
        }
        .mc-score-track {
            background: #292d47;
            border-radius: 99px;
            height: 8px;
            overflow: hidden;
            width: 100%;
        }
        .mc-score-fill {
            background: linear-gradient(90deg, var(--mc-violet), var(--mc-magenta), #ff8fc7);
            border-radius: 99px;
            height: 100%;
        }
        .mc-status {
            align-items: center;
            background: rgba(21, 26, 44, .82);
            border: 1px solid rgba(99, 110, 151, .3);
            border-radius: 99px;
            color: var(--mc-text);
            display: inline-flex;
            font-size: .7rem;
            font-weight: 700;
            gap: .35rem;
            margin: .2rem .25rem .2rem 0;
            padding: .42rem .65rem;
        }
        .mc-dot {
            border-radius: 50%;
            display: inline-block;
            height: 7px;
            width: 7px;
        }
        .mc-dot-green { background: var(--mc-green); box-shadow: 0 0 8px var(--mc-green); }
        .mc-dot-red { background: var(--mc-rose); box-shadow: 0 0 8px var(--mc-rose); }
        .mc-dot-amber { background: var(--mc-amber); box-shadow: 0 0 8px var(--mc-amber); }
        .mc-dot-gray { background: #7b8492; }
        .mc-condition {
            background: rgba(26, 31, 51, .88);
            border: 1px solid rgba(99, 110, 151, .26);
            border-radius: 11px;
            font-size: .8rem;
            margin: .3rem 0;
            padding: .55rem .65rem;
        }
        .mc-condition-pass { color: var(--mc-green); }
        .mc-condition-fail { color: #ff8797; }
        .mc-callout {
            background: linear-gradient(90deg, rgba(97, 50, 118, .28), rgba(19, 25, 44, .75));
            border: 1px solid rgba(181, 99, 234, .25);
            border-left: 3px solid var(--mc-magenta);
            border-radius: 11px;
            color: #eacff3;
            font-size: .82rem;
            margin: .8rem 0;
            padding: .7rem .8rem;
        }
        .mc-kpi-grid {
            display: grid;
            gap: .8rem;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            margin: .8rem 0 1.25rem;
        }
        .mc-kpi-card {
            background:
                radial-gradient(circle at 92% 8%, var(--glow, rgba(117, 81, 255, .24)), transparent 8rem),
                linear-gradient(145deg, #1a2762, #111a42);
            border: 1px solid rgba(157, 167, 255, .23);
            border-radius: 16px;
            box-shadow: inset 0 1px rgba(255, 255, 255, .035), 0 14px 28px rgba(0, 0, 0, .16);
            min-height: 127px;
            overflow: hidden;
            padding: .9rem;
            position: relative;
        }
        .mc-kpi-card::after {
            background: linear-gradient(90deg, transparent, var(--accent, var(--mc-violet)));
            bottom: 0;
            content: "";
            height: 2px;
            left: 0;
            opacity: .75;
            position: absolute;
            width: 100%;
        }
        .mc-kpi-label {
            color: #a5adc4;
            font-size: .67rem;
            font-weight: 700;
            letter-spacing: .1em;
            text-transform: uppercase;
        }
        .mc-kpi-value {
            color: var(--mc-text);
            font-size: clamp(1.15rem, 2vw, 1.65rem);
            font-weight: 800;
            letter-spacing: -.045em;
            line-height: 1.1;
            margin: .45rem 0 .25rem;
        }
        .mc-kpi-detail {
            color: var(--mc-green);
            font-size: .72rem;
            font-weight: 700;
        }
        .mc-kpi-detail.neutral { color: var(--mc-muted); }
        .mc-kpi-detail.warn { color: var(--mc-amber); }
        .mc-section-shell {
            background: rgba(14, 18, 32, .38);
            border: 1px solid rgba(99, 110, 151, .16);
            border-radius: 17px;
            margin: .65rem 0 1rem;
            padding: .25rem .85rem .85rem;
        }
        .mc-vision-card-title {
            color: var(--mc-text);
            font-size: 1rem;
            font-weight: 700;
            letter-spacing: -.02em;
            margin: 0 0 .35rem;
        }
        .mc-vision-card-subtitle {
            color: var(--mc-muted);
            font-size: .74rem;
            line-height: 1.45;
            margin-bottom: .85rem;
        }
        .mc-vision-icon {
            align-items: center;
            background: linear-gradient(135deg, #7551ff, #01befe);
            border-radius: 8px;
            color: white;
            display: inline-flex;
            font-size: .75rem;
            height: 26px;
            justify-content: center;
            margin-right: .45rem;
            vertical-align: middle;
            width: 26px;
        }
        .mc-vision-progress {
            background: #2d356d;
            border-radius: 99px;
            height: 6px;
            margin-top: .35rem;
            overflow: hidden;
        }
        .mc-vision-progress > span {
            background: linear-gradient(90deg, #7551ff, #01befe);
            border-radius: inherit;
            display: block;
            height: 100%;
        }
        .mc-vision-status-row {
            align-items: center;
            border-bottom: 1px solid rgba(157, 167, 255, .12);
            display: flex;
            justify-content: space-between;
            padding: .55rem 0;
        }
        .mc-vision-status-row:last-child { border-bottom: 0; }
        .mc-vision-status-label { color: var(--mc-muted); font-size: .73rem; }
        .mc-vision-status-value { color: var(--mc-text); font-size: .75rem; font-weight: 700; }
        .mc-chart-card {
            background: linear-gradient(150deg, rgba(31, 39, 84, .90), rgba(15, 21, 53, .96));
            border: 1px solid var(--mc-line);
            border-radius: 15px;
            box-shadow: 0 12px 28px rgba(0, 0, 0, .18), inset 0 1px rgba(255, 255, 255, .035);
            margin: .7rem 0 1rem;
            overflow: hidden;
            padding: 1rem 1rem .65rem;
        }
        .mc-chart-header { align-items: flex-start; display: flex; justify-content: space-between; margin-bottom: .55rem; }
        .mc-chart-title { color: var(--mc-text); font-size: .96rem; font-weight: 700; letter-spacing: -.02em; }
        .mc-chart-note { color: var(--mc-muted); font-size: .70rem; margin-top: .18rem; }
        .mc-chart-badge {
            background: rgba(1, 181, 116, .12);
            border-radius: 7px;
            color: #01b574;
            font-size: .66rem;
            font-weight: 700;
            padding: .32rem .46rem;
        }
        .mc-chart-svg { display: block; height: auto; width: 100%; }
        .mc-chart-axis { fill: #8f9bbd; font-size: 10px; }
        .mc-chart-grid { stroke: rgba(226, 232, 255, .10); stroke-dasharray: 3 5; stroke-width: 1; }
        .mc-trades-card {
            background: linear-gradient(150deg, rgba(31, 39, 84, .90), rgba(15, 21, 53, .96));
            border: 1px solid var(--mc-line);
            border-radius: 15px;
            box-shadow: 0 12px 28px rgba(0, 0, 0, .18), inset 0 1px rgba(255, 255, 255, .035);
            margin: .7rem 0 1rem;
            overflow-x: auto;
            padding: .45rem .85rem .75rem;
        }
        .mc-trades-table { border-collapse: collapse; min-width: 680px; width: 100%; }
        .mc-trades-table th {
            border-bottom: 1px solid rgba(226, 232, 255, .13);
            color: #a0aec0;
            font-size: .64rem;
            font-weight: 700;
            letter-spacing: .08em;
            padding: .72rem .5rem;
            text-align: left;
            text-transform: uppercase;
        }
        .mc-trades-table td {
            border-bottom: 1px solid rgba(226, 232, 255, .08);
            color: #ffffff;
            font-size: .74rem;
            padding: .72rem .5rem;
            white-space: nowrap;
        }
        .mc-trades-table tbody tr:last-child td { border-bottom: 0; }
        .mc-trades-table tbody tr:hover { background: rgba(117, 81, 255, .08); }
        .mc-trade-id { color: #ffffff; font-weight: 700; }
        .mc-pnl-positive { color: #01b574 !important; font-weight: 700; }
        .mc-pnl-negative { color: #ff3d71 !important; font-weight: 700; }
        .mc-reason-badge {
            background: rgba(117, 81, 255, .16);
            border: 1px solid rgba(117, 81, 255, .25);
            border-radius: 6px;
            color: #d9d4ff;
            font-size: .64rem;
            font-weight: 700;
            padding: .26rem .42rem;
        }
        @media (max-width: 640px) {
            .block-container { padding: .9rem .7rem 2.2rem; }
            [data-testid="stMetric"] { min-height: 76px; padding: .6rem .7rem; }
            [data-testid="stMetricValue"] { font-size: 1.05rem; }
            [data-testid="stHorizontalBlock"] {
                flex-wrap: wrap;
                gap: .35rem .55rem;
            }
            [data-testid="stHorizontalBlock"] > [data-testid="column"] {
                flex: 1 1 100% !important;
                min-width: 100% !important;
                width: 100% !important;
            }
            .mc-data-metric { min-height: 68px; padding: .65rem .7rem; }
            .mc-data-label { font-size: .64rem; }
            .mc-data-value { font-size: .98rem; line-height: 1.25; }
            .mc-decision-value { font-size: 2rem; }
            .mc-topbar { border-radius: 14px; padding: .6rem .7rem; }
            .mc-nav { display: none; }
            .mc-top-status { font-size: .62rem; }
            .mc-kpi-grid { gap: .55rem; grid-template-columns: repeat(2, minmax(0, 1fr)); }
            .mc-kpi-card { border-radius: 13px; min-height: 108px; padding: .72rem; }
            .mc-section-shell { border-radius: 14px; padding: .15rem .55rem .65rem; }
            .mc-vision-card-title { font-size: .92rem; }
            .mc-navbar-pill { display: none; }
            .mc-chart-card { border-radius: 13px; padding: .8rem .7rem .55rem; }
            .mc-trades-card { border-radius: 13px; padding: .35rem .55rem .6rem; }
        }
        @media (max-width: 767px) {
            :root { color-scheme: dark; }
            [data-testid="stHeader"] {
                height: calc(70px + env(safe-area-inset-top));
                padding-top: env(safe-area-inset-top);
            }
            [data-testid="stHeader"]::before {
                font-size: 1.3rem;
                top: calc(20px + env(safe-area-inset-top));
            }
            [data-testid="stHeader"]::after {
                font-size: .5rem;
                letter-spacing: .08em;
                top: calc(47px + env(safe-area-inset-top));
            }
            [data-testid="stToolbar"] {
                padding-right: max(.55rem, env(safe-area-inset-right));
            }
            .stApp {
                padding-bottom: env(safe-area-inset-bottom);
            }
            .block-container {
                max-width: 100%;
                padding-left: max(.7rem, env(safe-area-inset-left));
                padding-right: max(.7rem, env(safe-area-inset-right));
                padding-bottom: max(2.2rem, env(safe-area-inset-bottom));
            }
            [data-testid="stSidebar"] {
                width: min(88vw, 22rem) !important;
            }
            [data-testid="stSidebar"] > div:first-child {
                padding-top: max(1rem, env(safe-area-inset-top));
                padding-bottom: max(1rem, env(safe-area-inset-bottom));
            }
            [data-testid="stSidebar"] [data-testid="stRadio"] label {
                min-height: 48px;
                padding: .72rem .78rem;
                font-size: .82rem;
            }
            [data-testid="stSidebar"] [data-testid="stRadio"] label::before {
                height: 30px;
                width: 30px;
            }
            [data-testid="stButton"] button,
            [data-testid="stDownloadButton"] button,
            [data-testid="stTextInput"] input,
            [data-testid="stTextArea"] textarea {
                min-height: 44px;
            }
            [data-testid="stExpander"] summary {
                min-height: 48px;
                padding: .75rem .8rem;
            }
            [data-testid="stDataFrame"] {
                max-width: 100%;
                overflow-x: auto;
            }
            .mc-kpi-grid { grid-template-columns: 1fr; }
            .mc-data-value { overflow-wrap: anywhere; }
            .mc-trades-card {
                overflow-x: visible;
                padding: .35rem .45rem .55rem;
            }
            .mc-trades-table { min-width: 0; table-layout: fixed; }
            .mc-trades-table thead {
                position: absolute;
                height: 1px;
                width: 1px;
                overflow: hidden;
                clip: rect(0 0 0 0);
                white-space: nowrap;
            }
            .mc-trades-table,
            .mc-trades-table tbody,
            .mc-trades-table tr,
            .mc-trades-table td {
                display: block;
                width: 100%;
            }
            .mc-trades-table tbody { display: grid; gap: .55rem; }
            .mc-trades-table tr {
                background: rgba(3, 12, 29, .35);
                border: 1px solid rgba(226, 232, 255, .10);
                border-radius: 12px;
                display: grid;
                gap: .15rem .6rem;
                grid-template-columns: repeat(2, minmax(0, 1fr));
                padding: .55rem;
            }
            .mc-trades-table td {
                border-bottom: 0;
                font-size: .72rem;
                overflow-wrap: anywhere;
                padding: .25rem .15rem;
                white-space: normal;
            }
            .mc-trades-table td::before {
                color: var(--mc-muted);
                display: block;
                font-size: .57rem;
                font-weight: 700;
                letter-spacing: .08em;
                margin-bottom: .1rem;
                text-transform: uppercase;
            }
            .mc-trades-table td:nth-child(1) {
                grid-column: 1 / -1;
            }
            .mc-trades-table td:nth-child(1)::before { content: "Trade"; }
            .mc-trades-table td:nth-child(2)::before { content: "Entry"; }
            .mc-trades-table td:nth-child(3)::before { content: "Exit"; }
            .mc-trades-table td:nth-child(4)::before { content: "Size"; }
            .mc-trades-table td:nth-child(5)::before { content: "Fees"; }
            .mc-trades-table td:nth-child(6)::before { content: "Net P/L"; }
            .mc-trades-table td:nth-child(7)::before { content: "Exit reason"; }
        }
        @media (min-width: 768px) and (max-width: 1100px) {
            .block-container {
                padding-left: 1.2rem;
                padding-right: 1.2rem;
            }
            [data-testid="stSidebar"] { min-width: 15rem; }
            .mc-data-value { font-size: 1.02rem; }
        }
        @media (prefers-reduced-motion: reduce) {
            *, *::before, *::after {
                animation-duration: .01ms !important;
                animation-iteration-count: 1 !important;
                scroll-behavior: auto !important;
                transition-duration: .01ms !important;
            }
        }
        /* THEME_OVERRIDES */
        </style>
        """.replace("/* THEME_OVERRIDES */", theme_overrides),
        unsafe_allow_html=True,
    )


def render_orbit_overview_styles():
    st.markdown(
        """
        <style>
        .orbit-overview {
            margin: .15rem auto 2.5rem;
            max-width: 1380px;
        }
        .orbit-shell {
            align-items: center;
            border-bottom: 1px solid rgba(153, 181, 255, .14);
            display: flex;
            gap: 1rem;
            justify-content: space-between;
            margin-bottom: 1.45rem;
            padding: .35rem 0 .9rem;
        }
        .orbit-brand {
            align-items: center;
            display: flex;
            gap: .7rem;
        }
        .orbit-brand img {
            border-radius: 12px;
            box-shadow: 0 0 22px rgba(1, 190, 254, .24);
            height: 40px;
            width: 40px;
        }
        .orbit-brand-name {
            color: #fff;
            font-size: 1.12rem;
            font-weight: 800;
            letter-spacing: -.035em;
        }
        .orbit-brand-subtitle {
            color: #8e9abc;
            font-size: .67rem;
            letter-spacing: .1em;
            margin-top: .18rem;
            text-transform: uppercase;
        }
        .orbit-shell-meta {
            align-items: center;
            color: #a6b3d4;
            display: flex;
            font-size: .72rem;
            gap: .7rem;
        }
        .orbit-shell-meta span {
            border: 1px solid rgba(1, 190, 254, .24);
            border-radius: 999px;
            padding: .38rem .62rem;
        }
        .orbit-shell-meta .orbit-mode {
            border-color: rgba(1, 181, 116, .38);
            color: #56e5b1;
        }
        .orbit-hero {
            align-items: stretch;
            display: grid;
            gap: 1.1rem;
            grid-template-columns: minmax(0, 1.55fr) minmax(290px, .8fr);
            margin-bottom: 1.1rem;
        }
        .orbit-mission {
            background:
                radial-gradient(circle at 78% 35%, rgba(117, 81, 255, .2), transparent 14rem),
                linear-gradient(135deg, rgba(13, 28, 67, .96), rgba(7, 15, 39, .97));
            border: 1px solid rgba(87, 181, 255, .28);
            border-radius: 22px;
            min-height: 280px;
            overflow: hidden;
            padding: 1.45rem;
            position: relative;
        }
        .orbit-mission::after {
            border: 1px solid rgba(1, 190, 254, .12);
            border-radius: 50%;
            content: "";
            height: 360px;
            position: absolute;
            right: -96px;
            top: -38px;
            width: 360px;
        }
        .orbit-eyebrow {
            color: #42d8ff;
            font-size: .67rem;
            font-weight: 800;
            letter-spacing: .15em;
            text-transform: uppercase;
        }
        .orbit-mission h1 {
            color: #fff;
            font-size: clamp(2rem, 5vw, 3.7rem);
            letter-spacing: -.065em;
            line-height: .98;
            margin: .55rem 0 .7rem;
            max-width: 560px;
        }
        .orbit-mission p {
            color: #9cadd0;
            font-size: .88rem;
            line-height: 1.55;
            margin: 0;
            max-width: 480px;
        }
        .orbit-mission-footer {
            align-items: center;
            bottom: 1.35rem;
            display: flex;
            gap: .65rem;
            position: absolute;
        }
        .orbit-state {
            align-items: center;
            background: rgba(1, 181, 116, .1);
            border: 1px solid rgba(1, 181, 116, .32);
            border-radius: 999px;
            color: #63edbb;
            display: inline-flex;
            font-size: .7rem;
            font-weight: 700;
            gap: .4rem;
            padding: .42rem .68rem;
        }
        .orbit-state-dot {
            background: #44e2ad;
            border-radius: 50%;
            box-shadow: 0 0 10px #44e2ad;
            height: 7px;
            width: 7px;
        }
        .orbit-mission-footer small {
            color: #7182a8;
            font-size: .68rem;
        }
        .orbit-core {
            align-items: center;
            display: flex;
            inset: 0 1.7rem 0 auto;
            justify-content: center;
            position: absolute;
            width: 220px;
        }
        .orbit-core::before, .orbit-core::after {
            border: 1px solid rgba(1, 190, 254, .42);
            border-radius: 50%;
            content: "";
            height: 178px;
            position: absolute;
            transform: rotate(-24deg);
            width: 82px;
        }
        .orbit-core::after {
            border-color: rgba(189, 93, 255, .45);
            height: 210px;
            transform: rotate(62deg);
            width: 100px;
        }
        .orbit-core img {
            border: 1px solid rgba(255, 255, 255, .2);
            border-radius: 50%;
            box-shadow: 0 0 28px rgba(1, 190, 254, .55), 0 0 56px rgba(117, 81, 255, .24);
            height: 92px;
            position: relative;
            width: 92px;
            z-index: 1;
        }
        .orbit-side-stack {
            display: grid;
            gap: 1.1rem;
            grid-template-rows: 1fr 1fr;
        }
        .orbit-panel {
            background: rgba(12, 23, 53, .8);
            border: 1px solid rgba(153, 181, 255, .16);
            border-radius: 18px;
            padding: 1rem 1.1rem;
        }
        .orbit-panel-title {
            align-items: center;
            color: #a7b5d8;
            display: flex;
            font-size: .68rem;
            font-weight: 800;
            justify-content: space-between;
            letter-spacing: .12em;
            text-transform: uppercase;
        }
        .orbit-panel-title b {
            color: #4de0ff;
            font-size: .62rem;
            letter-spacing: .05em;
        }
        .orbit-big-value {
            color: #fff;
            font-size: 1.8rem;
            font-weight: 800;
            letter-spacing: -.045em;
            margin-top: .55rem;
        }
        .orbit-panel-note {
            color: #8190b2;
            font-size: .7rem;
            margin-top: .2rem;
        }
        .st-key-overview_nav_grid {
            margin: 0 auto 1.2rem;
            max-width: 1380px;
            padding-bottom: .2rem;
        }
        .st-key-overview_nav_grid [data-testid="stHorizontalBlock"] {
            display: grid;
            gap: .8rem;
            grid-template-columns: repeat(4, minmax(0, 1fr));
        }
        .st-key-overview_nav_grid [data-testid="stHorizontalBlock"] > div {
            min-width: 0;
            width: auto !important;
        }
        .st-key-overview_nav_grid .overview-nav-link {
            background:
                radial-gradient(circle at 92% 8%, rgba(117, 81, 255, .24), transparent 8rem),
                linear-gradient(145deg, #1a2762, #111a42);
            border: 1px solid rgba(157, 167, 255, .23);
            border-radius: 16px;
            box-shadow: inset 0 1px rgba(255, 255, 255, .035), 0 14px 28px rgba(0, 0, 0, .16);
            color: var(--mc-text);
            min-height: 127px;
            overflow: hidden;
            padding: .9rem;
            position: relative;
            transition: border-color .2s ease, transform .2s ease, box-shadow .2s ease;
        }
        .st-key-overview_nav_grid .overview-nav-link::after {
            background: linear-gradient(90deg, transparent, var(--mc-violet));
            bottom: 0;
            content: "";
            height: 2px;
            left: 0;
            opacity: .75;
            position: absolute;
            width: 100%;
        }
        .st-key-overview_nav_grid .overview-nav-link:hover,
        .st-key-overview_nav_grid .overview-nav-link:focus-visible {
            border-color: rgba(1, 190, 254, .72);
            box-shadow: inset 0 1px rgba(255, 255, 255, .06), 0 0 20px rgba(1, 190, 254, .16);
            transform: translateY(-1px);
        }
        .st-key-overview_nav_grid .overview-nav-link {
            font-size: .67rem;
            font-weight: 700;
            letter-spacing: .1em;
            text-transform: uppercase;
        }
        .st-key-overview_live_monitor {
            margin: 0 auto 1.2rem;
            max-width: 1380px;
        }
        .st-key-overview_live_monitor [data-testid="stHorizontalBlock"] {
            align-items: stretch;
            gap: 1.1rem;
        }
        .st-key-overview_live_monitor [data-testid="stHorizontalBlock"] > div {
            min-width: 0;
        }
        .st-key-overview_live_monitor [data-testid="stButton"] button {
            align-items: center;
            background:
                radial-gradient(circle at 92% 8%, rgba(233, 174, 250, .24), transparent 8rem),
                linear-gradient(145deg, #1a2762, #111a42);
            border: 1px solid rgba(157, 167, 255, .23);
            border-radius: 16px;
            box-shadow: inset 0 1px rgba(255, 255, 255, .035), 0 14px 28px rgba(0, 0, 0, .16);
            color: var(--mc-text);
            display: flex;
            min-height: 238px;
            overflow: hidden;
            padding: 1.2rem;
            position: relative;
            transition: border-color .2s ease, transform .2s ease, box-shadow .2s ease;
        }
        .st-key-overview_live_monitor [data-testid="stButton"] button::after {
            background: linear-gradient(90deg, transparent, var(--mc-magenta));
            bottom: 0;
            content: "";
            height: 2px;
            left: 0;
            opacity: .8;
            position: absolute;
            width: 100%;
        }
        .st-key-overview_live_monitor [data-testid="stButton"] button:hover,
        .st-key-overview_live_monitor [data-testid="stButton"] button:focus-visible {
            border-color: rgba(233, 174, 250, .72);
            box-shadow: inset 0 1px rgba(255, 255, 255, .06), 0 0 20px rgba(233, 174, 250, .16);
            transform: translateY(-1px);
        }
        .st-key-overview_live_monitor [data-testid="stButton"] button p {
            font-size: clamp(1.35rem, 3vw, 2rem);
            font-weight: 800;
            letter-spacing: -.04em;
            text-transform: uppercase;
        }
        .st-key-overview_live_monitor .mc-chart-card {
            margin: 0;
            min-height: 238px;
        }
        .st-key-overview_market {
            margin: 0 auto 1.2rem;
            max-width: 1380px;
            position: relative;
        }
        .st-key-overview_market .mc-chart-card {
            margin: 0;
            pointer-events: none;
        }
        .overview-market-chart-link {
            border-radius: 16px;
            display: block;
            inset: 0;
            position: absolute;
        }
        .st-key-overview_market .stElementContainer:has(.overview-market-chart-link) {
            inset: 0;
            position: absolute;
            z-index: 2;
        }
        .st-key-overview_market .stElementContainer:has(.overview-market-chart-link) .stMarkdown,
        .st-key-overview_market .stElementContainer:has(.overview-market-chart-link) .stMarkdownContainer,
        .st-key-overview_market .stElementContainer:has(.overview-market-chart-link) p {
            height: 100%;
        }
        .st-key-overview_market .stElementContainer:has(.overview-market-chart-link) .stMarkdown {
            position: relative;
        }
        .overview-market-chart-link:focus-visible {
            border: 2px solid var(--mc-cyan);
            box-shadow: 0 0 0 4px rgba(1, 190, 254, .22);
            outline: none;
        }
        .orbit-progress {
            background: rgba(255, 255, 255, .08);
            border-radius: 999px;
            height: 6px;
            margin-top: .75rem;
            overflow: hidden;
        }
        .orbit-progress span {
            background: linear-gradient(90deg, #01befe, #9a6cff);
            border-radius: inherit;
            display: block;
            height: 100%;
        }
        .orbit-grid {
            display: grid;
            gap: 1.1rem;
            grid-template-columns: minmax(0, 1.35fr) minmax(260px, .75fr) minmax(220px, .65fr);
        }
        .orbit-chart-panel {
            grid-column: span 2;
        }
        .orbit-chart-panel .mc-chart-card {
            border: 0;
            box-shadow: none;
            margin: 0;
            padding: 0;
        }
        .orbit-chart-panel .mc-chart-header {
            padding: 0 0 .35rem;
        }
        .orbit-activity {
            display: grid;
            gap: .7rem;
            margin-top: .8rem;
        }
        .orbit-activity-row {
            align-items: center;
            border-top: 1px solid rgba(153, 181, 255, .1);
            display: flex;
            gap: .65rem;
            padding-top: .62rem;
        }
        .orbit-activity-icon {
            align-items: center;
            background: rgba(1, 190, 254, .1);
            border: 1px solid rgba(1, 190, 254, .25);
            border-radius: 9px;
            color: #4de0ff;
            display: inline-flex;
            flex: 0 0 27px;
            font-size: .72rem;
            height: 27px;
            justify-content: center;
        }
        .orbit-activity-copy {
            color: #dce6ff;
            font-size: .73rem;
            line-height: 1.35;
        }
        .orbit-activity-copy small {
            color: #7485aa;
            display: block;
            font-size: .63rem;
            margin-top: .12rem;
        }
        @media (max-width: 900px) {
            .orbit-hero { grid-template-columns: 1fr; }
            .orbit-side-stack { grid-template-columns: 1fr 1fr; grid-template-rows: auto; }
            .orbit-grid { grid-template-columns: 1fr 1fr; }
            .orbit-chart-panel { grid-column: span 2; }
        }
        @media (max-width: 600px) {
            .orbit-overview { margin-top: -.4rem; }
            .orbit-shell { margin-bottom: 1rem; padding-bottom: .7rem; }
            .orbit-brand img { height: 33px; width: 33px; }
            .orbit-brand-name { font-size: .98rem; }
            .orbit-brand-subtitle, .orbit-shell-meta span:first-child { display: none; }
            .orbit-shell-meta { font-size: .65rem; }
            .orbit-mission { min-height: 300px; padding: 1.05rem; }
            .st-key-overview_nav_grid {
                padding-bottom: 4.6rem;
            }
            .st-key-overview_nav_grid [data-testid="stHorizontalBlock"] {
                gap: .55rem;
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
            .st-key-overview_nav_grid [data-testid="stLinkButton"] a {
                min-height: 108px;
            }
            .st-key-overview_live_monitor,
            .st-key-overview_market {
                padding-bottom: 4.6rem;
            }
            .st-key-overview_live_monitor [data-testid="stHorizontalBlock"] {
                gap: .7rem;
            }
            .st-key-overview_live_monitor [data-testid="stButton"] button,
            .st-key-overview_live_monitor .mc-chart-card,
            .st-key-overview_market .mc-chart-card {
                min-height: 190px;
            }
            .orbit-mission h1 { font-size: 1.85rem; max-width: 63%; }
            .orbit-mission p { font-size: .78rem; max-width: 64%; }
            .orbit-core { opacity: .78; right: -2.2rem; transform: scale(.77); }
            .orbit-mission-footer {
                bottom: auto;
                flex-wrap: wrap;
                margin-top: 1.2rem;
                position: relative;
            }
            .orbit-side-stack, .orbit-grid { grid-template-columns: 1fr; }
            .orbit-chart-panel { grid-column: auto; }
            .orbit-panel { padding: .9rem; }
            .orbit-big-value { font-size: 1.55rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_mc_section(kicker, title, caption=None):
    st.markdown(f'<div class="mc-kicker">{kicker}</div>', unsafe_allow_html=True)
    st.subheader(title)
    if caption:
        st.caption(caption)


def render_mc_status(label, state="green", detail=None):
    state_class = {
        "green": "mc-dot-green",
        "red": "mc-dot-red",
        "amber": "mc-dot-amber",
        "gray": "mc-dot-gray",
    }.get(state, "mc-dot-gray")
    suffix = f" · {detail}" if detail else ""
    st.markdown(
        f'<span class="mc-status"><span class="mc-dot {state_class}">'
        f"</span>{label}{suffix}</span>",
        unsafe_allow_html=True,
    )


def render_mc_kpi_card(label, value, detail, accent, glow, detail_class=""):
    st.markdown(
        f'<div class="mc-kpi-card" style="--accent:{accent};--glow:{glow}">'
        f'<div class="mc-kpi-label">{label}</div>'
        f'<div class="mc-kpi-value">{value}</div>'
        f'<div class="mc-kpi-detail {detail_class}">{detail}</div>'
        "</div>",
        unsafe_allow_html=True,
    )


def render_mc_metrics(metrics, columns=2):
    for start in range(0, len(metrics), columns):
        row = metrics[start:start + columns]
        columns_view = st.columns(len(row))
        for column, (label, value) in zip(columns_view, row):
            with column:
                render_metric(label, value)


def render_mc_line_chart(title, caption, values, series_label, badge="LIVE DATA"):
    sampled = list(values)
    if len(sampled) > 96:
        step = max(1, len(sampled) // 96)
        sampled = sampled[::step]
        if sampled[-1] != values[-1]:
            sampled.append(values[-1])
    if not sampled:
        st.info("No chart values are available.")
        return

    chart_width, chart_height = 720, 245
    left, top, right, bottom = 45, 18, 18, 34
    plot_width = chart_width - left - right
    plot_height = chart_height - top - bottom
    low, high = min(sampled), max(sampled)
    if high == low:
        high += 1
        low -= 1
    points = []
    for index, value in enumerate(sampled):
        x = left + plot_width * index / max(1, len(sampled) - 1)
        y = top + (high - value) / (high - low) * plot_height
        points.append((x, y))
    path = " ".join(
        f"{'M' if index == 0 else 'L'} {x:.2f} {y:.2f}"
        for index, (x, y) in enumerate(points)
    )
    grid = "".join(
        f'<line class="mc-chart-grid" x1="{left}" x2="{chart_width - right}" '
        f'y1="{top + plot_height * index / 4:.2f}" '
        f'y2="{top + plot_height * index / 4:.2f}" />'
        for index in range(5)
    )
    last_x, last_y = points[-1]
    st.markdown(
        f'<div class="mc-chart-card"><div class="mc-chart-header"><div>'
        f'<div class="mc-chart-title">{escape(title)}</div>'
        f'<div class="mc-chart-note">{escape(caption)}</div></div>'
        f'<span class="mc-chart-badge">{escape(badge)}</span></div>'
        f'<svg class="mc-chart-svg" viewBox="0 0 {chart_width} {chart_height}" '
        f'role="img" aria-label="{escape(series_label)} chart">'
        f'<defs><linearGradient id="mcLine" x1="0" x2="0" y1="0" y2="1">'
        '<stop offset="0%" stop-color="#7551ff" stop-opacity=".48"/>'
        '<stop offset="100%" stop-color="#7551ff" stop-opacity="0"/>'
        '</linearGradient></defs>'
        f'{grid}'
        f'<path d="{path} L {last_x:.2f} {top + plot_height:.2f} '
        f'L {left:.2f} {top + plot_height:.2f} Z" fill="url(#mcLine)"/>'
        f'<path d="{path}" fill="none" stroke="#7551ff" stroke-linecap="round" '
        'stroke-linejoin="round" stroke-width="3"><title>'
        f'{escape(series_label)}: {sampled[-1]:,.4f}</title></path>'
        f'<circle cx="{last_x:.2f}" cy="{last_y:.2f}" fill="#01befe" r="5" '
        'stroke="#ffffff" stroke-width="2"><title>'
        f'Latest {escape(series_label)}: {sampled[-1]:,.4f}</title></circle>'
        f'<text class="mc-chart-axis" x="{left}" y="{chart_height - 10}">'
        f'{sampled[0]:,.2f}</text><text class="mc-chart-axis" '
        f'x="{chart_width - right - 46}" y="{chart_height - 10}">'
        f'{sampled[-1]:,.2f}</text></svg></div>',
        unsafe_allow_html=True,
    )


def render_mc_bar_chart(
    title,
    caption,
    values,
    series_label,
    badge="HISTORICAL BATCH BACKTEST",
):
    if not values:
        st.info("No completed paper trades are available.")
        return
    sampled = list(values[-24:])
    chart_width, chart_height = 720, 245
    left, top, right, bottom = 38, 18, 18, 34
    plot_width = chart_width - left - right
    plot_height = chart_height - top - bottom
    ceiling = max(max(abs(value) for value in sampled), 0.0001)
    zero_y = top + plot_height / 2
    bar_width = max(5, plot_width / len(sampled) * .55)
    bars = []
    for index, value in enumerate(sampled):
        x = left + (index + .5) * plot_width / len(sampled) - bar_width / 2
        height = abs(value) / ceiling * (plot_height / 2 - 6)
        y = zero_y - height if value >= 0 else zero_y
        color = "#01b574" if value >= 0 else "#ff3d71"
        bars.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width:.2f}" '
            f'height="{height:.2f}" rx="3" fill="{color}"><title>'
            f'{escape(series_label)} {index + 1}: {value:+.4f}</title></rect>'
        )
    st.markdown(
        f'<div class="mc-chart-card"><div class="mc-chart-header"><div>'
        f'<div class="mc-chart-title">{escape(title)}</div>'
        f'<div class="mc-chart-note">{escape(caption)}</div></div>'
        f'<span class="mc-chart-badge">{escape(badge)}</span></div>'
        f'<svg class="mc-chart-svg" viewBox="0 0 {chart_width} {chart_height}" '
        f'role="img" aria-label="{escape(series_label)} bar chart">'
        f'<line class="mc-chart-grid" x1="{left}" x2="{chart_width - right}" '
        f'y1="{zero_y:.2f}" y2="{zero_y:.2f}" />{"".join(bars)}'
        f'<text class="mc-chart-axis" x="{left}" y="{chart_height - 10}">'
        f'Last {len(sampled)} trades</text><text class="mc-chart-axis" '
        f'x="{chart_width - right - 76}" y="{chart_height - 10}">'
        'Net P/L</text></svg></div>',
        unsafe_allow_html=True,
    )


def render_mc_conditions(evaluation):
    for start in range(0, len(ANALYSIS_CONDITION_LABELS), 2):
        row = list(ANALYSIS_CONDITION_LABELS.items())[start:start + 2]
        columns_view = st.columns(len(row))
        for column, (key, label) in zip(columns_view, row):
            passed = bool(evaluation[key])
            with column:
                status = "PASSED" if passed else "FAILED"
                status_class = (
                    "mc-condition-pass" if passed else "mc-condition-fail"
                )
                st.markdown(
                    f'<div class="mc-condition {status_class}">'
                    f"{'✓' if passed else '×'} {label}<br>"
                    f"<strong>{status}</strong></div>",
                    unsafe_allow_html=True,
                )


def render_mc_live_market(market_data, candles):
    render_mc_section(
        "LIVE MARKET DISPLAY",
        "BTC/CAD market display",
        "Public Kraken XBT/CAD data for display only; it is not connected "
        "to the paper backtest or any order endpoint.",
    )
    if not candles:
        st.error(
            "Kraken display data is unavailable. "
            f"{market_data.last_error or 'No candles were returned.'}"
        )
        return

    latest = candles[-1]
    previous = candles[-2] if len(candles) > 1 else None
    change = (
        ((latest["close"] / previous["close"]) - 1) * 100
        if previous and previous["close"]
        else None
    )
    render_mc_metrics(
        [
            ("BTC/CAD price", f"${latest['close']:,.2f}"),
            ("Market regime", "Not classified (live display)"),
            ("Candle timestamp", format_market_timestamp(latest["timestamp"])),
            ("Loaded candles", str(len(candles))),
            ("1-candle change", f"{change:+.3f}%" if change is not None else "N/A"),
            ("Source", market_data.pair_name or "XBT/CAD · Kraken"),
        ],
        columns=2,
    )
    render_mc_line_chart(
        "BTC/CAD close",
        "Public Kraken market display · not connected to order placement.",
        [candle["close"] for candle in candles],
        "BTC/CAD close",
        "KRAKEN PUBLIC DATA",
    )


def render_mc_account(results):
    render_mc_section(
        "HISTORICAL BATCH BACKTEST",
        "Simulated account performance",
    )
    return_percent = results["profit"] / results["starting_capital"] * 100
    render_mc_metrics(
        [
            ("Starting capital", f"${results['starting_capital']:.2f}"),
            ("Current balance", f"${results['ending_capital']:.4f}"),
            ("Profit / loss", f"${results['profit']:+.4f}"),
            ("Return", f"{return_percent:+.2f}%"),
            ("Trades", str(results["trades"])),
            ("Wins / losses", f"{results['wins']} / {results['losses']}"),
            ("Win rate", f"{results['win_rate']:.2f}%"),
            ("Maximum drawdown", f"{results['max_drawdown']:.2f}%"),
            ("Fees", f"${results['total_fees']:.4f}"),
            ("Estimated slippage", f"${results['total_slippage']:.4f}"),
        ],
        columns=2,
    )


def render_mc_equity(results):
    render_mc_line_chart(
        "Paper account trajectory",
        f"Historical batch backtest · ${results['equity_curve'][0]:.2f} "
        f"starting balance across {len(results['equity_curve']) - 1} "
        "evaluated candles.",
        results["equity_curve"],
        "Account value",
        "HISTORICAL BATCH BACKTEST",
    )


def render_mc_trade_pnl_chart(results):
    render_mc_bar_chart(
        "Net P/L by completed trade",
        "Historical batch backtest · completed simulated trades",
        [
            trade["net_profit_loss"]
            for trade in results["trades_history"]
        ],
        "Net P/L",
    )


def render_mc_trade_history(results):
    render_mc_section(
        "TRADE HISTORY",
        "Completed backtest-trade audit",
    )
    if not results["trades_history"]:
        st.info("No completed simulated trades.")
        return
    for trade in reversed(results["trades_history"]):
        net = trade["net_profit_loss"]
        with st.expander(
            f"Trade {trade['trade_number']} · "
            f"{'PROFIT' if net >= 0 else 'LOSS'} · "
            f"${net:+.4f} · {trade['reason']}",
            expanded=False,
        ):
            render_mc_metrics(
                [
                    ("Entry", f"Candle {trade['entry_candle']}"),
                    ("Exit", f"Candle {trade['exit_candle']}"),
                    ("Entry price", f"${trade['entry_price']:.4f}"),
                    ("Exit price", f"${trade['exit_price']:.4f}"),
                    ("Position size", f"{trade['position_size']:.8f}"),
                    ("Gross P/L", f"${trade['gross_profit_loss']:+.4f}"),
                    ("Fees", f"${trade['fees']:.4f}"),
                    ("Slippage", f"${trade['estimated_slippage']:.4f}"),
                    ("Net P/L", f"${net:+.4f}"),
                    ("Exit reason", trade["reason"]),
                    ("Strategy score", f"{trade['strategy_score']}/100"),
                    ("RSI at entry", f"{trade['rsi_at_entry']:.2f}"),
                ],
                columns=2,
            )


def render_mc_research_lab(results, market_data):
    with st.expander("RESEARCH LAB · HISTORICAL BTC/CAD STUDIES", expanded=False):
        if results is None:
            st.warning(
                "Historical research is unavailable. "
                f"{market_data.last_error or ''}"
            )
            return
        st.caption(
            "RESEARCH-ONLY · Historical batch backtest periods using public "
            "Yahoo Finance daily data. Regimes are research labels only; they "
            "do not change strategy decisions or auto-promote candidates."
        )
        aggregate = results["aggregate"]
        render_mc_metrics(
            [
                ("Completed periods", str(len(results["periods"]))),
                ("Total independent return", f"{aggregate['total_return']:+.2f}%"),
                ("Average period return", f"{aggregate['average_return']:+.2f}%"),
                ("Average win rate", f"{aggregate['average_win_rate']:.2f}%"),
                (
                    "Average market benchmark",
                    f"{sum(p['market_return'] for p in results['periods']) / max(len(results['periods']), 1):+.2f}%",
                ),
                (
                    "Average net strategy",
                    f"{sum(p['return_percent'] for p in results['periods']) / max(len(results['periods']), 1):+.2f}%",
                ),
                (
                    "Average net delta vs benchmark",
                    f"{(sum(p['return_percent'] - p['market_return'] for p in results['periods']) / max(len(results['periods']), 1)):+.2f}%",
                ),
                ("Gross P/L", f"${aggregate['total_gross_profit_before_costs']:+.4f}"),
                ("Net fees", f"${aggregate['total_fees']:.4f}"),
                ("Slippage", f"${aggregate['total_slippage']:.4f}"),
                ("Worst drawdown", f"{aggregate['worst_drawdown']:.2f}%"),
            ],
            columns=2,
        )
        st.caption(
            "Benchmark comparison uses the same completed historical periods. "
            "Strategy return is net of modeled fees and slippage; it is not a "
            "live or genuine-observation result."
        )
        st.markdown("#### Regime performance")
        for regime in ("Bull", "Sideways", "Bear"):
            periods = results["regime_summary"].get(regime, [])
            with st.expander(f"{regime} · {len(periods)} periods"):
                if not periods:
                    st.info(f"No completed {regime} period is currently available.")
                    continue
                render_mc_metrics(
                    [
                        ("Market return", f"{sum(p['market_return'] for p in periods) / len(periods):+.2f}%"),
                        ("Strategy return", f"{sum(p['return_percent'] for p in periods) / len(periods):+.2f}%"),
                        ("Trades", str(sum(p["trades"] for p in periods))),
                        ("Win rate", f"{sum(p['wins'] for p in periods) / max(sum(p['trades'] for p in periods), 1) * 100:.2f}%"),
                        ("Gross P/L", f"${sum(p['gross_profit_before_costs'] for p in periods):+.4f}"),
                        ("Net P/L", f"${sum(p['net_profit'] for p in periods):+.4f}"),
                        ("Fees", f"${sum(p['total_fees'] for p in periods):.4f}"),
                        ("Slippage", f"${sum(p['total_slippage'] for p in periods):.4f}"),
                    ],
                    columns=2,
                )
        st.markdown("#### Ten-period audit")
        st.dataframe(
            [
                {
                    "Period": period["period"],
                    "Dates": f"{period['start_date']} → {period['end_date']}",
                    "Regime": period["regime"],
                    "Market": f"{period['market_return']:+.2f}%",
                    "Strategy": f"{period['return_percent']:+.2f}%",
                    "Gross": f"${period['gross_profit_before_costs']:+.4f}",
                    "Fees": f"${period['total_fees']:.4f}",
                    "Slippage": f"${period['total_slippage']:.4f}",
                    "Net": f"${period['net_profit']:+.4f}",
                    "Trades": period["trades"],
                    "Drawdown": f"{period['max_drawdown']:.2f}%",
                }
                for period in results["periods"]
            ],
            hide_index=True,
            width="stretch",
        )


def render_mc_system_health(
    strategy_ready,
    paper_ready,
    live_candles,
    market_data,
    historical_results,
    historical_market_data,
):
    render_mc_section(
        "SYSTEM HEALTH",
        "Operational health",
        "The canonical view for genuine observation, safety, and engine health.",
    )
    render_mc_live_observation_status()

    snapshot = load_live_observation_status()
    st.markdown("#### Market-data health")
    health = getattr(market_data, "health", {}) or {}
    health_state = health.get("status", "UNAVAILABLE" if not live_candles else "UNKNOWN")
    render_mc_status(
        "Market data",
        "green" if health_state == "HEALTHY" else (
            "amber" if health_state == "DEGRADED" else "red"
        ),
        health_state,
    )
    render_mc_metrics(
        [
            ("Current source", market_data.pair_name or "Kraken XBT/CAD"),
            ("Data age", (
                f"{health['data_age_seconds']:.0f}s"
                if health.get("data_age_seconds") is not None
                else "Unknown"
            )),
            ("Latest candle", (
                format_market_timestamp(live_candles[-1]["timestamp"])
                if live_candles else "Unavailable"
            )),
            ("Loaded candles", str(len(live_candles))),
            ("Provider error", getattr(market_data, "last_error", None) or "None"),
        ],
        columns=2,
    )

    st.markdown("#### Paper engine")
    engine_available = snapshot.get("available", False) and not snapshot.get("store_error")
    render_mc_status(
        "Paper engine",
        "green" if paper_ready else "red",
        "AVAILABLE" if paper_ready else "UNAVAILABLE",
    )
    render_mc_metrics(
        [
            ("Paper cash", f"${snapshot['cash']:,.2f}" if engine_available else "Unavailable"),
            ("Position", f"{snapshot['position']:.8f}" if engine_available else "Unavailable"),
            ("Current paper balance", f"${snapshot['cash']:,.2f}" if engine_available else "Unavailable"),
        ],
        columns=2,
    )

    st.markdown("#### Evidence storage")
    storage_health = snapshot.get("paper_storage") or {}
    storage_state = storage_health.get("status", "UNKNOWN")
    storage_indicator = {"HEALTHY": "green", "UNAVAILABLE": "red"}.get(
        storage_state, "amber"
    )
    render_mc_status("Evidence storage", storage_indicator, storage_state)
    if storage_state == "UNAVAILABLE":
        render_mc_metrics(
            [
                ("Outage code", storage_health.get("error_code") or "Unavailable"),
                ("Safe reason", storage_health.get("last_error") or "Unavailable"),
                ("Operation", storage_health.get("operation") or "Unavailable"),
            ],
            columns=2,
        )

    st.markdown("#### Strategy engine")
    render_mc_status(
        "Strategy engine",
        "green" if strategy_ready else "red",
        "AVAILABLE" if strategy_ready else "UNAVAILABLE",
    )

    st.markdown("#### Safety")
    render_mc_metrics(
        [
            ("Paper trading", "ENABLED" if PAPER_TRADING else "DISABLED"),
            ("Live trading", "DISABLED" if not LIVE_TRADING else "ENABLED"),
            ("Live execution", "NOT AVAILABLE"),
        ],
        columns=2,
    )

    st.markdown("#### Diagnostics")
    render_mc_metrics(
        [
            ("Observation store", "AVAILABLE" if snapshot.get("available") else "NOT STARTED"),
            ("Historical data", "AVAILABLE" if historical_results else "UNAVAILABLE"),
            (
                "Historical source issue",
                getattr(historical_market_data, "last_error", None) or "None",
            ),
        ],
        columns=2,
    )
    render_mc_provider_health()


def render_mc_provider_health():
    provider_health = get_provider_health()
    with st.expander("Provider diagnostics", expanded=False):
        provider_state = provider_health["availability"]
        render_mc_status(
            f"PROVIDER: {provider_state}",
            "green"
            if provider_state == "HEALTHY"
            else "amber"
            if provider_state == "DEGRADED"
            else "red",
        )
        render_mc_metrics(
            [
                ("Provider", provider_health["provider"]),
                ("Requests", str(provider_health["requests"])),
                (
                    "Success rate",
                    (
                        f"{provider_health['success_rate_percent']:.1f}%"
                        if provider_health["success_rate_percent"] != UNKNOWN
                        else UNKNOWN
                    ),
                ),
                (
                    "Last latency",
                    (
                        f"{provider_health['last_latency_ms']:.1f} ms"
                        if provider_health["last_latency_ms"] != UNKNOWN
                        else UNKNOWN
                    ),
                ),
                ("Last outcome", provider_health["last_outcome"]),
                (
                    "Latest failure category",
                    format_failure_category(provider_health["last_failure_category"]),
                ),
            ],
            columns=3,
        )
        st.caption(
            "Failure category counts are aggregate telemetry only; provider "
            "response details are not shown."
        )
        st.metric(
            "Failure category counts",
            format_failure_category_counts(provider_health["failure_categories"]),
        )


def render_mc_header(
    selected_section,
    latest_evaluation,
    market_data,
    live_candles,
):
    page_title = dashboard_display_name(selected_section)
    st.markdown(
            '<div class="mc-topbar">'
            '<div><div class="mc-breadcrumb">Pages <span>/</span> '
            f'<strong>{escape(page_title)}</strong></div>'
            f'<div class="mc-navbar-title">{escape(page_title)}</div></div>'
            '<div class="mc-navbar-actions">'
            '<span class="mc-navbar-pill">BTC/CAD</span>'
            '</div></div>'
            f'<div class="mc-title">{escape(page_title)}</div>'
            "",
            unsafe_allow_html=True,
        )


NAVIGATION_ITEMS = (
    "OVERVIEW",
    "LIVE MONITOR",
    "STRATEGY",
    "POSITIONS",
    "PERFORMANCE",
    "RISK",
    "OPTIONS REVIEW",
    "MARKET",
    "RESEARCH",
    "BACKTEST",
    "SYSTEM",
    "SETTINGS",
)

NAVIGATION_DISPLAY_NAMES = {}


def dashboard_display_name(section):
    return NAVIGATION_DISPLAY_NAMES.get(section, section)


def render_mc_navigation():
    pending_destination = st.session_state.pop("overview_destination", None)
    query_destination = st.query_params.get("section")
    if query_destination in NAVIGATION_ITEMS:
        pending_destination = query_destination
    elif query_destination is None and pending_destination is None:
        # The query string is the browser-history source of truth.  In
        # particular, a Back navigation to the hub must not inherit the
        # detail view from Streamlit session state.
        st.session_state.dashboard_section = "OVERVIEW"
    if pending_destination in NAVIGATION_ITEMS:
        st.session_state.dashboard_section = pending_destination
    selected = st.session_state.get("dashboard_section", "OVERVIEW")
    if selected not in NAVIGATION_ITEMS:
        selected = "OVERVIEW"
        st.session_state.dashboard_section = selected
    return selected


def render_mc_return_to_overview():
    if st.button(
        "Return to Overview",
        icon=":material/home:",
        key="return_to_overview",
        help="Return to the main KOVA dashboard",
    ):
        st.session_state.dashboard_section = "OVERVIEW"
        st.query_params.clear()
        st.rerun()


def render_kova_voice_assistant(
    results,
    latest_evaluation,
    market_data,
    live_candles,
    historical_results,
):
    """Render the persistent, read-only browser voice assistant."""
    context = build_assistant_context(
        results,
        latest_evaluation,
        market_data,
        live_candles,
        historical_results,
    )
    if "assistant_messages" not in st.session_state:
        st.session_state.assistant_messages = []
    if "kova_voice_last_event" not in st.session_state:
        st.session_state.kova_voice_last_event = None
    if "kova_voice_response" not in st.session_state:
        st.session_state.kova_voice_response = None

    event = KOVA_VOICE_COMPONENT(
        context=context,
        history=st.session_state.assistant_messages[-8:],
        response=st.session_state.kova_voice_response,
        appearance=st.session_state.get("dashboard_appearance", "Dark"),
        # Voice is deliberately scoped to read-only answers. It has no control
        # event path, so spoken START/PAUSE/STOP requests cannot mutate state.
        permission="READ_ONLY",
        key="kova_voice_assistant",
    )
    if (
        isinstance(event, dict)
        and event.get("type") == "user_message"
        and event.get("id") != st.session_state.kova_voice_last_event
    ):
        question = str(event.get("text", "")).strip()
        st.session_state.kova_voice_last_event = event.get("id")
        if question:
            response_text = answer_question(
                question,
                context,
                st.session_state.assistant_messages,
            )
            st.session_state.assistant_messages.extend([
                {"role": "user", "content": question},
                {"role": "assistant", "content": response_text},
            ])
            st.session_state.kova_voice_response = {
                "id": event.get("id"),
                "text": response_text,
            }
            st.rerun()


def render_mc_command_cards(results, latest_evaluation):
    render_mc_section(
        "COMMAND STATUS",
        "System at a glance",
    )
    return_percent = results["profit"] / results["starting_capital"] * 100
    cards = st.columns(4)
    card_data = [
        (
            "Backtest account",
            f"${results['ending_capital']:.4f}",
            f"{return_percent:+.2f}% · historical batch backtest",
            "#9a6cff",
            "rgba(139, 92, 246, .30)",
            "",
        ),
        (
            "Completed trades",
            str(results["trades"]),
            f"{results['wins']} wins · {results['losses']} losses",
            "#43d8ff",
            "rgba(69, 200, 255, .24)",
            "neutral",
        ),
        (
            "AI strategy score",
            f"{latest_evaluation['strategy_score']}/100",
            latest_evaluation["decision"],
            "#ee62d0",
            "rgba(228, 72, 201, .27)",
            "warn" if latest_evaluation["decision"] != "BUY" else "",
        ),
        (
            "Maximum drawdown",
            f"{results['max_drawdown']:.2f}%",
            "Backtest drawdown",
            "#f5a74f",
            "rgba(247, 185, 76, .22)",
            "warn",
        ),
    ]
    for column, card in zip(cards, card_data):
        with column:
            render_mc_kpi_card(*card)


def render_mc_evaluation_history(results):
    render_mc_section(
        "STRATEGY TELEMETRY",
        "Recent evaluations",
    )
    evaluations = results["evaluation_history"]
    if not evaluations:
        st.info("No strategy evaluations are available.")
        return
    st.dataframe(
        [
            {
                "Evaluation": evaluation["evaluation_number"],
                "Candle": evaluation["candle"],
                "Time": format_market_timestamp(evaluation["timestamp"]),
                "Price": f"${evaluation['current_price']:,.2f}",
                "Score": evaluation["strategy_score"],
                "Decision": evaluation["decision"],
                "RSI": f"{evaluation['rsi']:.2f}",
                "Trend": "PASS" if evaluation["long_term_trend"] else "FAIL",
                "Momentum": "PASS" if evaluation["short_term_momentum"] else "FAIL",
                "Volume": "PASS" if evaluation["volume"] else "FAIL",
            }
            for evaluation in evaluations[-12:]
        ],
        hide_index=True,
        width="stretch",
    )


def render_mc_autonomous_portfolio_decision(
    results,
    latest_evaluation,
    live_candles,
    observation_snapshot,
):
    """Show the portfolio-decision audit without connecting it to execution."""
    render_mc_section(
        "AUTONOMOUS PORTFOLIO DECISION",
        "Decision council audit",
        "Read-only portfolio analysis. The genuine paper signal is separated "
        "from historical strategy and research evidence.",
    )
    signal_payload = (
        (observation_snapshot.get("last_signal") or {}).get("payload", {})
        if observation_snapshot.get("available")
        else {}
    )
    if not isinstance(signal_payload, dict):
        signal_payload = {}
    paper_decision = (
        "BUY" if signal_payload.get("entry_eligible") is True
        else "HOLD" if signal_payload.get("entry_eligible") is False
        else "UNAVAILABLE"
    )
    paper_score = signal_payload.get("strategy_score")
    paper_score_text = (
        f"{float(paper_score):.0f}/100"
        if isinstance(paper_score, (int, float))
        else "Unavailable"
    )
    trend = bool(latest_evaluation.get("long_term_trend"))
    momentum = bool(latest_evaluation.get("short_term_momentum"))
    regime = (
        "BULLISH" if trend and momentum
        else "MIXED" if trend or momentum
        else "DEFENSIVE"
    )
    cash = observation_snapshot.get("cash")
    position = float(observation_snapshot.get("position", 0) or 0)
    current_price = (
        float(live_candles[-1]["close"])
        if live_candles and live_candles[-1].get("close") is not None
        else None
    )
    current_exposure = position * current_price if current_price else None
    configured_allocation = MAX_POSITION_PERCENT * 100
    render_mc_metrics(
        [
            ("Genuine paper decision", paper_decision),
            ("Genuine paper score", paper_score_text),
            ("Historical council score", f"{latest_evaluation['strategy_score']}/100"),
            ("Historical regime context", regime),
            (
                "Configured allocation ceiling",
                f"{configured_allocation:.0f}% of paper capital",
            ),
            (
                "Paper sizing capacity",
                f"${float(cash) * MAX_POSITION_PERCENT:,.4f}"
                if isinstance(cash, (int, float))
                else "Unavailable",
            ),
            (
                "Current paper exposure",
                f"${current_exposure:,.4f}"
                if current_exposure is not None
                else "No active exposure",
            ),
            (
                "Concentration / diversification",
                "SINGLE-ASSET SCOPE · diversification not assessed",
            ),
            (
                "Risk Governor",
                "PAPER GUARDRAILS ACTIVE · 2% stop · 4% target",
            ),
            ("Live execution", "DISABLED"),
        ],
        columns=2,
    )
    st.info(
        "The paper decision is the only current operational signal. Historical "
        "scores and research experiments are reference evidence and cannot "
        "auto-promote into production."
    )


def render_mc_position_snapshot(results, latest_evaluation, live_candles):
    render_mc_section(
        "POSITION SNAPSHOT",
        "Current paper position",
    )
    snapshot = load_live_observation_status()
    position = float(snapshot.get("position", 0) or 0)
    entry_price = snapshot.get("entry_price")
    current_price = live_candles[-1]["close"] if live_candles else None
    has_position = snapshot.get("available") and position > 0
    unrealized = (
        (current_price - entry_price) * position
        if has_position and current_price is not None and entry_price
        else None
    )
    render_mc_metrics(
        [
            ("Position", "ACTIVE" if has_position else "NO ACTIVE PAPER POSITION"),
            (
                "Position size",
                f"{position:.8f} BTC" if has_position else "Not applicable",
            ),
            (
                "Entry price",
                f"${entry_price:,.2f}" if has_position and entry_price else "Not applicable",
            ),
            (
                "Current price",
                f"${current_price:,.2f}" if current_price is not None else "Unavailable",
            ),
            ("Stop loss", f"{STOP_LOSS_PERCENT:.1%} configured"),
            ("Take profit", f"{TAKE_PROFIT_PERCENT:.1%} configured"),
            (
                "Unrealized P/L",
                f"${unrealized:+.4f}" if unrealized is not None else "Not applicable",
            ),
            ("Last signal", latest_evaluation["decision"]),
        ],
        columns=2,
    )


def render_mc_recent_trades_table(results):
    render_mc_section(
        "HISTORICAL BATCH BACKTEST",
        "Completed simulated trades",
        "Historical simulation only — not genuine paper observation evidence.",
    )
    trades = results["trades_history"]
    if not trades:
        st.info("No completed simulated trades.")
        return
    rows = []
    for trade in reversed(trades):
        net = trade["net_profit_loss"]
        pnl_class = "mc-pnl-positive" if net >= 0 else "mc-pnl-negative"
        rows.append(
            "<tr>"
            f'<td class="mc-trade-id">#{trade["trade_number"]:02d}</td>'
            f"<td>${trade['entry_price']:,.4f}</td>"
            f"<td>${trade['exit_price']:,.4f}</td>"
            f"<td>{trade['position_size']:.8f}</td>"
            f"<td>${trade['fees']:.4f}</td>"
            f'<td class="{pnl_class}">${net:+.4f}</td>'
            f'<td><span class="mc-reason-badge">{escape(trade["reason"])}</span></td>'
            "</tr>"
        )
    st.markdown(
        '<div class="mc-trades-card"><table class="mc-trades-table">'
        "<thead><tr><th>Trade</th><th>Entry</th><th>Exit</th><th>Size</th>"
        "<th>Fees</th><th>Net P/L</th><th>Exit reason</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>",
        unsafe_allow_html=True,
    )


def render_mc_strategy_page(results, latest_evaluation, live_candles):
    observation_snapshot = load_live_observation_status()
    render_mc_ai_decision(latest_evaluation, observation_snapshot)
    render_mc_autonomous_portfolio_decision(
        results,
        latest_evaluation,
        live_candles,
        observation_snapshot,
    )
    render_mc_evaluation_history(results)


def render_mc_market_indicators(latest_evaluation):
    render_mc_section(
        "MARKET INDICATORS",
        "Current indicator snapshot",
        "Only indicators already exposed by the strategy evaluation are shown.",
    )
    render_mc_metrics(
        [
            ("RSI", f"{latest_evaluation['rsi']:.2f}"),
            ("EMA9", "N/A"),
            ("EMA21", f"${latest_evaluation['ema21']:,.2f}"),
            ("EMA50", f"${latest_evaluation['ema50']:,.2f}"),
            ("EMA200", f"${latest_evaluation['ema200']:,.2f}"),
            ("Volume", "N/A"),
            (
                "Trend state",
                "BULLISH" if latest_evaluation["long_term_trend"] else "NOT BULLISH",
            ),
            (
                "Momentum state",
                "BULLISH" if latest_evaluation["short_term_momentum"] else "WEAK",
            ),
        ],
        columns=2,
    )


def render_mc_orbit_status(snapshot):
    with st.container(border=True):
        st.markdown(
            '<div class="mc-vision-card-title"><span class="mc-vision-icon">'
            "◈</span>Orbit status</div>",
            unsafe_allow_html=True,
        )
        render_mc_metrics(
            [
                (
                    "Observation",
                    snapshot["status"].replace("_", " "),
                ),
                (
                    "Paper balance",
                    f"${snapshot['cash']:,.4f}"
                    if snapshot.get("available")
                    else "Unavailable",
                ),
                (
                    "Current position",
                    "ACTIVE"
                    if float(snapshot.get("position", 0) or 0) > 0
                    else "NO ACTIVE PAPER POSITION",
                ),
                ("Genuine signals", str(snapshot.get("signals", 0))),
                ("Completed paper trades", str(snapshot.get("trades", 0))),
                (
                    "Healthy data",
                    f"{snapshot.get('healthy_ratio', 0) * 100:.1f}%",
                ),
            ],
            columns=3,
        )


def render_mc_observation_progress(snapshot):
    with st.container(border=True):
        st.markdown(
            '<div class="mc-vision-card-title">Observation progress</div>',
            unsafe_allow_html=True,
        )
        criteria = snapshot["criteria"]
        if not snapshot.get("available"):
            st.info("No genuine paper observation evidence yet.")
            return
        render_mc_metrics(
            [
                (
                    "Genuine trades",
                    f"{snapshot['trades']} / {criteria['min_completed_trades']}",
                ),
                (
                    "Observation days",
                    f"{snapshot['observation_days']:.2f} / "
                    f"{criteria['min_observation_days']}",
                ),
                (
                    "Healthy data",
                    f"{snapshot['healthy_ratio'] * 100:.1f}% / "
                    f"{criteria['min_healthy_ratio'] * 100:.0f}%",
                ),
                (
                    "Maximum window",
                    f"{snapshot['observation_days']:.2f} / "
                    f"{criteria['max_observation_days']} days",
                ),
            ],
            columns=2,
        )


def render_mc_overview_navigation():
    destinations = (
        ("Positions", "POSITIONS", "Paper position"),
        ("Strategy", "STRATEGY", "Decision details"),
        ("Performance", "PERFORMANCE", "Results and charts"),
        ("Risk", "RISK", "Guardrail status"),
        ("Options Review", "OPTIONS REVIEW", "Quote-only review"),
        ("Settings", "SETTINGS", "Preferences"),
    )

    with st.container(key="overview_nav_grid"):
        st.markdown(
            '<div role="navigation" aria-label="Explore dashboard details">'
        )
        for row_start in range(0, len(destinations), 4):
            columns = st.columns(4, gap="small")
            for column, (label, destination, description) in zip(
                columns, destinations[row_start:row_start + 4]
            ):
                with column:
                    st.markdown(
                        f'<a class="overview-nav-link" '
                        f'href="?section={destination}" target="_self" '
                        f'title="{escape(description)}">{escape(label)}</a>',
                        unsafe_allow_html=True,
                    )
        st.markdown("</div>", unsafe_allow_html=True)


def render_mc_overview_market(market_data, live_candles):
    with st.container(key="overview_market"):
        if not live_candles:
            st.error(
                "Kraken display data is unavailable. "
                f"{market_data.last_error or 'No candles were returned.'}"
            )
            return
        render_mc_line_chart(
            "BTC/CAD close",
            "Public Kraken market display · not connected to order placement.",
            [candle["close"] for candle in live_candles[-48:]],
            "BTC/CAD close",
            "KRAKEN PUBLIC DATA",
        )
        st.markdown(
            '<a class="overview-market-chart-link" href="?section=MARKET" '
            'target="_self" aria-label="Open detailed Market page"></a>',
            unsafe_allow_html=True,
        )


def _observed_paper_decision(snapshot):
    signal = snapshot.get("last_signal") or {}
    payload = signal.get("payload") if isinstance(signal, dict) else {}
    if not isinstance(payload, dict):
        return "UNAVAILABLE", None
    eligible = payload.get("entry_eligible")
    score = payload.get("strategy_score")
    if not isinstance(eligible, bool):
        return "UNAVAILABLE", score
    return ("BUY" if eligible else "HOLD"), score


def render_mc_overview_live_monitor(latest_evaluation, market_data, live_candles):
    observation_snapshot = load_live_observation_status()
    observed_decision, _ = _observed_paper_decision(observation_snapshot)
    with st.container(key="overview_live_monitor"):
        columns = st.columns((.72, 1.28), gap="large")
        with columns[0]:
            if st.button(
                f"CURRENT DECISION\n\n{observed_decision}",
                key="overview_live_monitor_decision",
                use_container_width=True,
                help="Open the detailed Live Monitor view",
            ):
                st.session_state.overview_destination = "LIVE MONITOR"
                st.rerun()
        with columns[1]:
            if not live_candles:
                st.error(
                    "Kraken display data is unavailable. "
                    f"{market_data.last_error or 'No candles were returned.'}"
                )
                return
            render_mc_line_chart(
                "BTC/CAD close",
                "Public Kraken market display · not connected to order placement.",
                [candle["close"] for candle in live_candles],
                "BTC/CAD close",
                "KRAKEN PUBLIC DATA",
            )


def render_mc_overview_page(
    results,
    latest_evaluation,
    market_data,
    live_candles,
    historical_results,
    historical_market_data,
):
    render_orbit_overview_styles()
    snapshot = load_live_observation_status()
    criteria = snapshot["criteria"]
    trade_progress = min(
        100,
        snapshot.get("trades", 0) / max(criteria["min_completed_trades"], 1) * 100,
    )
    days_progress = min(
        100,
        snapshot.get("observation_days", 0)
        / max(criteria["min_observation_days"], 1)
        * 100,
    )
    observation_state = (
        snapshot["status"].replace("_", " ").title()
        if snapshot.get("available")
        else "Not started"
    )
    last_trade = snapshot.get("last_completed_trade") or {}
    activity_text = (
        f"Paper trade {last_trade.get('trade_number', '')} closed · "
        f"{last_trade.get('reason', 'completed')}"
        if last_trade
        else "No completed genuine paper trade yet"
    )
    st.markdown(
        f"""
        <main class="orbit-overview">
          <header class="orbit-shell">
            <div class="orbit-brand">
              <img src="/app/static/kova/kova-ai-avatar-192.png" alt="Kova Orb">
              <div><div class="orbit-brand-name">Kova</div>
              <div class="orbit-brand-subtitle">Orbit Summary · paper operations</div></div>
            </div>
              <div class="orbit-shell-meta"><span>BTC/CAD</span></div>
          </header>
          <section class="orbit-hero">
            <article class="orbit-panel orbit-observation-card">
              <div class="orbit-eyebrow">Observation status</div>
              <h1>Paper operations progress</h1>
              <p>Genuine paper-observation evidence is tracked here; public market display data and historical simulations remain separate.</p>
              <div class="orbit-mission-footer">
                <span class="orbit-state"><i class="orbit-state-dot"></i>{escape(observation_state)}</span>
              </div>
            </article>
            <div class="orbit-side-stack">
              <article class="orbit-panel">
                <div class="orbit-panel-title">Observation progress <b>{snapshot.get('trades', 0)} / {criteria['min_completed_trades']} trades</b></div>
                <div class="orbit-big-value">{snapshot.get('observation_days', 0):.2f} days</div>
                <div class="orbit-panel-note">Minimum {criteria['min_observation_days']} days · {snapshot.get('healthy_ratio', 0) * 100:.1f}% healthy data</div>
                <div class="orbit-progress"><span style="width:{days_progress:.1f}%"></span></div>
              </article>
            </div>
          </section>
        </main>
        """,
        unsafe_allow_html=True,
    )
    render_mc_performance_page(results)
    render_mc_overview_market(market_data, live_candles)
    activity_column = st.container()
    with activity_column:
        st.markdown(
            f"""
            <section class="orbit-panel">
              <div class="orbit-panel-title">Recent activity</div>
              <div class="orbit-activity">
                <div class="orbit-activity-row">
                  <span class="orbit-activity-icon">↗</span>
                  <div class="orbit-activity-copy">{escape(activity_text)}
                  </div>
                </div>
                <div class="orbit-activity-row">
                  <span class="orbit-activity-icon">◌</span>
                  <div class="orbit-activity-copy">Trade evidence progress
                    <small>{snapshot.get('trades', 0)} of {criteria['min_completed_trades']} required · {trade_progress:.0f}%</small>
                  </div>
                </div>
              </div>
            </section>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('<div style="height:.8rem"></div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <section class="orbit-panel">
              <div class="orbit-panel-title">Observation evidence</div>
              <div class="orbit-panel-note">Signals {snapshot.get('signals', 0)} · completed paper trades {snapshot.get('trades', 0)}</div>
              <div class="orbit-progress"><span style="width:{trade_progress:.1f}%"></span></div>
            </section>
            """,
            unsafe_allow_html=True,
        )
    render_mc_overview_live_monitor(latest_evaluation, market_data, live_candles)
    render_mc_overview_navigation()

def render_mc_performance_page(results):
    trades = results["trades_history"]
    net_values = [trade["net_profit_loss"] for trade in trades]
    average_trade = sum(net_values) / len(net_values) if net_values else 0
    render_mc_section(
        "PERFORMANCE",
        "Historical batch backtest results summary",
    )
    return_percent = results["profit"] / results["starting_capital"] * 100
    render_mc_metrics(
        [
            ("Total return", f"{return_percent:+.2f}%"),
            ("Net P/L", f"${results['profit']:+.4f}"),
            ("Gross P/L", f"${sum(trade['gross_profit_loss'] for trade in trades):+.4f}"),
            ("Win rate", f"{results['win_rate']:.2f}%"),
            ("Completed trades", str(results["trades"])),
            ("Average trade", f"${average_trade:+.4f}"),
            ("Best trade", f"${max(net_values):+.4f}" if net_values else "N/A"),
            ("Worst trade", f"${min(net_values):+.4f}" if net_values else "N/A"),
            ("Maximum drawdown", f"{results['max_drawdown']:.2f}%"),
            ("Fees", f"${results['total_fees']:.4f}"),
            ("Slippage", f"${results['total_slippage']:.4f}"),
        ],
        columns=2,
    )
    render_mc_equity(results)
    render_mc_trade_pnl_chart(results)
    render_mc_recent_trades_table(results)


def render_mc_backtest_page(results):
    render_mc_section(
        "HISTORICAL BATCH BACKTEST",
        "Historical simulation detail",
    )
    render_mc_metrics(
        [
            ("Starting capital", f"${results['starting_capital']:.2f}"),
            ("Evaluated candles", str(len(results["equity_curve"]) - 1)),
            ("Stop loss", f"{STOP_LOSS_PERCENT:.1%}"),
            ("Take profit", f"{TAKE_PROFIT_PERCENT:.1%}"),
            ("Position sizing", f"{MAX_POSITION_PERCENT:.1%}"),
            ("Fees", f"{results['total_fees']:.4f} total"),
            ("Slippage", f"{results['total_slippage']:.4f} total"),
            ("Exit model", "Stop loss / take profit / period end"),
        ],
        columns=2,
    )
    render_mc_trade_history(results)


def render_mc_position_and_operations(results, latest_evaluation, live_candles):
    render_mc_section(
        "RECENT ACTIVITY",
        "Latest paper event",
        "Completed trades from the historical batch backtest.",
    )
    if not results["trades_history"]:
        st.info("No completed backtest trades.")
        return
    last_trade = results["trades_history"][-1]
    st.caption(
        f"Trade {last_trade['trade_number']} · {last_trade['reason']} · "
        f"net ${last_trade['net_profit_loss']:+.4f}"
    )


def render_mc_ai_decision(latest_evaluation, observation_snapshot=None):
    render_mc_section(
        "AI DECISION ENGINE",
        "Strategy decision",
    )
    with st.container(border=True):
        decision, observed_score = _observed_paper_decision(
            observation_snapshot or load_live_observation_status()
        )
        score = max(
            0,
            min(
                100,
                observed_score
                if isinstance(observed_score, (int, float))
                else latest_evaluation["strategy_score"],
            ),
        )
        st.caption(
            "Genuine paper-observation signal · public market display remains "
            "separate from order placement."
        )
        st.markdown(
            f'<div class="mc-decision"><div class="mc-decision-label">'
            f"CURRENT DECISION</div><div class=\"mc-decision-value\">"
            f"{decision}</div><div class=\"mc-score-row\"><span>Score</span>"
            f"<strong>{score}/100</strong></div><div class=\"mc-score-track\">"
            f'<div class="mc-score-fill" style="width:{score}%"></div></div>'
            f"</div>",
            unsafe_allow_html=True,
        )
        render_mc_conditions(latest_evaluation)
        render_mc_metrics(
            [
                ("RSI condition", "PASS" if latest_evaluation["rsi_condition"] else "FAIL"),
                ("Trend condition", "PASS" if latest_evaluation["long_term_trend"] else "FAIL"),
                ("Momentum", "PASS" if latest_evaluation["short_term_momentum"] else "FAIL"),
                ("Volume", "PASS" if latest_evaluation["volume"] else "FAIL"),
                ("Price / EMA21", "PASS" if latest_evaluation["price_above_ema21"] else "FAIL"),
                ("RSI value", f"{latest_evaluation['rsi']:.2f}"),
            ],
            columns=2,
        )


def render_mc_risk_monitor(results, live_candles, observation_snapshot=None):
    render_mc_section(
        "RISK MONITOR",
        "Guardrail status",
        "Configured limits are shown separately from current observed risk.",
    )
    st.markdown("#### Configured limits")
    render_mc_metrics(
        [
            ("Maximum drawdown", f"{results['max_drawdown']:.2f}%"),
            ("Maximum daily loss", f"{MAX_DAILY_LOSS_PERCENT:.1%}"),
            ("Maximum position", f"{MAX_POSITION_PERCENT:.0%}"),
            ("Maximum daily trades", str(MAX_TRADES_PER_DAY)),
        ],
        columns=2,
    )
    st.markdown("#### Current observed risk")
    snapshot = observation_snapshot or load_live_observation_status()
    engine_available = snapshot.get("available", False) and not snapshot.get(
        "store_error"
    )
    position = float(snapshot.get("position", 0) or 0) if engine_available else 0
    current_price = (
        float(live_candles[-1]["close"])
        if live_candles and live_candles[-1].get("close") is not None
        else None
    )
    entry_price = (
        float(snapshot["entry_price"])
        if engine_available and snapshot.get("entry_price") is not None
        else None
    )
    exposure = position * current_price if position and current_price else None
    stop_risk = (
        position * entry_price * STOP_LOSS_PERCENT
        if position and entry_price
        else None
    )
    if position > 0:
        st.info("Active genuine paper position. No live order is available.")
    else:
        st.info(
            "No active position — position exposure, position risk, and "
            "risk utilization are not applicable."
        )
    render_mc_metrics(
        [
            ("Current position", f"{position:.8f} BTC" if position > 0 else "No active position"),
            (
                "Current position risk",
                f"${stop_risk:,.4f} to configured stop" if stop_risk is not None else "Not applicable",
            ),
            (
                "Position exposure",
                f"${exposure:,.4f}" if exposure is not None else "Not applicable",
            ),
            ("Daily loss status", "OBSERVED PAPER STATE"),
            (
                "Daily trade count",
                f"{snapshot.get('trades_today', 0)} / {MAX_TRADES_PER_DAY}"
                if engine_available
                else "Unavailable",
            ),
        ],
        columns=2,
    )


def render_mc_options_review(candidates=None):
    """Display quote-only option analysis; never exposes execution controls."""
    render_mc_section(
        "DEFINED-RISK OPTIONS",
        "Candidate review",
        "Read-only payoff bounds from normalized public quote snapshots. "
        "Rejected candidates remain visible so stale or unsupported data is never hidden.",
    )
    if candidates is None:
        symbol = st.text_input(
            "Underlying symbol",
            value=os.getenv("OPTIONS_REVIEW_SYMBOL", "SPY"),
            max_chars=12,
            help="Quotes come from Yahoo Finance's public, read-only options chain.",
        ).strip()
        refresh_quotes = st.button(
            "Refresh quotes",
            help="Discard the short-lived quote snapshot and request a new public chain.",
        )
        if refresh_quotes:
            fetch_cached_public_option_quote_candidates.clear()
        provider_snapshot = fetch_cached_public_option_quote_candidates(symbol)
        render_mc_metrics(
            [
                ("Quote source", provider_snapshot["source"]),
                ("Underlying", provider_snapshot["symbol"] or "Not specified"),
                ("Snapshot age", _option_snapshot_age(provider_snapshot)),
                (
                    "Provider status",
                    "AVAILABLE" if provider_snapshot["available"] else "UNAVAILABLE",
                ),
                (
                    "Nearest expiration",
                    provider_snapshot.get("expiration", "Unavailable"),
                ),
            ],
            columns=2,
        )
        if not provider_snapshot["available"]:
            st.error(
                "REJECTED · Public option quote provider · "
                f"{provider_snapshot['error']}"
            )
            st.info(
                "No quote has been invented or substituted. Try another symbol "
                "or refresh when the public source is available."
            )
            return
        candidates = provider_snapshot["candidates"]

    reviewed = review_defined_risk_option_candidates(candidates or [])
    if not reviewed:
        st.info(
            "No option quote candidates are available. This view does not "
            "invent quotes and cannot place brokerage orders."
        )
        return

    accepted = [item for item in reviewed if item["status"] == "ACCEPTED"]
    rejected = [item for item in reviewed if item["status"] == "REJECTED"]
    render_mc_metrics(
        [
            ("Candidates reviewed", str(len(reviewed))),
            ("Accepted for comparison", str(len(accepted))),
            ("Rejected visibly", str(len(rejected))),
            ("Execution", "DISABLED"),
        ],
        columns=2,
    )
    for item in reviewed:
        if item["status"] == "REJECTED":
            with st.container(border=True):
                st.error(
                    f"REJECTED · {item['instrument']} · {item['strategy']} · "
                    f"{item['rejection_reason']}"
                )
            continue
        analysis = item["analysis"]
        with st.container(border=True):
            st.markdown(f"#### {item['instrument']} · {analysis['strategy']}")
            render_mc_metrics(
                [
                    ("Quote status", "NORMALIZED / FRESH"),
                    ("Strategy", analysis["strategy"]),
                    ("Break-even", _format_option_price(analysis["break_even"])),
                    ("Maximum profit", _format_option_money(analysis["maximum_profit"])),
                    ("Maximum loss", _format_option_money(analysis["maximum_loss"])),
                    ("Exposure", _format_option_money(analysis["exposure"])),
                    ("Cost", _format_option_money(analysis["cost"])),
                    ("Slippage", _format_option_money(analysis["slippage"])),
                    ("Days to expiration", str(analysis["days_to_expiration"])),
                ],
                columns=3,
            )
    st.caption(
        "Analysis only. No brokerage connection, order ticket, credentials, "
        "margin calculation, or live-trading action is available here."
    )


def _format_option_money(value):
    return "Unlimited" if value == float("inf") else f"${value:,.2f}"


def _format_option_price(value):
    return "N/A" if value is None else f"${value:,.2f}"


def render_mc_research_catalogue():
    studies = (
        (
            "RSI ≥60 study",
            "Does a higher-RSI entry condition produce evidence worth considering?",
        ),
        (
            "Cooldown study",
            "Does a cooldown after a trade improve execution quality?",
        ),
        (
            "Trade sequencing study",
            "Do recent trade sequences reveal repeatable clustering effects?",
        ),
        (
            "Regime study",
            "How does performance vary across Bull, Sideways, and Bear periods?",
        ),
        (
            "Bear-market opportunity study",
            "Are there evidence-backed opportunities during Bear conditions?",
        ),
        (
            "Exit-parameter robustness study",
            "How robust are exit assumptions across independent periods?",
        ),
        (
            "Exit period robustness",
            "Do exit results hold across different historical windows?",
        ),
        (
            "Economic-value selection study",
            "Does a candidate add value after execution economics?",
        ),
        (
            "Execution economics study",
            "How do modeled fees and slippage affect paper outcomes?",
        ),
    )
    render_mc_section(
        "RESEARCH HISTORY",
        "Completed study catalogue",
        "Historical evidence catalogue · no candidates are promoted automatically.",
    )
    for start in range(0, len(studies), 2):
        row = st.columns(2, gap="large")
        for column, (title, question) in zip(row, studies[start:start + 2]):
            with column:
                with st.container(border=True):
                    st.markdown(
                        f'<div class="mc-vision-card-title">{escape(title)}</div>'
                        f'<div class="mc-vision-card-subtitle">'
                        f'<strong>Research question:</strong> {escape(question)}'
                        "</div>",
                        unsafe_allow_html=True,
                    )
                    render_mc_metrics(
                        [
                            ("Evidence class", "Historical"),
                            ("Result", "See study report"),
                        ],
                        columns=2,
                    )


def render_mc_research_status(historical_results):
    with st.expander("Evidence boundary", expanded=False):
        render_mc_section(
            "EVIDENCE CLASSIFICATION",
            "Research interpretation",
        )
        render_mc_metrics(
            [
                ("Latest completed study", "Historical studies"),
                ("Current classification", "Not eligible for production"),
                ("Validation status", "Read-only evidence"),
                (
                    "Historical periods",
                    str(len(historical_results["periods"]))
                    if historical_results else "N/A",
                ),
            ],
            columns=2,
        )
        st.markdown("#### Approved research providers")
        st.caption(
            "Only successful, fresh API responses can supply research inputs. "
            "Configuration alone is not a connection and no estimates are substituted."
        )
        providers = provider_catalog()
        for provider in providers:
            status = provider["status"]
            contract_status = provider.get("contract_status", "NOT_ASSESSED")
            indicator = (
                "red"
                if contract_status == "INVALID" or status in {"UNCONFIGURED", "FAILED"}
                else "green"
                if status == "AVAILABLE"
                else "amber"
            )
            render_mc_status(
                f"{provider['provider']} · {status}",
                indicator,
            )
            render_mc_metrics(
                [
                    ("Domains", ", ".join(provider["domains"])),
                    ("Source", provider["source"]),
                    ("Freshness", (
                        f"{provider['freshness_seconds']:.0f}s"
                        if provider["freshness_seconds"] is not None else "Not fetched"
                    )),
                    ("Quality", provider["quality"]),
                    ("Uncertainty", provider["uncertainty"]),
                    ("Contract status", contract_status),
                    (
                        "Contract reason",
                        provider.get("uncertainty", "not assessed")
                        if contract_status == "INVALID"
                        else "Not assessed until a response is fetched",
                    ),
                ],
                columns=2,
            )
            if provider["error"]:
                st.caption(provider["error"])
            if contract_status == "INVALID":
                st.warning(
                    "Provider contract drift or throttling detected. "
                    "Research readiness remains blocked until the adapter is reviewed."
                )
        readiness = research_readiness(providers)
        if readiness["ready"]:
            st.success("Critical research inputs are available from supported providers.")
        else:
            st.warning(
                "Portfolio scoring and strategy-council research are BLOCKED: "
                + ", ".join(readiness["missing_domains"])
            )


def render_mc_pre_live_diagnostics(
    market_data,
    live_candles,
    historical_results,
    historical_market_data,
):
    with st.expander("Pre-live readiness diagnostics", expanded=False):
        st.caption(
            "Checks for paper safety, evidence integrity, provider availability, "
            "and capability boundaries. Controls below are paper-only and require "
            "an authenticated session plus explicit confirmation; without that "
            "boundary, this dashboard cannot interrupt the active genuine observation."
        )
        snapshot = load_live_observation_status()
        health = getattr(market_data, "health", {}) or {}
        observation_state = (
            snapshot.get("status", "NOT STARTED")
            if snapshot.get("available")
            else "NOT STARTED"
        )
        evidence_state = (
            "RECONCILED"
            if snapshot.get("evidence_reconciled", False)
            else "BLOCKED"
        )
        market_state = health.get(
            "status",
            "HEALTHY" if live_candles else "UNAVAILABLE",
        )
        render_mc_metrics(
            [
                ("Paper mode", "ENABLED" if PAPER_TRADING else "DISABLED"),
                ("Live execution", "DISABLED"),
                ("Observation", observation_state),
                ("Evidence reconciliation", evidence_state),
                ("Risk settings", "2% stop · 4% target"),
                ("Starting capital", f"${STARTING_CAPITAL:.2f}"),
                ("Market provider", market_data.pair_name or "Kraken XBT/CAD"),
                ("Market quality", market_state),
                (
                    "Latest market candle",
                    (
                        format_market_timestamp(live_candles[-1]["timestamp"])
                        if live_candles
                        else "Unavailable"
                    ),
                ),
                ("Historical research", "AVAILABLE" if historical_results else "UNAVAILABLE"),
                (
                    "Historical source",
                    getattr(historical_market_data, "source", None)
                    or "Yahoo BTC/CAD · preflight controlled",
                ),
                ("Options boundary", "DEFINED-RISK · QUOTE-ONLY"),
                ("Experimental strategies", "SEPARATE · NO AUTO-PROMOTION"),
                 ("Voice authority", "USER-ACTIVATED · READ-ONLY"),
                 (
                     "Authenticated runner controls",
                     "AVAILABLE" if _authenticated_user_key() else "AUTHENTICATION REQUIRED",
                 ),
            ],
            columns=2,
        )
        st.markdown("#### Paper observation controls")
        st.info(
            "These controls affect the genuine paper observation loop only. "
            "They never place live orders, change strategy settings, or operate "
            "options or margin."
        )
        control_status = snapshot.get("status", "NOT_STARTED")
        st.caption(f"Current persisted state: **{control_status}**")
        if not _authenticated_user_key():
            st.warning("Sign in to unlock paper controls. KOVA voice remains read-only.")
        else:
            control_columns = st.columns(3)
            control_specs = (
                ("START", "Start or resume the paper observation loop."),
                ("PAUSE", "Pause after the current runner cycle."),
                ("STOP", "Permanently stop this observation period."),
            )
            for column, (action, help_text) in zip(control_columns, control_specs):
                with column:
                    st.checkbox(
                        f"Confirm {action}",
                        key=f"confirm_paper_{action.lower()}",
                        help=help_text,
                    )
                    if st.button(
                        action,
                        key=f"paper_control_{action.lower()}",
                        disabled=(
                            (action == "START" and control_status not in {"NOT_STARTED", "PAUSED"})
                            or (action == "PAUSE" and control_status != "RUNNING")
                            or (action == "STOP" and control_status not in {"NOT_STARTED", "RUNNING", "PAUSED"})
                        ),
                        use_container_width=True,
                    ):
                        try:
                            updated = _apply_authenticated_paper_control(action)
                            st.success(f"Paper observation: {updated['status']}")
                            st.rerun()
                        except (ObservationControlError, ValueError, OSError) as error:
                            st.error(f"Control refused: {error}")
            st.markdown(
                '<div data-testid="paper-controls-ready">Paper controls ready</div>',
                unsafe_allow_html=True,
            )
        st.warning(
            "Live trading, live options, margin, and undefined-risk options remain disabled."
        )


def render_mc_settings_page(
    results,
    market_data,
    live_candles,
    historical_results,
    historical_market_data,
):
    execution_model = StrategyBacktester(STARTING_CAPITAL)
    render_mc_section(
        "SETTINGS & PREFERENCES",
        "Dashboard preferences",
    )

    with st.container(border=True):
        st.markdown(
            '<div class="mc-vision-card-title">Appearance</div>',
            unsafe_allow_html=True,
        )
        st.radio(
            "Appearance",
            APPEARANCE_OPTIONS,
            horizontal=True,
            key="dashboard_appearance",
            on_change=_save_dashboard_appearance_preference,
            label_visibility="collapsed",
        )
        st.caption("Choose the preferred dashboard appearance.")

    trading_column, mode_column = st.columns(2, gap="large")
    with trading_column:
        with st.container(border=True):
            st.markdown(
                '<div class="mc-vision-card-title">Trading configuration</div>'
                "",
                unsafe_allow_html=True,
            )
            render_mc_metrics(
                [
                    ("Starting capital", f"${results['starting_capital']:.2f}"),
                    ("Stop loss", f"{STOP_LOSS_PERCENT:.1%}"),
                    ("Take profit", f"{TAKE_PROFIT_PERCENT:.1%}"),
                    ("Maximum position", f"{MAX_POSITION_PERCENT:.0%}"),
                    ("Daily loss limit", f"{MAX_DAILY_LOSS_PERCENT:.0%}"),
                    ("Daily trade limit", str(MAX_TRADES_PER_DAY)),
                ],
                columns=2,
            )
    with mode_column:
        with st.container(border=True):
            st.markdown(
                '<div class="mc-vision-card-title">Trading mode</div>'
                "",
                unsafe_allow_html=True,
            )
            render_mc_status("PAPER TRADING: ENABLED", "green")
            render_mc_status("Live trading", "red", "DISABLED")

    with st.container(border=True):
        st.markdown(
            '<div class="mc-vision-card-title">Genuine paper engine</div>',
            unsafe_allow_html=True,
        )
        snapshot = load_live_observation_status()
        engine_available = snapshot.get("available", False) and not snapshot.get(
            "store_error"
        )
        render_mc_metrics(
            [
                (
                    "Observation state",
                    snapshot.get("status", "NOT STARTED")
                    if snapshot.get("available")
                    else "NOT STARTED",
                ),
                (
                    "Runner",
                    snapshot.get("runner_status", "UNKNOWN")
                    if snapshot.get("available")
                    else "NOT STARTED",
                ),
                (
                    "Current decision",
                    snapshot.get("last_signal", "Unavailable")
                    if engine_available
                    else "Unavailable",
                ),
                (
                    "Paper balance",
                    f"${snapshot['cash']:,.4f}" if engine_available else "Unavailable",
                ),
                (
                    "Position",
                    f"{float(snapshot.get('position', 0) or 0):.8f} BTC"
                    if engine_available
                    else "Unavailable",
                ),
                (
                    "Evidence",
                    f"{snapshot.get('trades', 0)} completed trades"
                    if engine_available
                    else "Unavailable",
                ),
            ],
            columns=2,
        )
        if snapshot.get("store_error"):
            st.error(snapshot["store_error"])

    economics_column, data_column = st.columns(2, gap="large")
    with economics_column:
        with st.container(border=True):
            st.markdown(
                '<div class="mc-vision-card-title">Execution economics</div>'
                "",
                unsafe_allow_html=True,
            )
            render_mc_metrics(
                [
                    ("Fee assumption", f"{execution_model.fee_percent:.2%}"),
                    ("Slippage assumption", f"{execution_model.slippage_percent:.2%}"),
                    ("Entry cost treatment", "MODELED"),
                    ("Exit cost treatment", "MODELED"),
                    ("Break-even reference", "COST-AWARE"),
                    ("Production results fees", f"${results['total_fees']:.4f}"),
                ],
                columns=2,
            )
    with data_column:
        with st.container(border=True):
            st.markdown(
                '<div class="mc-vision-card-title">Data configuration</div>'
                "",
                unsafe_allow_html=True,
            )
            render_mc_metrics(
                [
                    ("Live market source", market_data.pair_name or "Kraken XBT/CAD"),
                    ("Historical market source", "Yahoo BTC/CAD"),
                    ("Historical window", getattr(historical_market_data, "data_range", "N/A")),
                    ("Refresh behavior", "ON DASHBOARD RERUN"),
                    ("Yahoo preflight", "REQUIRED"),
                    ("Data mode", "DISPLAY ONLY"),
                ],
                columns=2,
            )

    preferences_column, research_column = st.columns(2, gap="large")
    with preferences_column:
        with st.container(border=True):
            st.markdown(
                '<div class="mc-vision-card-title">Display preferences</div>'
                "",
                unsafe_allow_html=True,
            )
            render_mc_metrics(
                [
                    ("Dashboard density", "STANDARD"),
                    ("Chart style", "VISION SVG"),
                    ("Recent trades", "ALL COMPLETED"),
                    ("Refresh interval", "ON RERUN"),
                ],
                columns=2,
            )
    with research_column:
        with st.container(border=True):
            st.markdown(
                '<div class="mc-vision-card-title">Research configuration</div>'
                "",
                unsafe_allow_html=True,
            )
            render_mc_metrics(
                [
                    ("Active production control", f"{STOP_LOSS_PERCENT:.1%} / {TAKE_PROFIT_PERCENT:.1%}"),
                    ("Research candidates", "Not eligible for production"),
                    ("Promotion state", "NO AUTO-PROMOTION"),
                    ("Validation policy", "READ-ONLY EVIDENCE"),
                ],
                columns=2,
            )

    st.markdown('<div class="mc-kicker">ADVANCED OPERATIONS</div>', unsafe_allow_html=True)
    st.subheader("Operations library")
    st.caption("Detailed operational views are grouped here to keep the main dashboard focused.")
    with st.expander("System health", expanded=False):
        render_mc_system_health(
            strategy_ready=bool(results),
            paper_ready=bool(
                PAPER_TRADING and not LIVE_TRADING and historical_results is not None
            ),
            live_candles=live_candles,
            market_data=market_data,
            historical_results=historical_results,
            historical_market_data=historical_market_data,
        )
    with st.expander("Backtest detail", expanded=False):
        render_mc_backtest_page(results)
    with st.expander("Research catalogue", expanded=False):
        render_mc_research_catalogue()
        render_mc_research_lab(historical_results, historical_market_data)
        render_mc_research_status(historical_results)
    render_mc_pre_live_diagnostics(
        market_data,
        live_candles,
        historical_results,
        historical_market_data,
    )


def render_mc_trade_activity(results):
    render_mc_section(
        "TRADE ACTIVITY",
        "Recent backtest activity",
        "Only completed trades from the historical batch backtest are shown.",
    )
    if not results["trades_history"]:
        st.info("No completed backtest trades.")
        return
    for trade in reversed(results["trades_history"][-6:]):
        net = trade["net_profit_loss"]
        state = "mc-condition-pass" if net >= 0 else "mc-condition-fail"
        st.markdown(
            f'<div class="mc-condition {state}"><code>'
            f"TRADE {trade['trade_number']:02d} · "
            f"{trade['reason']} · NET ${net:+.4f}</code></div>",
            unsafe_allow_html=True,
        )


def render_dashboard():
    st.set_page_config(
        page_title="Kova",
        page_icon="◉",
        layout="wide",
        initial_sidebar_state="auto",
    )
    if "dashboard_appearance" not in st.session_state:
        st.session_state.dashboard_appearance = (
            _load_saved_dashboard_appearance() or "Dark"
        )
    elif st.session_state.dashboard_appearance not in APPEARANCE_OPTIONS:
        st.session_state.dashboard_appearance = "Dark"
    inject_mission_control_theme(st.session_state.dashboard_appearance)

    results = run_strategy_backtest()
    latest_evaluation = results["evaluation_history"][-1]

    market_data, live_candles = load_kraken_market_data()

    real_market_results = run_live_market_backtest(live_candles)
    historical_market_data, historical_candles = (
        load_historical_btc_cad_data()
    )
    historical_market_results = run_historical_market_backtest(
        historical_candles
    )

    selected_section = render_mc_navigation()
    if selected_section != "OVERVIEW":
        render_mc_header(
            selected_section,
            latest_evaluation,
            market_data,
            live_candles,
        )
        render_mc_return_to_overview()
    if selected_section == "OVERVIEW":
        render_mc_overview_page(
            results,
            latest_evaluation,
            market_data,
            live_candles,
            historical_market_results,
            historical_market_data,
        )
    elif selected_section == "LIVE MONITOR":
        signal_column, market_column = st.columns((.88, 1.32), gap="large")
        with signal_column:
            render_mc_ai_decision(
                latest_evaluation,
                load_live_observation_status(),
            )
            render_mc_position_snapshot(results, latest_evaluation, live_candles)
        with market_column:
            render_mc_live_market(market_data, live_candles)
            render_mc_market_indicators(latest_evaluation)
    elif selected_section == "STRATEGY":
        render_mc_strategy_page(results, latest_evaluation, live_candles)
    elif selected_section == "POSITIONS":
        render_mc_position_snapshot(results, latest_evaluation, live_candles)
        render_mc_recent_trades_table(results)
    elif selected_section == "PERFORMANCE":
        render_mc_performance_page(results)
    elif selected_section == "RISK":
        render_mc_risk_monitor(
            results,
            live_candles,
            load_live_observation_status(),
        )
    elif selected_section == "OPTIONS REVIEW":
        render_mc_options_review()
    elif selected_section == "MARKET":
        render_mc_live_market(market_data, live_candles)
        render_mc_market_indicators(latest_evaluation)
    elif selected_section == "RESEARCH":
        render_mc_research_catalogue()
        render_mc_research_lab(
            historical_market_results,
            historical_market_data,
        )
        render_mc_research_status(historical_market_results)
    elif selected_section == "BACKTEST":
        render_mc_backtest_page(results)
    elif selected_section == "SYSTEM":
        render_mc_system_health(
            strategy_ready=bool(results),
            paper_ready=bool(
                PAPER_TRADING and not LIVE_TRADING and real_market_results is not None
            ),
            live_candles=live_candles,
            market_data=market_data,
            historical_results=historical_market_results,
            historical_market_data=historical_market_data,
        )
    elif selected_section == "SETTINGS":
        render_mc_settings_page(
            results,
            market_data,
            live_candles,
            historical_market_results,
            historical_market_data,
        )
    render_kova_voice_assistant(
        results,
        latest_evaluation,
        market_data,
        live_candles,
        historical_market_results,
    )

if __name__ == "__main__":
    render_dashboard()
