# Kova BTC/CAD Operations Centre

Kova is a read-only Streamlit operations dashboard for a frozen BTC/CAD
paper-trading strategy. It presents three explicitly separate sources:

- **Live market display** — public Kraken BTC/CAD candles and health diagnostics.
- **Historical batch backtest** — a repeatable historical simulation, never paper evidence.
- **Genuine paper observation** — append-only evidence emitted only by the
  incremental paper runner after it is armed on live public data.

## Safety boundaries

- Paper mode is enabled; live execution is disabled.
- The frozen starting capital, strategy thresholds, costs, sizing, exits, and
  risk controls are defined in `config.py`.
- The dashboard's authenticated paper controls can start, pause, or stop the
  paper observation loop only. Kova voice remains read-only and cannot invoke
  controls. Neither surface can trade live, write observations, edit
  configuration, or access credentials.
- Historical records and paper-operational records remain separate datasets.
- Observation completion requires genuine completed trades, elapsed duration,
  and data-health criteria. Historical results cannot satisfy those criteria.

## Run and validate

- `uv run streamlit run dashboard.py --server.address 0.0.0.0 --server.port 5000`
  — Streamlit dashboard.
- `uv run python paper_observation_runner.py` — public-data paper observation
  runner; it requires explicit observation criteria environment variables.
- `pnpm --filter @workspace/api-server run dev` — authenticated read-only API.
- `uv run python release_check.py --allow-missing-published-url` — complete
  local release gate.

The release check runs regression, offline-report coverage, public-data
preflight, API restart/readiness, and PWA asset validation. A published URL,
when available, must be supplied so hosted PWA assets are checked rather than
skipped.

## Architecture map

- `config.py` — frozen operational policy.
- `strategy.py`, `risk_manager.py`, `strategy_backtest.py` — strategy,
  risk policy, and historical execution simulation.
- `incremental_paper_engine.py`, `paper_observation_runner.py`,
  `observation_controller.py` — genuine paper observation runtime.
- `observation_store.py`, `paper_observation_adapter.py` — validated
  append-only JSONL evidence boundary.
- `market_data_health.py`, `kraken_live_data.py` — public market data and
  integrity checks.
- `research_providers.py` — opt-in, read-only approved API adapters with
  normalized provenance, freshness, quality, uncertainty, and fail-closed
  readiness status.
- `dashboard.py`, `ai_operations_assistant.py`, `static/` — read-only Kova
  operations UI and PWA assets.
- `artifacts/api-server/` — optional authenticated observation API. `/healthz`
  is liveness; `/readyz` shows whether its observation authentication boundary
  is configured.

## Operational notes

The evidence store is intentionally JSONL, not a transactional database.
Appends are file-locked and fsynced to protect idempotency during overlapping
process attempts. Engine state and evidence are still separate durable files;
reconciliation failures stop observation safely rather than manufacturing or
repairing evidence. Any storage migration requires a separate controlled task.

No profitability, observation completion, or real-money readiness is implied
by dashboard status or historical research. Before any future live-trading
proposal, use separate reviews for exchange authentication, custody, order
execution, monitoring, incident response, regulatory obligations, and a
controlled migration from paper evidence.