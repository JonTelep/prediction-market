# prediction-market — Conventions (extracted 2026-07-15, HEAD f06ac00)

Ground truth for this repository. Every claim carries a `file:line` anchor or a command that was actually run. Consumers should treat this as a snapshot at commit `f06ac00`.

## Commands

- Full test suite: `uv run pytest tests/` — **147 passed in 1.43s** at HEAD (verified by running it). No network access occurs in tests; respx raises on unmocked routes.
- Dev-dependency install: `uv sync --extra dev`. **The README's `uv sync --dev` is wrong** — pytest/ruff live under `[project.optional-dependencies].dev` (pyproject.toml:22-28), which `--dev` does not install (verified: fresh `uv sync --dev` → `error: Failed to spawn: pytest`).
- Lint: `uv run ruff check src/ tests/` — passes at HEAD. Format: `uv run ruff format src/ tests/` — **not currently enforced**: `ruff format --check` would reformat 20 of 46 files. Do not mass-reformat in a functional change; it will drown the diff.
- No CI exists (`.github/workflows/` absent). The Makefile is the command source of truth: `make test|lint|fmt` map to the uv commands above; `make build|run|monitor|scan|backfill|shell` are podman targets (Makefile:1-27).
- CLI entry point: `prediction-market` → `prediction_market.cli:main` (pyproject.toml:31). Subcommands: `monitor [--agent info-leak|manipulation]`, `scan`, `backfill --days N`, `markets`, `reports [--severity ...|--agent ...|--limit N]`, `report ID [--json]` (cli.py:95-477).
- No codegen, spec-sync, or migration rituals exist.

## Hard invariants

- **SQLite schema is a single `SCHEMA_SQL` executescript with `CREATE TABLE IF NOT EXISTS`** (store/database.py:14-132, applied at :148). There is no migration framework and no schema-version table. Adding a column to an existing table requires explicit `ALTER TABLE` handling — `IF NOT EXISTS` silently skips changed definitions on existing DBs.
- **Pragmas on every connection**: `journal_mode=WAL`, `synchronous=NORMAL`, `foreign_keys=ON` (store/database.py:143-145). `get_database()` (database.py:155-157) just re-runs `init_database` — there is no lightweight connection path.
- **Timestamps are TEXT `"%Y-%m-%d %H:%M:%S"` strings compared lexicographically** (store/queries.py:15-29, comment at 16-19). All new queries must format times the same way or range filters silently break.
- **Polymarket string-typed JSON quirk**: Gamma returns `outcomes`, `outcomePrices`, `clobTokenIds` as JSON-encoded strings; `GammaMarket` normalizes via `_parse_json_list` field validator (data/polymarket/models.py:18-30, 59-62). Order-book/trade `price`/`size` arrive as strings with `*_float` conversion properties (models.py:71-83, 180-219).
- **Config precedence**: env vars > custom TOML (`--config`) > `config/default.toml` (config.py:106-140). Only `DATABASE_PATH` is deep-merged; `congress_api_key`, `courtlistener_token`, `newsapi_key`, `webhook_url`, `log_level` come exclusively from env vars — TOML values for these are ignored (config.py:126-139). `ReportingConfig.webhook_url` (config.py:58) is dead; the orchestrator reads only top-level `AppConfig.webhook_url` (orchestrator.py:591).
- **Missing API keys degrade soft**: CongressClient, CourtCalendarClient, and the NewsAPI half of NewsChecker warn once at init and return `[]` from every method (congress.py:113-118, court_calendar.py:97-102, news_checker.py:91-96). They never raise. Code must not interpret `[]` as "no events exist."
- **No HTTP retry anywhere**: all clients do a single request + `raise_for_status()` (gamma_client.py:34-39, clob_client.py:38-44, data_client.py). Rate limiting exists only for the three Polymarket clients via `TokenBucketRateLimiter` (rate_limiter.py:9-45).

## Known-broken wiring (verified at HEAD — the most important facts in this document)

The headline pipeline has never executed. All of the following were verified by direct read at HEAD:

1. **Orchestrator calls `agent.run(markets=..., db=..., ...)`** (orchestrator.py:547-555) but no agent class defines `run` — `BaseAgent` exposes only `start()/stop()/_loop()/tick()` (agents/base.py:63-102). The `AttributeError` is swallowed by `except Exception` and retried forever with backoff (orchestrator.py:562-574).
2. **`_build_agents()` constructs agents with one positional arg** — `InfoLeakDetector(self.config)`, `ManipulationGuard(self.config)` (orchestrator.py:609, 616) — but both constructors require `db` (info_leak_detector.py:117-121, manipulation_guard.py:141-148). The `TypeError` is swallowed (orchestrator.py:608-619); `_build_agents` returns `[]` and `start()` logs "No agents were initialised".
3. **`InfoLeakDetector.__init__` calls `super().__init__(name=..., tick_interval=..., config=...)`** (info_leak_detector.py:122-126) — `BaseAgent.__init__` accepts `(config, db, sinks)` (base.py:27-37). Constructing it raises `TypeError` unconditionally. It also defines `on_start`/`on_stop` hooks (info_leak_detector.py:139-174) that `BaseAgent` never calls, and never implements the abstract `tick_interval_seconds`.
4. **InfoLeakDetector calls 7 nonexistent `store.queries` functions**: `get_latest_snapshot`, `get_scheduled_events_in_range`, `get_market`, `get_latest_orderbook_snapshot`, `insert_anomaly_report`, `get_active_political_markets`, `upsert_rolling_stats` (info_leak_detector.py:143,201,268,314,351,452,490,501,510). Real queries.py exports: `get_recent_snapshots`, `get_price_history`, `get_volume_history`, `get_recent_orderbooks`, `get_market_trades`, `get_anomaly_reports`, `save_anomaly_report`, `get_rolling_stats`, `save_rolling_stats` (queries.py:63-375). Even `get_rolling_stats` is called with a wrong signature (`stat_types=[...]`, no `market_id` — info_leak_detector.py:143-145 vs queries.py:335-338).
5. **InfoLeakDetector misuses the analyzers**: `PriceAnalyzer(window_days=...)`, `VolumeAnalyzer(window_days=...)`, `RollingStats(window_days=...)` (info_leak_detector.py:51-53) — none accept that kwarg (`PriceAnalyzer(thresholds)` price_analyzer.py:74-78; `RollingStats(window: timedelta)` timeseries.py:54). It calls `analyzer.update(price, timestamp)` expecting a returned anomaly (info_leak_detector.py:214-219); real signature is `update(market_id, price, timestamp)` returning `None`, with detection in a separate `check_anomaly(market_id)` (price_analyzer.py:89-168).
6. **InfoLeakDetector builds `AnomalyReport` without required `id` and `market_question` fields** (info_leak_detector.py:436-448 vs anomaly_report.py:21-24) — `TypeError`.
7. **`scheduled_events` table has no writer** (schema at database.py:85-95; no INSERT anywhere in src/ or scripts/). The event-amplifier path can never fire on real data. `CongressClient`, `CourtCalendarClient`, `WhiteHouseClient` are exported (data/external/__init__.py:8-25) but **never instantiated outside their own modules**.
8. **WebSocket clients are dead code**: `ClobWebSocket` (ws_market.py) and `RtdsWebSocket` (ws_rtds.py) are referenced only in their own files despite docstrings claiming they are the "primary data source" (ws_market.py:5-6).
9. **Two divergent `LiquidityAnalyzer` implementations**: the live one inline in agents/manipulation_guard.py:33-122, and a dead one in analysis/liquidity_analyzer.py (different HHI scale — fraction-squared vs 0–10,000; different depth formula). `CorrelationDetector` (analysis/correlation.py) is likewise never imported by anything.
10. **Two severity scales coexist**: `AnomalyReport.severity_from_score` staircases a 0–1 score (anomaly_report.py:36-52, used by ManipulationGuard at manipulation_guard.py:395); `InfoLeakDetector._classify_severity` staircases a raw z-norm score ≥8/≥6/≥4 (info_leak_detector.py:522-531). Nothing keeps `severity` consistent with `anomaly_score` in the DB.
11. **`scripts/backfill_markets.py` re-implements political classification** (lines 39-88) instead of importing `PoliticalFilter`, with **behaviorally different volume semantics**: the script returns non-political below `min_volume` (lines 84-85); `PoliticalFilter.classify` keeps `is_political=True` and only appends a reason (political_filter.py:66-68). `scripts/snapshot_political_markets.py:42-77` likewise duplicates upsert SQL that exists in store/snapshots.py.
12. **Shipped `data/prediction_market.db`**: 65 real markets (all `first_seen 2026-02-25 04:36:34`), **zero rows** in snapshots, orderbook_snapshots, trades, scheduled_events, anomaly_reports, rolling_stats — the pipeline has never run end-to-end.

ManipulationGuard is the one agent whose constructor and internals match reality (manipulation_guard.py:141-159 matches base.py:27-37); it is broken only by the orchestrator's `agent.run()` call and single-arg construction.

## Boundaries

- `agents/base.py` imports config + reporting only; DB access is via injected `aiosqlite.Connection` — agents must not import `store.database` or `data.*` clients (base.py respects this; manipulation_guard imports `data.polymarket.models`/`rate_limiter` for API access, orchestrator.py:20-46 owns client construction).
- `store/*` imports nothing from `agents`, `orchestrator`, or `reporting` (no cycles). `store/snapshots.py` imports `data.polymarket.models` only.
- Boundary violations that exist today (do not imitate): ManipulationGuard runs raw SQL against `orderbook_snapshots`/`markets` (manipulation_guard.py:219-235, 247-294) bypassing store/queries; the CLI does the same (cli.py:220-237, 309-336, 390-403); the orchestrator reaches into `PoliticalFilter._min_volume` (orchestrator.py:414).
- Client construction convention: `__init__(self, config: AppConfig, http_client: httpx.AsyncClient | None = None)` with `_owns_client` flag and `async def close()` — followed by all seven clients. There is no ABC/Protocol/registry; there are no extension seams. `_extract_keywords`/`_parse_datetime` are copy-pasted across congress.py:26-89, court_calendar.py:26-73, white_house.py:33-80.

## Testing idioms & gotchas

- Runner: pytest with `asyncio_mode = "auto"` (pyproject.toml) — `@pytest.mark.asyncio` decorators throughout the suite are redundant but conventional here.
- HTTP mocking: respx everywhere (`@respx.mock` + route mocks; `.side_effect = [...]` for pagination sequences, e.g. tests/unit/test_data_client.py:71-74). Unmocked routes raise — no test can silently hit the network.
- "integration" = real temp SQLite (`tmp_path / "test.db"`) or multi-module orchestration, **never live APIs**. The registered `integration` marker (pyproject.toml) is never applied to any test — the split is by directory only.
- Fixtures: `tests/conftest.py` provides `app_config` (real `load_config()` with tmp_path overrides, conftest.py:13-19), `sample_political_market`, `sample_nonpolitical_market`, `sample_orderbook`, `thin_orderbook`. Static API payloads live in `tests/fixtures/*.json` (gamma_markets, holders, orderbook, price_history, trades).
- No golden files, no skips/xfails at HEAD.
- **Coverage holes**: no test file exists for `info_leak_detector`, `manipulation_guard`, `agents/base`, `analysis/liquidity_analyzer`, the websocket clients, `congress`/`court_calendar`/`white_house` clients, or the orchestrator's `start()` agent path (tests/integration/test_orchestrator.py covers only `scan_once`/`backfill`). This is why the suite is green while the pipeline is broken.
- Weak assertions to tighten when touched: `test_price_analyzer.py:43-49` (`data is not None` only), `test_orchestrator.py:105` (`total_points >= 0`), loose `pytest.approx(abs=100)` in test_liquidity_analyzer.py, timing-sensitive sleeps in test_rate_limiter.py.
- Timing tests use real `asyncio.sleep` (test_rate_limiter.py) — flake-prone; don't add more wall-clock-dependent tests.

## Seams (for extension)

- `ReportSink` ABC (reporting/sink.py:24-32) — the only real designed seam. Implement `write` (+optional `close`); existing implementations to imitate: `FileSink` (:35-56), `StdoutSink` (:59-69), `WebhookSink` (:72-133, swallows failures after 3 retries), `CompositeSink` (:136-160, per-sink error isolation).
- `BaseAgent` (agents/base.py) — subclass with `name`, `tick_interval_seconds`, `tick()`; `emit()` persists to `anomaly_reports` then fans out to sinks (base.py:106-157). `ManipulationGuard` is the reference implementation.
- `TokenBucketRateLimiter` (data/polymarket/rate_limiter.py:9-45) — reusable, currently used only by the three Polymarket clients.

## Vocabulary

- **market** = one Polymarket binary market (`GammaMarket`, keyed `id`/`slug`/`condition_id`). **event** = external scheduled political event (`ScheduledEvent`), *not* Polymarket's event grouping.
- **condition_id** = market-level CLOB identifier; **token_id / asset_id / clob_token_ids** = per-outcome identifiers. Positional convention `clob_token_ids[0]`=YES, `[1]`=NO is assumed, unvalidated (orchestrator.py:357-358, 496-499).
- **snapshot** = point-in-time price/volume row (`snapshots` table); **orderbook snapshot** = separate table + 300s sub-cadence inside the 60s snapshot loop (orchestrator.py:489-529). **backfill** = historical price-history + trades ingestion. **tick** = one agent evaluation cycle.
- Agent names as stored in `anomaly_reports.agent`: `info_leak`, `manipulation`; CLI `--agent` flags use `info-leak`/`manipulation` (cli.py:95-120).
- Severity levels: low/medium/high/critical, ordered by `SEVERITY_ORDER` (store/models.py:20-25); CLI `--severity` is a minimum-severity filter.

## Discrepancies found (docstring/README vs code)

- README architecture diagram shows WS feeds, external sources, and agents wired to the orchestrator — WS clients and the three government-calendar clients are dead code; agents never construct (see Known-broken wiring).
- README Quick Start `uv sync --dev` does not install dev tools (needs `--extra dev`).
- `gamma_client.search_markets` (gamma_client.py:102-108) filters by slug; it is not a search.
- `white_house.py:3` claims scraping; implementation calls the WordPress REST API. `_BRIEFINGS_URL` (white_house.py:30) is unused.
- `news_checker.py:12-13` claims NewsAPI free-tier 100 req/day awareness; nothing enforces it (only reactive 429 handling at news_checker.py:332-338).
- price_analyzer docstring claims EWMA-based scoring; z-scores actually come from `RollingStats` population stdev of log-returns (price_analyzer.py:122-168); EWMA is reporting context only. `RollingStats.std` uses population (÷n) variance (timeseries.py:99).
- Containerfile has no `HEALTHCHECK`; default CMD is `scan` (Containerfile:28-29).
