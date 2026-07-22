# PHASE1-END-TO-END.md — Make the surveillance pipeline real

A series of standalone implementation prompts for `prediction-market`, designed to be
handed one at a time to a Claude agent for implementation, with each result reviewed
before the next prompt runs. Phase context: at HEAD `f06ac00` the headline pipeline has
never executed — the orchestrator's agent wiring is broken in ways the (green, 147-test)
suite never exercises. Ground truth for every claim below is `docs/CONVENTIONS.md`;
methodology rationale is `docs/RESEARCH-BRIEF.md` (implementers do not need to read the
brief — every decision it informed is stated inline here).

**The non-negotiable outcome of this phase:** `prediction-market monitor`'s pipeline
executes end-to-end — `Orchestrator` constructs both agents fail-fast (construction
errors abort startup loudly instead of being swallowed), `BaseAgent.start()` drives
their ticks, and a seeded anomaly produces persisted `anomaly_reports` rows from
**both** agents. Enforced permanently by `tests/integration/test_pipeline.py`
(`test_info_leak_pipeline_emits_report`, `test_manipulation_pipeline_emits_report`,
`test_orchestrator_constructs_agents`), written in Prompt 3, which must stay green
forever.

**Scope guard — this series deliberately does NOT:**
- Wire the WebSocket clients (`ws_market.py`, `ws_rtds.py`) — Phase 3; Prompt 6 only
  corrects their docstrings to stop claiming they are the "primary data source".
- Add wallet-level/on-chain analysis, Sybil clustering, or Polygon RPC — Phase 2
  (see docs/RESEARCH-BRIEF.md §Deferred).
- Add CUSUM/Bayesian change-point detection — Phase 2. Prompt 5's logit transform is
  the only methodology change in this phase.
- Add a schema-migration framework, a backtest harness against known insider cases,
  or live-API contract verification (CLOB V2 cutover risk is documented in the brief).
- Mass-reformat the codebase. `ruff format` is not enforced at HEAD (20 of 46 files
  differ); format only files you materially edit, and never as a standalone diff.

**How to use:** paste one prompt (the full section between `---` separators) into a
fresh session. Prompts assume the ones before them have landed — run them in order.
Each prompt is self-contained: it references only artifacts that exist in the codebase
by the time it runs. If a referenced helper, function, or file does not exist, that
signals a prerequisite session did not land — stop and report rather than improvise.

**Review loop:** after each prompt lands, review the diff for correctness, test honesty
(do the tests actually exercise the claim, or could they pass with the claim false?),
and scope creep. Loop fixes back to the same agent until the prompt's acceptance
criteria pass, then move on.

**Sequencing:** 1 → 2 → 3 strictly; 4, 5, 6 each require 3 and are mutually
independent (disjoint files); 7 runs last (it touches test files Prompts 5 and 6
also touch).

**Human checkpoint recommended:** Prompt 2 (full-file rewrite of the info-leak
detector) and Prompt 3 (orchestrator lifecycle cutover — the largest blast radius in
the series). For both: run in plan mode and review the plan before letting the agent
code.

---

## Prompt 1: Store-layer query functions the agents need

**Goal:** `store/queries.py` gains the five read/write functions the surveillance agents
require, and the `scheduled_events` table gains a uniqueness guarantee so event
ingestion is idempotent. No agent or orchestrator code changes in this prompt — this is
pure store-layer foundation plus its tests.

**Affected packages:** `src/prediction_market/store/` (`queries.py`, `database.py`),
`tests/integration/test_queries.py`.

**Details:**
- Follow the existing idioms in `store/queries.py` exactly: module-level `async def`
  functions taking `db: aiosqlite.Connection` first, `_fetch_all_dicts`/`_fetch_one_dict`
  helpers (queries.py:32-60), and TEXT timestamps formatted `"%Y-%m-%d %H:%M:%S"`
  compared lexicographically (`_hours_ago`/`_utcnow`, queries.py:15-29 — the comment at
  16-19 explains why). Every new time parameter must be formatted through the same
  pattern; an ISO-8601 string with a `T` separator will silently fail range filters.
- Add these functions (signatures are decisions, not suggestions):
  - `async def get_latest_snapshot(db, market_id: str) -> dict | None` — newest row
    from `snapshots` for the market by `timestamp` DESC LIMIT 1.
  - `async def get_latest_orderbook_snapshot(db, market_id: str) -> dict | None` —
    newest row from `orderbook_snapshots` for the market (across all token_ids) by
    `timestamp` DESC LIMIT 1.
  - `async def get_active_political_markets(db) -> list[dict]` — rows from `markets`
    WHERE `active = 1 AND political_confidence > 0`, ORDER BY `volume` DESC, with
    `clob_token_ids` JSON-decoded into a `token_ids` list key on each dict (fall back
    to `[]` on decode failure, mirroring `get_rolling_stats`'s swallow-and-default at
    queries.py:368-370). The `political_confidence > 0` predicate matches what the
    CLI already treats as "tracked" (cli.py:220-237). This function returns full
    market rows (including `question`) — consumers must not re-fetch per market.
  - `async def get_scheduled_events_in_range(db, start: datetime, end: datetime) -> list[dict]`
    — rows from `scheduled_events` WHERE `event_date` between the two formatted
    timestamps, ORDER BY `event_date` ASC, `keywords` JSON-decoded to a list.
  - `async def save_scheduled_events(db, events: list[ScheduledEvent]) -> int` —
    `ScheduledEvent` is the dataclass at
    `src/prediction_market/data/external/models.py:9-24` (fields: source, event_type,
    title, description, event_date, url, keywords). Insert via
    `INSERT OR IGNORE` + `executemany`, `event_date` formatted
    `"%Y-%m-%d %H:%M:%S"`, `keywords` JSON-encoded. Commit once. Return the number of
    newly inserted rows using the same rowcount-fallback idiom as
    `save_trades_batch` (store/snapshots.py:247-295). This is the table's **only**
    writer — nothing else in the repo writes `scheduled_events` today (verified at
    HEAD; see docs/CONVENTIONS.md "Known-broken wiring" item 7).
- Idempotency: add
  `CREATE UNIQUE INDEX IF NOT EXISTS idx_events_unique ON scheduled_events(source, title, event_date);`
  to `SCHEMA_SQL` alongside the existing index block (database.py:124-131). The table
  is empty in every known DB (including the shipped `data/prediction_market.db`), so
  the unique index applies cleanly to existing files via the normal
  `init_database` path — no migration machinery is needed or wanted.
- Scope creep prohibitions: do NOT modify `InfoLeakDetector`, `ManipulationGuard`, or
  the orchestrator in this prompt (later sessions own those). Do NOT rename or "fix"
  the existing broken calls in `info_leak_detector.py` — that file is rewritten
  wholesale in a later session. Do NOT add functions beyond the five listed (in
  particular, no per-market `get_market` helper — nothing in this phase consumes one).

**Testing:** extend `tests/integration/test_queries.py` (temp-SQLite idiom already used
throughout that file). Claims to prove:
- `get_latest_snapshot` returns the row with the greatest timestamp when three
  snapshots with distinct timestamps exist, and `None` for an unknown market.
- `get_active_political_markets` excludes `active = 0` rows and
  `political_confidence = 0` rows, orders by volume descending, and `token_ids` is a
  decoded Python list (not a JSON string).
- `save_scheduled_events` returns 2 on first insert of two events; calling it again
  with the same two events returns 0 and the table still holds exactly 2 rows (this is
  the idempotency proof — it fails without the unique index).
- `get_scheduled_events_in_range` returns an event dated inside the range and excludes
  one dated outside it; `keywords` round-trips as a list. Construct the in-range event
  from a `datetime` and query with `datetime` bounds — this proves the string
  formatting is consistent end-to-end.

**Invariants:** every existing function in `queries.py` is byte-identical (additions
only). `SCHEMA_SQL` changes are limited to the one new index line. No new
dependencies. All 147 existing tests pass unmodified — any expectation change is a red
flag; justify each one.

**Guardrails (mandatory):**
- Run `uv run pytest tests/` and make it pass before finishing (147 existing + your
  new tests).
- Run `uv run ruff check src/ tests/` and make it pass.
- If any existing `queries.py` function or the schema differs from what is described
  above, stop and report the discrepancy instead of improvising.
- Do not commit or push unless explicitly asked.

---

## Prompt 2: Rewrite InfoLeakDetector onto the real BaseAgent contract

**Goal:** `src/prediction_market/agents/info_leak_detector.py` becomes a working agent.
At HEAD the class cannot even construct: its `__init__` calls
`super().__init__(name=..., tick_interval=..., config=...)` (info_leak_detector.py:122-126)
against a `BaseAgent.__init__` that accepts `(config, db, sinks)` (agents/base.py:27-37),
it calls seven `store.queries` functions that don't exist, and it misuses the analyzers'
APIs. This prompt is a **full-file rewrite** — treat the existing file as a statement of
intent (its module docstring at lines 1-16 describes the right pipeline *steps*; ignore
its claim that z-scores come from "a 7-day EWMA baseline" — they come from
`RollingStats` of returns, and the rewritten class docstring must not repeat that
inaccuracy), not as code to patch. `ManipulationGuard` (agents/manipulation_guard.py:141-171) is the reference
implementation of the contract.

**Prerequisite check:** `store/queries.py` must export `get_latest_snapshot`,
`get_latest_orderbook_snapshot`, `get_active_political_markets`, and
`get_scheduled_events_in_range`. If any is missing, a prerequisite session did not land
— stop and report.

**Affected packages:** `src/prediction_market/agents/` (`info_leak_detector.py`,
`__init__.py`), `src/prediction_market/analysis/` (`price_analyzer.py`,
`volume_analyzer.py` — one additive method each, see Details),
`src/prediction_market/config.py`, `config/default.toml`,
`tests/unit/test_info_leak_detector.py` (new).

**Details:**
- **Contract** (mirror ManipulationGuard exactly):
  `__init__(self, config: AppConfig, db: aiosqlite.Connection, sinks: list[ReportSink] | None = None, news_checker: NewsChecker | None = None)`
  calling `super().__init__(config, db, sinks)`; `name` property returns `"info_leak"`;
  `tick_interval_seconds` property returns
  `self.config.polling.snapshot_interval_seconds`. The `news_checker` parameter
  defaults to `NewsChecker(config)` and exists so tests can inject a stub — keep it.
  Delete the `on_start`/`on_stop` methods: `BaseAgent` has no such hooks
  (base.py:63-102), and rolling-state persistence is out of scope (see prohibition
  below).
- **Analyzers, used correctly this time:** hold ONE `PriceAnalyzer(config.thresholds)`
  and ONE `VolumeAnalyzer(config.thresholds)` on the instance — both are already
  multi-market, keyed internally by `market_id` (price_analyzer.py:74-120,
  volume_analyzer.py:57-79). The real API is
  `update(market_id, value, timestamp) -> None` then
  `check_anomaly(market_id) -> PriceAnomaly | None` (fields: `market_id`, `z_score`,
  `current_price`, `baseline_price`, `price_return`, `timestamp` —
  price_analyzer.py:23-41) / `VolumeAnomaly | None` (volume_analyzer.py:24-40). Delete
  the `_MarketState` class — the analyzers own per-market state.
- **Add one small method to each analyzer:**
  `def current_z_score(self, market_id: str) -> float | None` on both `PriceAnalyzer`
  and `VolumeAnalyzer`, returning the z-score of the latest observation regardless of
  threshold (None during warm-up, i.e. under 3 observations — same guard as
  `check_anomaly`, price_analyzer.py:139). Reason: the combined score needs both
  z-scores even when only one crossed its trigger threshold, and `check_anomaly`
  returns None below threshold.
- **Tick pipeline** (one `tick()` pass):
  1. `markets = await queries.get_active_political_markets(self.db)`.
  2. Per market: `snap = await queries.get_latest_snapshot(self.db, market_id)`; skip
     if None. Track `self._last_processed: dict[str, str]` of snapshot timestamps and
     skip if this timestamp was already processed (the snapshot loop writes every 60s;
     an unchanged row must not be double-counted into the rolling stats).
  3. Price observation: `snap["price_yes"]`; skip market if None. Volume observation:
     the per-interval delta `max(0.0, volume_total - previous volume_total)` tracked in
     `self._last_volume_total: dict[str, float]` — `volume_24hr` is a rolling 24-hour
     aggregate and feeding it to `VolumeAnalyzer.update(market_id, hourly_volume, ...)`
     (volume_analyzer.py:67) would z-score the wrong series. Skip the volume update
     (not the price update) on the first observation of a market, when there is no
     previous total to difference against.
  4. `price_z = price_analyzer.current_z_score(mid)`,
     `vol_z = volume_analyzer.current_z_score(mid)` (treat None as 0.0). Trigger when
     `abs(price_z) >= thresholds.price_zscore` (2.5) OR
     `abs(vol_z) >= thresholds.volume_zscore` (3.0) (config.py:31-32). Not triggered →
     next market.
  5. `combined = math.sqrt(price_z**2 + vol_z**2)`.
  6. Event amplifier: `queries.get_scheduled_events_in_range(db, ts - proximity, ts + proximity)`
     with `proximity = timedelta(hours=thresholds.event_proximity_hours)` (24). Any
     events → `combined *= thresholds.event_amplifier` (1.5) and record the matched
     events (source, title, event_date) for the report's `calendar_matches`.
  7. News dampener: `result = await self._news_checker.check_news_exists(keywords, before_time=ts)`
     — the real method and its signature are at news_checker.py:103
     (`keywords: list[str], before_time: datetime, window_hours: int = 2`), and the
     result field is **`news_found`**, not `has_prior_news`
     (data/external/models.py:38-50). Keywords come from a new module-level
     `_extract_keywords(question: str) -> list[str]` in the detector file: split the
     question on non-alphanumeric characters, lowercase, drop tokens shorter than 4
     characters, drop tokens in a module-level `_STOPWORDS` frozenset — use exactly
     `{"will", "this", "that", "with", "have", "been", "before", "after", "than",
     "them", "they", "there", "what", "when", "does", "from", "into", "over",
     "under", "between"}` — and cap at the first 6 remaining tokens in question
     order. (The external clients each carry a private `_extract_keywords` copy,
     e.g. congress.py:26-89 — do NOT import those; they're event-shaped, and
     consolidating them is a different session's job.)
     If `result.news_found`: `combined *= (1.0 - thresholds.news_dampener)` (×0.7 —
     the move is publicly explained, so it is *less* suspicious).
  8. Thin-liquidity annotation: `get_latest_orderbook_snapshot`;
     `thin = row is not None and row["susceptibility_score"] is not None and row["susceptibility_score"] >= thresholds.susceptibility_threshold`.
     Annotation only — it does not change the score.
  9. Emit when `combined >= thresholds.combined_score_min` (4.0) AND the market is not
     in cooldown.
- **Cooldown (new behavior, decided here):** add `alert_cooldown_minutes: int = 60` to
  `ThresholdConfig` (config.py:31-43) and a matching line in `config/default.toml`
  under `[thresholds]`. In-memory `self._last_alert: dict[str, datetime]`; suppress
  emission for a market alerted within the window. In-memory is a deliberate choice:
  a restart resets cooldowns and rolling baselines, which is acceptable and must be
  stated in the class docstring (warm-up is 3 snapshots ≈ 3 minutes at default
  cadence). Do NOT build DB-backed persistence for analyzer state or cooldowns —
  that is scope creep; if it ever becomes needed it belongs in a future phase.
- **Report construction:** `AnomalyReport` requires `id` and `market_question`
  (reporting/anomaly_report.py:13-34); use `AnomalyReport.new_id()`
  (anomaly_report.py:55). `anomaly_score` = the final combined score (raw scale —
  say so in `details`). `confidence` = the existing logistic mapping
  `1/(1+exp(-0.5*(score-4.0)))`, kept verbatim including its OverflowError guard
  (info_leak_detector.py:533-544). **Severity = `AnomalyReport.severity_from_score(confidence)`**
  (anomaly_report.py:36-52) — this retires the private `_classify_severity` staircase
  (info_leak_detector.py:522-531) and makes severity derivation uniform across both
  agents (ManipulationGuard already uses `severity_from_score`,
  manipulation_guard.py:395). Delete `_classify_severity`. Set **`agent="info_leak"`
  as a literal** — this is the value the `anomaly_reports.agent` column stores and
  the `reports --agent info_leak` CLI filter matches (cli.py:277); note that
  ManipulationGuard likewise hardcodes `agent="manipulation"`
  (manipulation_guard.py:416) independently of its `name` property — mirror that
  pattern, do not pass `agent=self.name` and assume they'll stay aligned. Set
  `summary` to one human-readable sentence containing the market question, the final
  combined score to two decimals, and which trigger(s) fired (price, volume, or
  both). Populate `price_evidence`
  (z, current, baseline, return), `volume_evidence` (z, delta, mean),
  `calendar_matches`, `news_check` (from `NewsCheckResult`), and
  `details` (raw and final scores, amplifiers/dampeners applied, `thin_liquidity`).
  Emit via `await self.emit(report)` — `BaseAgent.emit` already persists to
  `anomaly_reports` and fans out to sinks (base.py:106-157); do not call
  `queries.save_anomaly_report` yourself (double-write).
- **Exports:** add `InfoLeakDetector` to `agents/__init__.py` (currently only
  `BaseAgent` and `ManipulationGuard`).
- Do NOT touch `orchestrator.py` in this prompt — its wiring is still broken after
  this lands and a dedicated later session owns that cutover. Do NOT modify
  `ManipulationGuard`.

**Testing:** new `tests/unit/test_info_leak_detector.py`. Use a temp SQLite DB via
`store.database.init_database` and seed through `store.snapshots.save_market` /
`save_price_snapshot` (snapshots.py:21-137) and `store.queries.save_scheduled_events`.
Inject a stub `NewsChecker` (a tiny class whose `check_news_exists` returns a canned
`NewsCheckResult`) — no respx needed. Drive `await agent.tick()` directly; never
`start()` (wall-clock loops are flake bait). Claims to prove:
- **Warm-up:** 2 stable snapshots processed across 2 ticks → zero `anomaly_reports`
  rows.
- **Detection:** ≥6 stable snapshots (price_yes 0.50 ± 0.002) then one spike snapshot
  (price_yes 0.70) → exactly one report row with `agent = 'info_leak'`, the right
  `market_id`, and `anomaly_score >= 4.0`. Assert via a SQL count/select, not via a
  mock on `emit` — the DB row is the unfakeable outcome.
- **No double-processing:** two consecutive ticks with no new snapshot → no change in
  analyzer observation count / no new report.
- **Amplifier:** same spike scenario plus a seeded `scheduled_events` row inside ±24h
  → report `details` records the amplifier and `calendar_matches` is non-empty; the
  final score equals 1.5× the unamplified run's score (run both scenarios and compare).
- **Dampener:** stub NewsChecker returning `news_found=True` → final score is 0.7× the
  `news_found=False` run on identical data.
- **Cooldown:** after an emission, a second triggering snapshot within the window →
  still exactly one report row.
- **Severity uniformity:** the emitted report's severity equals
  `AnomalyReport.severity_from_score(report.confidence)`.
- **Report content:** the emitted row has `agent = 'info_leak'`, a non-empty
  `summary` containing the market question, and `_extract_keywords("Will Trump
  nominate Judy Shelton as the next Fed chair?")` returns exactly
  `["trump", "nominate", "judy", "shelton", "next", "chair"]` (hand-derived: "will"
  is a stopword, "as"/"the" are under 4 chars, "fed" is under 4 chars).
All 147 pre-existing tests pass unmodified except `tests/` files this prompt owns;
expectation changes elsewhere are a red flag — justify each.

**Invariants:** `agents/base.py`, `agents/manipulation_guard.py`, `orchestrator.py`,
`store/`, `reporting/` untouched. `PriceAnalyzer`/`VolumeAnalyzer` changes are limited
to the additive `current_z_score` method — `update`/`check_anomaly` behavior is
byte-identical (existing analyzer tests must pass unmodified). Agents still never
import `store.database` or construct their own DB connections (the connection is
injected — this is the layering rule in docs/CONVENTIONS.md §Boundaries).

**Guardrails (mandatory):**
- Run `uv run pytest tests/` and make it pass before finishing.
- Run `uv run ruff check src/ tests/` and make it pass.
- The detection test must be shown to FAIL if the emission threshold is not reached —
  temporarily assert with stable-only data to confirm zero rows, then restore. A
  detection test that passes on stable data is testing nothing.
- If `BaseAgent`, the analyzers, or the Prompt-1 query functions differ from what is
  stated above, stop and report — do not adapt around a mismatch.
- Do not commit or push unless explicitly asked.

---

## Prompt 3: Orchestrator agent-lifecycle cutover and the flagship pipeline test

**Goal:** the orchestrator actually runs the agents. At HEAD, `_build_agents()`
constructs both agents with a single positional argument and swallows the resulting
`TypeError` (orchestrator.py:600-621), and `_run_agent` awaits `agent.run(...)`
(orchestrator.py:547-555) — a method that exists on no agent — swallowing the
`AttributeError` forever. After this prompt, `monitor` boots both agents through
`BaseAgent.start()/stop()`, construction failures crash startup loudly, and the
phase's flagship test locks the whole pipeline. This is the load-bearing cutover of
the series.

**Prerequisite check:** `InfoLeakDetector` must construct as
`InfoLeakDetector(config, db, sinks)` and be importable from
`prediction_market.agents`. If not, a prerequisite session did not land — stop and
report.

**Affected packages:** `src/prediction_market/orchestrator.py`,
`tests/integration/test_pipeline.py` (new).

**Details:**
- **Construction, fail-fast:** change `_build_agents` to
  `_build_agents(self, db: aiosqlite.Connection, sinks: list[ReportSink]) -> list[BaseAgent]`,
  called from `start()` *after* `init_database` and sink construction. Build
  `InfoLeakDetector(self.config, db, sinks)` and
  `ManipulationGuard(self.config, db, sinks, http_client=self._http)`
  (ManipulationGuard's real signature is at manipulation_guard.py:141-148; passing the
  shared client avoids a second connection pool). **Remove the per-agent
  `try/except Exception` blocks entirely** — a mis-wired agent must raise out of
  `start()`. Keep the `_agent_filter` mapping (`"info-leak"`/`"manipulation"`/None,
  as the CLI passes it, cli.py:104). Replace the top-of-module
  `try/except ImportError` lazy imports (orchestrator.py:38-46) with normal imports —
  the "modules that may not exist yet" era is over.
- **Lifecycle:** delete `_run_agent` (orchestrator.py:535-574) entirely. `BaseAgent`
  already owns the loop: `start()` spawns a named task whose `_loop()` isolates
  per-tick exceptions via `on_error` (base.py:63-102) — the orchestrator's
  retry-with-backoff wrapper was solving a problem the base class already solves. In
  `Orchestrator.start()`: `for agent in self._agents: await agent.start()`. In
  `stop()`: `await agent.stop()` for each, and delete the dead
  `agent.close()` hook call (no agent defines `close`; see docs/CONVENTIONS.md).
  Startup ordering: DB init → clients → sinks → market discovery → agents → periodic
  loops (agents read snapshots, so the snapshot loop should already be scheduled or
  about to be — the first agent tick finding zero snapshots is fine, warm-up handles
  it).
- **Log line:** replace "No agents were initialised" logic — with fail-fast
  construction the empty case is only reachable via `--agent` filter mismatch, which
  should raise `ValueError` naming the valid filter values.
- **The flagship test — `tests/integration/test_pipeline.py`** (this file enforces the
  phase thesis; put that sentence in its module docstring and mark it permanent —
  future phases add to it, never delete it):
  - `test_orchestrator_constructs_agents`: real temp DB via the `app_config` fixture
    (conftest.py:13-19), construct `Orchestrator(config)`, invoke the internal
    wiring so `_build_agents` runs with a live connection. To make that testable,
    extract `async def _init_resources()` from `start()` with this exact boundary
    (decided, not optional): it contains DB init, HTTP/API client construction, sink
    construction, initial market discovery, and `_build_agents` — and it returns
    **before** signal-handler installation, before any `asyncio.create_task` for the
    periodic loops, and before any `agent.start()` call. `start()` becomes
    `await self._init_resources()` followed by the handlers/tasks/agent-start block.
    Assert exactly 2 agents,
    of the right classes; assert `--agent`-filter equivalents yield 1; assert an
    unknown filter raises `ValueError`.
  - `test_info_leak_pipeline_emits_report`: seed markets + a stable-then-spike
    snapshot sequence through `store.snapshots` helpers (the exact scenario shape
    from the detection test in `tests/unit/test_info_leak_detector.py` — reuse it),
    construct agents **via the orchestrator**, then
    drive `await agent.tick()` on the info-leak agent once per seeded snapshot
    timestamp (re-seeding the latest snapshot between ticks so each tick sees a new
    row). Assert an `anomaly_reports` row with `agent = 'info_leak'` exists via SQL.
  - `test_manipulation_pipeline_emits_report`: first **seed one market row** via
    `store.snapshots.save_market` with `active=1`, `closed=0`, a `condition_id`, and
    non-empty `clob_token_ids` — ManipulationGuard's tick starts from a
    `SELECT ... FROM markets WHERE active = 1 AND closed = 0` and silently no-ops on
    an empty table. Then respx-mock the CLOB and Data API endpoints (`/book` and the
    holders endpoint) with a thin order book (reuse the shape of the
    `thin_orderbook` fixture, conftest.py:86-100, as JSON) **and a non-empty,
    non-degenerate holders payload** (e.g. one holder at 0.6 plus two at 0.2 —
    `concentration_score` returns 1.0 on an *empty* holder list, so an unmocked or
    failing holders route would fake a high composite without ever exercising the
    HHI path). Drive one `await agent.tick()`, assert an `anomaly_reports` row with
    `agent = 'manipulation'`, **and assert via respx's call log that the holders
    route was called at least once** — that assertion is what proves the
    concentration path really ran. This is the first test ever to execute
    ManipulationGuard's tick path — expect to discover real bugs; fix what the test
    surfaces in `manipulation_guard.py` minimally and report each fix.
  - One lifecycle test: a trivial `BaseAgent` subclass with a counting `tick()` and
    `tick_interval_seconds = 0` (define it in the test file), `await start()`,
    `await asyncio.sleep(0.05)`, `await stop()` — assert ≥1 tick ran and the task is
    gone. This proves start/stop mechanics without wall-clock flakiness on the real
    agents.
- Do NOT change agent internals beyond what the manipulation-guard test forces. Do
  NOT touch the discovery/snapshot loops (orchestrator.py:448-529) — they work and
  are covered by `test_orchestrator.py`.

**Testing:** the four tests above, plus: existing
`tests/integration/test_orchestrator.py` (`scan_once`/`backfill`) passes unmodified —
those paths build their own clients and must be unaffected by the lifecycle changes.
Expectation changes in any existing test are a red flag; justify each one.

**Invariants:** `scan`, `backfill`, `markets`, `reports`, `report` CLI commands
byte-identical in behavior. `agents/base.py` untouched. `store/` untouched. The
snapshot/discovery loop cadence and DB write shapes unchanged.

**Guardrails (mandatory):**
- Run `uv run pytest tests/` and make it pass before finishing.
- Run `uv run ruff check src/ tests/` and make it pass.
- The flagship tests must assert on `anomaly_reports` rows read back via SQL — not on
  mocked `emit` calls. A pipeline test that mocks the pipeline's output is not a
  pipeline test.
- Prove fail-fast: one test constructs the orchestrator with a deliberately broken
  agent path (e.g. monkeypatch `InfoLeakDetector.__init__` to raise) and asserts the
  error propagates out of the wiring call instead of being logged-and-ignored.
- If `manipulation_guard.py`'s tick path turns out to need more than minimal fixes to
  pass its pipeline test, stop and report the list of defects instead of rewriting
  the agent.
- Do not commit or push unless explicitly asked.

---

## Prompt 4: Scheduled-events ingestion — wire the government-calendar clients

**Goal:** the `scheduled_events` table gets real data. At HEAD, `CongressClient`,
`CourtCalendarClient`, and `WhiteHouseClient` are fully implemented
(data/external/congress.py, court_calendar.py, white_house.py) but never instantiated
outside their own modules, and the table they should feed has no writer wired — which
means the info-leak detector's event amplifier can never fire on real data. After this
prompt, `monitor` runs a periodic event-refresh loop that populates the table.

**Prerequisite check:** `store.queries.save_scheduled_events` must exist, and the
orchestrator must start agents via `BaseAgent.start()` with no `_run_agent` method
present. If either is untrue, a prerequisite session did not land — stop and report.

**Affected packages:** `src/prediction_market/orchestrator.py`,
`src/prediction_market/store/database.py` (one stale comment only — see Testing),
`tests/integration/test_events_ingestion.py` (new).

**Details:**
- New orchestrator method `_periodic_event_refresh()`, registered as a named task
  (`"event-refresh"`) in `start()` alongside the existing loops. Interval:
  `self.config.polling.event_refresh_interval_seconds` (already defined — config.py:27,
  default.toml:8, default 7200). Copy the interruptible-sleep idiom exactly from
  `_periodic_market_discovery` (orchestrator.py:448-460): `asyncio.wait_for(shutdown.wait(), timeout=interval)`,
  `TimeoutError` means "do a cycle".
- **Refresh once at startup, before the first sleep** — otherwise the amplifier has no
  data for the first two hours of every run.
- Per cycle, gather from the three clients (constructed in `start()` with the shared
  `self._http` client, closed in `stop()` with the same per-resource try/except
  pattern the other closes use, orchestrator.py:195-238):
  - `congress.get_upcoming_hearings(days_ahead=7)` and
    `congress.get_upcoming_votes(days_ahead=7)` (congress.py:154, 237)
  - `court.get_upcoming_arguments(days_ahead=14)` (court_calendar.py:144)
  - `wh.get_schedule(days_ahead=7)` (white_house.py:168)
  Concatenate and `await save_scheduled_events(self._db, events)`; log the
  inserted-count. Wrap the cycle body in try/except-log so one failed cycle doesn't
  kill the loop (matching the snapshot loop's per-iteration resilience).
- Missing API keys are already handled: these clients warn once at init and return
  `[]` from every method (congress.py:113-118, court_calendar.py:97-102) — the loop
  must run without keys and simply insert nothing. Do not add key checks in the
  orchestrator.
- Scope creep prohibitions: do NOT add `get_recent_bills`/`get_recent_opinions`/
  `get_recent_actions` to the cycle — retrospective feeds are not calendar data and
  belong to a future news-context phase. Do NOT wire `NewsChecker` here (the detector
  owns it). Do NOT build keyword matching between events and markets — the amplifier
  currently ranges over all events by design; per-market relevance scoring is future
  work.

**Testing:** new `tests/integration/test_events_ingestion.py`, respx for HTTP. Claims
to prove:
- With respx-mocked Congress + CourtListener + White House payloads (derive minimal
  JSON from each client's parsing code; the clients' own modules show the expected
  shapes) and fake keys set on the config, one refresh cycle inserts rows whose
  `source` values are exactly the literals the clients emit: `"congress.gov"`
  (congress.py:195), `"courtlistener"` (court_calendar.py:200), and
  `"whitehouse.gov"` (white_house.py:229). Note: the schema comment at database.py:87
  says `-- 'congress', 'court', 'whitehouse'` — that comment is stale and wrong;
  update it to the three real literals (comment-only edit, no DDL change). Do NOT
  change the clients' `source=` values to match the comment.
- Running the same cycle twice inserts no duplicates (row count stable — exercises
  the `idx_events_unique` unique index through the real path).
- With empty-string keys for Congress and CourtListener but the White House routes
  still respx-mocked with real payloads, a cycle completes without raising, inserts
  zero rows with `source` in (`"congress.gov"`, `"courtlistener"`), **and still
  inserts the White House rows** — `WhiteHouseClient` needs no key
  (white_house.py:132-138), and without this positive assertion the test cannot
  distinguish "keyed sources correctly gated" from "everything silently no-op'd"
  (white_house.py:164-166 swallows any fetch error into `None`).
- Detector integration: after ingestion, `get_scheduled_events_in_range` around a
  seeded event's date returns it (this is the seam the amplifier consumes — proven
  end-to-end by the amplifier test in `tests/unit/test_info_leak_detector.py`, so a
  read-back check suffices here).

**Invariants:** discovery/snapshot loops untouched. Clients themselves
(`congress.py`, `court_calendar.py`, `white_house.py`) untouched — if their parsing
fails against your fixture JSON, fix the fixture to match the client, not the client
(their shapes were verified against their own parsing code; a genuine client bug is a
stop-and-report). `scan_once`/`backfill` unaffected.

**Guardrails (mandatory):**
- Run `uv run pytest tests/` and make it pass before finishing.
- Run `uv run ruff check src/ tests/` and make it pass.
- The dedup test must run the real cycle twice — calling `save_scheduled_events`
  twice directly tests the store layer, not this prompt's wiring.
- If any anchor above (line numbers, signatures, source literals) doesn't match the
  code you find, stop and report instead of adapting silently.
- Do not commit or push unless explicitly asked.

---

## Prompt 5: Logit-return price anomalies and honest variance

**Goal:** price anomaly detection moves from log-returns to **logit-returns**
(log-odds differences), and `RollingStats` switches from population to sample
standard deviation. Rationale (decided; do not relitigate): prediction-market prices
live in [0,1], so raw/log price-change variance shrinks mechanically near the
boundaries — a fixed z-threshold under-flags mid-range moves and mis-scales boundary
moves. `logit(p) = ln(p/(1-p))` maps (0,1) → ℝ and is the standard space for
statistical modeling of prediction-market prices (see docs/RESEARCH-BRIEF.md §4 for
sources; the brief is context, not required reading). Sample stdev (÷ n−1) replaces
population stdev (÷ n) because with the small warm-up windows this system uses (3+
observations), population variance systematically understates spread and inflates
z-scores.

**Prerequisite check:** `tests/unit/test_info_leak_detector.py` and
`tests/integration/test_pipeline.py` must exist and be green — this prompt's changes
must be proven against them. If either file is missing, a prerequisite session did
not land — stop and report.

**Affected packages:** `src/prediction_market/analysis/` (`price_analyzer.py`,
`timeseries.py`), `tests/unit/test_price_analyzer.py`, `tests/unit/test_timeseries.py`,
`tests/unit/test_volume_analyzer.py` (only if stdev-change ripples require it),
`tests/unit/test_info_leak_detector.py` (only if scenario magnitudes require
re-tuning).

**Details:**
- In `PriceAnalyzer.update` (price_analyzer.py:89-120): replace the log-return
  computation `math.log(price / state.last_price)` with a logit-return:
  `logit(clamp(price)) - logit(clamp(last_price))` where
  `clamp(p) = min(max(p, 0.005), 0.995)` and `logit(p) = math.log(p / (1 - p))`.
  The clamp bound 0.005 is a decision — it caps a single observation's |logit| at
  ~5.29 and keeps resolved-market prices (0.0/1.0) finite. Put `_logit` and `_clamp`
  as module-level functions with the reasoning in their docstrings. The existing
  guard `last_price > 0 and price > 0` becomes unnecessary post-clamp but keep the
  `last_price is not None` check.
- `PriceAnomaly.price_return` now carries the logit-return. Keep the field name (it
  is persisted into report evidence dicts; renaming is a compatibility break for
  nothing) but update its docstring (price_analyzer.py:31-32) to say "logit-return
  (log-odds difference)".
- The EWMA-on-raw-price baseline (`state.ewma.update(price)`, price_analyzer.py:118)
  stays as-is — it is reporting context (`baseline_price`), not part of the z-score.
  While in the file, fix the class docstring that claims the EWMA drives anomaly
  scoring (it never did; the `RollingStats` of returns does — see
  docs/CONVENTIONS.md §Discrepancies).
- In `RollingStats.std` (timeseries.py:89-100): divide by `n - 1` instead of `n`;
  update the docstring from "Population standard deviation" to sample. The `n < 2 →
  0.0` guard stays and is now also the divide-by-zero guard.
- Z-score thresholds (2.5/3.0) are scale-free and stay unchanged — state this in the
  diff summary so the reviewer doesn't hunt for a re-tuning that isn't there.
- Scope creep prohibitions: do NOT condition volatility on time-to-resolution or
  liquidity tier (Phase 2, per the research brief). Do NOT touch `EWMA`'s variance
  recurrence (timeseries.py:225-238) — it is internally consistent and unused by the
  z-score path. Do NOT alter `VolumeAnalyzer`'s input semantics (the detector already
  feeds it per-interval deltas).

**Testing:** claims to prove, with **independently hand-computed constants stated as
literals in the tests** (compute them by hand or with a calculator while writing the
test — a test that recomputes the expectation with the same formula as the
implementation proves nothing):
- `RollingStats` sample stdev: for values [1, 2, 3, 4] in-window, `std == pytest.approx(1.2909944487)`
  (= √(5/3)), and mean 2.5. Update any existing `test_timeseries.py` expectations that
  encoded population stdev — each such change must be re-derived by hand and the
  derivation noted in the test.
- Logit math: `_logit(0.5) == 0.0`; `_logit(0.995) == pytest.approx(5.2933048)`;
  `_clamp(1.0) == 0.995`; a price sequence hitting exact 0.0/1.0 produces finite
  returns and no exception.
- **The boundary-compression claim itself:** two five-observation series with
  identical stable baselines and identical absolute price jumps of +0.02 — one
  jumping 0.50→0.52, one 0.95→0.97 — must yield a *larger* final logit-return for the
  0.95→0.97 jump (hand-check: Δlogit ≈ 0.080 vs ≈ 0.532). This is the test that fails
  under the old log-return code path... verify it does by inspection of the old
  formula, and say so in a comment.
- Regression: a stable price series still yields `check_anomaly is None`; the
  spike-detection scenario from `test_price_analyzer.py` still detects (adjust the
  magnitude only if the logit scale requires it, with the adjustment justified in a
  comment).
- Detector end-to-end unaffected in kind: `tests/unit/test_info_leak_detector.py` and
  `tests/integration/test_pipeline.py` still pass — if scenario magnitudes need
  re-tuning to stay above thresholds, tune the *fixture data*, never the asserted
  behavior.
- Tighten while here: replace `test_serialization`'s `data is not None`
  (test_price_analyzer.py:43-49) with assertions on actual round-trip content
  (`to_dict` → `from_dict` → same tracked markets and equal `current_z_score` for a
  seeded market).

**Invariants:** `update`/`check_anomaly` signatures unchanged. `PriceAnomaly` field
names unchanged. `VolumeAnalyzer` behavior changes only via the shared
`RollingStats.std` denominator. No config changes. No agent or store changes.

**Guardrails (mandatory):**
- Run `uv run pytest tests/` and make it pass before finishing.
- Run `uv run ruff check src/ tests/` and make it pass.
- Every changed numeric expectation in existing tests must carry a comment showing
  the hand derivation. Expectation changes without derivations are the known dodge
  here and will be rejected in review.
- If the detector or pipeline tests from earlier prompts are missing, stop and report
  (prerequisites didn't land).
- Do not commit or push unless explicitly asked.

---

## Prompt 6: Kill the duplicates and make the docs stop lying

**Goal:** one implementation per concept, and documentation that matches reality.
Three duplications exist at HEAD: (a) `scripts/backfill_markets.py:39-88` re-implements
`PoliticalFilter.classify` with *behaviorally different* volume semantics (the script
returns non-political below min-volume, scripts/backfill_markets.py:80-85; the library
keeps `is_political=True` and appends a reason, political_filter.py:66-68); (b)
`scripts/snapshot_political_markets.py:42-77` re-implements the upsert SQL that
`store/snapshots.py` already provides; (c) `analysis/liquidity_analyzer.py` is a dead
twin of the live `LiquidityAnalyzer` inline in `agents/manipulation_guard.py:33-122` —
and the test suite currently tests **only the dead one**
(tests/unit/test_liquidity_analyzer.py imports `prediction_market.analysis.liquidity_analyzer`).

**Prerequisite check:** `tests/integration/test_pipeline.py` must exist (its
manipulation test is this prompt's safety net for the STEP-0 change), and
`orchestrator._build_agents` must be the fail-fast version (no per-agent
`try/except Exception`). If either is missing, a prerequisite session did not land —
stop and report.

**Affected packages:** `scripts/` (both scripts),
`src/prediction_market/analysis/` (`liquidity_analyzer.py` deleted, `__init__.py`),
`src/prediction_market/data/political_filter.py` (one additive property),
`src/prediction_market/data/polymarket/` (`ws_market.py`, `ws_rtds.py`,
`gamma_client.py` — docstrings only), `src/prediction_market/agents/manipulation_guard.py`
(concentration_score only, gated by STEP 0), `src/prediction_market/orchestrator.py`
(one-line private-access fix), `tests/unit/test_liquidity_analyzer.py`, `README.md`.

**STEP 0 (mandatory, before any edit):** read `tests/fixtures/holders.json` and the
`MarketHolder` model (data/polymarket/models.py:222-230) and determine the scale of
`pctSupply` in the fixture: fraction (0–1) or percentage (0–100). The live
`concentration_score` (manipulation_guard.py:78-90) squares `pct_supply` raw and is
only correct for fractions. **Fallback scope, decided now:** if the fixture shows
values > 1 (percentage scale), normalize inside `concentration_score` by dividing
each value by 100 before squaring, and add the percentage-input test case. If the
fixture shows fractions, change nothing in the formula and add the fraction test
case. Do not import or resurrect the dead analyzer's sum-heuristic scale detection —
pick the one scale the fixture evidences and document it in the docstring. Report
which branch you took.

**Details:**
- **Scripts, deduplicated:**
  - Add a read-only property `min_volume` to `PoliticalFilter` exposing `_min_volume`
    (political_filter.py:26-32), and fix the orchestrator's private access
    `self._filter._min_volume` (orchestrator.py:414) to use it.
  - `backfill_markets.py`: delete `classify_political` (lines 39-88) and use
    `PoliticalFilter`. Selection rule after the change: include a market iff
    `classification.is_political` is True AND
    `market.volume >= political_filter.min_volume`. **Know what this changes — the
    two classifiers are NOT equivalent** (verified at HEAD): the script scores any
    keyword match a flat +0.3 and treats *any* non-empty reason list as political
    (backfill_markets.py:76-81 — no confidence gate), while the library scores
    keywords incrementally (`min(0.3, matches * 0.1)`, political_filter.py:58-60)
    and gates on `confidence >= 0.3` (political_filter.py:64). Consequence: a market
    matching only 1–2 keywords (confidence 0.1–0.2, no tag/category hit) was
    backfilled before and will NOT be after. This is the intended unification — the
    library classifier is what `monitor`/`scan` use, and one source of truth wins.
    State this behavior change in your summary; do not "fix" the library to match
    the script.
  - `snapshot_political_markets.py`: replace the raw `INSERT OR REPLACE` /
    `INSERT INTO snapshots` SQL (lines 42-77) with
    `store.snapshots.save_market(db, market, classification)` and
    `save_price_snapshot(...)` (snapshots.py:21-137 — match their real signatures by
    reading them first). Behavior note to preserve: `save_market` is an upsert that
    doesn't clobber `first_seen` on conflict; the script's old SQL did — this is a
    deliberate improvement, mention it in the summary.
- **Dead analyzer removed:** delete `src/prediction_market/analysis/liquidity_analyzer.py`
  and its export from `analysis/__init__.py`. Rewrite
  `tests/unit/test_liquidity_analyzer.py` to import
  `LiquidityAnalyzer` from `prediction_market.agents.manipulation_guard` and test the
  **live** formulas with hand-computed constants:
  - `depth_score`: total ≤ 5,000 → 1.0; ≥ 500,000 → 0.0; total = 50,000 →
    `pytest.approx(0.5)` exactly (= 1 − ln(10)/ln(100) — note the derivation in the
    test); 0 depth → 1.0.
  - `spread_score`: `spread_pct=None` → 1.0; 0.05 → 0.5; ≥ 0.10 → 1.0.
  - `imbalance_score` passthrough of `abs(imbalance)`.
  - `concentration_score`: single 100% holder → 1.0; four equal holders → 0.25;
    empty list → **1.0** (that is what the live code returns,
    manipulation_guard.py:84-85 — unknown holders are treated as maximally risky;
    do not "fix" this branch, it is deliberate) — expressed in whichever scale
    STEP 0 established.
  - `compute_susceptibility`: one hand-computed composite from known sub-scores and
    the default weights (0.30/0.25/0.25/0.20, config.py:40-43).
  Replace the old file's loose tolerances (`pytest.approx(abs=100)`) with exact-value
  assertions — the formulas are deterministic.
  Keep `analysis/correlation.py` and its tests untouched: it is dead-but-planned
  (lead-lag detection, Phase 3 per docs/RESEARCH-BRIEF.md) and, unlike the liquidity
  twin, has no live duplicate to conflict with.
- **Docstring honesty (no behavior changes):** `ws_market.py:5-6` and `ws_rtds.py:5-6`
  claim the WebSocket clients are the "primary data source with REST polling as
  fallback" — rewrite to state they are implemented but not yet wired (Phase 3), REST
  polling is the live path. `gamma_client.search_markets` (gamma_client.py:102-108)
  claims search but filters by slug — rename the docstring's claim (keep the method
  name; callers may exist downstream). `white_house.py:3` says "Scrapes" — it calls
  the WordPress REST API; fix, and delete the unused `_BRIEFINGS_URL` constant
  (white_house.py:30). `news_checker.py:12-13` implies free-tier limit awareness that
  doesn't exist in code — reword to "handles 429/426 responses reactively".
- **README:** fix `uv sync --dev` → `uv sync --extra dev` (README.md:35 — verified
  broken: `--dev` does not install `[project.optional-dependencies].dev`). In the
  architecture diagram section, mark the WS-feeds box as "(planned — not yet wired)".
  Add two sentences under monitoring: agents warm up over ~3 snapshots, and repeat
  alerts per market are suppressed by `alert_cooldown_minutes` (default 60).

**Testing:** claims to prove:
- Backfill selection semantics: for a market list containing four fixtures —
  (a) tag-matched political + high-volume, (b) tag-matched political + low-volume,
  (c) non-political, (d) **keyword-only** market with exactly 2 keyword matches and
  no tag/category hit — the script's new selection logic must select only (a).
  (b) proves the volume gate survived; (d) proves the documented behavior change
  (the old flat-+0.3/no-gate script selected it; the unified `PoliticalFilter` at
  confidence 0.2 must not — assert exclusion deliberately, with a comment naming it
  as the intended unification change). Write this as a unit test of the selection
  helper if you extract one, otherwise as a focused test on the script's filter step.
- Rewritten `test_liquidity_analyzer.py` as specified above — these are the first
  tests the live analyzer has ever had.
- All other existing tests pass unmodified; `test_pipeline.py` in particular (its
  manipulation test exercises `concentration_score` through the real path and will
  catch a botched STEP-0 normalization).

**Invariants:** `manipulation_guard.py` is untouched except (possibly)
`concentration_score` per STEP 0. `political_filter.classify` semantics unchanged
(the property is additive). No schema, config, agent-contract, or orchestrator-loop
changes (the orchestrator diff is the one-line property swap). Scripts produce the
same market selections and equivalent DB rows as before (modulo the documented
`first_seen` improvement).

**Guardrails (mandatory):**
- Run `uv run pytest tests/` and make it pass before finishing.
- Run `uv run ruff check src/ tests/` and make it pass.
- Editing the scripts without executing them does not count as done: run
  `uv run python scripts/snapshot_political_markets.py --help` and
  `uv run python scripts/backfill_markets.py --help` (or their import paths if no
  `--help` exists — at minimum import both modules) to prove they still load; state
  in the summary that live API runs were not exercised.
- Report the STEP-0 branch taken and the fixture evidence for it.
- Do not commit or push unless explicitly asked.

---

## Prompt 7: HTTP retry, YES/NO token-order validation, and honest assertions

**Goal:** the data layer stops treating every transient failure as fatal and stops
assuming Polymarket's outcome ordering. At HEAD no client retries anything — a single
`raise_for_status()` per call (gamma_client.py:34-39, clob_client.py:38-44) — and the
orchestrator hard-assumes `clob_token_ids[0]` is the YES token
(orchestrator.py:357-358, 496-499) without checking the market's `outcomes` labels; a
market listed with inverted outcome order would silently record NO prices as YES
prices — a data-accuracy bug in a system whose whole point is accurate data.

**Prerequisite check:** `tests/integration/test_pipeline.py` must exist; `_logit`
and `_clamp` must exist at module level in `analysis/price_analyzer.py`; and
`tests/unit/test_liquidity_analyzer.py` must import `LiquidityAnalyzer` from
`prediction_market.agents.manipulation_guard`. If any of these is missing, a
prerequisite session did not land — stop and report (this prompt edits test files
those sessions own).

**Affected packages:** `src/prediction_market/data/retry.py` (new),
`src/prediction_market/data/polymarket/` (`gamma_client.py`, `clob_client.py`,
`data_client.py`, `models.py`), `src/prediction_market/data/external/`
(`congress.py`, `court_calendar.py`, `white_house.py`, `news_checker.py`),
`src/prediction_market/orchestrator.py`, `tests/unit/test_retry.py` (new),
`tests/unit/test_models.py`, `tests/unit/test_price_analyzer.py`,
`tests/integration/test_orchestrator.py`.

**Details:**
- **Retry helper** — new module `src/prediction_market/data/retry.py`, one function:
  `async def get_with_retry(client: httpx.AsyncClient, url: str, *, params: dict | None = None, headers: dict | None = None, attempts: int = 3, base_delay: float = 0.5) -> httpx.Response`.
  Semantics (decisions, not options): attempt the GET; retry on
  `httpx.TransportError` and on response status in {429, 500, 502, 503, 504}; do NOT
  retry other 4xx (a 404 is an answer, not a failure). Backoff
  `base_delay * 2**attempt` seconds, honoring a numeric `Retry-After` header when
  present (use the larger of the two). After the final attempt, raise the last
  exception (or call `raise_for_status()` on the last response) — the helper never
  swallows. No new dependencies — hand-roll it; do not add tenacity.
- Wire it into every client's internal GET helper, replacing the direct
  `client.get(...)` call while keeping each helper's existing error-translation
  behavior: the Polymarket clients keep raising `HTTPStatusError` outward
  (tests at test_clob_client.py:126-137 depend on it — they must pass unmodified);
  the external clients keep their catch-log-return-None shape
  (congress.py:142-152, white_house.py:153-166). Rate-limiter acquisition stays
  *outside* the retry loop where it currently sits — a retried request re-enters
  `client.get` but must not double-acquire… actually it must: each network attempt
  consumes rate budget. Decision: pass no limiter into the helper; the Polymarket
  clients' `_get` acquires once before calling `get_with_retry`. This slightly
  undercounts under retries, which is acceptable at 4000–9000 req/10s budgets — note
  it in the helper docstring rather than building limiter plumbing.
- **Token-order validation** — add two properties to `GammaMarket`
  (data/polymarket/models.py:33-68): `yes_token_id` and `no_token_id`. Logic: if
  `outcomes` has exactly two entries and `clob_token_ids` has ≥2, map
  case-insensitively by label (`"yes"`/`"no"`); if labels are anything else (e.g.
  candidate names) or lengths mismatch, fall back to positional `[0]`/`[1]` — the
  positional convention is correct for standard binary markets; the properties exist
  to catch *inverted* Yes/No listings. Return `None` when no token exists. Replace
  the four positional accesses in the orchestrator (orchestrator.py:357-358 backfill;
  496-499 snapshot loop) with the properties, logging a warning once per market when
  the label-map and positional order disagree (that log line is the observable for
  the test). ManipulationGuard reads token ids from DB rows, not `GammaMarket` —
  leave it alone; rows are written in the same order Gamma returned, and normalizing
  stored order is explicitly out of scope (existing DBs would silently disagree).
- **Assertion tightening** (the weak spots identified in docs/CONVENTIONS.md
  §Testing): `test_orchestrator.py:105`'s `assert total_points >= 0` on the backfill
  happy path becomes an exact-count assertion derived from the mocked price-history
  fixture (count the points in the fixture; assert equality). If
  `test_price_analyzer.py::test_serialization` already asserts real round-trip
  content, leave it; if it still only asserts `data is not None`, tighten it to
  assert a `to_dict` → `from_dict` round trip preserves tracked markets and their
  z-scores.
- Scope creep prohibitions: do NOT add retry to WebSocket clients (they have their
  own reconnect logic and are unwired). Do NOT add POST retry (only `WebhookSink`
  POSTs, and it already retries — sink.py:99-122). Do NOT introduce a circuit
  breaker, metrics, or jitter — three attempts with deterministic backoff is the
  whole design.

**Testing:** claims to prove:
- `test_retry.py` (respx, `side_effect` sequences): 500-then-200 → returns the 200
  and made exactly 2 requests; 404 → raises immediately, exactly 1 request;
  three 503s → raises `HTTPStatusError`, exactly 3 requests; `Retry-After: 0` on a
  429 → still retries. Patch `asyncio.sleep` or use `base_delay=0` so the suite stays
  fast — wall-clock backoff sleeps in tests are the flake pattern this repo already
  suffers in `test_rate_limiter.py`; don't add more.
- Client integration: one test per Polymarket client showing a 500-then-200 sequence
  (respx `side_effect` list) now succeeds where it previously raised (e.g.
  `get_order_book`). The existing hard-failure test `test_get_order_book_http_error`
  (test_clob_client.py:126-137) mocks a single `return_value=httpx.Response(500)` —
  respx repeats a `return_value` for every matching call, so all retry attempts see
  500 and the test passes **unmodified**. Confirm that by running it; do not rewrite
  it, and never touch its assertion.
- `yes_token_id`/`no_token_id`: standard order → `[0]`/`[1]`; **inverted** outcomes
  `["No", "Yes"]` → `yes_token_id == clob_token_ids[1]` (this is the test that fails
  under positional-only logic); non-Yes/No labels → positional fallback; empty
  token list → `None`.
- Backfill exact-count assertion as described.
- Full suite green; `test_pipeline.py` untouched and green.

**Invariants:** no public client method signatures change. External clients still
never raise for missing keys. `TokenBucketRateLimiter` untouched. No new
dependencies in `pyproject.toml`. Stored `clob_token_ids` order in the `markets`
table unchanged.

**Guardrails (mandatory):**
- Run `uv run pytest tests/` and make it pass before finishing.
- Run `uv run ruff check src/ tests/` and make it pass.
- Retry tests must assert **request counts** via respx's call log, not just outcomes —
  a retry helper that never retries passes outcome-only tests.
- If any anchor above (line numbers, signatures) doesn't match the code you find,
  stop and report instead of adapting silently.
- Do not commit or push unless explicitly asked.
