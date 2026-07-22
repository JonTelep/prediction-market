# PHASE2-WALLET-BACKTEST.md — Wallet attribution, change-point detection, and the backtest harness

A series of standalone implementation prompts for `prediction-market`, designed to be
handed one at a time to a Claude agent for implementation, with each result reviewed
before the next prompt runs. Phase context: Phase 1 (docs/prompts/PHASE1-END-TO-END.md)
made the live pipeline real — both agents construct fail-fast, tick through
`BaseAgent`, and persist `anomaly_reports`; the suite is 292 tests green at the merge
of PR #4. Phase 2 delivers the top three deferred items from docs/RESEARCH-BRIEF.md:
wallet-level analysis via the Data API's `proxyWallet` attribution, a CUSUM
change-point detector alongside the logit z-scores, and a point-in-time backtest
harness that can replay documented insider cases. Methodology rationale lives in the
brief; implementers do not need to read it — every decision it informed is stated
inline here.

**The non-negotiable outcome of this phase:** the system attributes every stored
trade to a wallet and can replay an archived market history point-in-time through the
full detection stack — flagging a seeded insider-trading case with wallet-level
evidence *before* its event timestamp, while emitting **zero** reports on a
volatility-matched benign control. Enforced permanently by
`tests/backtest/test_replay_validation.py`
(`test_seeded_insider_case_flagged`, `test_benign_case_not_flagged`), written in
Prompt 10, which must stay green forever — any future detector change that either
misses the insider case (too blunt) or flags the benign control (too twitchy) breaks
the build.

**Scope guard — this series deliberately does NOT:**
- Wire the WebSocket clients (`ws_market.py`, `ws_rtds.py`) — Phase 3.
- Add on-chain Polygon RPC work: holder reconstruction past the `/holders` 20-cap,
  USDC funding-trace Sybil clustering — Phase 3.
- Wire `analysis/correlation.py` (lead-lag across correlated markets) — Phase 3.
- Add wallet win-rate / realized-profitability features (Mitts & Ofir signals 2–3).
  These require storing market *resolution outcomes*, which no table holds today —
  Phase 3, together with a `resolutions` capture path.
- Integrate the `simulation/` package (copulas, particle filter, Monte Carlo). It is
  currently dead code exercised only by its own unit tests; whether to wire or delete
  it is a separate product decision. Do not import from `prediction_market.simulation`
  anywhere in this series.
- Add a schema-migration framework. Prompt 1 adds one narrow `_ensure_columns` helper
  for a single column; that is the ceiling.
- Mass-reformat. `ruff format` is not enforced; format only files you materially edit.

**How to use:** paste one prompt (the full section between `---` separators) into a
fresh session. Prompts assume the ones before them (per the sequencing line) have
landed — run them in dependency order. Each prompt is self-contained: it references
only artifacts that exist in the codebase by the time it runs. If a referenced
helper, function, config field, or file does not exist, that signals a prerequisite
session did not land — stop and report rather than improvise.

**Review loop:** after each prompt lands, review the diff for correctness, test
honesty (do the tests actually exercise the claim, or could they pass with the claim
false?), and scope creep. Loop fixes back to the same agent until the prompt's
acceptance criteria pass, then move on.

**Sequencing:** 1 → 2 and 1 → 3; 4 is independent of 1–3 and may run any time before
5; 5 requires 3 and 4; 6 requires 5; 7 requires 2 and 6; 8 requires 6 and 7;
9 requires 3 and 5; 10 requires 8; 11 runs last, after **all** of 1–10,
live-network, with the human present.

**Human checkpoint recommended:** Prompt 5 (changes the live scoring path of
InfoLeakDetector — largest live blast radius) and Prompt 6 (the replay engine's
architecture decides what "point-in-time" means for every later result). For both:
run in plan mode and review the plan before letting the agent code.

**Cross-cutting rituals (restated per prompt, but learn them once):**
- Test suite: `uv run pytest tests/ -q` — 292 passed at series start. Lint:
  `uv run ruff check src/ tests/` has **8 pre-existing errors — 6 inside
  `src/prediction_market/simulation/` (unused imports/locals) and 2 F401 in
  `tests/unit/test_particle_filter.py`**. Those are known and out of scope; your
  diff must introduce zero new errors.
- All DB timestamps are TEXT `"%Y-%m-%d %H:%M:%S"` (UTC) compared lexicographically
  (`store/queries.py:18-32`). An ISO string with a `T` separator silently breaks
  range filters.
- Any new `ThresholdConfig` field must be mirrored in `config/default.toml` — the
  TOML file mirrors every pydantic default exactly, and drift between them is a bug.

---

## Prompt 1: Make the trades table trustworthy — wallet attribution and timestamp normalization

**Goal:** Every stored trade carries the wallet that made it, and trade timestamps
obey the repo's lexicographic TEXT-timestamp convention. Today both are broken:
`Trade` (data/polymarket/models.py:223-241) has no field aliased to the Data API's
`proxyWallet` key, so the `owner` column and its `idx_trades_owner` index
(store/database.py:79,127) fill with empty strings against the real API; and
`save_trade`/`save_trades_batch` (store/snapshots.py:207-294) store `match_time` raw
— the API emits ISO-8601 with a `T`/`Z` (see tests/fixtures/trades.json) while
`get_market_trades` filters `match_time >= cutoff` with `"%Y-%m-%d %H:%M:%S"`
strings (queries.py:290-316), so hour-range filters are silently wrong. This prompt
is pure model + store foundation; no agent, orchestrator, or CLI code changes.

**Affected packages:** `src/prediction_market/data/polymarket/models.py`,
`src/prediction_market/store/` (`database.py`, `snapshots.py`, `queries.py`),
`tests/fixtures/trades.json`, `tests/unit/test_models.py`,
`tests/integration/test_snapshots.py`, `tests/integration/test_queries.py`.

**Details:**
- **`Trade` model:** add `proxy_wallet: str = Field("", alias="proxyWallet")`
  alongside the existing fields (models.py:223-241; `populate_by_name` is already
  set). Keep `owner` exactly as is — it is a distinct API field, not a synonym.
- **Schema:** add `proxy_wallet TEXT NOT NULL DEFAULT ''` to the `trades` table in
  `SCHEMA_SQL` (database.py:70-83) and a new index
  `idx_trades_wallet ON trades(proxy_wallet)` next to the existing index block
  (database.py:124-132). **There is no migration framework**, and `CREATE TABLE IF
  NOT EXISTS` silently skips changed definitions on existing DBs (the shipped
  `data/prediction_market.db` predates this column). Add a private helper
  `_ensure_columns(db)` in database.py, called from `init_database` after the
  executescript: `PRAGMA table_info(trades)`; if `proxy_wallet` is absent, run
  `ALTER TABLE trades ADD COLUMN proxy_wallet TEXT NOT NULL DEFAULT ''`. The index
  uses `CREATE INDEX IF NOT EXISTS` and needs no guard. This helper is intentionally
  single-purpose — do not generalize it into a migration system (scope guard).
- **Timestamp normalization:** add `_normalize_match_time(raw: str) -> str` in
  snapshots.py: parse ISO-8601 (`datetime.fromisoformat` after `Z`→`+00:00`, the
  same trick as `Trade.match_datetime`, models.py:256-262) or an all-digits
  epoch-seconds string (`datetime.fromtimestamp(int(raw), tz=UTC)`); convert to UTC
  and return `strftime("%Y-%m-%d %H:%M:%S")`. On parse failure, return the raw
  string unchanged and log at debug — never raise on dirty API data. Apply it to
  `match_time` in both `save_trade` (snapshots.py:207-244) and `save_trades_batch`
  (snapshots.py:247-294), which also now write `proxy_wallet`.
- **New time-anchored queries** in queries.py, following the module's exact idioms
  (`db: aiosqlite.Connection` first, `_fetch_all_dicts`, TEXT timestamps —
  queries.py:35-58). These take explicit `start`/`end` TEXT timestamps instead of
  wall-clock `hours` **so they work identically in live mode and in the Phase-2
  replay harness** (which processes historical time):
  - `async def get_wallet_trades(db, wallet: str, limit: int = 200) -> list[dict]`
    — all trades where `proxy_wallet = ?`, ordered `match_time DESC`, LIMIT.
  - `async def get_market_wallet_summary(db, market_id: str, start: str, end: str) -> list[dict]`
    — one row per wallet in the window: `proxy_wallet`, `trade_count`,
    `total_volume_usd` (SUM of `volume_usd`), `buy_volume_usd` (SUM where
    `side='BUY'`), `first_trade` (MIN `match_time` **across the wallet's entire
    trades-table history, not just the window** — a correlated subquery or a JOIN
    against a per-wallet MIN; this is the "fresh wallet" input and windowing it
    would make every wallet look fresh), `last_trade` (MAX in window). Exclude
    rows with empty `proxy_wallet` from the summary. Order by `total_volume_usd`
    DESC.
- **Fixture:** add a `"proxyWallet"` key to all 4 records in
  tests/fixtures/trades.json (distinct addresses, e.g. `0xwallet1`…`0xwallet4`).
  Do not remove or rename existing keys — other tests parse this file.
- Also update `get_market_trades` (queries.py:290-316) to include `proxy_wallet` in
  its SELECT list — additive only; do not change its wall-clock `hours` contract.

**Testing (claims to prove, not activities):**
- `Trade.model_validate` on a fixture record populates `proxy_wallet` from
  `proxyWallet`; a record without the key yields `""` (no ValidationError).
- After `save_trades_batch` of the fixture trades into a temp DB
  (`tmp_path / "test.db"` idiom, per tests/integration/test_snapshots.py), the
  stored `match_time` values are space-separated `"%Y-%m-%d %H:%M:%S"` — assert the
  literal stored string for one known input (`"2026-02-20T10:00:00Z"` →
  `"2026-02-20 10:00:00"`), and assert `proxy_wallet` round-trips.
- Normalization edge cases: epoch-seconds string input; garbage input stored
  unchanged (the claim is "never raises", so the test must feed actual garbage).
- `get_market_wallet_summary` boundary proof: a trade exactly at `start` and one
  exactly at `end` are both counted in the window aggregates; one second outside
  either bound is excluded. (This is the lexicographic-filter regression proof — it
  fails against unnormalized `T`-format storage.)
- `get_market_wallet_summary`: aggregates are numerically checked against
  hand-computed values from the fixture; `first_trade` reflects a trade *outside*
  the query window (construct one older trade for the same wallet to prove the
  all-history MIN); empty-`proxy_wallet` trades are absent from the summary.
- Re-running `init_database` against a DB file created with the *old* schema (create
  it in-test by executing the pre-change CREATE TABLE for `trades`) succeeds and
  adds the column — the `_ensure_columns` upgrade proof. Then call `init_database`
  a **second** time on the upgraded DB and assert no exception — the idempotency
  proof (a double-`ALTER TABLE` would raise).
- All existing tests pass with at most mechanical updates; expectation changes are a
  red flag — justify each one.

**Invariants:** no changes to `markets`, `snapshots`, `orderbook_snapshots`,
`scheduled_events`, `anomaly_reports`, `rolling_stats` schemas. `get_market_trades`'s
signature and `hours` semantics unchanged. No agent/orchestrator/CLI files touched.
Reading a legacy row whose `match_time` is still `T`-format must not crash any query
(normalization applies at write time only).

**Guardrails (mandatory):**
- Run `uv run pytest tests/ -q` and make it pass before finishing (292 tests at
  series start — the count only goes up).
- Run `uv run ruff check src/ tests/` — zero new errors. The 8 pre-existing
  ones (6 in `src/prediction_market/simulation/`, 2 F401 in
  `tests/unit/test_particle_filter.py`) are known and out of scope.
- All DB timestamps are TEXT `"%Y-%m-%d %H:%M:%S"` UTC compared lexicographically —
  every new query parameter goes through that format.
- If the codebase contradicts anything stated above (an anchor doesn't match, a
  function is missing), stop and report the discrepancy instead of improvising.
- Do not commit or push unless explicitly asked.

---

## Prompt 2: Wallet-scoped Data API endpoints

**Goal:** `DataClient` can ask Polymarket's Data API the wallet-shaped questions:
trades filtered by wallet, a wallet's positions, and a wallet's activity feed. Pure
client + model work; nothing consumes these methods yet (the case-archiver prompt
later in this series and the Phase-3 wallet investigations are the consumers).

**Prerequisite check:** `Trade.proxy_wallet` must exist in
`data/polymarket/models.py` with alias `proxyWallet`. If it does not, a prerequisite
session did not land — stop and report.

**Affected packages:** `src/prediction_market/data/polymarket/`
(`data_client.py`, `models.py`), `tests/unit/test_data_client.py`,
`tests/unit/test_models.py`, `tests/fixtures/` (two new fixture files).

**Details:**
- **Honesty preamble — read this before coding:** the exact Data API parameter names
  below are taken from Polymarket's public API documentation and the research brief
  (docs/RESEARCH-BRIEF.md §3); they cannot be verified live from a test environment
  (respx raises on any unmocked route, and this repo's tests never touch the
  network). Implement them as specified, encode the expectation in respx tests, and
  leave verification-against-reality to the live-network validation session at the
  end of this phase (it is explicitly tasked with reporting any param mismatch back
  into this client). Do not silently "fix" param names on a hunch.
- **Extend `get_trades`** (data_client.py:47-84): add keyword-only params
  `user: str | None = None` (sent as `user`, a proxy-wallet address) and
  `taker_only: bool = False` (sent as `takerOnly=true` only when True). Existing
  params unchanged — note the Python keyword `market_id` is sent on the wire as
  `market` (data_client.py:69); do not "fix" that mismatch by renaming the Python
  param (call sites depend on it). Thread both
  through `get_all_trades` (data_client.py:86-122) without changing its
  cursor-pagination logic (last trade's `.id` as cursor, stop on short batch).
- **New models** in models.py, following the house pydantic style (string-typed
  numerics + `*_float` properties, `Field(alias=...)`, `populate_by_name` —
  models.py:223-262 is the pattern):
  - `WalletPosition`: `proxy_wallet` (alias `proxyWallet`), `asset` (alias
    `asset`), `condition_id` (alias `conditionId`), `size`, `avg_price` (alias
    `avgPrice`), `initial_value` (alias `initialValue`), `current_value` (alias
    `currentValue`), `cash_pnl` (alias `cashPnl`), `percent_pnl` (alias
    `percentPnl`), `outcome`, `title`. All `str = ""` defaults except
    floats-as-strings following the Trade pattern; add `size_float`,
    `avg_price_float`, `cash_pnl_float` properties.
  - `WalletActivity`: `proxy_wallet` (alias `proxyWallet`), `timestamp`, `type`,
    `condition_id` (alias `conditionId`), `size`, `usdc_size` (alias `usdcSize`),
    `price`, `side`, `outcome`, `transaction_hash` (alias `transactionHash`).
    Same string-typed style, plus `usdc_size_float`.
- **New `DataClient` methods**, mirroring the existing method shape (rate-limited
  `self._get`, docstrings, data_client.py:39-45):
  - `async def get_wallet_positions(self, wallet: str, condition_id: str | None = None, limit: int = 100) -> list[WalletPosition]`
    — `GET /positions`, params `user=wallet`, optional `market=condition_id`,
    `limit`.
  - `async def get_wallet_activity(self, wallet: str, limit: int = 100, offset: int = 0) -> list[WalletActivity]`
    — `GET /activity`, params `user=wallet`, `limit`, `offset`.
  Both tolerate a non-list JSON response by returning `[]` and logging one
  `logger.warning` naming the endpoint (the module logger already exists at
  data_client.py:15).
- **Fixtures:** `tests/fixtures/positions.json` (3 records) and
  `tests/fixtures/activity.json` (4 records, mixed `TRADE`/`REDEEM` types), keys in
  API camelCase form.
- Scope creep, prohibited by name: do not add retry/param plumbing to the Gamma or
  CLOB clients, do not build a wallet "service" layer over these calls, and do not
  wire any agent to them — consumers come later in the phase.

**Testing:**
- respx-mocked (`@respx.mock`, the idiom throughout tests/unit/test_data_client.py):
  `get_trades(user=..., taker_only=True)` sends exactly `user` and `takerOnly=true`
  query params (assert on `route.calls[0].request.url.params`); `taker_only=False`
  sends **no** `takerOnly` param.
- `get_wallet_positions`/`get_wallet_activity` parse their fixtures into typed
  models with correct alias mapping (spot-check one aliased field each, e.g.
  `cashPnl` → `cash_pnl`) and send `user` as the wallet param.
- Non-list JSON body (e.g. an error dict) → `[]`, no exception, and the warning
  was actually logged (`caplog.at_level("WARNING")` assertion — the log line is a
  stated requirement, not decoration).
- Pagination: `get_all_trades(user=...)` threads `user` into every page request
  (assert across two mocked pages via `side_effect=[...]`, the idiom at
  test_data_client.py's pagination tests).

**Invariants:** existing `get_trades`/`get_all_trades` call sites (orchestrator
backfill at orchestrator.py:409-419) behave identically — new params are
keyword-only with inert defaults. No store, agent, or orchestrator files touched.
`TokenBucketRateLimiter` wiring (data_client.py:29-32) untouched.

**Guardrails (mandatory):**
- Run `uv run pytest tests/ -q` and make it pass before finishing.
- Run `uv run ruff check src/ tests/` — zero new errors. The 8 pre-existing
  ones (6 in `src/prediction_market/simulation/`, 2 F401 in
  `tests/unit/test_particle_filter.py`) are known and out of scope.
- respx raises on unmocked routes — if a test hits the network, your mocks are
  wrong; fix the mocks, never disable respx.
- If the codebase contradicts anything stated above, stop and report instead of
  improvising.
- Do not commit or push unless explicitly asked.

---

## Prompt 3: Wallet profiler — per-wallet anomaly features

**Goal:** A pure, stateless feature-extraction module,
`analysis/wallet_profiler.py`, that turns a window of wallet-attributed trade rows
into ranked per-wallet anomaly features — the Phase-2 adaptation of the
Mitts & Ofir composite (bet size, concentration, timing) minus the
profitability signals (deferred to Phase 3; they need resolution outcomes we don't
store). No agent consumes it yet — wiring is a later prompt.

**Prerequisite check:** `store/queries.py` must export `get_market_wallet_summary`
(time-anchored, returning `proxy_wallet`/`trade_count`/`total_volume_usd`/
`buy_volume_usd`/`first_trade`/`last_trade` rows). If it does not, stop and report.

**Affected packages:** `src/prediction_market/analysis/`
(`wallet_profiler.py` (new), `__init__.py`), `src/prediction_market/config.py`,
`config/default.toml`, `tests/unit/test_wallet_profiler.py` (new).

**Details:**
- **Input contract:** the profiler consumes the exact dict rows
  `get_market_wallet_summary` returns — it must not issue queries or import
  `store`/`aiosqlite` (`analysis/` stays a pure computation layer; the existing
  analyzers PriceAnalyzer/VolumeAnalyzer take plain values the same way).
- **`WalletFeatures` dataclass** (frozen): `wallet: str`, `trade_count: int`,
  `total_volume_usd: float`, `volume_share: float`, `directional_concentration:
  float`, `is_fresh: bool`, `first_trade: str`, `score: float`. Provide `to_dict()`
  returning JSON-serializable primitives (it will be embedded in
  `AnomalyReport.details` later; keep it flat).
- **`profile_wallets(summaries: list[dict], *, window_end: str, thresholds: ThresholdConfig) -> list[WalletFeatures]`**
  — module-level function, sorted by `score` DESC. Feature definitions (author
  decisions, not suggestions):
  - `volume_share` = wallet `total_volume_usd` / Σ over **all** summary rows
    (0.0 if the denominator is 0).
  - `directional_concentration` = `max(buy, sell) / total` mapped from [0.5, 1.0]
    onto [0.0, 1.0] via `(c - 0.5) * 2` (a 50/50 wallet scores 0, a one-sided
    wallet scores 1), where `buy = buy_volume_usd`, `sell = total - buy`.
    A wallet with `total_volume_usd == 0` scores 0.
  - `is_fresh` = `first_trade` (the all-history MIN from the summary) is within
    `thresholds.wallet_freshness_hours` before `window_end`. Both are TEXT
    `"%Y-%m-%d %H:%M:%S"` timestamps — parse with
    `datetime.strptime(s, "%Y-%m-%d %H:%M:%S")`; an unparseable `first_trade`
    means not fresh (legacy `T`-format rows must not crash the profiler).
  - `score` = `0.40 * volume_share + 0.30 * directional_concentration + 0.30 *
    (1.0 if is_fresh else 0.0)`, clamped to [0, 1]. Weights are fixed constants in
    the module (named `_W_SHARE`, `_W_CONCENTRATION`, `_W_FRESH`), not config —
    three config knobs nobody will tune is speculative surface.
  - Wallets with `total_volume_usd < thresholds.wallet_min_volume_usd` are
    **excluded from the returned list** (dust filter) but still count toward the
    `volume_share` denominator — share is share of real market flow.
- **New `ThresholdConfig` fields** (config.py:30-44), with the same
  field-plus-default style, and mirrored in `config/default.toml` under
  `[thresholds]` with a `# Wallet profiling (Phase 2)` comment block:
  `wallet_freshness_hours: int = 48`, `wallet_min_volume_usd: float = 1000.0`,
  `wallet_score_min: float = 0.6`, `wallet_top_k: int = 3`,
  `wallet_lookback_hours: int = 24`. The last three are consumed by the detector
  wiring in a later prompt — defining all five here keeps the config change in one
  reviewed place.
- Export `WalletFeatures`, `profile_wallets` from `analysis/__init__.py`
  (`__init__.py:16-31` is the pattern — import block plus `__all__`).
- Scope creep, prohibited by name: no win-rate/binomial p-value features (Phase 3),
  no Sybil/funding clustering (Phase 3), no DB access, no agent wiring (later
  prompt).

**Testing:**
- Hand-computed numeric proof: three constructed summary rows with known volumes →
  assert exact `volume_share`, `directional_concentration`, and `score` values
  (compute the arithmetic in the test by hand, as literals — not by re-running the
  formula).
- The dominant-insider shape: one fresh, one-sided wallet holding 60% of window
  volume scores ≥ 0.9 and ranks first; a diffuse 50/50 old wallet scores < 0.2.
- Dust exclusion: a wallet below `wallet_min_volume_usd` is absent from results but
  its volume still deflates the others' `volume_share` (assert the exact share).
- Freshness boundary: `first_trade` exactly `wallet_freshness_hours` before
  `window_end` → fresh; one second earlier → not fresh. Unparseable `first_trade`
  → `is_fresh is False`, no exception.
- Empty input → `[]`. Zero total volume → all shares 0, no ZeroDivisionError.
- `to_dict()` output survives `json.dumps` unmodified.
- Boundary-of-the-layer proof (mechanical, not prose): a test reads the source of
  `analysis/wallet_profiler.py` (`inspect.getsource` or `Path.read_text`) and
  asserts neither `aiosqlite` nor `prediction_market.store` appears in it — the
  purity rule above must be enforceable by a command, not trust.

**Invariants:** no existing analyzer files modified (price_analyzer.py,
volume_analyzer.py, timeseries.py, correlation.py untouched). No store, agent,
orchestrator, or CLI changes. `config/default.toml` and `ThresholdConfig` remain
mirror images — every new field appears in both.

**Guardrails (mandatory):**
- Run `uv run pytest tests/ -q` and make it pass before finishing.
- Run `uv run ruff check src/ tests/` — zero new errors. The 8 pre-existing
  ones (6 in `src/prediction_market/simulation/`, 2 F401 in
  `tests/unit/test_particle_filter.py`) are known and out of scope.
- If `get_market_wallet_summary`'s row shape differs from the contract above, stop
  and report — do not adapt the profiler to a different shape silently.
- Do not commit or push unless explicitly asked.

---

## Prompt 4: CUSUM change-point detector on logit-returns

**Goal:** `analysis/changepoint.py` provides a two-sided CUSUM detector over
logit-return series — materially faster than rolling z-scores at flagging sustained
regime shifts (a run of +1.5σ moves never trips a 2.5σ z-threshold but accumulates
in CUSUM within a few observations). Standalone module + tests; detector wiring is
a later prompt. This is the only new detection mathematics in the phase.

**Affected packages:** `src/prediction_market/analysis/` (`changepoint.py` (new),
`timeseries.py`, `price_analyzer.py`, `__init__.py`), `src/prediction_market/config.py`,
`config/default.toml`, `tests/unit/test_changepoint.py` (new),
`tests/unit/test_timeseries.py`.

**Details:**
- **Promote the logit helpers first** (mechanical prerequisite inside this prompt):
  `price_analyzer.py` privately defines `_clamp` (clamps to [0.005, 0.995],
  price_analyzer.py:25-35) and `_logit` (price_analyzer.py:38-50). Move the
  implementations into `timeseries.py` as public `clamp_probability(p: float) ->
  float` and `logit(p: float) -> float` (docstrings included), re-export from
  `analysis/__init__.py`, and make price_analyzer.py import them
  (`from .timeseries import clamp_probability as _clamp, logit as _logit` keeps its
  call sites byte-identical). Behavior must not change — the existing
  test_price_analyzer.py suite is the no-regression proof.
- **`CusumAlarm` dataclass** (frozen): `market_id: str`, `direction: str`
  (`"up"`/`"down"`), `statistic: float` (the S value that crossed), `threshold:
  float`, `observations_since_reset: int`, `timestamp: datetime`.
- **`CusumDetector`** class, surface deliberately parallel to `PriceAnalyzer`
  (price_analyzer.py:85-309) so the detector wiring later is symmetrical:
  - `__init__(self, thresholds: ThresholdConfig | None = None)` — reads `cusum_k`,
    `cusum_h`, `cusum_min_observations`, `rolling_window_days` (baseline window)
    from thresholds.
  - `update(self, market_id, price, timestamp=None) -> None` — computes the
    logit-return from the previous price exactly as PriceAnalyzer does
    (`logit(clamp(p)) - logit(clamp(p_prev))`, cf. price_analyzer.py:151), feeds a
    per-market baseline `RollingStats` (timeseries.py:43-188; sample stdev), and —
    only once the baseline has ≥ `cusum_min_observations` returns — standardizes
    the return `z = (r - mean) / std` (guard `std < 1e-12` → z = 0, mirroring
    `compute_z_score`, timeseries.py:16-32) and accumulates
    `S⁺ = max(0, S⁺ + z - k)` and `S⁻ = max(0, S⁻ - z - k)`.
    **Decision — the baseline includes every observed return, including the ones
    that trip the alarm.** Excluding post-alarm returns from the baseline
    ("contamination control") is a Phase-3 refinement; do not build it.
  - `check_alarm(self, market_id) -> CusumAlarm | None` — returns an alarm when
    `S⁺ >= h` (direction "up") or `S⁻ >= h` ("down"); **resets both S statistics to
    0 and the observation counter on alarm** (one alarm per excursion, not one per
    tick). Returns None during warm-up.
  - `current_statistics(self, market_id) -> tuple[float, float] | None` — (S⁺, S⁻)
    for diagnostics, None if untracked.
  - `to_dict()` / `from_dict(data, thresholds=None)` — full state round-trip,
    following the serialization pattern of PriceAnalyzer
    (price_analyzer.py:262-303), including each market's RollingStats via its own
    `to_dict`/`from_dict` (timeseries.py:136-172).
- **New `ThresholdConfig` fields** (config.py:30-44), mirrored in
  `config/default.toml` `[thresholds]` under a `# CUSUM change-point (Phase 2)`
  comment: `cusum_k: float = 0.5` (slack, in σ), `cusum_h: float = 5.0` (decision
  threshold, in accumulated σ), `cusum_min_observations: int = 5`,
  `cusum_amplifier: float = 1.3` (consumed by detector wiring in a later prompt;
  defined here so the config lands in one place).
- Export `CusumDetector`, `CusumAlarm` from `analysis/__init__.py`.
- Scope creep, prohibited by name: no BOCPD/Bayesian variant, no
  volatility-conditioning on time-to-resolution (both are documented Phase-3
  candidates in the research brief), no agent wiring.

**Testing:**
- **The headline claim — CUSUM catches what z-scores miss:** construct a series of
  20 baseline returns (constant small alternation, near-zero σ is disallowed — use
  a fixed ±0.01 pattern), then append 6 consecutive returns of +1.5× the baseline σ.
  Assert `CusumDetector` alarms within those 6 observations **and** that
  `PriceAnalyzer` configured with the default `price_zscore=2.5` threshold, fed the
  identical price series, never returns an anomaly. This pairing is the reason the
  module exists; the test must construct both detectors explicitly.
- Stationary noise stays silent: 100 deterministic small-alternation returns → no
  alarm (do not use `random` without a fixed seed anywhere in tests).
- Reset semantics: after an alarm fires, `current_statistics` reads (0.0, 0.0), and
  an immediately following quiet observation does not re-alarm.
- Direction: a sustained negative shift produces `direction == "down"`.
- Warm-up: fewer than `cusum_min_observations` returns → `check_alarm` is None even
  for huge moves.
- `to_dict` → `from_dict` → identical `current_statistics` and identical alarm
  behavior on the next observation (serialize mid-excursion to prove S state
  survives).
- Logit-helper promotion: full existing test_price_analyzer.py passes unmodified —
  that suite is the no-regression proof for the move; expectation changes there are
  a red flag requiring justification. Add direct unit tests for public
  `clamp_probability`/`logit` in test_timeseries.py (clamp bounds 0.005/0.995,
  logit(0.5) == 0).

**Invariants:** `PriceAnalyzer`'s observable behavior byte-identical (same anomalies
on the same series). `volume_analyzer.py`, `correlation.py` untouched. No agent,
store, orchestrator, or CLI changes. `ThresholdConfig` and `config/default.toml`
stay mirror images.

**Guardrails (mandatory):**
- Run `uv run pytest tests/ -q` and make it pass before finishing.
- Run `uv run ruff check src/ tests/` — zero new errors. The 8 pre-existing
  ones (6 in `src/prediction_market/simulation/`, 2 F401 in
  `tests/unit/test_particle_filter.py`) are known and out of scope.
- No wall-clock-dependent or unseeded-random tests — the repo already has
  flake-prone timing tests (test_rate_limiter.py); do not add more.
- If the codebase contradicts anything stated above, stop and report instead of
  improvising.
- Do not commit or push unless explicitly asked.

---

## Prompt 5: Multi-signal corroboration in InfoLeakDetector — CUSUM amplifier + wallet evidence

**Goal:** `InfoLeakDetector` corroborates its z-score composite with the two new
signal families: an active CUSUM alarm amplifies the combined score, and — once a
report is warranted — wallet-level evidence from the trade window is attached to the
report and can bump confidence. This changes the live scoring path; it is the
series' largest live blast radius and a recommended plan-mode checkpoint.

**Prerequisite check:** `analysis` must export `CusumDetector` and
`profile_wallets`/`WalletFeatures`; `store.queries` must export
`get_market_wallet_summary`; `ThresholdConfig` must have `cusum_amplifier`,
`wallet_lookback_hours`, `wallet_top_k`, `wallet_score_min`.
Any of these missing ⇒ a prerequisite session did not land — stop and report.

**Affected packages:** `src/prediction_market/agents/info_leak_detector.py`,
`tests/unit/test_info_leak_detector.py`, `tests/integration/test_pipeline.py`
(additive assertions only).

**Details:**
- **CUSUM wiring:** construct `self._cusum = CusumDetector(config.thresholds)` in
  `__init__` next to the existing analyzers (info_leak_detector.py:130-131). In
  `_process_market`, immediately after `self._price_analyzer.update(...)`
  (info_leak_detector.py:192-193): `self._cusum.update(market_id, price, ts)` and
  then `alarm = self._cusum.check_alarm(market_id)` — **unconditionally, exactly
  once per market per tick, BEFORE the trigger gate** (`if not (price_triggered or
  volume_triggered): return`, info_leak_detector.py:198-201). This placement is
  state hygiene, not style: `check_alarm` resets S on fire, and if it were only
  called on z-triggered ticks, a stale excursion from days ago would sit
  accumulated until an unrelated marginal trigger read it. Store `alarm` in a
  local; never call `check_alarm` twice in one tick.
  **Scope decision, stated so nobody over-reads this prompt:** in live wiring this
  phase, CUSUM is a *corroborating amplifier only* — an alarm on a tick where
  neither z-trigger fired does not by itself proceed past the gate or emit a
  report. (Its sub-threshold-run detection value is exercised standalone and in
  the backtest harness; promoting it to an independent live alert path is a
  Phase-3 decision, made after the backtest evidence exists.) After the existing
  amplifier/dampener chain (event amplifier then news dampener,
  info_leak_detector.py:209-235): if `alarm` is not None, multiply `combined` by
  `thresholds.cusum_amplifier` and append to `amplifiers_applied` following the
  existing string pattern (:220-222). The CUSUM amplifier applies after the news
  dampener — corroboration multiplies the *net* evidence, and the existing
  amplifier/dampener code stays untouched.
- **Wallet corroboration — only after the gate.** The existing threshold gate
  (`combined < thresholds.combined_score_min: return`, info_leak_detector.py:245-246)
  and cooldown (:248-251) stay exactly where they are. Once a report will be
  emitted, and before building it:
  - `window_end = ts` formatted `"%Y-%m-%d %H:%M:%S"`, `window_start = ts -
    timedelta(hours=thresholds.wallet_lookback_hours)` — **anchored to the
    snapshot timestamp, never `datetime.now()`**. This is not a style preference:
    the Phase-2 replay harness drives this detector over historical time, and a
    wall-clock window would silently query zero trades there while looking correct
    live.
  - `summaries = await queries.get_market_wallet_summary(self.db, market_id,
    window_start, window_end)`; `features = profile_wallets(summaries,
    window_end=window_end, thresholds=self.config.thresholds)`.
  - **The data flows into the report through `_emit_report`'s signature — this is
    the decided mechanism, not one option:** add keyword-only params
    `wallet_features: list[WalletFeatures]` and `cusum_alarm: CusumAlarm | None`
    to `_emit_report`, passed from `_process_market` (the existing call at
    info_leak_detector.py:255-265 area). Inside `_emit_report`:
    - `details["wallet_evidence"] = [f.to_dict() for f in
      wallet_features[:thresholds.wallet_top_k]]`.
    - `details["cusum"] = {"direction": alarm.direction, "statistic":
      alarm.statistic, "threshold": alarm.threshold}` when the alarm is not None,
      else `None`.
    - `details["snapshot_timestamp"] = ts.strftime("%Y-%m-%d %H:%M:%S")` — the
      historical time the report is *about*. `AnomalyReport.created_at` is
      wall-clock emission time (anomaly_report.py:33-34) and the two diverge by
      construction under replay; every downstream consumer that needs "when did
      this anomaly happen" reads this key, never `created_at`.
    - `details` is JSON-serialized by `BaseAgent._persist_report`
      (base.py:147-151) — everything attached must be plain primitives, which
      `WalletFeatures.to_dict()` guarantees.
    - After the existing `confidence = self._score_to_confidence(combined)`
      (info_leak_detector.py:294): if `wallet_features` is non-empty and
      `wallet_features[0].score >= thresholds.wallet_score_min`, then
      `confidence = min(1.0, confidence + 0.10)` and append a sentence naming
      the wallet and its volume share to the report summary. The bump happens
      **before** `severity_from_score(confidence)` (:295) so severity and
      confidence stay consistent.
  - **Soft-fail is mandatory:** an empty trades table (the live DB today has zero
    trades) yields `wallet_evidence: []` and no confidence bump — never an
    exception, never a skipped report.
- Scope creep, prohibited by name: do not wire `CusumDetector` or wallet evidence
  into `ManipulationGuard` (its wash-trading upgrade is a separate future series);
  do not add new columns to `anomaly_reports` (wallet evidence rides inside the
  existing `details` JSON column); do not touch the analyzers' internals.

**Testing:**
- Extend tests/unit/test_info_leak_detector.py (existing harness constructs the
  detector against a temp DB):
  - CUSUM amplifier proof: a series engineered to trip CUSUM but sit just under
    `combined_score_min` without the amplifier → report emitted **only** with the
    amplifier applied; `details["cusum"]["direction"]` present. (This proves the
    multiplication happened and in the right place — compute the pre-amplifier
    combined in the test and assert the boundary.)
  - Wallet evidence attachment: seed the temp DB with trades (via
    `save_trades_batch`) inside the lookback window whose match_times are near the
    *snapshot's* timestamp — then also assert the negative control: identical
    trades timestamped near `datetime.now()` but far from the snapshot ts produce
    **empty** wallet evidence (the snapshot-anchored-window proof).
  - Confidence bump boundary: top wallet score exactly at `wallet_score_min` bumps;
    below does not; confidence caps at 1.0.
  - Soft-fail: empty trades table → report still emitted, `wallet_evidence == []`.
- tests/integration/test_pipeline.py: `test_info_leak_pipeline_emits_report`
  (test_pipeline.py:125) must pass with at most additive assertions (e.g.
  `details` now contains a `wallet_evidence` key). Weakening or removing any
  existing assertion in that file is a red flag — it is the Phase-1 flagship.
- All existing tests pass; expectation changes are a red flag — justify each one.

**Invariants:** the emitted report count and market selection for any series that
never trips CUSUM and has no stored trades are **identical to pre-change behavior**
(amplifier multiplies only on alarm; wallet path activates only past the gate).
`agent="info_leak"` string unchanged (info_leak_detector.py:354). No store,
analysis, orchestrator, reporting, or CLI files touched. `ManipulationGuard`
untouched.

**Guardrails (mandatory):**
- Run `uv run pytest tests/ -q` and make it pass before finishing.
- Run `uv run ruff check src/ tests/` — zero new errors. The 8 pre-existing
  ones (6 in `src/prediction_market/simulation/`, 2 F401 in
  `tests/unit/test_particle_filter.py`) are known and out of scope.
- All DB timestamps are TEXT `"%Y-%m-%d %H:%M:%S"` UTC compared lexicographically —
  the wallet window bounds must be formatted through that pattern.
- If the codebase contradicts anything stated above, stop and report instead of
  improvising.
- Do not commit or push unless explicitly asked.

---

## Prompt 6: Point-in-time replay harness

**Goal:** A new `backtest/` package can replay an archived market case through the
real `InfoLeakDetector` against a throwaway SQLite DB, strictly point-in-time: at
each step the detector sees only data timestamped at or before that step. This is
the engine every later evaluation stands on; run it in plan mode and review the
plan first (recommended human checkpoint). Case *creation* tooling is a separate
prompt — this one defines the format and consumes hand-built fixtures.

**Prerequisite check:** `InfoLeakDetector` must window its wallet query off the
snapshot timestamp (grep for `wallet_lookback_hours` in info_leak_detector.py),
set `details["snapshot_timestamp"]` in `_emit_report`, and accept an injected
`news_checker` (info_leak_detector.py:122-137). Any missing ⇒ stop and report.

**Affected packages:** `src/prediction_market/backtest/` (new: `__init__.py`,
`case_format.py`, `replay.py`), `src/prediction_market/store/snapshots.py` (one
additive param), `tests/backtest/` (new: `__init__.py`, `test_replay.py`),
`tests/fixtures/cases/minimal/` (new hand-built fixture case).

**Details:**
- **Case format** (`case_format.py`) — a case is a directory:
  - `case.toml` — manifest: `[case]` with `slug`, `market_id`, `condition_id`,
    `question`, `archived_at`, `notes`; optional `[label]` with `window_start`,
    `window_end`, `event_time` (all `"%Y-%m-%d %H:%M:%S"` TEXT) — the labeled
    anomaly window for scoring (consumed by the evaluation prompt; parsed and
    exposed now).
  - `market.json` — one Gamma-style market dict (parseable by
    `GammaMarket.model_validate`; the string-typed JSON quirks are already handled
    by its validators, data/polymarket/models.py:18-30 and :59-62).
  - `snapshots.json` — the replay spine: a time-ascending list of
    `{"timestamp": "...", "price_yes": float, "price_no": float,
    "volume_total": float}` rows. (Snapshots, not raw price history — CLOB
    `/prices-history` carries no volume, so the archiver derives per-interval
    volume from trades; hand-built cases state it directly.)
  - `trades.json` — list of Data-API-shaped trade dicts (camelCase keys,
    parseable by `Trade.model_validate`, including `proxyWallet`).
  - `events.json` (optional) — list of `ScheduledEvent`-shaped dicts. **The
    loader constructs real `ScheduledEvent` instances** (the dataclass in
    data/external/models.py — fields include `source`, `event_type`, `title`,
    `description`, `event_date: datetime`, `url`, `keywords`), parsing
    `event_date` with `datetime.strptime(s, "%Y-%m-%d %H:%M:%S")` —
    `save_scheduled_events` (queries.py:361-402) does attribute access and
    `.strftime()` on its inputs, so raw dicts would crash it.
  - Loader: `@dataclass Case` (manifest fields + parsed models + snapshot rows) and
    `load_case(path: Path) -> Case`, validating on load and raising `ValueError`
    naming the file and problem: snapshots out of time order; `snapshots.json`
    missing; **manifest `market_id` ≠ `market.json`'s `id`** (without this check,
    a drifted hand-built fixture replays against a market row
    `get_active_political_markets` never returns and silently emits zero reports).
    A writer `save_case(path: Path, case: Case)` (round-trip partner; the archiver
    prompt reuses it).
- **`save_price_snapshot` gains an explicit timestamp** (snapshots.py:98-136): add
  keyword-only `timestamp: str | None = None`, defaulting to the current `_utcnow()`
  behavior. Replay passes historical timestamps; live callers change zero call
  sites.
- **Replay engine** (`replay.py`):
  - `class _ListSink(ReportSink)` — captures reports in `self.reports: list`
    (implement `write`; `ReportSink` ABC at reporting/sink.py:24-32).
  - `class _NullNewsChecker` — `async def check_news_exists(self, keywords,
    before_time=None)` returns `NewsCheckResult(news_found=False)` (the dataclass
    at data/external/models.py:39-49; the detector reads `.news_found` and
    `.articles`, info_leak_detector.py:226-235). Replay is offline by definition;
    the news dampener is out of scope for replayed history, and this stub
    documents that.
  - `async def replay_case(case: Case, config: AppConfig) -> ReplayResult`:
    1. Create a fresh DB via `init_database` at a caller-supplied/tmp path.
    2. Insert the market via `save_market` with political classification
       `{"confidence": 1.0, "reasons": ["backtest case"]}`, and **force
       `active=1, closed=0`** on the stored row regardless of the archived
       market's real (resolved) state — `get_active_political_markets` filters
       `active = 1 AND political_confidence > 0` (queries.py:66-94) and a replayed
       case is by construction a market under surveillance.
    3. Insert all `events.json` rows up front via `save_scheduled_events`
       (queries.py:361-402). **Decision:** scheduled events are public calendar
       announcements known in advance; inserting them wholesale is not lookahead.
       Trades and snapshots, which *are* market activity, stream strictly by time.
    4. Construct one `InfoLeakDetector(config, db, sinks=[_ListSink()],
       news_checker=_NullNewsChecker())`.
    5. For each snapshot row in order: `save_price_snapshot(db, market_id,
       price_yes=..., price_no=..., volume_total=..., timestamp=row_ts)`; then
       `save_trades_batch` for all not-yet-inserted trades with
       `match_time <= row_ts` (batch is `INSERT OR IGNORE` — idempotent,
       snapshots.py:247-294); then `await detector.tick()`.
    6. Return `ReplayResult`: `reports: list[AnomalyReport]` (from the sink, in
       emission order), `steps: int`, `case: Case`.
  - The engine drives the real `tick()` — it must not reach into detector
    internals, monkeypatch time, or reimplement scoring. If the detector needs
    something replay can't supply, that is a finding to report, not to shim.
- Scope creep, prohibited by name: no ManipulationGuard replay (it fetches live
  order books over HTTP mid-tick — replaying it needs an order-book case format,
  Phase 3); no CLI command yet (evaluation prompt); no multi-market cases (one
  market per case in this phase).
- **Fixture** `tests/fixtures/cases/minimal/`: hand-built, ~30 snapshots at a fixed
  hourly cadence with a quiet series then an obvious sustained jump plus a volume
  spike, 6–8 trades across 2 wallets, one scheduled event, `[label]` filled in.
  Values chosen so the default thresholds flag it (work the arithmetic; do not
  brute-force).

**Testing (`tests/backtest/test_replay.py`):**
- Round-trip: `load_case(save_case(tmp, case))` equals the original (field-wise).
- Malformed cases raise `ValueError` naming the offending file: out-of-order
  snapshots; missing `snapshots.json`.
- `replay_case` on the minimal fixture emits ≥ 1 report, and every report's
  `market_id` matches the case.
- **Report comparisons throughout this file use
  `(details["snapshot_timestamp"], severity, summary)` tuples** — never `id`
  (random uuid) and never `created_at` (wall-clock emission time, which differs
  across runs by construction).
- **The no-lookahead proofs (the point of the engine):**
  - Prefix property: replaying a copy of the case truncated to its first N
    snapshots yields exactly the same report tuples as the first-N-step prefix of
    the full replay. Run for N = the step just *before* the fixture's anomaly —
    the truncated run must emit **zero** reports.
  - Trade-window property: give the minimal fixture one enormous, fresh,
    one-sided trade timestamped *after* the anomaly step. The report emitted at
    the anomaly step must show **no trace of it** in
    `details["wallet_evidence"]` (its wallet absent, shares computed without
    it). Note the SQL window (`match_time` bounds) is what enforces this —
    insertion order alone would not — so this assertion, not insertion
    choreography, is the honest lookahead check on the trade side.
- `save_price_snapshot` regression: calling it without `timestamp` still stamps
  `_utcnow()` (assert the stored value parses as within the test's runtime, not
  the historical range).
- Determinism: two `replay_case` runs of the same case produce identical report
  tuple sequences (the tuple basis above — `id`/`created_at` excluded).

**Invariants:** no changes to any agent's scoring logic. `save_price_snapshot`
call sites in orchestrator.py (:386-402 backfill, snapshot loop) are untouched and
behaviorally identical. `store/queries.py` untouched. Nothing imports
`prediction_market.simulation`.

**Guardrails (mandatory):**
- Run `uv run pytest tests/ -q` and make it pass before finishing.
- Run `uv run ruff check src/ tests/` — zero new errors. The 8 pre-existing
  ones (6 in `src/prediction_market/simulation/`, 2 F401 in
  `tests/unit/test_particle_filter.py`) are known and out of scope.
- All DB timestamps TEXT `"%Y-%m-%d %H:%M:%S"` UTC, lexicographic — case files use
  the same format everywhere.
- The trade-window test must fail if the future trade's wallet appears in the
  anomaly-step report — verify by temporarily widening the SQL window bound (then
  restore); state in your summary that you performed this check.
- If the codebase contradicts anything stated above, stop and report instead of
  improvising.
- Do not commit or push unless explicitly asked.

---

## Prompt 7: Case archiver — `archive-case` and `cases` CLI commands

**Goal:** One command freezes a real Polymarket market into the on-disk case format:
`prediction-market archive-case <slug>` fetches the market, its full CLOB price
history, and its complete trade tape, derives the snapshot spine, and writes a case
directory that `load_case` accepts. A companion `cases` command lists archived
cases. This is the bridge from live APIs to the replay harness.

**Prerequisite check:** `backtest/case_format.py` must export
`Case`/`load_case`/`save_case`, and `DataClient.get_all_trades` must accept
`user`/`taker_only` passthrough (data_client.py). Missing ⇒ stop and report.

**Affected packages:** `src/prediction_market/backtest/archiver.py` (new),
`src/prediction_market/cli.py`, `.gitignore`, `tests/backtest/test_archiver.py`
(new).

**Details:**
- **`archiver.py`:** `async def archive_case(config: AppConfig, slug: str, output_dir:
  Path, *, max_trade_pages: int = 200) -> Path`:
  1. Resolve the market: `GammaClient.search_markets(slug)`
     (gamma_client.py:103-114 — slug-exact filter, not text search; it does not
     constrain `active`/`closed`, so resolved markets are reachable). Zero matches
     ⇒ raise `ValueError` telling the operator to check the slug. Multiple ⇒ take
     the exact slug match; none-exact ⇒ error listing candidates.
  2. Price history: for the YES token (`market.yes_token_id`, models.py:86-99 —
     the property itself already falls back to `clob_token_ids[0]`; if it returns
     `None` the market has no CLOB tokens at all — treat that exactly like empty
     price history below, do not raise), call
     `ClobClient.get_price_history(token_id, interval="max", fidelity=10)`
     (clob_client.py:98-129 passes `interval`/`fidelity` through unvalidated).
     **`interval="max"` is an assumption about the live API, not a verified
     fact** — the docstring documents only `1m/1h/1d` examples, and this repo's
     tests are fully mocked. Say exactly that in the archiver docstring, and know
     that the live validation session at the end of this phase is tasked with
     confirming it and reporting a mismatch back. Distinguish outcomes honestly:
     an empty `history` on a 200 response is recorded as
     `"price_history": "empty"` in the manifest `notes` (resolved pre-CLOB-V2
     markets may legitimately return nothing); an HTTP error (4xx after retries)
     is recorded as `"price_history": "error: <status>"` — never conflate the
     two, and neither crashes the archive.
  3. Trades: `DataClient.get_all_trades(condition_id=market.condition_id,
     max_pages=max_trade_pages)` — with the per-page limit of 100
     (data_client.py:86-122) the default caps at 20,000 trades; log a WARNING
     naming the cap if the final page came back full (the tape may be truncated —
     no silent caps).
  4. Derive `snapshots.json` from price points + trades: one row per price point,
     `price_yes = p`, `price_no = 1 - p`, `volume_total` = cumulative running sum
     of `volume_usd` of all trades with `match_time <=` that point's timestamp
     (convert CLOB epoch `t` to `"%Y-%m-%d %H:%M:%S"` UTC). This mirrors what the
     live snapshot loop would have recorded, from the only data the APIs still
     serve — say exactly that in the module docstring.
  5. Write via `save_case`; manifest `archived_at` = now, `notes` documents
     page-cap status and any empty responses. `[label]` is left absent — labeling
     is a human judgment recorded later.
  6. Client lifecycle: construct Gamma/Clob/Data clients + a shared httpx client
     locally (the same clients `Orchestrator.backfill` builds for itself,
     orchestrator.py:337-428) — but **unlike** `backfill`, which does not
     guarantee teardown on exception, close them in a `finally` block. Do not
     copy the orchestrator's unguarded teardown; the archiver must not leak
     clients on a failed fetch.
- **CLI** (cli.py — follow the existing subcommand pattern, e.g. `backfill` at
  cli.py:175-200): `archive-case SLUG --output data/cases --max-trade-pages 200`
  runs the archiver and prints the written path plus counts (snapshots, trades,
  events); `cases --dir data/cases` lists case directories with slug, question,
  snapshot/trade counts, and whether `[label]` is present, via `load_case`.
- **`.gitignore`:** add `data/cases/` (archives are large and regenerable).
  The existing file previously had a bare-`data/`-pattern bug (fixed in commit
  2523222) — the entry must be exactly `data/cases/`, anchored, nothing broader.
- Scope creep, prohibited by name: no NO-token price history (YES only — `price_no
  = 1 - price_yes` matches how the detector consumes snapshots); no order-book
  archiving (needs live WS capture, Phase 3); no auto-labeling of `[label]`.

**Testing (`tests/backtest/test_archiver.py`, all respx-mocked — unmocked routes
raise, so the tests also prove exactly which endpoints are called):**
- Happy path: mock `/markets?slug=...` (reuse a record from
  tests/fixtures/gamma_markets.json), `/prices-history` (fixture
  price_history.json shape), `/trades` (two pages via `side_effect=[...]`,
  second page short) → resulting directory passes `load_case`; `snapshots.json`
  volume column is the hand-computed cumulative sum of the mocked trades; the
  final case round-trips into `replay_case` without error (smoke — one call, no
  report assertions).
- Empty price history → case still written, manifest notes record it, zero
  snapshots.
- Slug with zero Gamma matches → `ValueError` with the slug in the message.
- Trade-cap warning: mock exactly `max_trade_pages` full pages (use
  `max_trade_pages=2`, tiny limit) and assert the truncation warning was logged
  (`caplog`).
- CLI: invoke `archive-case`/`cases` via `click.testing.CliRunner` against the
  mocked routes and a tmp dir; assert exit code 0 and the printed counts.

**Invariants:** no orchestrator, agent, store, or analysis changes. Existing CLI
subcommands and flags unchanged. `search_markets`, `get_price_history`,
`get_all_trades` signatures unchanged (the archiver is a consumer, not an editor).

**Guardrails (mandatory):**
- Run `uv run pytest tests/ -q` and make it pass before finishing.
- Run `uv run ruff check src/ tests/` — zero new errors. The 8 pre-existing
  ones (6 in `src/prediction_market/simulation/`, 2 F401 in
  `tests/unit/test_particle_filter.py`) are known and out of scope.
- respx raises on unmocked routes — no test may touch the network.
- Editing the CLI without invoking it does not count as done — the CliRunner tests
  are mandatory, not optional.
- If the codebase contradicts anything stated above, stop and report instead of
  improvising.
- Do not commit or push unless explicitly asked.

---

## Prompt 8: Backtest evaluation — `backtest` CLI, case scoring, and synthetic null controls

**Goal:** Replays become measurements. `prediction-market backtest --case <dir>`
replays a case, scores the result against the manifest's labeled window (detected?
lead time? false alarms?), and — with `--null-runs N` — measures the detector's
false-positive rate on synthetic volatility-matched null markets. This
operationalizes the research brief's false-positive discipline: an alert only means
something relative to a matched reference.

**Prerequisite check:** `backtest.replay.replay_case`, `backtest.case_format.Case`
(with parsed `[label]` fields), and the fixture case
`tests/fixtures/cases/minimal/` must exist. Missing ⇒ stop and report.

**Affected packages:** `src/prediction_market/backtest/` (`metrics.py` (new),
`synthetic.py` (new)), `src/prediction_market/cli.py`,
`tests/backtest/test_metrics.py` (new), `tests/backtest/test_synthetic.py` (new).

**Details:**
- **`metrics.py`:**
  - `@dataclass ReplayEvaluation`: `case_slug: str`, `detected: bool`,
    `first_hit_time: str | None`, `lead_time_minutes: float | None` (event_time −
    first hit; None when no event_time or no hit), `hits: int` (reports inside
    `[window_start, window_end]`), `false_alarms: int` (reports outside it),
    `total_reports: int`, `to_dict()`.
  - `evaluate_replay(result: ReplayResult, case: Case) -> ReplayEvaluation` —
    **report times come from `report.details["snapshot_timestamp"]`** (the
    historical TEXT timestamp the detector stamps on every report), compared
    lexicographically against the label's TEXT timestamps.
    `AnomalyReport.created_at` is wall-clock emission time (a `datetime`,
    anomaly_report.py:33-34) and is meaningless under replay — a report missing
    `snapshot_timestamp` is a hard `ValueError` (it means a non-Phase-2 detector
    produced it), not a silent fallback to `created_at`. A case with no `[label]` yields `detected=False`,
    `hits=0`, and `false_alarms=total_reports` **with a `labeled: false` flag in
    `to_dict()`** so downstream consumers can't mistake "unlabeled" for "clean
    miss" — put `labeled: bool` on the dataclass.
  - **Decision — "detected" means:** ≥ 1 report with severity `medium` or higher
    (per `SEVERITY_ORDER`, store/models.py:20-25) whose `created_at` falls inside
    the labeled window and at or before `event_time` when present. Low-severity
    murmurs inside the window count as hits but do not flip `detected`.
- **`synthetic.py`:**
  - `def generate_null_case(template: Case, *, seed: int) -> Case` — a benign
    twin: same market metadata (slug suffixed `-null-<seed>`), same snapshot
    timestamps and cadence, prices regenerated as a logit-space Gaussian random
    walk (`random.Random(seed)`, never the global RNG) whose per-step σ equals the
    **sample** standard deviation of the template's own logit-returns (reuse
    `logit`/`clamp_probability` from `analysis.timeseries` and `RollingStats`'s
    sample-stdev convention — compute directly with `statistics.stdev`), starting
    from the template's first price; `volume_total` rebuilt as a smooth cumulative
    series with the template's mean per-step volume increment and no spikes;
    trades: the template's trades resampled to spread uniformly across the
    timeline with wallets replaced by `0xnull<i>` round-robin across 10 wallets
    (kills concentration and freshness structure); events: **copied as-is**
    (a null market shares the public calendar — the detector must stay quiet
    *despite* the event amplifier, which is precisely the falsifiable part of the
    control).
  - `def estimate_false_positive_rate(template: Case, config: AppConfig, *,
    runs: int, base_seed: int = 1000) -> dict` — replays `runs` null cases
    (seeds `base_seed + i`), returns `{"runs": N, "paths_with_reports": k,
    "reports_per_path_mean": x, "fp_path_rate": k/N}`. Async (drives
    `replay_case`).
- **CLI:** `backtest --case DIR [--null-runs N] [--seed 1000] [--json]` — prints
  the evaluation (question, labeled window, detected/lead-time/hits/false-alarms,
  each report's timestamp+severity+summary line), then the null-control block when
  requested. `--json` emits `ReplayEvaluation.to_dict()` (+ the FP dict) for
  scripting. Follow the click patterns at cli.py:175-200; async entry via the same
  `asyncio.run` idiom the other commands use.
- Scope creep, prohibited by name: no precision/recall sweeps across threshold
  grids (a tuning study, not phase infrastructure); no plotting; no persistence of
  evaluations to the DB (stdout/JSON only this phase).

**Testing:**
- `evaluate_replay` against hand-built `ReplayResult`s (construct `AnomalyReport`s
  directly — reporting/anomaly_report.py:13-34): report inside window before
  event_time + severity high → `detected=True`, correct positive
  `lead_time_minutes` (hand-computed literal); only-low-severity hit →
  `detected=False`, `hits=1`; report outside window → `false_alarms=1`; unlabeled
  case → `labeled=False` and `detected=False`; no-reports → all zeros.
- Boundary: report exactly at `window_start` and exactly at `event_time` both
  count (lexicographic `>=`/`<=`).
- `generate_null_case`: deterministic for a fixed seed (two calls byte-equal);
  different seeds differ; per-step logit-return sample stdev of the null is within
  25% of the template's (assert numerically); null trades show ≥ 10 distinct
  wallets and no wallet above 20% volume share; events identical to template's.
- `estimate_false_positive_rate` with **exactly** `template = the minimal
  fixture, runs=3, base_seed=1000` (these literals are fixed here so the test
  cannot be tuned after seeing a result): returns the right shape, and — the
  honest claim — `fp_path_rate` on these nulls is `0.0` with default thresholds.
  If the first unmodified run is not 0.0, that is a real calibration finding:
  state the raw observed value in your summary, assert that observed value with
  an explanatory comment, and change **nothing** to force a zero — not
  thresholds, not the seed, not the null-generation logic.
- CLI: CliRunner on the minimal fixture case — exit 0, output contains
  detected/lead-time lines; `--json` parses with `json.loads`.

**Invariants:** `replay_case` and the case format untouched. No agent, store, or
analysis changes. Existing CLI subcommands unchanged. Synthetic generation must not
mutate the template `Case` (assert-worthy: deep-copy in, compare after).

**Guardrails (mandatory):**
- Run `uv run pytest tests/ -q` and make it pass before finishing.
- Run `uv run ruff check src/ tests/` — zero new errors. The 8 pre-existing
  ones (6 in `src/prediction_market/simulation/`, 2 F401 in
  `tests/unit/test_particle_filter.py`) are known and out of scope.
- Seeded `random.Random` instances only — never the global RNG, never unseeded.
- Editing the CLI without invoking it does not count as done — CliRunner tests are
  mandatory.
- If the codebase contradicts anything stated above, stop and report instead of
  improvising.
- Do not commit or push unless explicitly asked.

---

## Prompt 9: Wallet surfacing — `wallets` and `wallet` CLI commands, report rendering

**Goal:** Wallet evidence becomes visible to the operator: a `wallets` command
ranks a market's wallets by profiler score, a `wallet` command shows one address's
cross-market history from the local DB, and the Markdown report formatter renders
the `wallet_evidence` section that InfoLeakDetector now attaches.

**Prerequisite check:** `analysis` must export `profile_wallets`/`WalletFeatures`;
`store.queries` must export `get_market_wallet_summary` and `get_wallet_trades`;
`InfoLeakDetector._emit_report` must set `details["wallet_evidence"]` and
`details["cusum"]` (the latter a dict with `direction`/`statistic`/`threshold`
keys, or None). Any missing ⇒ stop and report.

**STEP 0 (before touching any file):** capture the golden for the formatter
no-regression test. Run the *current* `human_formatter.format_report` on a
representative report whose `details` has none of the Phase-2 keys, and hardcode
its exact output as a string literal in the new test. If you have already edited
`human_formatter.py` when you read this, recover the pre-edit source with
`git show HEAD:src/prediction_market/reporting/human_formatter.py` (HEAD only
moves on commit, and this session does not commit). A golden produced by calling
the *new* code is worthless — it cannot detect the regression it exists to catch.

**Affected packages:** `src/prediction_market/cli.py`,
`src/prediction_market/reporting/human_formatter.py`,
`tests/unit/test_cli_wallets.py` (new), `tests/unit/test_anomaly_report.py` or the
formatter's existing test home (`tests/integration/test_reporting.py`) — additive.

**Details:**
- **`wallets MARKET_ID [--hours 168]`:** open the DB read-only the way `markets`
  does (direct connection + query, cli.py:208-259 is the established pattern for
  read-only CLI views), compute `end = now`, `start = now - hours` in the TEXT
  timestamp format, call `get_market_wallet_summary`, run `profile_wallets`
  (thresholds from the loaded config), and print a table: wallet (abbreviated
  `0x1234…abcd`), trades, volume USD, share %, concentration, fresh?, score —
  sorted by score. Empty result prints a clear "no wallet-attributed trades" line
  mentioning that trades ingested before Phase 2 lack attribution.
- **`wallet ADDRESS [--limit 50]`:** `get_wallet_trades` → table of match_time,
  market_id, side, outcome, price, volume USD; footer totals (trade count, total
  volume, buy/sell split). This is a local-DB view only — it must not call the
  Data API (`get_wallet_positions`/`get_wallet_activity` are for the live
  validation session and Phase 3 investigations; say so in the docstring).
- **Formatter:** in `human_formatter.format_report`
  (human_formatter.py:75-165), when `report.details.get("wallet_evidence")` is a
  non-empty list, render a `## Wallet Evidence` section — one line per wallet:
  address, share %, concentration, fresh flag, score — between the existing
  evidence sections and the calendar table; and when `details.get("cusum")` is
  truthy, one line noting the change-point direction and statistic. Reports
  without these keys (every pre-Phase-2 report, and ManipulationGuard's) render
  **byte-identically to today** — that is the compatibility line.
- Scope creep, prohibited by name: no new store queries (the two existing ones
  suffice); no webhook/JSON formatter changes (`json_formatter` already serializes
  `details` wholesale); no live-API calls from either command.

**Testing:**
- CliRunner + temp DB seeded via `save_trades_batch`: `wallets` output contains
  the top wallet's abbreviated address and its hand-computed share; `--hours`
  window excludes an old trade (seed one outside the window and assert its volume
  is absent from the totals); empty DB prints the no-data line, exit 0.
- `wallet` totals are hand-computed literals from the seeded trades; unknown
  address prints empty-state, exit 0.
- Formatter: a report with two wallet-evidence entries renders both lines plus the
  CUSUM line (golden substring assertions); **the no-regression proof:** a report
  dict without the new keys produces output equal to the STEP-0 golden literal —
  never a golden obtained by calling the new code twice.

**Invariants:** existing CLI subcommands, flags, and output unchanged.
`json_formatter.py`, `sink.py`, `anomaly_report.py` untouched. Formatter output
for reports lacking the new detail keys is byte-identical to pre-change output.

**Guardrails (mandatory):**
- Run `uv run pytest tests/ -q` and make it pass before finishing.
- Run `uv run ruff check src/ tests/` — zero new errors. The 8 pre-existing
  ones (6 in `src/prediction_market/simulation/`, 2 F401 in
  `tests/unit/test_particle_filter.py`) are known and out of scope.
- Editing the CLI without invoking it does not count as done — CliRunner tests are
  mandatory.
- All DB timestamps TEXT `"%Y-%m-%d %H:%M:%S"` UTC, lexicographic.
- If the codebase contradicts anything stated above, stop and report instead of
  improvising.
- Do not commit or push unless explicitly asked.

---

## Prompt 10: The Phase-2 flagship — seeded insider case vs benign control

**Goal:** The permanent test that enforces this phase's thesis:
`tests/backtest/test_replay_validation.py` replays two committed fixture cases
through the full stack — a seeded insider case that **must** be flagged with wallet
evidence before its event, and a volatility-matched benign control that **must**
stay silent. Any future change that misses the insider or flags the control breaks
the build. This test is to Phase 2 what `test_pipeline.py` is to Phase 1; mark it
as permanent in its module docstring, and note that Phase-3 detectors must be added
to it.

**Prerequisite check:** `backtest.replay.replay_case`,
`backtest.metrics.evaluate_replay`, and `tests/fixtures/cases/minimal/` must all
exist; `InfoLeakDetector` must attach `details["wallet_evidence"]`. Missing ⇒ stop
and report.

**Affected packages:** `scripts/generate_validation_cases.py` (new),
`tests/fixtures/cases/seeded_insider/` (new, committed),
`tests/fixtures/cases/benign_control/` (new, committed),
`tests/backtest/test_replay_validation.py` (new).

**Details:**
- **Generator script** (`scripts/generate_validation_cases.py`): argparse with
  `--seed` (default 42) and `--output tests/fixtures/cases`; builds both cases via
  `backtest.case_format.save_case` and a seeded `random.Random`. It exists so the
  fixtures are regenerable and reviewable — but **the committed fixtures are the
  source of truth; the test reads them from disk and never regenerates** (a test
  that generates its own inputs can drift in lockstep with a bug).
  **Seed selection is an authorized fixture-construction knob.** A random walk at
  an unlucky seed can spuriously cross the amplified threshold on the benign
  control — that is bad luck, not detector failure. The generator must therefore:
  replay-and-evaluate both candidate cases itself for seeds 42, 43, … 61 (in
  order, max 20), select the **first** seed where the insider case is detected
  with margin (its peak `combined` ≥ 1.25 × `combined_score_min`) *and* the
  benign control emits zero reports with margin (its peak `combined` ≤ 0.8 ×
  `combined_score_min`), and record the chosen seed and both peak-`combined`
  margins in each case's manifest `notes`. Committed fixtures are the generator's
  output at that recorded seed, untouched.
- **Shared spine (both cases):** 10 days of hourly snapshots (240 rows), start
  price 0.15, logit-space Gaussian walk with per-step σ = 0.05, volume growing
  ~5,000 USD per step with ±20% seeded jitter; 30 background wallets
  (`0xbg00`–`0xbg29`) trading small, two-sided, spread uniformly. One scheduled
  event at step 216 (`event_time`). **Know what the event amplifier actually is:**
  `_process_market` amplifies on pure date proximity — *any* event within
  ±`event_proximity_hours` of the snapshot fires it
  (info_leak_detector.py:209-222; `if events:` — there is **no** keyword matching
  against the market question in the amplifier path; keywords feed only the news
  dampener, which replay stubs out). So the event's content is cosmetic; its
  `event_date` is the load-bearing field, and the amplifier is *indiscriminately*
  active for every snapshot within ±24h of step 216 — in both cases. That is
  exactly what makes the benign control falsifiable: it must stay silent *with*
  the amplifier live.
  **Snapshot timestamps must be exact-hour-aligned** (`YYYY-MM-DD HH:00:00`): the
  60-minute alert cooldown uses a strict `<` (info_leak_detector.py:248-251), so
  exact-hour cadence does not suppress consecutive in-window alerts — any
  sub-hour jitter in generated timestamps would.
- **Benign control:** the spine, unmodified. `[label]` present with the same
  window as the insider case (window_start = step 192, window_end/event_time =
  step 216) — labeled precisely so `evaluate_replay` scores silence as 0 hits, 0
  false alarms.
- **Seeded insider case:** identical to the spine (same seed, same code path) up
  to step 192 (T−24h), then superimposed structure:
  - Price: 24 steps of +0.09 logit-units drift added to the walk (≈ +1.8σ per
    step sustained — engineered to trip CUSUM decisively and push the combined
    z-composite past `combined_score_min=4.0` with the event amplifier).
  - Volume: window per-step increments ×3 (volume z-spike).
  - Trades: wallet `0xinsider01`, first-ever trade at step 190 (fresh under the
    48h default), 12 BUY trades totaling ≥ 60% of the window's traded volume,
    one-sided.
  - `[label]`: window_start = step 192's timestamp, window_end = event_time =
    step 216's timestamp, notes citing the construction.
- **The flagship tests** (names are the contract, stated in the series header):
  - `test_seeded_insider_case_flagged`: replay + evaluate ⇒ `detected is True`;
    `lead_time_minutes > 0`; at least one qualifying report's
    `details["wallet_evidence"]` has `0xinsider01` ranked first with
    `score >= 0.6` and `is_fresh` true; `details["cusum"]` is **truthy** (an
    alarm dict, not the always-present-but-None default) on at least one report
    in the window.
  - `test_benign_case_not_flagged`: replay + evaluate ⇒ `total_reports == 0`.
    Not "no high-severity reports" — **zero reports**: the control shares the
    event calendar, so this asserts the detector doesn't alert on calendar
    proximity alone.
  - `test_validation_cases_are_committed`: both case dirs `load_case`
    successfully and their manifests carry `[label]` — a tripwire so a deleted or
    regenerated-and-broken fixture fails loudly rather than skipping.
- **If no seed in the 42–61 sweep satisfies both margins, and adjusting the
  *fixture construction* (drift size, volume multiplier, wallet share) within
  reason doesn't either — i.e. the detector genuinely cannot separate these cases
  at default thresholds — STOP. Do not change any threshold in
  `config/default.toml`, any detector constant, or any scoring formula to make
  the test pass: that is silent recalibration of the live system to satisfy a
  synthetic. Report the separation failure with the per-seed observed peak
  scores; that report is the deliverable in that branch.**
- Scope creep, prohibited by name: no additional cases beyond these two (real
  cases arrive via the live validation runbook); no threshold tuning; no changes
  under `src/` at all — if `src/` needs touching, a prerequisite is broken; stop
  and report.

**Testing:** this prompt *is* tests. Additional honesty checks on the fixtures
themselves, inside the same module: insider and benign snapshots are identical
before step 192 (row-wise equality — proves the control is matched, not merely
similar); the insider window's logit-return mean exceeds the benign window's
(proves the structure was actually injected).

**Invariants:** zero diffs under `src/` and zero diffs to existing tests or
fixtures. `uv run pytest tests/ -q` total runtime stays under ~60s — 2 × 240-step
replays are cheap against a tmp SQLite; if runtime explodes, something is wrong
(report it, don't mark the test slow).

**Guardrails (mandatory):**
- Run `uv run pytest tests/ -q` and make it pass before finishing.
- Run `uv run ruff check src/ tests/` — zero new errors. The 8 pre-existing
  ones (6 in `src/prediction_market/simulation/`, 2 F401 in
  `tests/unit/test_particle_filter.py`) are known and out of scope.
- The committed fixtures must be the generator's actual output at the default
  seed: regenerate once after the final generator edit and commit that output;
  hand-editing fixture JSON after generation is prohibited.
- If the codebase contradicts anything stated above, stop and report instead of
  improvising.
- Do not commit or push unless explicitly asked.

---

## Prompt 11: Live validation runbook — the Maduro-capture case

**Goal:** Run the finished Phase-2 stack against reality: archive the Polymarket
market behind the DOJ-charged Van Dyke insider case (account created Dec 26 2025;
~13 bets ≈ $33K placed Dec 27–Jan 2; ~$400K profit on the Jan 3 2026 Maduro
capture — see docs/RESEARCH-BRIEF.md §3, DOJ press release linked there), replay
it, and record what the detectors saw in a new `docs/CASES.md`. This is the first
real use of the system and the ground-truth check on everything above.

**This prompt is unlike the others — read before starting:**
- It **requires live network access** to Polymarket's public APIs and is run with
  the human present. It is surveillance research on public market data about a
  publicly documented, criminally charged case — analysis, not trading.
- Its deliverable is an **honest report**, whatever the outcome. "The data is no
  longer retrievable" and "the detector missed it" are both valid, valuable
  results. Fabricating, interpolating, or hand-editing archived data to make the
  replay work is the one unforgivable failure mode.
- Polymarket's CLOB had a V2 hard cutover on 2026-04-28 (docs/RESEARCH-BRIEF.md
  §3); this market resolved before it. Expect endpoints to possibly return
  nothing for it, and treat every empty response as a finding to record, not an
  obstacle to code around.

**Prerequisite check:** `prediction-market archive-case`, `backtest`, `cases`, and
`wallets` CLI commands must all exist and `uv run pytest tests/ -q` must be green
before starting. Missing/red ⇒ Phase 2 is not done — stop and report.

**Affected artifacts:** `data/cases/<slug>/` (gitignored archive),
`docs/CASES.md` (new, committed), the case's `case.toml` `[label]` (edited by
hand), and — only if the live APIs contradict the client's parameter expectations —
a reported (not silently patched) list of mismatches.

**Steps:**
1. **Resolve the market.** The exact slug is not recorded in this repo and must be
   discovered live. Try, in order: (a) the Polymarket website's search for
   Venezuela/Maduro markets resolving early January 2026 (the human can paste the
   URL slug); (b) `GammaClient.search_markets("<candidate-slug>")` for each
   candidate; (c) a scratch (uncommitted, scratchpad-dir) script paginating
   `GammaClient.get_all_markets(active=False, closed=True)` and grepping
   `question` for "Maduro" — and if that yields nothing, retry with
   `active=True, closed=True` (some APIs never flip `active` on resolution; the
   client's `active`/`closed` bools are independent params,
   gamma_client.py:80-101, and omitting `active` entirely is not expressible
   through this client). Note `get_all_markets` caps at `max_pages=50` ×100
   markets; raise it via its parameter if needed. Confirm the resolved market's
   question and end date with the human before archiving. If the market cannot be
   found at all, record that in docs/CASES.md and switch to the fallback case
   (step 6).
2. **Archive.** `uv run prediction-market archive-case <slug> --output data/cases
   --max-trade-pages 500`. Then `uv run prediction-market cases --dir data/cases`
   and sanity-read the counts. Record in docs/CASES.md exactly which of
   {market, price history, trades} came back non-empty, with counts — this
   doubles as the live verification of the Data-API parameter names used by the
   wallet client methods; report any 4xx/param mismatch for a follow-up client
   fix rather than patching ad hoc.
3. **Label.** Edit the case's `case.toml` `[label]` by hand from the public
   record: `window_start = "2025-12-27 00:00:00"`, `event_time = "2026-01-03
   00:00:00"` (refine the capture announcement hour from the DOJ release if it
   states one), `window_end` = event_time, notes citing the DOJ press release
   URL from docs/RESEARCH-BRIEF.md.
4. **Replay and evaluate.** `uv run prediction-market backtest --case
   data/cases/<slug> --null-runs 20 --json | tee` into the session log. Also run
   `uv run prediction-market wallets <market_id> --hours 200` against the
   archived DB window if trades were retrievable, and compare the top wallet's
   pattern (fresh, one-sided, ~13 trades, ~$33K) against the DOJ description —
   wallet identity is not in the public record; pattern match is the claim, and
   docs/CASES.md must phrase it as a pattern match, never as identification of a
   person.
5. **Write `docs/CASES.md`** (committed): a section per attempted case — data
   retrieved (counts + gaps), label provenance (DOJ citation), evaluation output
   (detected, lead time, hits, false alarms, FP rate on nulls), wallet-evidence
   summary, and an honest verdict line: flagged with lead time / flagged late /
   missed / not evaluable, with one paragraph on why. Close with the follow-ups
   the result implies (e.g. "trade tape truncated at N pages — Phase 3 on-chain
   reconstruction needed").
6. **Fallback if the Maduro data is gone:** repeat steps 1–5 for the Iran
   military-action markets (Feb–Jun 2026 — post-V2-cutover, so retrievability is
   more likely; Bubblemaps/60 Minutes documented 9 connected accounts, 98% win
   rate; Mitts & Ofir documented one wallet trading 71 minutes pre-announcement —
   label from those public timestamps, cited in docs/RESEARCH-BRIEF.md §3).
   Archive both cases if both are reachable — more ground truth is strictly
   better.

**Guardrails (mandatory):**
- Never fabricate, interpolate, or hand-edit archived market data. The `[label]`
  block in `case.toml` and `docs/CASES.md` are the only hand-written artifacts.
- Every empty or failed API response is recorded in docs/CASES.md — silence about
  a gap is a falsified result.
- `uv run pytest tests/ -q` must still be green at the end (this session should
  add no `src/` changes; if a live-API mismatch forces a client fix, that is a
  separate reviewed change — report it first).
- docs/CASES.md states findings about wallets as **pattern observations on public
  trading data**, never as identification of, or accusation against, a person.
- Do not commit or push unless explicitly asked; `data/cases/` stays gitignored.
