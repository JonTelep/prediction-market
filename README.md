# Prediction Market Surveillance

Automated surveillance system for Polymarket political prediction markets. Detects anomalous trading activity -- potential information leakage and market manipulation -- by cross-referencing price/volume movements with legislative calendars, court schedules, and public news feeds.

## Architecture

```
                    ┌──────────────────────────────────────────────────┐
                    │                 Orchestrator                     │
                    │  market discovery · snapshot loops · agent mgmt  │
                    └────────┬──────────────────┬──────────────┬──────┘
                             │                  │              │
              ┌──────────────▼───┐   ┌──────────▼──────┐  ┌───▼──────────────┐
              │   Data Clients   │   │     Agents       │  │    Reporting     │
              │ Gamma · CLOB     │   │ InfoLeakDetector │  │ JSON · Markdown  │
              │ Data  · WS feeds │   │ ManipulationGuard│  │ File · Webhook   │
              └────────┬─────────┘   └──────────┬───────┘  └───┬─────────────┘
                       │                        │              │
              ┌────────▼────────────────────────▼──────────────▼──────┐
              │                    SQLite (WAL)                       │
              │  markets · snapshots · orderbooks · trades · reports  │
              └──────────────────────────────────────────────────────-┘
                       │
              ┌────────▼─────────────────────┐
              │     External Sources          │
              │ Congress · Courts · WhiteHouse│
              │ GDELT · NewsAPI               │
              └──────────────────────────────-┘
```

## Quick Start

```bash
# Install dependencies
uv sync --dev

# Copy and fill in API keys
cp .env.example .env
# Edit .env with your keys (CONGRESS_API_KEY, NEWSAPI_KEY, etc.)

# Run a one-shot scan of political markets
uv run prediction-market scan

# Backfill 30 days of historical data
uv run prediction-market backfill --days 30

# Start continuous monitoring
uv run prediction-market monitor
```

## CLI Reference

```bash
# Continuous monitoring (both agents by default)
prediction-market monitor
prediction-market monitor --agent info-leak
prediction-market monitor --agent manipulation

# One-shot market scan — lists tracked political markets
prediction-market scan

# Backfill historical price data
prediction-market backfill --days 30

# List tracked markets
prediction-market markets

# Query anomaly reports
prediction-market reports
prediction-market reports --severity high

# View a single report by ID
prediction-market report <id>
```

## Container Usage

Build and run with Podman (or Docker):

```bash
# Build the image
make build

# One-shot scan
make scan

# Continuous monitoring
make monitor

# Backfill historical data (default 30 days)
make backfill
make backfill DAYS=90

# Debug shell inside the container
make shell
```

The container mounts `./data/` for SQLite persistence and report output.

## Makefile Targets

| Target     | Description                                  |
|------------|----------------------------------------------|
| `build`    | Build container image                        |
| `run`      | Run container with custom `CMD=`             |
| `monitor`  | Start continuous monitoring in container     |
| `scan`     | One-shot market scan in container            |
| `backfill` | Backfill historical data (`DAYS=30`)         |
| `test`     | Run test suite (`uv run pytest`)             |
| `lint`     | Lint with ruff                               |
| `fmt`      | Format with ruff                             |
| `clean`    | Remove caches and build artifacts            |
| `shell`    | Interactive shell in container               |

## Configuration

Configuration is loaded in priority order: environment variables > custom TOML > `config/default.toml`.

See [`.env.example`](.env.example) for all available environment variables. Key settings:

- `CONGRESS_API_KEY` — Free key from api.congress.gov
- `NEWSAPI_KEY` — Free key from newsapi.org (100 req/day)
- `COURTLISTENER_TOKEN` — Free token from courtlistener.com
- `DATABASE_PATH` — SQLite database path (default: `data/prediction_market.db`)
- `WEBHOOK_URL` — Optional Slack/Discord webhook for report delivery

Thresholds, polling intervals, and rate limits are configured in `config/default.toml`.

## Project Structure

```
prediction-market/
├── config/
│   ├── default.toml              # Default configuration
│   └── political_keywords.toml   # Political classification rules
├── src/prediction_market/
│   ├── agents/
│   │   ├── base.py               # BaseAgent async tick loop
│   │   ├── info_leak_detector.py # Information leak detection (60s)
│   │   └── manipulation_guard.py # Manipulation detection (300s)
│   ├── analysis/
│   │   ├── correlation.py        # Cross-market correlation detector
│   │   ├── liquidity_analyzer.py # HHI, depth, susceptibility scoring
│   │   ├── price_analyzer.py     # Z-score price anomaly detection
│   │   ├── timeseries.py         # RollingStats, EWMA primitives
│   │   └── volume_analyzer.py    # Volume spike detection
│   ├── data/
│   │   ├── external/             # Congress, courts, news integrations
│   │   ├── political_filter.py   # Market political classification
│   │   └── polymarket/           # Gamma, CLOB, Data API clients + WS
│   ├── reporting/
│   │   ├── anomaly_report.py     # Core AnomalyReport dataclass
│   │   ├── human_formatter.py    # Markdown output for analysts
│   │   ├── json_formatter.py     # Structured JSON output
│   │   └── sink.py               # File, stdout, webhook, composite sinks
│   ├── store/
│   │   ├── database.py           # SQLite init, schema, WAL mode
│   │   ├── queries.py            # Pre-built analytical queries
│   │   └── snapshots.py          # Periodic data writers
│   ├── cli.py                    # Click CLI entrypoint
│   ├── config.py                 # TOML + env var config loading
│   └── orchestrator.py           # Async coordinator + lifecycle
├── tests/
│   ├── fixtures/                 # JSON test data
│   ├── integration/              # DB, orchestrator, reporting tests
│   └── unit/                     # Pure logic tests
├── Containerfile                 # Multi-stage Podman/Docker build
├── Makefile                      # Dev/ops workflow targets
└── pyproject.toml                # Project metadata + dependencies
```
