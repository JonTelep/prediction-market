"""SQLite database connection, migrations, and WAL mode setup."""

from __future__ import annotations

import logging
from pathlib import Path

import aiosqlite

from prediction_market.config import AppConfig

logger = logging.getLogger(__name__)

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS markets (
    id TEXT PRIMARY KEY,
    question TEXT NOT NULL,
    description TEXT DEFAULT '',
    category TEXT DEFAULT '',
    tags TEXT DEFAULT '[]',  -- JSON array
    slug TEXT DEFAULT '',
    condition_id TEXT DEFAULT '',
    clob_token_ids TEXT DEFAULT '[]',  -- JSON array
    volume REAL DEFAULT 0,
    liquidity REAL DEFAULT 0,
    active INTEGER DEFAULT 1,
    closed INTEGER DEFAULT 0,
    political_confidence REAL DEFAULT 0,
    political_reasons TEXT DEFAULT '[]',  -- JSON array
    created_at TEXT DEFAULT '',
    end_date TEXT DEFAULT '',
    first_seen TEXT DEFAULT (datetime('now')),
    last_updated TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    market_id TEXT NOT NULL REFERENCES markets(id),
    timestamp TEXT NOT NULL DEFAULT (datetime('now')),
    price_yes REAL,
    price_no REAL,
    volume_24hr REAL,
    volume_total REAL,
    liquidity REAL,
    num_trades_1hr INTEGER DEFAULT 0,
    UNIQUE(market_id, timestamp)
);

CREATE TABLE IF NOT EXISTS orderbook_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    market_id TEXT NOT NULL REFERENCES markets(id),
    token_id TEXT NOT NULL,
    timestamp TEXT NOT NULL DEFAULT (datetime('now')),
    best_bid REAL,
    best_ask REAL,
    midpoint REAL,
    spread REAL,
    spread_pct REAL,
    total_bid_depth REAL,
    total_ask_depth REAL,
    depth_1pct REAL,
    depth_5pct REAL,
    depth_10pct REAL,
    imbalance REAL,
    hhi REAL,
    susceptibility_score REAL,
    UNIQUE(market_id, token_id, timestamp)
);

CREATE TABLE IF NOT EXISTS trades (
    id TEXT PRIMARY KEY,
    market_id TEXT NOT NULL REFERENCES markets(id),
    asset_id TEXT DEFAULT '',
    side TEXT DEFAULT '',
    size REAL DEFAULT 0,
    price REAL DEFAULT 0,
    volume_usd REAL DEFAULT 0,
    outcome TEXT DEFAULT '',
    owner TEXT DEFAULT '',
    match_time TEXT DEFAULT '',
    transaction_hash TEXT DEFAULT '',
    inserted_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS scheduled_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,  -- 'congress', 'court', 'whitehouse'
    event_type TEXT DEFAULT '',
    title TEXT NOT NULL,
    description TEXT DEFAULT '',
    event_date TEXT NOT NULL,
    url TEXT DEFAULT '',
    keywords TEXT DEFAULT '[]',  -- JSON array for matching to markets
    fetched_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS anomaly_reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent TEXT NOT NULL,  -- 'info_leak' or 'manipulation'
    market_id TEXT NOT NULL REFERENCES markets(id),
    severity TEXT NOT NULL DEFAULT 'medium',  -- 'low', 'medium', 'high', 'critical'
    anomaly_score REAL DEFAULT 0,
    confidence REAL DEFAULT 0,
    summary TEXT NOT NULL,
    details TEXT NOT NULL DEFAULT '{}',  -- Full JSON payload
    price_evidence TEXT DEFAULT '{}',
    volume_evidence TEXT DEFAULT '{}',
    calendar_matches TEXT DEFAULT '[]',
    news_check TEXT DEFAULT '{}',
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS rolling_stats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    market_id TEXT NOT NULL REFERENCES markets(id),
    stat_type TEXT NOT NULL,  -- 'price', 'volume', 'depth'
    window_days INTEGER NOT NULL DEFAULT 7,
    serialized_data TEXT NOT NULL DEFAULT '{}',  -- JSON blob
    updated_at TEXT DEFAULT (datetime('now')),
    UNIQUE(market_id, stat_type, window_days)
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_snapshots_market_time ON snapshots(market_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_orderbook_market_time ON orderbook_snapshots(market_id, token_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_trades_market_time ON trades(market_id, match_time);
CREATE INDEX IF NOT EXISTS idx_trades_owner ON trades(owner);
CREATE INDEX IF NOT EXISTS idx_events_date ON scheduled_events(event_date);
CREATE INDEX IF NOT EXISTS idx_reports_market ON anomaly_reports(market_id, created_at);
CREATE INDEX IF NOT EXISTS idx_reports_severity ON anomaly_reports(severity, created_at);
CREATE INDEX IF NOT EXISTS idx_reports_agent ON anomaly_reports(agent, created_at);
"""


async def init_database(config: AppConfig) -> aiosqlite.Connection:
    """Initialize the database with WAL mode and full schema."""
    db_path = Path(config.database.path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    db = await aiosqlite.connect(str(db_path))

    # Enable WAL mode for concurrent reads
    await db.execute("PRAGMA journal_mode=WAL")
    await db.execute("PRAGMA synchronous=NORMAL")
    await db.execute("PRAGMA foreign_keys=ON")

    # Create all tables
    await db.executescript(SCHEMA_SQL)
    await db.commit()

    logger.info("Database initialized at %s", db_path)
    return db


async def get_database(config: AppConfig) -> aiosqlite.Connection:
    """Get a database connection (creates DB if needed)."""
    return await init_database(config)
