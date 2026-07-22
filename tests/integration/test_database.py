"""Integration tests for database operations."""

import json

import aiosqlite
import pytest
import pytest_asyncio

from prediction_market.config import load_config
from prediction_market.store.database import init_database


@pytest.fixture
def db_config(tmp_path):
    config = load_config()
    config.database.path = str(tmp_path / "test.db")
    return config


@pytest_asyncio.fixture
async def db(db_config):
    conn = await init_database(db_config)
    yield conn
    await conn.close()


@pytest.mark.asyncio
async def test_database_creates_tables(db):
    cursor = await db.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    )
    tables = {row[0] for row in await cursor.fetchall()}
    expected = {
        "markets",
        "snapshots",
        "orderbook_snapshots",
        "trades",
        "scheduled_events",
        "anomaly_reports",
        "rolling_stats",
    }
    assert expected.issubset(tables)


@pytest.mark.asyncio
async def test_wal_mode(db):
    cursor = await db.execute("PRAGMA journal_mode")
    row = await cursor.fetchone()
    assert row[0] == "wal"


@pytest.mark.asyncio
async def test_insert_market(db):
    await db.execute(
        """INSERT INTO markets (id, question, category, volume, political_confidence)
           VALUES (?, ?, ?, ?, ?)""",
        ("m1", "Test market?", "politics", 100000, 0.8),
    )
    await db.commit()

    cursor = await db.execute("SELECT id, question, volume FROM markets WHERE id = ?", ("m1",))
    row = await cursor.fetchone()
    assert row[0] == "m1"
    assert row[1] == "Test market?"
    assert row[2] == 100000


@pytest.mark.asyncio
async def test_insert_snapshot(db):
    await db.execute(
        "INSERT INTO markets (id, question) VALUES (?, ?)",
        ("m1", "Test?"),
    )
    await db.execute(
        """INSERT INTO snapshots (market_id, price_yes, price_no, volume_24hr)
           VALUES (?, ?, ?, ?)""",
        ("m1", 0.65, 0.35, 25000),
    )
    await db.commit()

    cursor = await db.execute(
        "SELECT market_id, price_yes, volume_24hr FROM snapshots WHERE market_id = ?",
        ("m1",),
    )
    row = await cursor.fetchone()
    assert row[0] == "m1"
    assert row[1] == 0.65
    assert row[2] == 25000


@pytest.mark.asyncio
async def test_insert_anomaly_report(db):
    await db.execute(
        "INSERT INTO markets (id, question) VALUES (?, ?)",
        ("m1", "Test?"),
    )
    await db.execute(
        """INSERT INTO anomaly_reports (agent, market_id, severity, anomaly_score, confidence, summary, details)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        ("info_leak", "m1", "high", 5.2, 0.85, "Unusual price spike", json.dumps({"price_z": 3.1})),
    )
    await db.commit()

    cursor = await db.execute(
        "SELECT agent, severity, anomaly_score FROM anomaly_reports WHERE market_id = ?",
        ("m1",),
    )
    row = await cursor.fetchone()
    assert row[0] == "info_leak"
    assert row[1] == "high"
    assert row[2] == 5.2


@pytest.mark.asyncio
async def test_insert_rolling_stats(db):
    await db.execute(
        "INSERT INTO markets (id, question) VALUES (?, ?)",
        ("m1", "Test?"),
    )
    data = json.dumps({"mean": 0.65, "std": 0.02, "count": 168})
    await db.execute(
        """INSERT OR REPLACE INTO rolling_stats (market_id, stat_type, window_days, serialized_data)
           VALUES (?, ?, ?, ?)""",
        ("m1", "price", 7, data),
    )
    await db.commit()

    cursor = await db.execute(
        "SELECT serialized_data FROM rolling_stats WHERE market_id = ? AND stat_type = ?",
        ("m1", "price"),
    )
    row = await cursor.fetchone()
    parsed = json.loads(row[0])
    assert parsed["mean"] == 0.65


# Pre-change trades table definition, without proxy_wallet, for the
# _ensure_columns upgrade proof below.
_LEGACY_TRADES_TABLE_SQL = """
CREATE TABLE trades (
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
"""


@pytest.mark.asyncio
async def test_init_database_adds_proxy_wallet_to_legacy_db(db_config, tmp_path):
    # Simulate a database file created before the proxy_wallet column existed.
    legacy_conn = await aiosqlite.connect(db_config.database.path)
    await legacy_conn.execute(
        "CREATE TABLE markets (id TEXT PRIMARY KEY, question TEXT NOT NULL)"
    )
    await legacy_conn.execute(_LEGACY_TRADES_TABLE_SQL)
    await legacy_conn.commit()
    await legacy_conn.close()

    # init_database should upgrade the legacy schema in place, without error.
    db = await init_database(db_config)
    cursor = await db.execute("PRAGMA table_info(trades)")
    columns = {row[1] for row in await cursor.fetchall()}
    assert "proxy_wallet" in columns

    # A second init_database call against the now-upgraded DB must not
    # raise (a naive double ALTER TABLE would fail with "duplicate column").
    db2 = await init_database(db_config)
    cursor = await db2.execute("PRAGMA table_info(trades)")
    columns2 = {row[1] for row in await cursor.fetchall()}
    assert "proxy_wallet" in columns2

    await db.close()
    await db2.close()
