"""Integration tests for database operations."""

import json

import pytest
import pytest_asyncio

from prediction_market.config import AppConfig, load_config
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
