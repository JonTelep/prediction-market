"""Integration tests for pre-built analytical queries."""


import pytest
import pytest_asyncio

from prediction_market.config import load_config
from prediction_market.data.external.models import ScheduledEvent
from prediction_market.store.database import init_database
from prediction_market.store.queries import (
    get_active_political_markets,
    get_anomaly_reports,
    get_latest_orderbook_snapshot,
    get_latest_snapshot,
    get_market_trades,
    get_price_history,
    get_recent_orderbooks,
    get_recent_snapshots,
    get_rolling_stats,
    get_scheduled_events_in_range,
    get_volume_history,
    save_anomaly_report,
    save_rolling_stats,
    save_scheduled_events,
)


@pytest.fixture
def db_config(tmp_path):
    config = load_config()
    config.database.path = str(tmp_path / "test.db")
    return config


@pytest_asyncio.fixture
async def db(db_config):
    conn = await init_database(db_config)
    # Seed with a market
    await conn.execute(
        "INSERT INTO markets (id, question, category, volume) VALUES (?, ?, ?, ?)",
        ("q-market-1", "Will X happen?", "politics", 100000),
    )
    await conn.commit()
    yield conn
    await conn.close()


async def _insert_snapshots(db, market_id, count):
    """Insert `count` snapshots with recent timestamps."""
    from datetime import datetime, timedelta, timezone

    now = datetime.now(timezone.utc)
    for i in range(count):
        ts = (now - timedelta(hours=count - i)).strftime("%Y-%m-%d %H:%M:%S")
        await db.execute(
            "INSERT INTO snapshots (market_id, timestamp, price_yes, price_no, volume_24hr, volume_total, liquidity) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (market_id, ts, 0.60 + i * 0.01, 0.40 - i * 0.01, 10000 + i * 1000, 500000, 100000),
        )
    await db.commit()


async def _insert_orderbooks(db, market_id, token_id, count):
    from datetime import datetime, timedelta, timezone

    now = datetime.now(timezone.utc)
    for i in range(count):
        ts = (now - timedelta(hours=count - i)).strftime("%Y-%m-%d %H:%M:%S")
        await db.execute(
            "INSERT INTO orderbook_snapshots "
            "(market_id, token_id, timestamp, best_bid, best_ask, midpoint, spread, spread_pct, "
            "total_bid_depth, total_ask_depth, depth_1pct, depth_5pct, depth_10pct, imbalance, hhi, susceptibility_score) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (market_id, token_id, ts, 0.64, 0.66, 0.65, 0.02, 0.03,
             5000, 4000, 1000, 3000, 5000, 0.1, 2500, 0.5),
        )
    await db.commit()


async def _insert_trades(db, market_id, count):
    from datetime import datetime, timedelta, timezone

    now = datetime.now(timezone.utc)
    for i in range(count):
        ts = (now - timedelta(hours=count - i)).strftime("%Y-%m-%d %H:%M:%S")
        await db.execute(
            "INSERT INTO trades (id, market_id, asset_id, side, size, price, volume_usd, outcome, owner, match_time) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (f"qt-{i}", market_id, "tok1", "BUY", 100, 0.65, 65, "Yes", f"0xo{i}", ts),
        )
    await db.commit()


@pytest.mark.asyncio
async def test_get_recent_snapshots(db):
    await _insert_snapshots(db, "q-market-1", 5)
    rows = await get_recent_snapshots(db, "q-market-1", hours=48)
    assert len(rows) == 5
    assert "price_yes" in rows[0]
    # Verify ascending order
    assert rows[0]["price_yes"] <= rows[-1]["price_yes"]


@pytest.mark.asyncio
async def test_get_recent_snapshots_empty(db):
    rows = await get_recent_snapshots(db, "nonexistent", hours=24)
    assert rows == []


@pytest.mark.asyncio
async def test_get_price_history(db):
    await _insert_snapshots(db, "q-market-1", 3)
    rows = await get_price_history(db, "q-market-1", hours=48)
    assert len(rows) == 3
    assert set(rows[0].keys()) == {"timestamp", "price_yes", "price_no"}


@pytest.mark.asyncio
async def test_get_volume_history(db):
    await _insert_snapshots(db, "q-market-1", 3)
    rows = await get_volume_history(db, "q-market-1", hours=48)
    assert len(rows) == 3
    assert "volume_24hr" in rows[0]
    assert "liquidity" in rows[0]


@pytest.mark.asyncio
async def test_get_recent_orderbooks(db):
    await _insert_orderbooks(db, "q-market-1", "tok1", 4)
    rows = await get_recent_orderbooks(db, "q-market-1", "tok1", hours=48)
    assert len(rows) == 4
    assert rows[0]["best_bid"] == 0.64


@pytest.mark.asyncio
async def test_get_market_trades(db):
    await _insert_trades(db, "q-market-1", 6)
    rows = await get_market_trades(db, "q-market-1", hours=48)
    assert len(rows) == 6
    assert rows[0]["side"] == "BUY"


@pytest.mark.asyncio
async def test_save_and_get_anomaly_reports(db):
    report_dict = {
        "agent": "info_leak",
        "market_id": "q-market-1",
        "severity": "high",
        "anomaly_score": 5.0,
        "confidence": 0.85,
        "summary": "Unusual spike detected",
        "details": {"z_score": 3.2},
        "price_evidence": {"before": 0.55},
        "volume_evidence": {},
        "calendar_matches": [],
        "news_check": {},
    }
    row_id = await save_anomaly_report(db, report_dict)
    assert row_id > 0

    reports = await get_anomaly_reports(db, severity="high")
    assert len(reports) >= 1
    assert reports[0]["severity"] == "high"
    assert reports[0]["agent"] == "info_leak"


@pytest.mark.asyncio
async def test_get_anomaly_reports_filters(db):
    for agent, sev in [("info_leak", "high"), ("manipulation", "medium"), ("info_leak", "low")]:
        await save_anomaly_report(db, {
            "agent": agent, "market_id": "q-market-1",
            "severity": sev, "summary": f"Test {agent} {sev}",
        })

    all_reports = await get_anomaly_reports(db)
    assert len(all_reports) == 3

    high_only = await get_anomaly_reports(db, severity="high")
    assert all(r["severity"] == "high" for r in high_only)

    manip_only = await get_anomaly_reports(db, agent="manipulation")
    assert all(r["agent"] == "manipulation" for r in manip_only)


@pytest.mark.asyncio
async def test_save_and_get_rolling_stats(db):
    data = {"mean": 0.65, "std": 0.02, "count": 168}
    await save_rolling_stats(db, "q-market-1", "price", 7, data)

    result = await get_rolling_stats(db, "q-market-1", "price")
    assert result is not None
    assert result["serialized_data"]["mean"] == 0.65
    assert result["window_days"] == 7


@pytest.mark.asyncio
async def test_rolling_stats_upsert(db):
    await save_rolling_stats(db, "q-market-1", "price", 7, {"mean": 0.60})
    await save_rolling_stats(db, "q-market-1", "price", 7, {"mean": 0.65})

    result = await get_rolling_stats(db, "q-market-1", "price")
    assert result["serialized_data"]["mean"] == 0.65


@pytest.mark.asyncio
async def test_get_rolling_stats_not_found(db):
    result = await get_rolling_stats(db, "nonexistent", "price")
    assert result is None


@pytest.mark.asyncio
async def test_get_latest_snapshot_returns_newest(db):
    from datetime import datetime, timedelta, timezone

    now = datetime.now(timezone.utc)
    for i, price in enumerate([0.50, 0.55, 0.60]):
        ts = (now - timedelta(hours=3 - i)).strftime("%Y-%m-%d %H:%M:%S")
        await db.execute(
            "INSERT INTO snapshots (market_id, timestamp, price_yes, price_no) "
            "VALUES (?, ?, ?, ?)",
            ("q-market-1", ts, price, 1 - price),
        )
    await db.commit()

    result = await get_latest_snapshot(db, "q-market-1")
    assert result is not None
    assert result["price_yes"] == 0.60


@pytest.mark.asyncio
async def test_get_latest_snapshot_unknown_market(db):
    result = await get_latest_snapshot(db, "nonexistent")
    assert result is None


@pytest.mark.asyncio
async def test_get_latest_orderbook_snapshot_returns_newest(db):
    from datetime import datetime, timedelta, timezone

    now = datetime.now(timezone.utc)
    for i, bid in enumerate([0.60, 0.62, 0.64]):
        ts = (now - timedelta(hours=3 - i)).strftime("%Y-%m-%d %H:%M:%S")
        await db.execute(
            "INSERT INTO orderbook_snapshots (market_id, token_id, timestamp, best_bid) "
            "VALUES (?, ?, ?, ?)",
            ("q-market-1", "tok1", ts, bid),
        )
    await db.commit()

    result = await get_latest_orderbook_snapshot(db, "q-market-1")
    assert result is not None
    assert result["best_bid"] == 0.64


@pytest.mark.asyncio
async def test_get_latest_orderbook_snapshot_unknown_market(db):
    result = await get_latest_orderbook_snapshot(db, "nonexistent")
    assert result is None


@pytest.mark.asyncio
async def test_get_active_political_markets(db):
    await db.execute(
        "INSERT INTO markets (id, question, volume, active, political_confidence, clob_token_ids) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        ("inactive-market", "Inactive?", 500000, 0, 0.9, '["tokA", "tokB"]'),
    )
    await db.execute(
        "INSERT INTO markets (id, question, volume, active, political_confidence, clob_token_ids) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        ("nonpolitical-market", "Nonpolitical?", 900000, 1, 0.0, '["tokC", "tokD"]'),
    )
    await db.execute(
        "INSERT INTO markets (id, question, volume, active, political_confidence, clob_token_ids) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        ("high-volume-political", "High volume political?", 200000, 1, 0.8, '["tokE", "tokF"]'),
    )
    await db.commit()
    # Bump the seeded "q-market-1" market to be political too, with lower volume.
    await db.execute(
        "UPDATE markets SET political_confidence = 0.5, clob_token_ids = ? WHERE id = ?",
        ('["tokG"]', "q-market-1"),
    )
    await db.commit()

    rows = await get_active_political_markets(db)
    ids = [r["id"] for r in rows]

    assert "inactive-market" not in ids
    assert "nonpolitical-market" not in ids
    assert ids == ["high-volume-political", "q-market-1"]
    assert rows[0]["token_ids"] == ["tokE", "tokF"]
    assert isinstance(rows[0]["token_ids"], list)
    assert "question" in rows[0]


@pytest.mark.asyncio
async def test_save_scheduled_events_idempotent(db):
    from datetime import datetime, timezone

    events = [
        ScheduledEvent(
            source="congress",
            event_type="hearing",
            title="Judiciary Committee Hearing",
            description="A hearing",
            event_date=datetime(2026, 8, 1, 14, 0, tzinfo=timezone.utc),
            url="https://example.com/1",
            keywords=["judiciary", "hearing"],
        ),
        ScheduledEvent(
            source="court",
            event_type="ruling",
            title="Supreme Court Ruling",
            description="A ruling",
            event_date=datetime(2026, 8, 2, 10, 0, tzinfo=timezone.utc),
            url="https://example.com/2",
            keywords=["scotus"],
        ),
    ]

    inserted = await save_scheduled_events(db, events)
    assert inserted == 2

    cursor = await db.execute("SELECT COUNT(*) FROM scheduled_events")
    row = await cursor.fetchone()
    assert row[0] == 2

    # Re-inserting the same events should be a no-op due to the unique index.
    inserted_again = await save_scheduled_events(db, events)
    assert inserted_again == 0

    cursor = await db.execute("SELECT COUNT(*) FROM scheduled_events")
    row = await cursor.fetchone()
    assert row[0] == 2


@pytest.mark.asyncio
async def test_get_scheduled_events_in_range(db):
    from datetime import datetime, timezone

    in_range_event = ScheduledEvent(
        source="congress",
        event_type="hearing",
        title="In Range Hearing",
        description="Happens within the query window",
        event_date=datetime(2026, 8, 5, 12, 0, tzinfo=timezone.utc),
        url="https://example.com/in-range",
        keywords=["election", "senate"],
    )
    out_of_range_event = ScheduledEvent(
        source="congress",
        event_type="hearing",
        title="Out Of Range Hearing",
        description="Happens outside the query window",
        event_date=datetime(2026, 9, 15, 12, 0, tzinfo=timezone.utc),
        url="https://example.com/out-of-range",
        keywords=["unrelated"],
    )

    await save_scheduled_events(db, [in_range_event, out_of_range_event])

    start = datetime(2026, 8, 1, 0, 0, tzinfo=timezone.utc)
    end = datetime(2026, 8, 10, 0, 0, tzinfo=timezone.utc)
    rows = await get_scheduled_events_in_range(db, start, end)

    titles = [r["title"] for r in rows]
    assert "In Range Hearing" in titles
    assert "Out Of Range Hearing" not in titles
    assert rows[0]["keywords"] == ["election", "senate"]
    assert isinstance(rows[0]["keywords"], list)
