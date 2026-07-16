"""Tests for the Info-Leak Detector agent."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest
import pytest_asyncio

from prediction_market.agents.info_leak_detector import (
    InfoLeakDetector,
    _extract_keywords,
)
from prediction_market.config import load_config
from prediction_market.data.external.models import NewsCheckResult, ScheduledEvent
from prediction_market.reporting.anomaly_report import AnomalyReport
from prediction_market.store import queries
from prediction_market.store.database import init_database

QUESTION = "Will Trump nominate Judy Shelton as the next Fed chair?"


class StubNewsChecker:
    """Tiny stand-in for NewsChecker — no HTTP, canned NewsCheckResult."""

    def __init__(self, news_found: bool = False) -> None:
        self.news_found = news_found
        self.calls: list[tuple[list[str], datetime, int]] = []

    async def check_news_exists(
        self,
        keywords: list[str],
        before_time: datetime,
        window_hours: int = 2,
    ) -> NewsCheckResult:
        self.calls.append((keywords, before_time, window_hours))
        return NewsCheckResult(
            news_found=self.news_found,
            articles=[],
            earliest_article_time=None,
            query_keywords=list(keywords),
        )


async def _make_db(tmp_path, name: str, market_id: str = "m1"):
    config = load_config()
    config.database.path = str(tmp_path / f"{name}.db")
    conn = await init_database(config)
    await conn.execute(
        "INSERT INTO markets "
        "(id, question, volume, active, political_confidence, clob_token_ids) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (market_id, QUESTION, 500000, 1, 0.8, '["tok-yes", "tok-no"]'),
    )
    await conn.commit()
    return config, conn


@pytest.fixture
def db_config(tmp_path):
    config = load_config()
    config.database.path = str(tmp_path / "test.db")
    return config


@pytest_asyncio.fixture
async def db(db_config):
    conn = await init_database(db_config)
    await conn.execute(
        "INSERT INTO markets "
        "(id, question, volume, active, political_confidence, clob_token_ids) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        ("m1", QUESTION, 500000, 1, 0.8, '["tok-yes", "tok-no"]'),
    )
    await conn.commit()
    yield conn
    await conn.close()


async def _insert_snapshot(
    db,
    market_id: str,
    ts: datetime,
    price_yes: float,
    volume_total: float = 100000.0,
) -> None:
    """Insert a snapshot row with a fully controlled timestamp.

    store.snapshots.save_price_snapshot always stamps "now" and cannot be
    given an explicit timestamp, which these tests need (warm-up sequencing,
    no-double-processing, cooldown windows). Direct insert mirrors the
    pattern used in tests/integration/test_queries.py.
    """
    ts_str = ts.strftime("%Y-%m-%d %H:%M:%S")
    await db.execute(
        "INSERT INTO snapshots "
        "(market_id, timestamp, price_yes, price_no, volume_24hr, volume_total, liquidity) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (market_id, ts_str, price_yes, 1.0 - price_yes, 10000.0, volume_total, 50000.0),
    )
    await db.commit()


async def _seed_price_series(
    db,
    agent: InfoLeakDetector,
    market_id: str,
    now: datetime,
    n: int,
    var: float,
    spike: float,
) -> None:
    """Insert n stable snapshots (0.50 +/- var) then one spike, ticking after each."""
    for i in range(n):
        ts = now + timedelta(minutes=i)
        price = 0.50 + (var if i % 2 == 0 else -var)
        await _insert_snapshot(db, market_id, ts, price)
        await agent.tick()
    spike_ts = now + timedelta(minutes=n)
    await _insert_snapshot(db, market_id, spike_ts, spike)
    await agent.tick()


async def _report_rows(db) -> list[tuple]:
    cursor = await db.execute(
        "SELECT agent, market_id, severity, anomaly_score, confidence, summary, "
        "details, calendar_matches FROM anomaly_reports"
    )
    return await cursor.fetchall()


@pytest.mark.asyncio
async def test_extract_keywords():
    assert _extract_keywords(QUESTION) == [
        "trump",
        "nominate",
        "judy",
        "shelton",
        "next",
        "chair",
    ]


@pytest.mark.asyncio
async def test_warmup_produces_no_reports(db_config, db):
    """Two stable snapshots processed across two ticks -> zero reports."""
    agent = InfoLeakDetector(db_config, db, news_checker=StubNewsChecker())
    now = datetime.now(timezone.utc)

    await _insert_snapshot(db, "m1", now, 0.50)
    await agent.tick()
    await _insert_snapshot(db, "m1", now + timedelta(minutes=1), 0.501)
    await agent.tick()

    rows = await _report_rows(db)
    assert rows == []


@pytest.mark.asyncio
async def test_detection_emits_single_report(db_config, db):
    """>=6 stable snapshots then a spike -> exactly one report row."""
    agent = InfoLeakDetector(db_config, db, news_checker=StubNewsChecker())
    now = datetime.now(timezone.utc)

    await _seed_price_series(db, agent, "m1", now, n=20, var=0.002, spike=0.70)

    rows = await _report_rows(db)
    assert len(rows) == 1
    agent_name, market_id, _severity, score, _confidence, summary, _details, _cal = rows[0]
    assert agent_name == "info_leak"
    assert market_id == "m1"
    assert score >= 4.0
    assert QUESTION in summary


@pytest.mark.asyncio
async def test_detection_negative_control_stable_only(db_config, db):
    """Same pipeline fed only stable data must NOT emit — proves the
    detection test above is actually testing sensitivity, not a tautology."""
    agent = InfoLeakDetector(db_config, db, news_checker=StubNewsChecker())
    now = datetime.now(timezone.utc)

    for i in range(21):
        ts = now + timedelta(minutes=i)
        price = 0.50 + (0.002 if i % 2 == 0 else -0.002)
        await _insert_snapshot(db, "m1", ts, price)
        await agent.tick()

    rows = await _report_rows(db)
    assert rows == []


@pytest.mark.asyncio
async def test_no_double_processing(db_config, db):
    """Two ticks with no new snapshot must not re-count into the analyzer."""
    agent = InfoLeakDetector(db_config, db, news_checker=StubNewsChecker())
    now = datetime.now(timezone.utc)

    for i in range(4):
        ts = now + timedelta(minutes=i)
        price = 0.50 + (0.002 if i % 2 == 0 else -0.002)
        await _insert_snapshot(db, "m1", ts, price)
        await agent.tick()

    count_after_seed = agent._price_analyzer._states["m1"].return_stats.count

    await agent.tick()
    await agent.tick()

    count_after_repeat = agent._price_analyzer._states["m1"].return_stats.count
    assert count_after_repeat == count_after_seed

    rows = await _report_rows(db)
    assert rows == []


@pytest.mark.asyncio
async def test_event_amplifier_scales_score(tmp_path):
    """An identical spike scenario with a nearby scheduled event scores 1.5x."""
    now = datetime.now(timezone.utc)

    base_config, base_db = await _make_db(tmp_path, "amp_base")
    base_agent = InfoLeakDetector(base_config, base_db, news_checker=StubNewsChecker())
    await _seed_price_series(base_db, base_agent, "m1", now, n=20, var=0.002, spike=0.70)
    base_rows = await _report_rows(base_db)
    assert len(base_rows) == 1
    base_score = base_rows[0][3]
    await base_db.close()

    amp_config, amp_db = await _make_db(tmp_path, "amp_event")
    await queries.save_scheduled_events(
        amp_db,
        [
            ScheduledEvent(
                source="congress",
                event_type="hearing",
                title="Fed Chair Confirmation Hearing",
                description="",
                event_date=now + timedelta(hours=1),
                url="",
                keywords=[],
            )
        ],
    )
    amp_agent = InfoLeakDetector(amp_config, amp_db, news_checker=StubNewsChecker())
    await _seed_price_series(amp_db, amp_agent, "m1", now, n=20, var=0.002, spike=0.70)
    amp_rows = await _report_rows(amp_db)
    assert len(amp_rows) == 1
    amp_score = amp_rows[0][3]
    details = json.loads(amp_rows[0][6])
    calendar_matches = json.loads(amp_rows[0][7])
    await amp_db.close()

    assert calendar_matches != []
    assert details["amplifiers_applied"]
    assert amp_score == pytest.approx(base_score * 1.5, rel=1e-6)


@pytest.mark.asyncio
async def test_news_dampener_scales_score(tmp_path):
    """Identical data, news_found=True vs False, differs by exactly 0.7x."""
    now = datetime.now(timezone.utc)

    no_news_config, no_news_db = await _make_db(tmp_path, "dampen_no_news")
    no_news_agent = InfoLeakDetector(
        no_news_config, no_news_db, news_checker=StubNewsChecker(news_found=False)
    )
    await _seed_price_series(no_news_db, no_news_agent, "m1", now, n=50, var=0.002, spike=0.70)
    no_news_rows = await _report_rows(no_news_db)
    assert len(no_news_rows) == 1
    no_news_score = no_news_rows[0][3]
    await no_news_db.close()

    news_config, news_db = await _make_db(tmp_path, "dampen_news")
    news_agent = InfoLeakDetector(
        news_config, news_db, news_checker=StubNewsChecker(news_found=True)
    )
    await _seed_price_series(news_db, news_agent, "m1", now, n=50, var=0.002, spike=0.70)
    news_rows = await _report_rows(news_db)
    assert len(news_rows) == 1
    news_score = news_rows[0][3]
    details = json.loads(news_rows[0][6])
    await news_db.close()

    assert details["dampeners_applied"]
    assert news_score == pytest.approx(no_news_score * 0.7, rel=1e-6)


@pytest.mark.asyncio
async def test_cooldown_suppresses_second_emission(db_config, db):
    """A second triggering snapshot within the cooldown window emits nothing new."""
    agent = InfoLeakDetector(db_config, db, news_checker=StubNewsChecker())
    now = datetime.now(timezone.utc)

    await _seed_price_series(db, agent, "m1", now, n=20, var=0.002, spike=0.70)
    rows = await _report_rows(db)
    assert len(rows) == 1

    # A further move well within the 60-minute cooldown window.
    second_ts = now + timedelta(minutes=21)
    await _insert_snapshot(db, "m1", second_ts, 0.72)
    await agent.tick()

    rows_after = await _report_rows(db)
    assert len(rows_after) == 1


@pytest.mark.asyncio
async def test_severity_matches_confidence_mapping(db_config, db):
    agent = InfoLeakDetector(db_config, db, news_checker=StubNewsChecker())
    now = datetime.now(timezone.utc)

    await _seed_price_series(db, agent, "m1", now, n=20, var=0.002, spike=0.70)
    rows = await _report_rows(db)
    assert len(rows) == 1
    severity, confidence = rows[0][2], rows[0][4]

    assert severity == AnomalyReport.severity_from_score(confidence)


@pytest.mark.asyncio
async def test_report_content(db_config, db):
    agent = InfoLeakDetector(db_config, db, news_checker=StubNewsChecker())
    now = datetime.now(timezone.utc)

    await _seed_price_series(db, agent, "m1", now, n=20, var=0.002, spike=0.70)
    rows = await _report_rows(db)
    assert len(rows) == 1
    agent_name, market_id, _sev, _score, _conf, summary, _details, _cal = rows[0]

    assert agent_name == "info_leak"
    assert market_id == "m1"
    assert summary
    assert QUESTION in summary
