"""Flagship pipeline test: locks in Phase 1's thesis that the orchestrator
actually runs both surveillance agents end-to-end -- construction through
BaseAgent.start()/stop(), a tick producing a persisted anomaly_reports row,
and fail-fast construction. This file is permanent: future phases add to
it, never delete it.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import httpx
import pytest
import respx

from prediction_market.agents.base import BaseAgent
from prediction_market.agents.info_leak_detector import InfoLeakDetector
from prediction_market.agents.manipulation_guard import ManipulationGuard
from prediction_market.data.polymarket.models import GammaMarket
from prediction_market.orchestrator import Orchestrator
from prediction_market.store.snapshots import save_market

QUESTION = "Will the president sign the infrastructure bill by March 2026?"


async def _insert_snapshot(
    db,
    market_id: str,
    ts: datetime,
    price_yes: float,
    volume_total: float = 100000.0,
) -> None:
    """Insert a snapshot row with a fully controlled timestamp.

    Mirrors tests/unit/test_info_leak_detector.py::_insert_snapshot --
    store.snapshots.save_price_snapshot always stamps "now" and cannot be
    given an explicit timestamp.
    """
    ts_str = ts.strftime("%Y-%m-%d %H:%M:%S")
    await db.execute(
        "INSERT INTO snapshots "
        "(market_id, timestamp, price_yes, price_no, volume_24hr, volume_total, liquidity) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (market_id, ts_str, price_yes, 1.0 - price_yes, 10000.0, volume_total, 50000.0),
    )
    await db.commit()


async def _report_rows(db, agent: str) -> list[tuple]:
    cursor = await db.execute(
        "SELECT agent, market_id FROM anomaly_reports WHERE agent = ?",
        (agent,),
    )
    return await cursor.fetchall()


def _mock_empty_market_discovery(config) -> None:
    """Mock the Gamma markets endpoint to return zero markets.

    _init_resources() always runs market discovery; these tests care
    about agent wiring / tick behaviour, not discovery, so we keep the
    discovery call a no-op rather than let it hit the (absent) network.
    """
    respx.get(f"{config.apis.gamma_base_url}/markets").mock(
        return_value=httpx.Response(200, json=[])
    )


@pytest.mark.asyncio
@respx.mock
async def test_orchestrator_constructs_agents(app_config):
    """_build_agents wires both real agents through a live DB connection."""
    _mock_empty_market_discovery(app_config)
    orch = Orchestrator(app_config)
    await orch._init_resources()
    try:
        assert len(orch._agents) == 2
        classes = {type(a) for a in orch._agents}
        assert classes == {InfoLeakDetector, ManipulationGuard}
    finally:
        await orch.stop()

    info_leak_orch = Orchestrator(app_config, agent_filter="info-leak")
    await info_leak_orch._init_resources()
    try:
        assert len(info_leak_orch._agents) == 1
        assert isinstance(info_leak_orch._agents[0], InfoLeakDetector)
    finally:
        await info_leak_orch.stop()

    manipulation_orch = Orchestrator(app_config, agent_filter="manipulation")
    await manipulation_orch._init_resources()
    try:
        assert len(manipulation_orch._agents) == 1
        assert isinstance(manipulation_orch._agents[0], ManipulationGuard)
    finally:
        await manipulation_orch.stop()

    bad_orch = Orchestrator(app_config, agent_filter="not-a-real-agent")
    with pytest.raises(ValueError):
        await bad_orch._init_resources()
    await bad_orch.stop()


@pytest.mark.asyncio
@respx.mock
async def test_construction_failure_propagates(app_config, monkeypatch):
    """A mis-wired agent must crash startup loudly, not be logged-and-ignored."""
    _mock_empty_market_discovery(app_config)

    def _broken_init(self, *args, **kwargs):
        raise RuntimeError("deliberately broken agent construction")

    monkeypatch.setattr(InfoLeakDetector, "__init__", _broken_init)

    orch = Orchestrator(app_config)
    with pytest.raises(RuntimeError, match="deliberately broken"):
        await orch._init_resources()
    await orch.stop()


@pytest.mark.asyncio
@respx.mock
async def test_info_leak_pipeline_emits_report(app_config):
    """Seed a stable-then-spike snapshot sequence and drive the info-leak
    agent's tick() directly -- constructed via the orchestrator -- then
    assert an anomaly_reports row landed via SQL (not a mocked emit call).
    """
    _mock_empty_market_discovery(app_config)
    orch = Orchestrator(app_config, agent_filter="info-leak")
    await orch._init_resources()
    try:
        db = orch._db
        assert db is not None

        market = GammaMarket(
            id="pipeline-market-1",
            question=QUESTION,
            outcomes=["Yes", "No"],
            outcomePrices=["0.50", "0.50"],
            volume=500000.0,
            active=True,
            closed=False,
            conditionId="0xpipeline1",
            clobTokenIds=["token-yes-pipeline", "token-no-pipeline"],
        )
        await save_market(db, market, {"confidence": 0.8, "reasons": ["test"]})

        agent = orch._agents[0]
        assert isinstance(agent, InfoLeakDetector)

        now = datetime.now(timezone.utc)
        n = 20
        var = 0.002
        spike = 0.70
        for i in range(n):
            ts = now + timedelta(minutes=i)
            price = 0.50 + (var if i % 2 == 0 else -var)
            await _insert_snapshot(db, "pipeline-market-1", ts, price)
            await agent.tick()
        spike_ts = now + timedelta(minutes=n)
        await _insert_snapshot(db, "pipeline-market-1", spike_ts, spike)
        await agent.tick()

        rows = await _report_rows(db, "info_leak")
        assert len(rows) == 1
        assert rows[0][1] == "pipeline-market-1"
    finally:
        await orch.stop()


@pytest.mark.asyncio
@respx.mock
async def test_manipulation_pipeline_emits_report(app_config):
    """Seed one active market, mock the CLOB /book and Data-API holders
    endpoints with a thin book and a concentrated (non-degenerate) holder
    set, drive one tick(), and assert both the persisted report row and
    that the holders route actually got called -- proving the HHI path
    really ran rather than faking a high score off an empty/failed
    holders fetch (concentration_score([]) == 1.0).
    """
    _mock_empty_market_discovery(app_config)
    orch = Orchestrator(app_config, agent_filter="manipulation")
    await orch._init_resources()
    try:
        db = orch._db
        assert db is not None

        market = GammaMarket(
            id="pipeline-market-2",
            question="Will the manipulation guard catch a thin market?",
            outcomes=["Yes", "No"],
            outcomePrices=["0.50", "0.50"],
            volume=500000.0,
            active=True,
            closed=False,
            conditionId="0xpipeline2",
            clobTokenIds=["token-yes-manip", "token-no-manip"],
        )
        await save_market(db, market, {"confidence": 0.8, "reasons": ["test"]})

        thin_book = {
            "market": "pipeline-market-2",
            "asset_id": "token-yes-manip",
            "bids": [
                {"price": "0.60", "size": "50"},
                {"price": "0.55", "size": "100"},
            ],
            "asks": [
                {"price": "0.70", "size": "30"},
                {"price": "0.80", "size": "80"},
            ],
        }
        # Concentrated (whale-dominated) holder set: high enough HHI, combined
        # with the maximally-thin depth/spread scores from `thin_book` above,
        # to cross the composite susceptibility threshold (0.7). Multiple
        # non-trivial shares (not a single-holder or empty list) ensure the
        # sum-of-squares HHI path actually runs rather than hitting the
        # concentration_score([]) == 1.0 shortcut.
        holders_payload = [
            {"address": "0xwhale", "position": 85000, "value": 55000, "pctSupply": 0.85},
            {"address": "0xmed1", "position": 10000, "value": 6500, "pctSupply": 0.10},
            {"address": "0xmed2", "position": 5000, "value": 3250, "pctSupply": 0.05},
        ]

        book_route = respx.get(f"{app_config.apis.clob_base_url}/book").mock(
            return_value=httpx.Response(200, json=thin_book)
        )
        holders_route = respx.get(f"{app_config.apis.data_base_url}/holders").mock(
            return_value=httpx.Response(200, json=holders_payload)
        )

        agent = orch._agents[0]
        assert isinstance(agent, ManipulationGuard)

        await agent.tick()

        rows = await _report_rows(db, "manipulation")
        assert len(rows) == 1
        assert rows[0][1] == "pipeline-market-2"
        assert holders_route.called
        assert book_route.called
    finally:
        await orch.stop()


class _CountingAgent(BaseAgent):
    """Trivial BaseAgent subclass for proving start/stop mechanics without
    wall-clock flakiness on the real agents."""

    def __init__(self, config, db, sinks=None) -> None:
        super().__init__(config, db, sinks)
        self.tick_count = 0

    @property
    def name(self) -> str:
        return "counting_agent"

    @property
    def tick_interval_seconds(self) -> int:
        return 0

    async def tick(self) -> None:
        self.tick_count += 1


@pytest.mark.asyncio
async def test_agent_lifecycle_start_stop(app_config):
    """start() spawns the loop task; stop() cancels it after >=1 tick ran."""
    from prediction_market.store.database import init_database

    db = await init_database(app_config)
    try:
        agent = _CountingAgent(app_config, db)

        await agent.start()
        await asyncio.sleep(0.05)
        await agent.stop()

        assert agent.tick_count >= 1
        assert agent._task is None
    finally:
        await db.close()
