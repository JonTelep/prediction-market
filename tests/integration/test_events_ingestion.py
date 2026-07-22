"""Integration test for the scheduled-events ingestion loop.

Proves that Orchestrator._refresh_scheduled_events() -- the cycle body of
the periodic "event-refresh" task -- actually pulls from the three
government-calendar clients and persists rows through the real
save_scheduled_events()/get_scheduled_events_in_range() seam that the
info-leak detector's event amplifier consumes.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import httpx
import pytest
import respx

from prediction_market.orchestrator import Orchestrator
from prediction_market.store.queries import get_scheduled_events_in_range

CONGRESS_REPORT_PAYLOAD = {
    "reports": [
        {
            "title": "Oversight Hearing on Executive Actions",
            "chamber": "Senate",
            "citation": "S.Rept. 119-42",
            "url": "https://congress.gov/report/119-42",
            "updateDate": "2026-07-16T00:00:00Z",
        }
    ]
}

CONGRESS_MEETING_PAYLOAD: dict = {"committeeMeetings": []}

CONGRESS_AMENDMENT_PAYLOAD = {
    "amendments": [
        {
            "description": "Amendment to appropriations bill",
            "purpose": "Adjust funding levels",
            "number": "5",
            "type": "SA",
            "congress": 119,
            "url": "https://congress.gov/amendment/119-5",
            "latestAction": {"actionDate": "2026-07-17"},
        }
    ]
}

COURT_ARGUMENTS_PAYLOAD = {
    "results": [
        {
            "case_name": "Test v. State",
            "case_name_short": "Test v. State",
            "docket": {"docket_number": "23-123"},
            "court": "scotus",
            "date_argued": "2026-07-20",
            "absolute_url": "/opinion/123/test-v-state/",
        }
    ]
}

WHITE_HOUSE_POSTS_PAYLOAD = [
    {
        "title": {"rendered": "President's Public Schedule for July 16"},
        "content": {"rendered": "<p>Meets with senior staff in the Oval Office.</p>"},
        "link": "https://www.whitehouse.gov/briefing-room/schedule/2026/07/16/",
        "date_gmt": "2026-07-16T09:00:00",
    }
]


def _mock_empty_market_discovery(config) -> None:
    """_init_resources() always runs market discovery; keep it a no-op."""
    respx.get(f"{config.apis.gamma_base_url}/markets").mock(
        return_value=httpx.Response(200, json=[])
    )


def _mock_all_calendar_routes(config) -> None:
    respx.get(f"{config.apis.congress_base_url}/committee-report").mock(
        return_value=httpx.Response(200, json=CONGRESS_REPORT_PAYLOAD)
    )
    respx.get(f"{config.apis.congress_base_url}/committee-meeting").mock(
        return_value=httpx.Response(200, json=CONGRESS_MEETING_PAYLOAD)
    )
    respx.get(f"{config.apis.congress_base_url}/amendment").mock(
        return_value=httpx.Response(200, json=CONGRESS_AMENDMENT_PAYLOAD)
    )
    respx.get(f"{config.apis.courtlistener_base_url}/oral-arguments/").mock(
        return_value=httpx.Response(200, json=COURT_ARGUMENTS_PAYLOAD)
    )
    _mock_white_house_route()


def _mock_white_house_route() -> None:
    respx.get("https://www.whitehouse.gov/wp-json/wp/v2/posts").mock(
        return_value=httpx.Response(200, json=WHITE_HOUSE_POSTS_PAYLOAD)
    )


async def _scheduled_event_sources(db) -> list[str]:
    cursor = await db.execute("SELECT source FROM scheduled_events")
    rows = await cursor.fetchall()
    return [row[0] for row in rows]


@pytest.mark.asyncio
@respx.mock
async def test_refresh_cycle_inserts_rows_with_correct_source_literals(app_config):
    app_config.congress_api_key = "fake-congress-key"
    app_config.courtlistener_token = "fake-courtlistener-token"

    _mock_empty_market_discovery(app_config)
    _mock_all_calendar_routes(app_config)

    orch = Orchestrator(app_config)
    await orch._init_resources()
    try:
        await orch._refresh_scheduled_events()

        sources = await _scheduled_event_sources(orch._db)
        assert sources, "expected at least one scheduled event row"
        assert set(sources) <= {"congress.gov", "courtlistener", "whitehouse.gov"}
        assert "congress.gov" in sources
        assert "courtlistener" in sources
        assert "whitehouse.gov" in sources
    finally:
        await orch.stop()


@pytest.mark.asyncio
@respx.mock
async def test_refresh_cycle_run_twice_does_not_duplicate(app_config):
    app_config.congress_api_key = "fake-congress-key"
    app_config.courtlistener_token = "fake-courtlistener-token"

    _mock_empty_market_discovery(app_config)
    _mock_all_calendar_routes(app_config)

    orch = Orchestrator(app_config)
    await orch._init_resources()
    try:
        # Exercise the real cycle method twice -- not save_scheduled_events
        # directly -- to prove the wiring (not just the store layer) is
        # idempotent under the unique index.
        await orch._refresh_scheduled_events()
        first_count = len(await _scheduled_event_sources(orch._db))

        await orch._refresh_scheduled_events()
        second_count = len(await _scheduled_event_sources(orch._db))

        assert first_count > 0
        assert second_count == first_count
    finally:
        await orch.stop()


@pytest.mark.asyncio
@respx.mock
async def test_refresh_cycle_without_keys_skips_keyed_sources_but_keeps_white_house(
    app_config,
):
    app_config.congress_api_key = ""
    app_config.courtlistener_token = ""

    _mock_empty_market_discovery(app_config)
    # Only White House is reachable without a key; Congress/CourtListener
    # short-circuit before making any HTTP call, so their routes are
    # intentionally left unmocked (respx would raise if hit).
    _mock_white_house_route()

    orch = Orchestrator(app_config)
    await orch._init_resources()
    try:
        await orch._refresh_scheduled_events()

        sources = await _scheduled_event_sources(orch._db)
        assert "congress.gov" not in sources
        assert "courtlistener" not in sources
        assert "whitehouse.gov" in sources
    finally:
        await orch.stop()


@pytest.mark.asyncio
@respx.mock
async def test_ingested_events_are_readable_in_range(app_config):
    """The seam the info-leak detector's amplifier consumes: after a
    refresh cycle, get_scheduled_events_in_range() around a seeded event's
    date returns it.
    """
    app_config.congress_api_key = ""
    app_config.courtlistener_token = ""

    _mock_empty_market_discovery(app_config)
    _mock_white_house_route()

    orch = Orchestrator(app_config)
    await orch._init_resources()
    try:
        await orch._refresh_scheduled_events()

        # The White House fixture is dated 2026-07-16T09:00:00.
        seeded_date = datetime(2026, 7, 16, 9, 0, 0)
        rows = await get_scheduled_events_in_range(
            orch._db,
            seeded_date - timedelta(hours=1),
            seeded_date + timedelta(hours=1),
        )
        assert any(r["source"] == "whitehouse.gov" for r in rows)
    finally:
        await orch.stop()
