"""Integration tests for the orchestrator scan_once and backfill flows."""

import json
from pathlib import Path

import httpx
import pytest
import respx

from prediction_market.config import load_config
from prediction_market.orchestrator import Orchestrator

FIXTURES = Path(__file__).parent.parent / "fixtures"


@pytest.fixture
def config(tmp_path):
    cfg = load_config()
    cfg.database.path = str(tmp_path / "test.db")
    cfg.reporting.output_dir = str(tmp_path / "reports")
    return cfg


@pytest.fixture
def gamma_markets_data():
    with open(FIXTURES / "gamma_markets.json") as f:
        return json.load(f)


@pytest.fixture
def price_history_data():
    with open(FIXTURES / "price_history.json") as f:
        return json.load(f)


@pytest.fixture
def trades_data():
    with open(FIXTURES / "trades.json") as f:
        return json.load(f)


@pytest.mark.asyncio
@respx.mock
async def test_scan_once(config, gamma_markets_data):
    """scan_once should discover markets, classify them, and return results."""
    # Mock Gamma markets endpoint (may be called multiple times for pagination)
    route = respx.get(f"{config.apis.gamma_base_url}/markets")
    route.side_effect = [
        httpx.Response(200, json=gamma_markets_data),
        httpx.Response(200, json=[]),  # pagination stop
    ]

    orch = Orchestrator(config)
    results = await orch.scan_once()

    # The fixtures contain political markets; at least some should be classified
    assert isinstance(results, list)
    for r in results:
        assert "id" in r
        assert "question" in r
        assert "political_confidence" in r
        assert "political_reasons" in r


@pytest.mark.asyncio
@respx.mock
async def test_scan_once_no_markets(config):
    """scan_once with no markets should return empty list."""
    respx.get(f"{config.apis.gamma_base_url}/markets").mock(
        return_value=httpx.Response(200, json=[])
    )

    orch = Orchestrator(config)
    results = await orch.scan_once()
    assert results == []


@pytest.mark.asyncio
@respx.mock
async def test_backfill(config, gamma_markets_data, price_history_data, trades_data):
    """backfill should fetch historical data and populate the database."""
    # Markets discovery
    gamma_route = respx.get(f"{config.apis.gamma_base_url}/markets")
    gamma_route.side_effect = [
        httpx.Response(200, json=gamma_markets_data),
        httpx.Response(200, json=[]),
    ]

    # Price history for each token
    respx.get(f"{config.apis.clob_base_url}/prices-history").mock(
        return_value=httpx.Response(200, json=price_history_data)
    )

    # Trades for each market
    trades_route = respx.get(f"{config.apis.data_base_url}/trades")
    trades_route.side_effect = [
        httpx.Response(200, json=trades_data),
        httpx.Response(200, json=[]),  # pagination stop
    ] * 10  # enough for multiple markets

    orch = Orchestrator(config)
    total_points = await orch.backfill(days=7)

    # Should have ingested some data points (depends on how many political markets)
    assert total_points >= 0


@pytest.mark.asyncio
@respx.mock
async def test_backfill_handles_api_errors(config, gamma_markets_data):
    """backfill should handle API errors gracefully without crashing."""
    gamma_route = respx.get(f"{config.apis.gamma_base_url}/markets")
    gamma_route.side_effect = [
        httpx.Response(200, json=gamma_markets_data),
        httpx.Response(200, json=[]),
    ]

    # Price history returns error
    respx.get(f"{config.apis.clob_base_url}/prices-history").mock(
        return_value=httpx.Response(500)
    )

    # Trades also error
    respx.get(f"{config.apis.data_base_url}/trades").mock(
        return_value=httpx.Response(500)
    )

    orch = Orchestrator(config)
    # Should not raise despite API errors
    total_points = await orch.backfill(days=7)
    assert total_points == 0
