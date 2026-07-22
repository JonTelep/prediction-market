"""Integration tests for Gamma API client with mocked HTTP."""

import json
from pathlib import Path

import httpx
import pytest
import respx

from prediction_market.config import load_config
from prediction_market.data.polymarket.gamma_client import GammaClient

FIXTURES = Path(__file__).parent.parent / "fixtures"


@pytest.fixture
def config():
    return load_config()


@pytest.fixture
def markets_data():
    with open(FIXTURES / "gamma_markets.json") as f:
        return json.load(f)


@pytest.mark.asyncio
@respx.mock
async def test_get_markets(config, markets_data):
    respx.get(f"{config.apis.gamma_base_url}/markets").mock(
        return_value=httpx.Response(200, json=markets_data)
    )
    client = GammaClient(config)
    try:
        markets = await client.get_markets()
        assert len(markets) == 3
        assert markets[0].id == "fixture-market-1"
        assert markets[0].question.startswith("Will the president")
        assert markets[0].volume == 500000
    finally:
        await client.close()


@pytest.mark.asyncio
@respx.mock
async def test_get_single_market(config, markets_data):
    respx.get(f"{config.apis.gamma_base_url}/markets/fixture-market-1").mock(
        return_value=httpx.Response(200, json=markets_data[0])
    )
    client = GammaClient(config)
    try:
        market = await client.get_market("fixture-market-1")
        assert market is not None
        assert market.id == "fixture-market-1"
    finally:
        await client.close()


@pytest.mark.asyncio
@respx.mock
async def test_get_market_not_found(config):
    respx.get(f"{config.apis.gamma_base_url}/markets/nonexistent").mock(
        return_value=httpx.Response(404)
    )
    client = GammaClient(config)
    try:
        market = await client.get_market("nonexistent")
        assert market is None
    finally:
        await client.close()


@pytest.mark.asyncio
@respx.mock
async def test_get_markets_retries_on_500_then_succeeds(config, markets_data):
    route = respx.get(f"{config.apis.gamma_base_url}/markets")
    route.side_effect = [
        httpx.Response(500),
        httpx.Response(200, json=markets_data),
    ]
    client = GammaClient(config)
    try:
        markets = await client.get_markets()
        assert len(markets) == 3
        assert route.call_count == 2
    finally:
        await client.close()


@pytest.mark.asyncio
@respx.mock
async def test_get_all_markets_pagination(config, markets_data):
    # First page returns full batch, second page returns empty
    route = respx.get(f"{config.apis.gamma_base_url}/markets")
    route.side_effect = [
        httpx.Response(200, json=markets_data),
        httpx.Response(200, json=[]),
    ]
    client = GammaClient(config)
    try:
        markets = await client.get_all_markets()
        assert len(markets) == 3
    finally:
        await client.close()


@pytest.mark.asyncio
@respx.mock
async def test_search_markets_default_omits_closed_param(config, markets_data):
    route = respx.get(f"{config.apis.gamma_base_url}/markets").mock(
        return_value=httpx.Response(200, json=markets_data)
    )
    client = GammaClient(config)
    try:
        await client.search_markets("some-slug")
        params = route.calls[0].request.url.params
        assert params["slug"] == "some-slug"
        assert "closed" not in params
    finally:
        await client.close()


@pytest.mark.asyncio
@respx.mock
async def test_search_markets_closed_true_sends_param(config, markets_data):
    """Live Gamma /markets?slug= excludes closed markets unless closed=true
    is sent (verified against the real API 2026-07-20) -- resolved markets
    are only reachable with the param."""
    route = respx.get(f"{config.apis.gamma_base_url}/markets").mock(
        return_value=httpx.Response(200, json=markets_data)
    )
    client = GammaClient(config)
    try:
        await client.search_markets("some-slug", closed=True)
        params = route.calls[0].request.url.params
        assert params["slug"] == "some-slug"
        assert params["closed"] == "true"
    finally:
        await client.close()
