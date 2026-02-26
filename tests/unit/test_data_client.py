"""Unit tests for Data API client with mocked HTTP."""

import json
from pathlib import Path

import httpx
import pytest
import respx

from prediction_market.config import load_config
from prediction_market.data.polymarket.data_client import DataClient

FIXTURES = Path(__file__).parent.parent / "fixtures"


@pytest.fixture
def config():
    return load_config()


@pytest.fixture
def trades_data():
    with open(FIXTURES / "trades.json") as f:
        return json.load(f)


@pytest.fixture
def holders_data():
    with open(FIXTURES / "holders.json") as f:
        return json.load(f)


@pytest.mark.asyncio
@respx.mock
async def test_get_trades_list_response(config, trades_data):
    respx.get(f"{config.apis.data_base_url}/trades").mock(
        return_value=httpx.Response(200, json=trades_data)
    )
    client = DataClient(config)
    try:
        trades = await client.get_trades(market_id="fixture-market-1")
        assert len(trades) == 4
        assert trades[0].id == "trade-1"
        assert trades[0].side == "BUY"
        assert trades[0].price_float == pytest.approx(0.65)
        assert trades[0].size_float == pytest.approx(500.0)
    finally:
        await client.close()


@pytest.mark.asyncio
@respx.mock
async def test_get_trades_dict_response(config, trades_data):
    """API may return trades wrapped in a dict with 'data' key."""
    respx.get(f"{config.apis.data_base_url}/trades").mock(
        return_value=httpx.Response(200, json={"data": trades_data})
    )
    client = DataClient(config)
    try:
        trades = await client.get_trades(condition_id="0xabc")
        assert len(trades) == 4
    finally:
        await client.close()


@pytest.mark.asyncio
@respx.mock
async def test_get_all_trades_pagination(config, trades_data):
    route = respx.get(f"{config.apis.data_base_url}/trades")
    # First page returns full batch, second returns less than limit
    route.side_effect = [
        httpx.Response(200, json=trades_data),
        httpx.Response(200, json=[trades_data[0]]),
    ]
    client = DataClient(config)
    try:
        trades = await client.get_all_trades(market_id="m1", limit=4)
        assert len(trades) == 5  # 4 + 1
    finally:
        await client.close()


@pytest.mark.asyncio
@respx.mock
async def test_get_all_trades_empty(config):
    respx.get(f"{config.apis.data_base_url}/trades").mock(
        return_value=httpx.Response(200, json=[])
    )
    client = DataClient(config)
    try:
        trades = await client.get_all_trades(market_id="m1")
        assert trades == []
    finally:
        await client.close()


@pytest.mark.asyncio
@respx.mock
async def test_get_market_holders(config, holders_data):
    respx.get(f"{config.apis.data_base_url}/holders").mock(
        return_value=httpx.Response(200, json=holders_data)
    )
    client = DataClient(config)
    try:
        holders = await client.get_market_holders(condition_id="0xabc")
        assert len(holders) == 6
        assert holders[0].address == "0xwhale1"
        assert holders[0].position == 50000
        assert holders[0].pct_supply == pytest.approx(0.35)
    finally:
        await client.close()


@pytest.mark.asyncio
@respx.mock
async def test_get_market_holders_dict_response(config, holders_data):
    respx.get(f"{config.apis.data_base_url}/holders").mock(
        return_value=httpx.Response(200, json={"data": holders_data})
    )
    client = DataClient(config)
    try:
        holders = await client.get_market_holders(condition_id="0xabc", token_id="tok1")
        assert len(holders) == 6
    finally:
        await client.close()


@pytest.mark.asyncio
@respx.mock
async def test_get_open_interest(config):
    respx.get(f"{config.apis.data_base_url}/open-interest").mock(
        return_value=httpx.Response(200, json={"assetId": "tok1", "openInterest": 1500000})
    )
    client = DataClient(config)
    try:
        oi = await client.get_open_interest(condition_id="0xabc")
        assert oi.open_interest == pytest.approx(1500000)
    finally:
        await client.close()


@pytest.mark.asyncio
@respx.mock
async def test_get_open_interest_list_response(config):
    respx.get(f"{config.apis.data_base_url}/open-interest").mock(
        return_value=httpx.Response(
            200, json=[{"assetId": "tok1", "openInterest": 750000}]
        )
    )
    client = DataClient(config)
    try:
        oi = await client.get_open_interest(condition_id="0xabc")
        assert oi.open_interest == pytest.approx(750000)
    finally:
        await client.close()
