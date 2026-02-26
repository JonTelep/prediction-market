"""Unit tests for CLOB API client with mocked HTTP."""

import json
from pathlib import Path

import httpx
import pytest
import respx

from prediction_market.config import load_config
from prediction_market.data.polymarket.clob_client import ClobClient

FIXTURES = Path(__file__).parent.parent / "fixtures"


@pytest.fixture
def config():
    return load_config()


@pytest.fixture
def orderbook_data():
    with open(FIXTURES / "orderbook.json") as f:
        return json.load(f)


@pytest.fixture
def price_history_data():
    with open(FIXTURES / "price_history.json") as f:
        return json.load(f)


@pytest.mark.asyncio
@respx.mock
async def test_get_order_book(config, orderbook_data):
    respx.get(f"{config.apis.clob_base_url}/book").mock(
        return_value=httpx.Response(200, json=orderbook_data)
    )
    client = ClobClient(config)
    try:
        ob = await client.get_order_book("token-yes-1")
        assert len(ob.bids) == 6
        assert len(ob.asks) == 6
        assert ob.best_bid == 0.64
        assert ob.best_ask == 0.66
        assert ob.midpoint == pytest.approx(0.65)
        assert ob.spread == pytest.approx(0.02)
    finally:
        await client.close()


@pytest.mark.asyncio
@respx.mock
async def test_get_midpoint(config):
    respx.get(f"{config.apis.clob_base_url}/midpoint").mock(
        return_value=httpx.Response(200, json={"mid": "0.65"})
    )
    client = ClobClient(config)
    try:
        mid = await client.get_midpoint("token-yes-1")
        assert mid == pytest.approx(0.65)
    finally:
        await client.close()


@pytest.mark.asyncio
@respx.mock
async def test_get_spread(config):
    respx.get(f"{config.apis.clob_base_url}/spread").mock(
        return_value=httpx.Response(200, json={"spread": "0.02"})
    )
    client = ClobClient(config)
    try:
        spread = await client.get_spread("token-yes-1")
        assert spread == pytest.approx(0.02)
    finally:
        await client.close()


@pytest.mark.asyncio
@respx.mock
async def test_get_price(config):
    respx.get(f"{config.apis.clob_base_url}/price").mock(
        return_value=httpx.Response(200, json={"price": "0.72"})
    )
    client = ClobClient(config)
    try:
        price = await client.get_price("token-yes-1")
        assert price == pytest.approx(0.72)
    finally:
        await client.close()


@pytest.mark.asyncio
@respx.mock
async def test_get_price_history(config, price_history_data):
    respx.get(f"{config.apis.clob_base_url}/prices-history").mock(
        return_value=httpx.Response(200, json=price_history_data)
    )
    client = ClobClient(config)
    try:
        history = await client.get_price_history(
            "token-yes-1", start_ts=1708300800, end_ts=1709424000
        )
        assert len(history.history) == 14
        assert history.history[0].p == 0.60
        assert history.history[-1].p == 0.78
    finally:
        await client.close()


@pytest.mark.asyncio
@respx.mock
async def test_get_price_history_no_timestamps(config, price_history_data):
    respx.get(f"{config.apis.clob_base_url}/prices-history").mock(
        return_value=httpx.Response(200, json=price_history_data)
    )
    client = ClobClient(config)
    try:
        history = await client.get_price_history("token-yes-1")
        assert len(history.history) == 14
    finally:
        await client.close()


@pytest.mark.asyncio
@respx.mock
async def test_get_order_book_http_error(config):
    respx.get(f"{config.apis.clob_base_url}/book").mock(
        return_value=httpx.Response(500)
    )
    client = ClobClient(config)
    try:
        with pytest.raises(httpx.HTTPStatusError):
            await client.get_order_book("token-yes-1")
    finally:
        await client.close()
