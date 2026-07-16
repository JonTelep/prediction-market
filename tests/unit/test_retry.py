"""Unit tests for the HTTP retry helper."""

from __future__ import annotations

import httpx
import pytest
import respx

from prediction_market.data.retry import get_with_retry

URL = "https://example.test/resource"


@pytest.mark.asyncio
@respx.mock
async def test_500_then_200_returns_200_after_two_requests():
    route = respx.get(URL)
    route.side_effect = [
        httpx.Response(500),
        httpx.Response(200, json={"ok": True}),
    ]

    async with httpx.AsyncClient() as client:
        resp = await get_with_retry(client, URL, base_delay=0.0)

    assert resp.status_code == 200
    assert resp.json() == {"ok": True}
    assert route.call_count == 2


@pytest.mark.asyncio
@respx.mock
async def test_404_raises_immediately_with_one_request():
    route = respx.get(URL).mock(return_value=httpx.Response(404))

    async with httpx.AsyncClient() as client:
        with pytest.raises(httpx.HTTPStatusError) as exc_info:
            await get_with_retry(client, URL, base_delay=0.0)

    assert exc_info.value.response.status_code == 404
    assert route.call_count == 1


@pytest.mark.asyncio
@respx.mock
async def test_three_503s_raises_after_three_requests():
    route = respx.get(URL).mock(return_value=httpx.Response(503))

    async with httpx.AsyncClient() as client:
        with pytest.raises(httpx.HTTPStatusError) as exc_info:
            await get_with_retry(client, URL, base_delay=0.0)

    assert exc_info.value.response.status_code == 503
    assert route.call_count == 3


@pytest.mark.asyncio
@respx.mock
async def test_retry_after_zero_on_429_still_retries():
    route = respx.get(URL)
    route.side_effect = [
        httpx.Response(429, headers={"Retry-After": "0"}),
        httpx.Response(200, json={"ok": True}),
    ]

    async with httpx.AsyncClient() as client:
        resp = await get_with_retry(client, URL, base_delay=0.0)

    assert resp.status_code == 200
    assert route.call_count == 2


@pytest.mark.asyncio
@respx.mock
async def test_transport_error_retries_then_succeeds():
    route = respx.get(URL)
    route.side_effect = [
        httpx.ConnectError("boom"),
        httpx.Response(200, json={"ok": True}),
    ]

    async with httpx.AsyncClient() as client:
        resp = await get_with_retry(client, URL, base_delay=0.0)

    assert resp.status_code == 200
    assert route.call_count == 2


@pytest.mark.asyncio
@respx.mock
async def test_transport_error_exhausts_attempts_and_raises():
    route = respx.get(URL)
    route.side_effect = httpx.ConnectError("boom")

    async with httpx.AsyncClient() as client:
        with pytest.raises(httpx.ConnectError):
            await get_with_retry(client, URL, base_delay=0.0, attempts=3)

    assert route.call_count == 3
