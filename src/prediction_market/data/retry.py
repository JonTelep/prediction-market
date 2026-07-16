"""HTTP GET retry helper for transient failures.

Retries are appropriate for transport-level failures (connection resets,
timeouts) and for a small set of "try again later" HTTP statuses (429 rate
limiting, 500/502/503/504 server errors). Any other 4xx response -- most
notably 404 -- is not retried: it is an answer, not a failure, and callers
(e.g. ``GammaClient.get_market``) rely on seeing it on the first attempt.

Rate limiting note: this helper does not accept or acquire from a rate
limiter. The Polymarket clients acquire once, before calling
``get_with_retry``, and retried attempts re-enter the network without a
second acquisition. This slightly undercounts request-budget consumption
under retries, which is acceptable given the 4000-9000 req/10s budgets
configured for Polymarket's APIs -- not worth threading limiter plumbing
through the retry loop for.
"""

from __future__ import annotations

import asyncio
import logging

import httpx

logger = logging.getLogger(__name__)

_RETRYABLE_STATUSES = frozenset({429, 500, 502, 503, 504})

# Bound as a module-level alias (rather than calling asyncio.sleep directly)
# so tests can monkeypatch just this retry backoff without touching
# asyncio.sleep globally -- other timing-sensitive tests (e.g.
# test_rate_limiter.py) depend on real sleeps elsewhere.
_sleep = asyncio.sleep


async def get_with_retry(
    client: httpx.AsyncClient,
    url: str,
    *,
    params: dict | None = None,
    headers: dict | None = None,
    attempts: int = 3,
    base_delay: float = 0.5,
) -> httpx.Response:
    """GET *url* via *client*, retrying transient failures.

    Retries on ``httpx.TransportError`` and on responses with status in
    {429, 500, 502, 503, 504}, up to *attempts* total tries. Backoff between
    attempts is ``base_delay * 2**attempt`` seconds, honoring a numeric
    ``Retry-After`` response header when present (the larger of the two
    delays is used). Any other error response (e.g. 404) is not retried --
    it is raised via ``raise_for_status()`` on the first attempt. After the
    final attempt this function always raises -- either the last transport
    exception or an ``httpx.HTTPStatusError`` via ``raise_for_status()`` --
    it never swallows a failure.
    """
    for attempt in range(attempts):
        try:
            response = await client.get(url, params=params, headers=headers)
        except httpx.TransportError:
            if attempt == attempts - 1:
                raise
            await _sleep_backoff(attempt, base_delay, None)
            continue

        if response.status_code < 400:
            return response

        if response.status_code not in _RETRYABLE_STATUSES:
            response.raise_for_status()
            return response  # pragma: no cover - raise_for_status always raises above

        if attempt == attempts - 1:
            response.raise_for_status()
            return response  # pragma: no cover - raise_for_status always raises above

        await _sleep_backoff(attempt, base_delay, response.headers.get("Retry-After"))

    # Unreachable: the loop above always returns or raises.
    raise AssertionError("get_with_retry exhausted attempts without returning or raising")


async def _sleep_backoff(attempt: int, base_delay: float, retry_after: str | None) -> None:
    delay = base_delay * (2**attempt)
    if retry_after is not None:
        try:
            delay = max(delay, float(retry_after))
        except ValueError:
            pass
    if delay > 0:
        await _sleep(delay)
