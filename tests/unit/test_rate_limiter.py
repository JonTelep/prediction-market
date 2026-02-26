"""Tests for token-bucket rate limiter."""

import asyncio
import time

import pytest

from prediction_market.data.polymarket.rate_limiter import TokenBucketRateLimiter


@pytest.mark.asyncio
async def test_initial_tokens():
    limiter = TokenBucketRateLimiter(max_tokens=100, window_seconds=10)
    assert limiter.available_tokens == pytest.approx(100, abs=1)


@pytest.mark.asyncio
async def test_acquire_consumes_tokens():
    limiter = TokenBucketRateLimiter(max_tokens=100, window_seconds=10)
    await limiter.acquire(10)
    assert limiter.available_tokens < 100


@pytest.mark.asyncio
async def test_acquire_multiple():
    limiter = TokenBucketRateLimiter(max_tokens=10, window_seconds=1)
    for _ in range(10):
        await limiter.acquire(1)
    # Should be near zero (some refill may have happened)
    assert limiter.available_tokens < 2


@pytest.mark.asyncio
async def test_refill_over_time():
    limiter = TokenBucketRateLimiter(max_tokens=100, window_seconds=1)
    await limiter.acquire(100)
    await asyncio.sleep(0.1)
    # Should have refilled ~10 tokens
    tokens = limiter.available_tokens
    assert tokens > 5


@pytest.mark.asyncio
async def test_does_not_exceed_max():
    limiter = TokenBucketRateLimiter(max_tokens=10, window_seconds=1)
    await asyncio.sleep(0.2)
    assert limiter.available_tokens <= 10


@pytest.mark.asyncio
async def test_acquire_waits_when_empty():
    limiter = TokenBucketRateLimiter(max_tokens=10, window_seconds=0.5)
    await limiter.acquire(10)
    start = time.monotonic()
    await limiter.acquire(1)
    elapsed = time.monotonic() - start
    assert elapsed > 0.01  # Had to wait for refill
