"""Token-bucket rate limiter for API calls."""

from __future__ import annotations

import asyncio
import time


class TokenBucketRateLimiter:
    """Async token-bucket rate limiter.

    Allows `max_tokens` requests per `window_seconds`.
    Tokens refill continuously.
    """

    def __init__(self, max_tokens: int, window_seconds: float) -> None:
        self.max_tokens = max_tokens
        self.window_seconds = window_seconds
        self.refill_rate = max_tokens / window_seconds  # tokens per second
        self._tokens = float(max_tokens)
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self.max_tokens, self._tokens + elapsed * self.refill_rate)
        self._last_refill = now

    async def acquire(self, tokens: int = 1) -> None:
        """Wait until `tokens` are available, then consume them."""
        async with self._lock:
            while True:
                self._refill()
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return
                wait_time = (tokens - self._tokens) / self.refill_rate
                await asyncio.sleep(wait_time)

    @property
    def available_tokens(self) -> float:
        self._refill()
        return self._tokens
