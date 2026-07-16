"""CLOB API client for order book, pricing, and price history data."""

from __future__ import annotations

import logging
from typing import Any

import httpx

from prediction_market.config import AppConfig
from prediction_market.data.polymarket.models import ClobPriceHistory, OrderBook
from prediction_market.data.polymarket.rate_limiter import TokenBucketRateLimiter

logger = logging.getLogger(__name__)


class ClobClient:
    """Client for the CLOB API (order books, prices, spreads, price history).

    Base URL: https://clob.polymarket.com
    """

    def __init__(self, config: AppConfig, http_client: httpx.AsyncClient | None = None) -> None:
        self.config = config
        self.base_url = config.apis.clob_base_url
        self._client = http_client or httpx.AsyncClient(timeout=30.0)
        self._owns_client = http_client is None
        self._limiter = TokenBucketRateLimiter(
            max_tokens=config.rate_limits.clob_requests_per_window,
            window_seconds=config.rate_limits.clob_window_seconds,
        )

    async def close(self) -> None:
        """Close the underlying HTTP client if we own it."""
        if self._owns_client:
            await self._client.aclose()

    async def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        """Make a rate-limited GET request to the CLOB API."""
        await self._limiter.acquire()
        url = f"{self.base_url}{path}"
        resp = await self._client.get(url, params=params)
        resp.raise_for_status()
        return resp.json()

    async def get_order_book(self, token_id: str) -> OrderBook:
        """Fetch the current order book for a token.

        Args:
            token_id: The CLOB token ID (asset ID) for the outcome.

        Returns:
            OrderBook with bids and asks populated.
        """
        data = await self._get("/book", params={"token_id": token_id})
        return OrderBook.model_validate(data)

    async def get_price(self, token_id: str) -> float:
        """Fetch the last traded price for a token.

        Args:
            token_id: The CLOB token ID.

        Returns:
            Last traded price as a float.
        """
        data = await self._get("/price", params={"token_id": token_id})
        return float(data.get("price", 0.0))

    async def get_midpoint(self, token_id: str) -> float:
        """Fetch the midpoint price for a token.

        This uses the CLOB midpoint endpoint, which returns the average
        of the best bid and best ask.

        Args:
            token_id: The CLOB token ID.

        Returns:
            Midpoint price as a float.
        """
        data = await self._get("/midpoint", params={"token_id": token_id})
        return float(data.get("mid", 0.0))

    async def get_spread(self, token_id: str) -> float:
        """Fetch the current bid-ask spread for a token.

        Args:
            token_id: The CLOB token ID.

        Returns:
            Spread as a float (ask - bid).
        """
        data = await self._get("/spread", params={"token_id": token_id})
        return float(data.get("spread", 0.0))

    async def get_price_history(
        self,
        token_id: str,
        start_ts: int | None = None,
        end_ts: int | None = None,
        interval: str = "1d",
        fidelity: int = 60,
    ) -> ClobPriceHistory:
        """Fetch historical price data for a token.

        Args:
            token_id: The CLOB token ID.
            start_ts: Start timestamp (Unix seconds). Omit for API default.
            end_ts: End timestamp (Unix seconds). Omit for API default.
            interval: Time interval grouping (e.g. '1m', '1h', '1d').
            fidelity: Number of data points to return within the interval.

        Returns:
            ClobPriceHistory containing timestamped price points.
        """
        params: dict[str, Any] = {
            "token_id": token_id,
            "interval": interval,
            "fidelity": fidelity,
        }
        if start_ts is not None:
            params["startTs"] = start_ts
        if end_ts is not None:
            params["endTs"] = end_ts

        data = await self._get("/prices-history", params=params)
        return ClobPriceHistory.model_validate(data)
