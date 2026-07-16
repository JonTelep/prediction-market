"""Data API client for trades, market holders, and open interest."""

from __future__ import annotations

import logging
from typing import Any

import httpx

from prediction_market.config import AppConfig
from prediction_market.data.polymarket.models import MarketHolder, OpenInterest, Trade
from prediction_market.data.polymarket.rate_limiter import TokenBucketRateLimiter

logger = logging.getLogger(__name__)


class DataClient:
    """Client for the Data API (trades, holders, open interest).

    Base URL: https://data-api.polymarket.com
    """

    def __init__(self, config: AppConfig, http_client: httpx.AsyncClient | None = None) -> None:
        self.config = config
        self.base_url = config.apis.data_base_url
        self._client = http_client or httpx.AsyncClient(timeout=30.0)
        self._owns_client = http_client is None
        self._limiter = TokenBucketRateLimiter(
            max_tokens=config.rate_limits.data_requests_per_window,
            window_seconds=config.rate_limits.data_window_seconds,
        )

    async def close(self) -> None:
        """Close the underlying HTTP client if we own it."""
        if self._owns_client:
            await self._client.aclose()

    async def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        """Make a rate-limited GET request to the Data API."""
        await self._limiter.acquire()
        url = f"{self.base_url}{path}"
        resp = await self._client.get(url, params=params)
        resp.raise_for_status()
        return resp.json()

    async def get_trades(
        self,
        market_id: str | None = None,
        condition_id: str | None = None,
        limit: int = 100,
        cursor: str | None = None,
    ) -> list[Trade]:
        """Fetch trades for a market or condition.

        At least one of market_id or condition_id should be provided.

        Args:
            market_id: The CLOB market slug/ID to filter by.
            condition_id: The condition ID to filter by.
            limit: Maximum number of trades to return per page.
            cursor: Pagination cursor from a previous response.

        Returns:
            List of Trade objects, newest first.
        """
        params: dict[str, Any] = {"limit": limit}
        if market_id is not None:
            params["market"] = market_id
        if condition_id is not None:
            params["condition_id"] = condition_id
        if cursor is not None:
            params["cursor"] = cursor

        data = await self._get("/trades", params=params)

        # The API may return a list directly or a dict with a "data" key
        trades_list: list[dict[str, Any]] = []
        if isinstance(data, list):
            trades_list = data
        elif isinstance(data, dict):
            trades_list = data.get("data", data.get("trades", []))

        return [Trade.model_validate(t) for t in trades_list]

    async def get_all_trades(
        self,
        market_id: str | None = None,
        condition_id: str | None = None,
        max_pages: int = 50,
        limit: int = 100,
    ) -> list[Trade]:
        """Paginate through all trades for a market.

        Args:
            market_id: The CLOB market slug/ID.
            condition_id: The condition ID.
            max_pages: Maximum pages to fetch (safety limit).
            limit: Page size.

        Returns:
            Aggregated list of Trade objects.
        """
        all_trades: list[Trade] = []
        cursor: str | None = None

        for _ in range(max_pages):
            batch = await self.get_trades(
                market_id=market_id,
                condition_id=condition_id,
                limit=limit,
                cursor=cursor,
            )
            if not batch:
                break
            all_trades.extend(batch)
            if len(batch) < limit:
                break
            # Use the last trade ID as the cursor for the next page
            cursor = batch[-1].id

        return all_trades

    async def get_market_holders(
        self,
        condition_id: str,
        token_id: str | None = None,
        limit: int = 100,
    ) -> list[MarketHolder]:
        """Fetch current holders for a market condition.

        Args:
            condition_id: The condition ID of the market.
            token_id: Optional token ID to filter to a specific outcome.
            limit: Maximum number of holders to return.

        Returns:
            List of MarketHolder objects sorted by position size.
        """
        params: dict[str, Any] = {
            "condition_id": condition_id,
            "limit": limit,
        }
        if token_id is not None:
            params["token_id"] = token_id

        data = await self._get("/holders", params=params)

        holders_list: list[dict[str, Any]] = []
        if isinstance(data, list):
            holders_list = data
        elif isinstance(data, dict):
            holders_list = data.get("data", data.get("holders", []))

        return [MarketHolder.model_validate(h) for h in holders_list]

    async def get_open_interest(self, condition_id: str) -> OpenInterest:
        """Fetch open interest for a market condition.

        Args:
            condition_id: The condition ID of the market.

        Returns:
            OpenInterest with total value currently at stake.
        """
        data = await self._get("/open-interest", params={"condition_id": condition_id})

        if isinstance(data, dict):
            return OpenInterest.model_validate(data)

        # If the API returns a list, take the first entry
        if isinstance(data, list) and data:
            return OpenInterest.model_validate(data[0])

        return OpenInterest()
