"""Gamma API client for market discovery and metadata."""

from __future__ import annotations

import logging
from typing import Any

import httpx

from prediction_market.config import AppConfig
from prediction_market.data.polymarket.models import GammaMarket
from prediction_market.data.polymarket.rate_limiter import TokenBucketRateLimiter

logger = logging.getLogger(__name__)


class GammaClient:
    """Client for the Gamma API (market discovery, metadata, tags)."""

    def __init__(self, config: AppConfig, http_client: httpx.AsyncClient | None = None) -> None:
        self.config = config
        self.base_url = config.apis.gamma_base_url
        self._client = http_client or httpx.AsyncClient(timeout=30.0)
        self._owns_client = http_client is None
        self._limiter = TokenBucketRateLimiter(
            max_tokens=config.rate_limits.gamma_requests_per_window,
            window_seconds=config.rate_limits.gamma_window_seconds,
        )

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        await self._limiter.acquire()
        url = f"{self.base_url}{path}"
        resp = await self._client.get(url, params=params)
        resp.raise_for_status()
        return resp.json()

    async def get_markets(
        self,
        limit: int = 100,
        offset: int = 0,
        active: bool = True,
        closed: bool = False,
        order: str = "volume",
        ascending: bool = False,
        tag: str | None = None,
    ) -> list[GammaMarket]:
        """Fetch markets from Gamma API."""
        params: dict[str, Any] = {
            "limit": limit,
            "offset": offset,
            "active": str(active).lower(),
            "closed": str(closed).lower(),
            "order": order,
            "ascending": str(ascending).lower(),
        }
        if tag:
            params["tag"] = tag
        data = await self._get("/markets", params=params)
        if isinstance(data, list):
            return [GammaMarket.model_validate(m) for m in data]
        return []

    async def get_market(self, market_id: str) -> GammaMarket | None:
        """Fetch a single market by ID."""
        try:
            data = await self._get(f"/markets/{market_id}")
            if isinstance(data, dict):
                return GammaMarket.model_validate(data)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return None
            raise
        return None

    async def get_all_markets(
        self,
        active: bool = True,
        closed: bool = False,
        tag: str | None = None,
        max_pages: int = 50,
    ) -> list[GammaMarket]:
        """Paginate through all markets."""
        all_markets: list[GammaMarket] = []
        offset = 0
        limit = 100
        for _ in range(max_pages):
            batch = await self.get_markets(
                limit=limit, offset=offset, active=active, closed=closed, tag=tag
            )
            if not batch:
                break
            all_markets.extend(batch)
            if len(batch) < limit:
                break
            offset += limit
        return all_markets

    async def search_markets(self, query: str, limit: int = 50) -> list[GammaMarket]:
        """Look up markets by exact slug match.

        Despite the name, this is not a text/keyword search -- it filters
        the Gamma ``/markets`` endpoint by the ``slug`` query param, so
        *query* must be (or closely match) a market's slug.
        """
        params: dict[str, Any] = {"limit": limit}
        data = await self._get("/markets", params={**params, "slug": query})
        if isinstance(data, list):
            return [GammaMarket.model_validate(m) for m in data]
        return []
