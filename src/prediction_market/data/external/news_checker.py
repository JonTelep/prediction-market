"""GDELT + NewsAPI integration for verifying news existence around market moves.

The critical question this module answers: did publicly available news
about a topic exist BEFORE a suspicious prediction market movement?
If a market moves significantly and no public news preceded it, that
is a stronger signal of potential information leakage.

Data sources:
  - GDELT (api.gdeltproject.org): No auth needed, generous rate limits,
    global news coverage with 15-minute update frequency.
  - NewsAPI (newsapi.org/v2): Requires API key. Free-tier request limits
    are not tracked or enforced proactively here; the client handles
    429/426 responses reactively when NewsAPI returns them.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

from prediction_market.config import AppConfig
from prediction_market.data.external.models import NewsArticle, NewsCheckResult
from prediction_market.data.retry import get_with_retry

logger = logging.getLogger(__name__)


def _parse_datetime(value: str | None) -> datetime:
    """Parse a datetime string from various news API formats."""
    if not value:
        return datetime.now(timezone.utc)
    # Try ISO 8601 first (NewsAPI format)
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, AttributeError):
        pass
    # GDELT uses YYYYMMDDHHmmSS format
    for fmt in (
        "%Y%m%d%H%M%S",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
    ):
        try:
            dt = datetime.strptime(value, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            continue
    return datetime.now(timezone.utc)


class NewsChecker:
    """Async client for checking news existence around market movements.

    Combines GDELT (free, no auth) and NewsAPI (key required, rate limited)
    to determine whether news about specific keywords existed before a
    given time -- the core check for distinguishing informed trading from
    reaction to public news.

    Usage:
        checker = NewsChecker(config)
        result = await checker.check_news_exists(
            keywords=["Supreme Court", "abortion"],
            before_time=market_move_time,
            window_hours=2,
        )
        if not result.news_found:
            # Market moved BEFORE any public news -- suspicious
            ...
    """

    def __init__(
        self,
        config: AppConfig,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.config = config
        self._gdelt_base_url = config.apis.gdelt_base_url.rstrip("/")
        self._newsapi_base_url = config.apis.newsapi_base_url.rstrip("/")
        self._newsapi_key = config.newsapi_key
        self._client = http_client or httpx.AsyncClient(timeout=30.0)
        self._owns_client = http_client is None

        if not self._newsapi_key:
            logger.warning(
                "NewsAPI key is not configured. "
                "NewsChecker will only use GDELT for news lookups. "
                "Set NEWSAPI_KEY in your environment for broader coverage."
            )

    async def close(self) -> None:
        """Close the underlying HTTP client if we own it."""
        if self._owns_client:
            await self._client.aclose()

    async def check_news_exists(
        self,
        keywords: list[str],
        before_time: datetime,
        window_hours: int = 2,
    ) -> NewsCheckResult:
        """Check if news about the given keywords existed BEFORE a specified time.

        This is the primary method for the surveillance pipeline. It looks
        for news articles published in the window [before_time - window_hours,
        before_time] to determine if public information was available before
        a suspicious market move.

        Args:
            keywords: Search terms to look for (e.g. ["Supreme Court", "ruling"]).
            before_time: The timestamp of the market movement. We check if
                news existed BEFORE this time.
            window_hours: How many hours before ``before_time`` to search.
                Default is 2 hours.

        Returns:
            NewsCheckResult indicating whether any articles were found,
            the list of matching articles, and the earliest article time.
        """
        if not keywords:
            return NewsCheckResult(
                news_found=False,
                articles=[],
                earliest_article_time=None,
                query_keywords=[],
            )

        # Ensure before_time has timezone info
        if before_time.tzinfo is None:
            before_time = before_time.replace(tzinfo=timezone.utc)

        start_time = before_time - timedelta(hours=window_hours)
        end_time = before_time

        # Query both sources in sequence (to stay within rate limits)
        all_articles: list[NewsArticle] = []

        gdelt_articles = await self.search_gdelt(keywords, start_time, end_time)
        all_articles.extend(gdelt_articles)

        newsapi_articles = await self.search_newsapi(keywords, start_time, end_time)
        all_articles.extend(newsapi_articles)

        # Filter to only articles strictly before the market move
        pre_move_articles = [
            a for a in all_articles if a.published_at < before_time
        ]

        # Sort by publication time (earliest first)
        pre_move_articles.sort(key=lambda a: a.published_at)

        # Deduplicate by URL
        seen_urls: set[str] = set()
        unique_articles: list[NewsArticle] = []
        for article in pre_move_articles:
            if article.url in seen_urls:
                continue
            seen_urls.add(article.url)
            unique_articles.append(article)

        earliest_time = unique_articles[0].published_at if unique_articles else None

        return NewsCheckResult(
            news_found=len(unique_articles) > 0,
            articles=unique_articles,
            earliest_article_time=earliest_time,
            query_keywords=list(keywords),
        )

    async def search_gdelt(
        self,
        keywords: list[str],
        start_time: datetime,
        end_time: datetime,
    ) -> list[NewsArticle]:
        """Search the GDELT Global Knowledge Graph for matching articles.

        GDELT's DOC API provides full-text search across global news with
        approximately 15-minute lag. No authentication required.

        Args:
            keywords: Search terms to combine with AND logic.
            start_time: Start of the search window (UTC).
            end_time: End of the search window (UTC).

        Returns:
            List of NewsArticle objects from GDELT results.
        """
        if not keywords:
            return []

        # Ensure timezone awareness
        if start_time.tzinfo is None:
            start_time = start_time.replace(tzinfo=timezone.utc)
        if end_time.tzinfo is None:
            end_time = end_time.replace(tzinfo=timezone.utc)

        # Build the GDELT DOC 2.0 query
        # Format: keywords joined by space (implicit AND in GDELT)
        query = " ".join(keywords)

        # GDELT uses YYYYMMDDHHMMSS format for time bounds
        start_str = start_time.strftime("%Y%m%d%H%M%S")
        end_str = end_time.strftime("%Y%m%d%H%M%S")

        params: dict[str, Any] = {
            "query": query,
            "mode": "ArtList",
            "maxrecords": 75,
            "format": "json",
            "startdatetime": start_str,
            "enddatetime": end_str,
            "sort": "DateDesc",
        }

        try:
            resp = await get_with_retry(
                self._client,
                f"{self._gdelt_base_url}/doc/doc",
                params=params,
            )
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPStatusError as exc:
            logger.error(
                "GDELT API HTTP error %d: %s",
                exc.response.status_code,
                exc.response.text[:200],
            )
            return []
        except httpx.RequestError as exc:
            logger.error("GDELT API request error: %s", exc)
            return []
        except Exception as exc:
            logger.error("GDELT API unexpected error: %s", exc)
            return []

        articles: list[NewsArticle] = []
        article_list = data.get("articles", [])
        if not isinstance(article_list, list):
            return []

        for item in article_list:
            title = item.get("title", "")
            url = item.get("url", "")
            date_str = item.get("seendate", "")
            source_name = item.get("domain", item.get("source", ""))
            snippet = item.get("socialimage", "")  # GDELT doesn't provide a snippet
            # Use title as snippet fallback
            if not snippet:
                snippet = title

            articles.append(
                NewsArticle(
                    title=title,
                    url=url,
                    published_at=_parse_datetime(date_str),
                    source=f"gdelt:{source_name}",
                    snippet=snippet,
                )
            )

        return articles

    async def search_newsapi(
        self,
        keywords: list[str],
        start_time: datetime,
        end_time: datetime,
    ) -> list[NewsArticle]:
        """Search NewsAPI for matching articles.

        Uses the /everything endpoint for broad article search. Requires
        a valid API key (100 requests/day on free tier).

        Args:
            keywords: Search terms to combine with AND logic.
            start_time: Start of the search window (UTC).
            end_time: End of the search window (UTC).

        Returns:
            List of NewsArticle objects from NewsAPI results.
        """
        if not self._newsapi_key:
            logger.debug("Skipping NewsAPI search: no API key configured.")
            return []

        if not keywords:
            return []

        # Ensure timezone awareness
        if start_time.tzinfo is None:
            start_time = start_time.replace(tzinfo=timezone.utc)
        if end_time.tzinfo is None:
            end_time = end_time.replace(tzinfo=timezone.utc)

        # NewsAPI uses AND by quoting phrases; combine with AND
        # For multi-word terms, quote them; single words left as-is
        query_parts: list[str] = []
        for kw in keywords[:5]:  # Limit to 5 keywords to keep query reasonable
            if " " in kw:
                query_parts.append(f'"{kw}"')
            else:
                query_parts.append(kw)
        query = " AND ".join(query_parts)

        params: dict[str, Any] = {
            "q": query,
            "from": start_time.strftime("%Y-%m-%dT%H:%M:%S"),
            "to": end_time.strftime("%Y-%m-%dT%H:%M:%S"),
            "sortBy": "publishedAt",
            "pageSize": 50,
            "language": "en",
        }

        try:
            resp = await get_with_retry(
                self._client,
                f"{self._newsapi_base_url}/everything",
                params=params,
                headers={"X-Api-Key": self._newsapi_key},
            )
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if status == 426:
                logger.warning(
                    "NewsAPI returned 426 (Upgrade Required) -- free tier "
                    "does not support searching articles older than ~1 month."
                )
            elif status == 429:
                logger.warning("NewsAPI rate limit exceeded (100/day on free tier).")
            else:
                logger.error(
                    "NewsAPI HTTP error %d: %s",
                    status,
                    exc.response.text[:200],
                )
            return []
        except httpx.RequestError as exc:
            logger.error("NewsAPI request error: %s", exc)
            return []
        except Exception as exc:
            logger.error("NewsAPI unexpected error: %s", exc)
            return []

        if data.get("status") != "ok":
            error_msg = data.get("message", "unknown error")
            logger.error("NewsAPI returned non-ok status: %s", error_msg)
            return []

        articles: list[NewsArticle] = []
        article_list = data.get("articles", [])
        if not isinstance(article_list, list):
            return []

        for item in article_list:
            title = item.get("title", "") or ""
            url = item.get("url", "") or ""
            published_at = item.get("publishedAt", "")
            source_info = item.get("source", {})
            source_name = (
                source_info.get("name", "")
                if isinstance(source_info, dict)
                else str(source_info)
            )
            description = item.get("description", "") or ""
            content = item.get("content", "") or ""
            snippet = description if description else content[:300]

            articles.append(
                NewsArticle(
                    title=title,
                    url=url,
                    published_at=_parse_datetime(published_at),
                    source=f"newsapi:{source_name}",
                    snippet=snippet,
                )
            )

        return articles
