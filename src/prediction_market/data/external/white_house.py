"""White House schedule integration for executive branch event tracking.

Calls the public WordPress REST API on whitehouse.gov (not scraping HTML)
to retrieve schedule pages and presidential-action posts. These events are
used to detect whether prediction market movements coincide with publicly
announced executive activity.

No authentication required -- this uses public API endpoints.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

from prediction_market.config import AppConfig
from prediction_market.data.external.models import ScheduledEvent
from prediction_market.data.retry import get_with_retry

logger = logging.getLogger(__name__)

_WH_BASE_URL = "https://www.whitehouse.gov"

# URLs for public White House data. These may change with administrations;
# the client handles 404s and other errors gracefully.
_SCHEDULE_URL = f"{_WH_BASE_URL}/wp-json/wp/v2/pages"
_ACTIONS_URL = f"{_WH_BASE_URL}/wp-json/wp/v2/posts"


def _extract_keywords(text: str) -> list[str]:
    """Extract simple keywords from a text string."""
    stopwords = {
        "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
        "has", "have", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "not", "no", "this", "that",
        "these", "those", "it", "its", "as", "if", "so", "than", "then",
        "also", "very", "just", "about", "into", "over", "such", "only",
        "other", "new", "some", "any", "all", "each", "every", "both", "few",
        "more", "most", "own", "same", "up", "out",
    }
    words = text.split()
    keywords: list[str] = []
    for word in words:
        cleaned = word.strip(".,;:!?()[]\"'")
        if len(cleaned) > 2 and cleaned.lower() not in stopwords:
            keywords.append(cleaned)
    seen: set[str] = set()
    unique: list[str] = []
    for kw in keywords:
        lower = kw.lower()
        if lower not in seen:
            seen.add(lower)
            unique.append(kw)
    return unique[:20]


def _parse_datetime(value: str | None) -> datetime:
    """Parse a datetime string from the WordPress REST API."""
    if not value:
        return datetime.now(timezone.utc)
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, AttributeError):
        pass
    for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(value, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            continue
    return datetime.now(timezone.utc)


def _strip_html(html: str) -> str:
    """Remove HTML tags from a string, producing plain text."""
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _classify_action_type(title: str, content: str) -> str:
    """Classify a White House post into a specific action type."""
    combined = (title + " " + content).lower()
    if "executive order" in combined:
        return "executive_order"
    if "proclamation" in combined:
        return "proclamation"
    if "memorandum" in combined or "memo" in combined:
        return "memorandum"
    if "statement" in combined or "remarks" in combined:
        return "statement"
    if "press briefing" in combined or "press conference" in combined:
        return "briefing"
    if "nomination" in combined or "appoint" in combined:
        return "nomination"
    if "veto" in combined:
        return "veto"
    if "sign" in combined and ("bill" in combined or "act" in combined or "law" in combined):
        return "bill_signing"
    return "presidential_action"


class WhiteHouseClient:
    """Async client for White House public schedule and actions.

    Calls the whitehouse.gov WordPress REST API (not HTML scraping) to
    retrieve publicly posted schedule items and presidential actions.

    No authentication is required. The client handles network errors,
    rate limits, and site structure changes gracefully by returning
    empty results.
    """

    def __init__(
        self,
        config: AppConfig,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.config = config
        self._client = http_client or httpx.AsyncClient(
            timeout=30.0,
            follow_redirects=True,
            headers={
                "User-Agent": (
                    "PredictionMarketSurveillance/1.0 "
                    "(research; +https://github.com/prediction-market)"
                ),
                "Accept": "application/json",
            },
        )
        self._owns_client = http_client is None

    async def close(self) -> None:
        """Close the underlying HTTP client if we own it."""
        if self._owns_client:
            await self._client.aclose()

    async def _get_json(self, url: str, params: dict[str, Any] | None = None) -> Any:
        """Make a GET request and return parsed JSON, or None on error."""
        try:
            resp = await get_with_retry(self._client, url, params=params)
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as exc:
            logger.error(
                "White House HTTP error %d for %s: %s",
                exc.response.status_code,
                url,
                exc.response.text[:200],
            )
            return None
        except httpx.RequestError as exc:
            logger.error("White House request error for %s: %s", url, exc)
            return None
        except Exception as exc:
            logger.error("White House unexpected error for %s: %s", url, exc)
            return None

    async def get_schedule(self, days_ahead: int = 7) -> list[ScheduledEvent]:
        """Fetch the White House public schedule for the next N days.

        Uses the WordPress REST API to query for schedule-related posts
        and pages. Falls back gracefully if the endpoint structure changes.

        Args:
            days_ahead: Number of days to look ahead.

        Returns:
            List of ScheduledEvent objects for upcoming schedule items.
        """
        now = datetime.now(timezone.utc)
        after_date = now.strftime("%Y-%m-%dT%H:%M:%S")
        before_date = (now + timedelta(days=days_ahead)).strftime(
            "%Y-%m-%dT%H:%M:%S"
        )

        events: list[ScheduledEvent] = []

        # Try the WP REST API posts endpoint, filtered by schedule-related
        # categories/search terms. The exact category IDs vary by
        # administration, so we search by keyword.
        for search_term in ("schedule", "daily guidance", "press schedule"):
            data = await self._get_json(
                _ACTIONS_URL,
                params={
                    "search": search_term,
                    "after": after_date,
                    "before": before_date,
                    "per_page": 20,
                    "orderby": "date",
                    "order": "asc",
                },
            )
            if not data or not isinstance(data, list):
                continue

            for post in data:
                title_raw = post.get("title", {})
                title = (
                    _strip_html(title_raw.get("rendered", ""))
                    if isinstance(title_raw, dict)
                    else str(title_raw)
                )

                content_raw = post.get("content", {})
                content = (
                    _strip_html(content_raw.get("rendered", ""))
                    if isinstance(content_raw, dict)
                    else ""
                )

                # Truncate content for the description field
                description = content[:500] if content else title

                link = post.get("link", "")
                date_str = post.get("date_gmt", post.get("date", ""))

                events.append(
                    ScheduledEvent(
                        source="whitehouse.gov",
                        event_type="schedule",
                        title=title,
                        description=description,
                        event_date=_parse_datetime(date_str),
                        url=link,
                        keywords=_extract_keywords(title + " " + content[:200]),
                    )
                )

        # Deduplicate by URL
        seen_urls: set[str] = set()
        unique_events: list[ScheduledEvent] = []
        for event in events:
            if event.url and event.url in seen_urls:
                continue
            seen_urls.add(event.url)
            unique_events.append(event)

        return unique_events

    async def get_recent_actions(self, days: int = 7) -> list[ScheduledEvent]:
        """Fetch recent presidential actions and statements.

        Queries the White House WordPress REST API for recent posts
        about executive orders, statements, proclamations, and other
        presidential actions.

        Args:
            days: Number of days to look back.

        Returns:
            List of ScheduledEvent objects for recent presidential actions.
        """
        now = datetime.now(timezone.utc)
        after_date = (now - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%S")

        events: list[ScheduledEvent] = []

        # Search for different types of presidential actions
        action_searches = [
            "executive order",
            "presidential action",
            "proclamation",
            "memorandum",
            "statement",
            "nomination",
        ]

        seen_urls: set[str] = set()

        for search_term in action_searches:
            data = await self._get_json(
                _ACTIONS_URL,
                params={
                    "search": search_term,
                    "after": after_date,
                    "per_page": 10,
                    "orderby": "date",
                    "order": "desc",
                },
            )
            if not data or not isinstance(data, list):
                continue

            for post in data:
                link = post.get("link", "")
                if link in seen_urls:
                    continue
                seen_urls.add(link)

                title_raw = post.get("title", {})
                title = (
                    _strip_html(title_raw.get("rendered", ""))
                    if isinstance(title_raw, dict)
                    else str(title_raw)
                )

                content_raw = post.get("content", {})
                content = (
                    _strip_html(content_raw.get("rendered", ""))
                    if isinstance(content_raw, dict)
                    else ""
                )

                excerpt_raw = post.get("excerpt", {})
                excerpt = (
                    _strip_html(excerpt_raw.get("rendered", ""))
                    if isinstance(excerpt_raw, dict)
                    else ""
                )

                description = excerpt[:500] if excerpt else content[:500]
                date_str = post.get("date_gmt", post.get("date", ""))
                action_type = _classify_action_type(title, content[:500])

                events.append(
                    ScheduledEvent(
                        source="whitehouse.gov",
                        event_type=action_type,
                        title=title,
                        description=description,
                        event_date=_parse_datetime(date_str),
                        url=link,
                        keywords=_extract_keywords(title + " " + excerpt[:200]),
                    )
                )

        return events
