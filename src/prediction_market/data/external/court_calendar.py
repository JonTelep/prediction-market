"""CourtListener API integration for federal court event tracking.

Fetches upcoming oral arguments, recent opinions, and docket searches
from the CourtListener API (courtlistener.com/api/rest/v4). These
events are used to detect whether prediction market movements coincide
with publicly scheduled or recently decided court activity.

API docs: https://www.courtlistener.com/help/api/rest/
Free auth token required: https://www.courtlistener.com/sign-up/
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

from prediction_market.config import AppConfig
from prediction_market.data.external.models import ScheduledEvent
from prediction_market.data.retry import get_with_retry

logger = logging.getLogger(__name__)


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
    """Parse a datetime string from the CourtListener API."""
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


class CourtCalendarClient:
    """Async client for the CourtListener REST API (v4).

    Retrieves upcoming oral arguments, recent opinions, and docket
    information to provide judicial-event context for market surveillance.

    The client gracefully handles a missing or empty auth token by
    logging a warning and returning empty results for all queries.
    """

    def __init__(
        self,
        config: AppConfig,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.config = config
        self.base_url = config.apis.courtlistener_base_url.rstrip("/")
        self._token = config.courtlistener_token
        self._client = http_client or httpx.AsyncClient(timeout=30.0)
        self._owns_client = http_client is None

        if not self._token:
            logger.warning(
                "CourtListener auth token is not configured. "
                "CourtCalendarClient will return empty results for all queries. "
                "Set COURTLISTENER_TOKEN in your environment to enable."
            )

    async def close(self) -> None:
        """Close the underlying HTTP client if we own it."""
        if self._owns_client:
            await self._client.aclose()

    def _auth_headers(self) -> dict[str, str]:
        """Build authorization headers."""
        if not self._token:
            return {}
        return {"Authorization": f"Token {self._token}"}

    async def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        """Make an authenticated GET request to the CourtListener API.

        Returns the parsed JSON response, or None on error.
        """
        if not self._token:
            return None

        url = f"{self.base_url}{path}"
        try:
            resp = await get_with_retry(
                self._client,
                url,
                params=params,
                headers=self._auth_headers(),
            )
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as exc:
            logger.error(
                "CourtListener API HTTP error %d for %s: %s",
                exc.response.status_code,
                path,
                exc.response.text[:200],
            )
            return None
        except httpx.RequestError as exc:
            logger.error("CourtListener API request error for %s: %s", path, exc)
            return None

    async def get_upcoming_arguments(
        self, days_ahead: int = 14
    ) -> list[ScheduledEvent]:
        """Fetch oral arguments scheduled in the next N days.

        Args:
            days_ahead: Number of days to look ahead for arguments.

        Returns:
            List of ScheduledEvent objects for upcoming oral arguments.
        """
        if not self._token:
            return []

        now = datetime.now(timezone.utc)
        date_after = now.strftime("%Y-%m-%d")
        date_before = (now + timedelta(days=days_ahead)).strftime("%Y-%m-%d")

        data = await self._get(
            "/oral-arguments/",
            params={
                "date_argued__gte": date_after,
                "date_argued__lte": date_before,
                "order_by": "date_argued",
                "page_size": 50,
            },
        )
        if not data:
            return []

        events: list[ScheduledEvent] = []
        results = data.get("results", [])
        if isinstance(results, list):
            for item in results:
                case_name = item.get("case_name", "") or item.get(
                    "case_name_short", ""
                )
                docket_number = item.get("docket", {})
                date_argued = item.get("date_argued", "")

                description = f"Oral argument: {case_name}"
                if isinstance(docket_number, dict):
                    dnum = docket_number.get("docket_number", "")
                    if dnum:
                        description += f" (Docket {dnum})"

                absolute_url = item.get("absolute_url", "")
                url = (
                    f"https://www.courtlistener.com{absolute_url}"
                    if absolute_url
                    else ""
                )

                events.append(
                    ScheduledEvent(
                        source="courtlistener",
                        event_type="oral_argument",
                        title=case_name,
                        description=description,
                        event_date=_parse_datetime(date_argued),
                        url=url,
                        keywords=_extract_keywords(case_name),
                    )
                )

        return events

    async def get_recent_opinions(self, days: int = 7) -> list[ScheduledEvent]:
        """Fetch opinions filed in the last N days.

        Args:
            days: Number of days to look back for recent opinions.

        Returns:
            List of ScheduledEvent objects for recent court opinions.
        """
        if not self._token:
            return []

        now = datetime.now(timezone.utc)
        date_after = (now - timedelta(days=days)).strftime("%Y-%m-%d")

        data = await self._get(
            "/opinions/",
            params={
                "date_created__gte": date_after,
                "order_by": "-date_created",
                "page_size": 50,
            },
        )
        if not data:
            return []

        events: list[ScheduledEvent] = []
        results = data.get("results", [])
        if isinstance(results, list):
            for item in results:
                # Opinions may have a cluster with case name
                cluster = item.get("cluster", "")
                case_name = ""
                opinion_url = ""

                if isinstance(cluster, dict):
                    case_name = cluster.get("case_name", "")
                    opinion_url = cluster.get("absolute_url", "")
                elif isinstance(cluster, str):
                    # cluster might be a URL reference
                    opinion_url = cluster

                if not case_name:
                    case_name = item.get("case_name", "Unknown Case")

                absolute_url = item.get("absolute_url", opinion_url)
                url = (
                    f"https://www.courtlistener.com{absolute_url}"
                    if absolute_url and not absolute_url.startswith("http")
                    else absolute_url or ""
                )

                date_created = item.get("date_created", "")
                court_id = item.get("court", "")
                opinion_type = item.get("type", "")

                description = f"Opinion ({opinion_type}): {case_name}"
                if court_id:
                    description += f" [{court_id}]"

                events.append(
                    ScheduledEvent(
                        source="courtlistener",
                        event_type="opinion",
                        title=case_name,
                        description=description,
                        event_date=_parse_datetime(date_created),
                        url=url,
                        keywords=_extract_keywords(case_name),
                    )
                )

        return events

    async def search_dockets(self, query: str) -> list[ScheduledEvent]:
        """Search dockets by a text query.

        Args:
            query: Free-text search query (e.g., case name, topic).

        Returns:
            List of ScheduledEvent objects for matching dockets.
        """
        if not self._token:
            return []

        if not query or not query.strip():
            return []

        data = await self._get(
            "/dockets/",
            params={
                "q": query.strip(),
                "order_by": "-date_modified",
                "page_size": 30,
            },
        )
        if not data:
            return []

        events: list[ScheduledEvent] = []
        results = data.get("results", [])
        if isinstance(results, list):
            for item in results:
                case_name = item.get("case_name", "") or item.get(
                    "case_name_short", ""
                )
                docket_number = item.get("docket_number", "")
                court = item.get("court", "")
                date_filed = item.get("date_filed", "")
                date_modified = item.get("date_modified", "")

                description = f"Docket {docket_number}: {case_name}"
                if court:
                    description += f" [{court}]"

                absolute_url = item.get("absolute_url", "")
                url = (
                    f"https://www.courtlistener.com{absolute_url}"
                    if absolute_url and not absolute_url.startswith("http")
                    else absolute_url or ""
                )

                events.append(
                    ScheduledEvent(
                        source="courtlistener",
                        event_type="docket",
                        title=case_name,
                        description=description,
                        event_date=_parse_datetime(date_modified or date_filed),
                        url=url,
                        keywords=_extract_keywords(case_name),
                    )
                )

        return events
