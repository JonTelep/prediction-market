"""Congress.gov API integration for legislative event tracking.

Fetches upcoming hearings, scheduled votes, and recent bills from the
Congress.gov API (api.congress.gov/v3). These events are used to detect
whether prediction market movements coincide with publicly scheduled
legislative activity.

API docs: https://api.congress.gov/
Free API key required: https://api.congress.gov/sign-up/
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

from prediction_market.config import AppConfig
from prediction_market.data.external.models import ScheduledEvent

logger = logging.getLogger(__name__)


def _extract_keywords(text: str) -> list[str]:
    """Extract simple keywords from a text string.

    Pulls out capitalized multi-word phrases and significant single words,
    filtering common stopwords. This is intentionally simple -- downstream
    consumers can apply more sophisticated NLP if needed.
    """
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
    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for kw in keywords:
        lower = kw.lower()
        if lower not in seen:
            seen.add(lower)
            unique.append(kw)
    return unique[:20]


def _parse_datetime(value: str | None) -> datetime:
    """Parse a date or datetime string from the Congress API.

    The API returns dates in various formats: ISO-8601 with timezone,
    date-only strings, etc. This handles common cases and falls back
    to UTC now on failure.
    """
    if not value:
        return datetime.now(timezone.utc)
    # Try ISO-8601 first
    for fmt in (
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d",
    ):
        try:
            dt = datetime.strptime(value, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            continue
    # Last resort: try fromisoformat which handles many variants
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, AttributeError):
        return datetime.now(timezone.utc)


class CongressClient:
    """Async client for the Congress.gov API (v3).

    Retrieves upcoming hearings, scheduled votes, and recent bills
    to provide legislative-event context for market surveillance.

    The client gracefully handles a missing or empty API key by logging
    a warning and returning empty results for all queries.
    """

    def __init__(
        self,
        config: AppConfig,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.config = config
        self.base_url = config.apis.congress_base_url.rstrip("/")
        self._api_key = config.congress_api_key
        self._client = http_client or httpx.AsyncClient(timeout=30.0)
        self._owns_client = http_client is None

        if not self._api_key:
            logger.warning(
                "Congress API key is not configured. "
                "CongressClient will return empty results for all queries. "
                "Set CONGRESS_API_KEY in your environment to enable."
            )

    async def close(self) -> None:
        """Close the underlying HTTP client if we own it."""
        if self._owns_client:
            await self._client.aclose()

    async def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        """Make a GET request to the Congress API with the API key.

        Returns the parsed JSON response, or None on error.
        """
        if not self._api_key:
            return None

        url = f"{self.base_url}{path}"
        request_params = dict(params) if params else {}
        request_params["api_key"] = self._api_key
        request_params["format"] = "json"

        try:
            resp = await self._client.get(url, params=request_params)
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as exc:
            logger.error(
                "Congress API HTTP error %d for %s: %s",
                exc.response.status_code,
                path,
                exc.response.text[:200],
            )
            return None
        except httpx.RequestError as exc:
            logger.error("Congress API request error for %s: %s", path, exc)
            return None

    async def get_upcoming_hearings(self, days_ahead: int = 7) -> list[ScheduledEvent]:
        """Fetch congressional hearings scheduled in the next N days.

        Args:
            days_ahead: Number of days to look ahead for hearings.

        Returns:
            List of ScheduledEvent objects for upcoming hearings.
        """
        if not self._api_key:
            return []

        now = datetime.now(timezone.utc)
        from_date = now.strftime("%Y-%m-%dT00:00:00Z")
        to_date = (now + timedelta(days=days_ahead)).strftime("%Y-%m-%dT23:59:59Z")

        data = await self._get(
            "/committee-report",
            params={
                "fromDateTime": from_date,
                "toDateTime": to_date,
                "limit": 50,
            },
        )
        if not data:
            return []

        events: list[ScheduledEvent] = []
        reports = data.get("reports", data.get("committeeReports", []))
        if isinstance(reports, list):
            for report in reports:
                title = report.get("title", "") or report.get("description", "")
                chamber = report.get("chamber", "")
                citation = report.get("citation", "")
                description = f"{chamber} {citation}".strip() if chamber or citation else title
                url = report.get("url", "")
                event_date = _parse_datetime(
                    report.get("updateDate") or report.get("date")
                )
                events.append(
                    ScheduledEvent(
                        source="congress.gov",
                        event_type="hearing",
                        title=title,
                        description=description,
                        event_date=event_date,
                        url=url,
                        keywords=_extract_keywords(title),
                    )
                )

        # Also try the committee-meeting endpoint for hearings
        meetings_data = await self._get(
            "/committee-meeting",
            params={
                "fromDateTime": from_date,
                "toDateTime": to_date,
                "limit": 50,
            },
        )
        if meetings_data:
            meetings = meetings_data.get("committeeMeetings", [])
            if isinstance(meetings, list):
                for meeting in meetings:
                    title = meeting.get("title", "")
                    chamber = meeting.get("chamber", "")
                    description = f"{chamber} committee meeting".strip()
                    url = meeting.get("url", "")
                    event_date = _parse_datetime(meeting.get("date"))
                    events.append(
                        ScheduledEvent(
                            source="congress.gov",
                            event_type="hearing",
                            title=title,
                            description=description,
                            event_date=event_date,
                            url=url,
                            keywords=_extract_keywords(title),
                        )
                    )

        return events

    async def get_upcoming_votes(self, days_ahead: int = 7) -> list[ScheduledEvent]:
        """Fetch votes scheduled or expected in the next N days.

        Uses the amendments and actions endpoints to find recent and
        upcoming vote-related activity.

        Args:
            days_ahead: Number of days to look ahead.

        Returns:
            List of ScheduledEvent objects for upcoming votes.
        """
        if not self._api_key:
            return []

        now = datetime.now(timezone.utc)
        from_date = now.strftime("%Y-%m-%dT00:00:00Z")
        to_date = (now + timedelta(days=days_ahead)).strftime("%Y-%m-%dT23:59:59Z")

        # Congress API doesn't have a direct "upcoming votes" endpoint,
        # so we query recent amendments (which precede votes) as a proxy.
        data = await self._get(
            "/amendment",
            params={
                "fromDateTime": from_date,
                "toDateTime": to_date,
                "limit": 50,
            },
        )
        if not data:
            return []

        events: list[ScheduledEvent] = []
        amendments = data.get("amendments", [])
        if isinstance(amendments, list):
            for amendment in amendments:
                title = amendment.get("description", "") or amendment.get(
                    "purpose", ""
                )
                number = amendment.get("number", "")
                amendment_type = amendment.get("type", "")
                congress = amendment.get("congress", "")
                description = f"{amendment_type} {number} ({congress}th Congress)"
                url = amendment.get("url", "")
                event_date = _parse_datetime(
                    amendment.get("latestAction", {}).get("actionDate")
                    if isinstance(amendment.get("latestAction"), dict)
                    else amendment.get("updateDate")
                )
                events.append(
                    ScheduledEvent(
                        source="congress.gov",
                        event_type="vote",
                        title=title or f"Amendment {amendment_type}{number}",
                        description=description,
                        event_date=event_date,
                        url=url,
                        keywords=_extract_keywords(title),
                    )
                )

        return events

    async def get_recent_bills(self, limit: int = 50) -> list[ScheduledEvent]:
        """Fetch recently introduced or updated bills.

        Args:
            limit: Maximum number of bills to return.

        Returns:
            List of ScheduledEvent objects for recent bills.
        """
        if not self._api_key:
            return []

        data = await self._get(
            "/bill",
            params={
                "limit": min(limit, 250),
                "sort": "updateDate+desc",
            },
        )
        if not data:
            return []

        events: list[ScheduledEvent] = []
        bills = data.get("bills", [])
        if isinstance(bills, list):
            for bill in bills:
                title = bill.get("title", "")
                bill_type = bill.get("type", "")
                number = bill.get("number", "")
                congress = bill.get("congress", "")
                latest_action = bill.get("latestAction", {})
                action_text = (
                    latest_action.get("text", "")
                    if isinstance(latest_action, dict)
                    else ""
                )
                description = (
                    f"{bill_type}{number} ({congress}th Congress): {action_text}"
                ).strip()
                url = bill.get("url", "")
                event_date = _parse_datetime(
                    latest_action.get("actionDate")
                    if isinstance(latest_action, dict)
                    else bill.get("updateDate")
                )
                events.append(
                    ScheduledEvent(
                        source="congress.gov",
                        event_type="bill",
                        title=title or f"{bill_type}{number}",
                        description=description,
                        event_date=event_date,
                        url=url,
                        keywords=_extract_keywords(title),
                    )
                )

        return events
