"""Shared data models for external data source integrations."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ScheduledEvent:
    """An event from an external calendar/schedule source.

    Used across Congress, court calendar, and White House integrations
    to represent upcoming or recent events that may correlate with
    prediction market movements.
    """

    source: str
    event_type: str
    title: str
    description: str
    event_date: datetime
    url: str
    keywords: list[str] = field(default_factory=list)


@dataclass
class NewsArticle:
    """A single news article returned from GDELT or NewsAPI."""

    title: str
    url: str
    published_at: datetime
    source: str
    snippet: str


@dataclass
class NewsCheckResult:
    """Result of checking whether news existed before a market move.

    The core question answered: did public information exist *before*
    the suspicious market activity, or did the move precede any
    publicly available news?
    """

    news_found: bool
    articles: list[NewsArticle] = field(default_factory=list)
    earliest_article_time: datetime | None = None
    query_keywords: list[str] = field(default_factory=list)
