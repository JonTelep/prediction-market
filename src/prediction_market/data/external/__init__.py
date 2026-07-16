"""External data source integrations for market surveillance.

Phase 5: Congress.gov, CourtListener, White House, GDELT, and NewsAPI
clients that provide event context and news verification for detecting
information leakage in prediction markets.
"""

from prediction_market.data.external.congress import CongressClient
from prediction_market.data.external.court_calendar import CourtCalendarClient
from prediction_market.data.external.models import (
    NewsArticle,
    NewsCheckResult,
    ScheduledEvent,
)
from prediction_market.data.external.news_checker import NewsChecker
from prediction_market.data.external.white_house import WhiteHouseClient

__all__ = [
    "CongressClient",
    "CourtCalendarClient",
    "NewsArticle",
    "NewsCheckResult",
    "NewsChecker",
    "ScheduledEvent",
    "WhiteHouseClient",
]
