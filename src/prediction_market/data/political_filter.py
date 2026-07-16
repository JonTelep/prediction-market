"""Classify markets as political based on tags, keywords, and categories."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from prediction_market.config import load_political_keywords
from prediction_market.data.polymarket.models import GammaMarket

logger = logging.getLogger(__name__)


@dataclass
class PoliticalClassification:
    """Result of political classification."""

    is_political: bool
    confidence: float  # 0.0-1.0
    reasons: list[str] = field(default_factory=list)


class PoliticalFilter:
    """Classifies Polymarket markets as political or not."""

    def __init__(self, keywords_config: dict | None = None) -> None:
        config = keywords_config or load_political_keywords()
        classification = config.get("classification", {})
        self._political_tags = {t.lower() for t in classification.get("political_tags", [])}
        self._title_keywords = {k.lower() for k in classification.get("title_keywords", [])}
        self._political_categories = {c.lower() for c in classification.get("political_categories", [])}
        self._min_volume = classification.get("min_volume_usd", 10000)

    @property
    def min_volume(self) -> float:
        """The minimum USD volume threshold used by the volume gate."""
        return self._min_volume

    def classify(self, market: GammaMarket) -> PoliticalClassification:
        """Classify a market as political or not."""
        reasons: list[str] = []
        score = 0.0

        # Check tags
        market_tags = {t.lower() for t in market.tag_labels}
        matching_tags = market_tags & self._political_tags
        if matching_tags:
            score += 0.4
            reasons.append(f"matching tags: {', '.join(matching_tags)}")

        # Check category
        if market.category.lower() in self._political_categories:
            score += 0.3
            reasons.append(f"political category: {market.category}")

        # Check title keywords
        title_lower = market.question.lower()
        desc_lower = market.description.lower() if market.description else ""
        matching_keywords = []
        for keyword in self._title_keywords:
            if keyword in title_lower or keyword in desc_lower:
                matching_keywords.append(keyword)
        if matching_keywords:
            keyword_score = min(0.3, len(matching_keywords) * 0.1)
            score += keyword_score
            reasons.append(f"keywords: {', '.join(matching_keywords[:5])}")

        confidence = min(1.0, score)
        is_political = confidence >= 0.3

        # Volume filter — still classified as political but flagged
        if is_political and market.volume < self._min_volume:
            reasons.append(f"low volume ({market.volume:.0f} < {self._min_volume})")

        return PoliticalClassification(
            is_political=is_political,
            confidence=confidence,
            reasons=reasons,
        )

    def filter_political(
        self, markets: list[GammaMarket], min_volume: bool = True
    ) -> list[GammaMarket]:
        """Filter a list of markets to only political ones."""
        result = []
        for market in markets:
            classification = self.classify(market)
            if classification.is_political:
                if min_volume and market.volume < self._min_volume:
                    continue
                result.append(market)
        return result
