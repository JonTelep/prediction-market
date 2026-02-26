"""Core anomaly report dataclass for surveillance findings."""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class AnomalyReport:
    """A single anomaly detected by a surveillance agent.

    Captures all evidence — price movements, volume spikes, calendar
    correlations, and news cross-references — into a self-contained
    record suitable for persistence, formatting, and dispatch.
    """

    id: str
    agent: str  # 'info_leak' or 'manipulation'
    market_id: str
    market_question: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    anomaly_score: float
    confidence: float  # 0.0-1.0
    summary: str
    details: dict[str, Any] = field(default_factory=dict)
    price_evidence: dict[str, Any] = field(default_factory=dict)
    volume_evidence: dict[str, Any] = field(default_factory=dict)
    calendar_matches: list[dict[str, Any]] = field(default_factory=list)
    news_check: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @staticmethod
    def severity_from_score(score: float) -> str:
        """Map a numeric anomaly score to a severity label.

        Score ranges:
            < 0.4  -> low
            < 0.7  -> medium
            < 0.9  -> high
            >= 0.9 -> critical
        """
        if score < 0.4:
            return "low"
        if score < 0.7:
            return "medium"
        if score < 0.9:
            return "high"
        return "critical"

    @staticmethod
    def new_id() -> str:
        """Generate a fresh UUID-4 string for a report."""
        return str(uuid.uuid4())

    # -- Serialisation --------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Convert the report to a plain dict (JSON-safe)."""
        data = asdict(self)
        data["created_at"] = self.created_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AnomalyReport:
        """Reconstruct a report from a plain dict."""
        data = data.copy()
        created_raw = data.get("created_at")
        if isinstance(created_raw, str):
            data["created_at"] = datetime.fromisoformat(created_raw)
        elif not isinstance(created_raw, datetime):
            data["created_at"] = datetime.now(timezone.utc)
        return cls(**data)

    def to_json(self, indent: int = 2) -> str:
        """Shortcut: serialise directly to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)
