"""Format AnomalyReport instances as structured JSON."""

from __future__ import annotations

import json
from typing import Any

from prediction_market.reporting.anomaly_report import AnomalyReport


def format_report(report: AnomalyReport) -> str:
    """Render a single report as pretty-printed JSON."""
    return json.dumps(report.to_dict(), indent=2, default=str)


def format_reports(reports: list[AnomalyReport]) -> str:
    """Render a list of reports as a JSON array.

    The wrapper object includes a count and an ISO-8601 generation
    timestamp so downstream consumers can quickly triage bulk output.
    """
    from datetime import datetime, timezone

    payload: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "count": len(reports),
        "reports": [r.to_dict() for r in reports],
    }
    return json.dumps(payload, indent=2, default=str)
