"""Format AnomalyReport instances as human-readable Markdown.

Output is designed for journalists, compliance teams, and analysts
who need to triage alerts quickly without parsing raw JSON.
"""

from __future__ import annotations

from prediction_market.reporting.anomaly_report import AnomalyReport

_SEVERITY_INDICATORS: dict[str, str] = {
    "low": "[LOW]",
    "medium": "[MEDIUM] (!)",
    "high": "[HIGH] (!!)",
    "critical": "[CRITICAL] (!!!)",
}

_CONFIDENCE_LABELS: dict[str, str] = {
    "very_low": "Very low (< 0.25)",
    "low": "Low (0.25 - 0.50)",
    "moderate": "Moderate (0.50 - 0.75)",
    "high": "High (> 0.75)",
}


def _confidence_label(confidence: float) -> str:
    if confidence < 0.25:
        return _CONFIDENCE_LABELS["very_low"]
    if confidence < 0.50:
        return _CONFIDENCE_LABELS["low"]
    if confidence < 0.75:
        return _CONFIDENCE_LABELS["moderate"]
    return _CONFIDENCE_LABELS["high"]


def _format_dict_section(title: str, data: dict, depth: int = 0) -> str:
    """Render a dict as a bulleted list under a heading."""
    if not data:
        return ""
    indent = "  " * depth
    lines = [f"{indent}### {title}", ""]
    for key, value in data.items():
        if isinstance(value, dict):
            lines.append(f"{indent}- **{key}**:")
            for k2, v2 in value.items():
                lines.append(f"{indent}  - {k2}: {v2}")
        elif isinstance(value, list):
            lines.append(f"{indent}- **{key}**: {', '.join(str(v) for v in value)}")
        else:
            lines.append(f"{indent}- **{key}**: {value}")
    lines.append("")
    return "\n".join(lines)


def _format_calendar_section(matches: list[dict]) -> str:
    """Render calendar matches as a Markdown table."""
    if not matches:
        return ""
    lines = [
        "### Calendar Matches",
        "",
        "| Source | Event | Date | Relevance |",
        "|--------|-------|------|-----------|",
    ]
    for m in matches:
        source = m.get("source", "unknown")
        title = m.get("title", "N/A")
        date = m.get("event_date", m.get("date", "N/A"))
        relevance = m.get("relevance", m.get("score", "N/A"))
        lines.append(f"| {source} | {title} | {date} | {relevance} |")
    lines.append("")
    return "\n".join(lines)


def format_report(report: AnomalyReport) -> str:
    """Render a single anomaly report as Markdown."""
    severity_indicator = _SEVERITY_INDICATORS.get(report.severity, report.severity.upper())
    created = report.created_at.strftime("%Y-%m-%d %H:%M:%S UTC")

    sections: list[str] = []

    # Header
    sections.append(f"# Anomaly Report {severity_indicator}")
    sections.append("")
    sections.append(f"**ID:** `{report.id}`  ")
    sections.append(f"**Agent:** {report.agent}  ")
    sections.append(f"**Created:** {created}  ")
    sections.append("")

    # Market info
    sections.append("## Market")
    sections.append("")
    sections.append(f"- **Market ID:** `{report.market_id}`")
    sections.append(f"- **Question:** {report.market_question}")
    sections.append("")

    # Score & confidence
    sections.append("## Assessment")
    sections.append("")
    sections.append(f"- **Severity:** {report.severity.upper()}")
    sections.append(f"- **Anomaly Score:** {report.anomaly_score:.3f}")
    sections.append(
        f"- **Confidence:** {report.confidence:.1%} -- {_confidence_label(report.confidence)}"
    )
    sections.append("")

    # Summary
    sections.append("## Summary")
    sections.append("")
    sections.append(report.summary)
    sections.append("")

    # Evidence sections
    price_section = _format_dict_section("Price Evidence", report.price_evidence)
    if price_section:
        sections.append(price_section)

    volume_section = _format_dict_section("Volume Evidence", report.volume_evidence)
    if volume_section:
        sections.append(volume_section)

    calendar_section = _format_calendar_section(report.calendar_matches)
    if calendar_section:
        sections.append(calendar_section)

    news_section = _format_dict_section("News Cross-Reference", report.news_check)
    if news_section:
        sections.append(news_section)

    # Full details (collapsed for brevity)
    if report.details:
        sections.append("### Details")
        sections.append("")
        sections.append("<details>")
        sections.append("<summary>Full payload (click to expand)</summary>")
        sections.append("")
        sections.append("```json")
        import json

        sections.append(json.dumps(report.details, indent=2, default=str))
        sections.append("```")
        sections.append("")
        sections.append("</details>")
        sections.append("")

    # Confidence note
    sections.append("---")
    sections.append("")
    if report.confidence < 0.5:
        sections.append(
            "*Note: Confidence is below 50%. This alert may be a false positive "
            "and should be verified manually before escalation.*"
        )
    elif report.confidence >= 0.75:
        sections.append(
            "*High-confidence alert. Evidence strongly suggests anomalous activity. "
            "Recommend immediate review.*"
        )
    else:
        sections.append(
            "*Moderate-confidence alert. Review supporting evidence before drawing conclusions.*"
        )
    sections.append("")

    return "\n".join(sections)
