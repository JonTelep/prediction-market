"""Integration tests for report formatters and sinks."""

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock

import httpx
import pytest
import respx

from prediction_market.reporting.anomaly_report import AnomalyReport
from prediction_market.reporting.human_formatter import format_report as format_md
from prediction_market.reporting.json_formatter import format_report as format_json
from prediction_market.reporting.json_formatter import format_reports
from prediction_market.reporting.sink import (
    CompositeSink,
    FileSink,
    StdoutSink,
    WebhookSink,
)


@pytest.fixture
def sample_report():
    return AnomalyReport(
        id="rpt-001",
        agent="info_leak",
        market_id="test-market-1",
        market_question="Will the bill pass?",
        severity="high",
        anomaly_score=5.5,
        confidence=0.85,
        summary="Unusual price spike detected.",
        details={"price_z": 3.2, "volume_z": 2.8},
        price_evidence={"before": 0.55, "after": 0.72},
        volume_evidence={"volume_24h": 150000},
        calendar_matches=[
            {"source": "congress", "title": "Committee vote", "date": "2026-02-21"}
        ],
        news_check={"news_found": False},
        created_at=datetime(2026, 2, 20, 12, 0, 0, tzinfo=timezone.utc),
    )


# -- JSON Formatter --


def test_json_format_single(sample_report):
    output = format_json(sample_report)
    parsed = json.loads(output)
    assert parsed["id"] == "rpt-001"
    assert parsed["severity"] == "high"
    assert parsed["anomaly_score"] == 5.5


def test_json_format_multiple(sample_report):
    output = format_reports([sample_report, sample_report])
    parsed = json.loads(output)
    assert parsed["count"] == 2
    assert "generated_at" in parsed
    assert len(parsed["reports"]) == 2


# -- Markdown Formatter --


def test_markdown_format_contains_header(sample_report):
    output = format_md(sample_report)
    assert "# Anomaly Report" in output
    assert "[HIGH]" in output


def test_markdown_format_contains_market(sample_report):
    output = format_md(sample_report)
    assert "test-market-1" in output
    assert "Will the bill pass?" in output


def test_markdown_format_contains_assessment(sample_report):
    output = format_md(sample_report)
    assert "5.500" in output
    assert "85.0%" in output
    assert "High" in output


def test_markdown_format_contains_evidence(sample_report):
    output = format_md(sample_report)
    assert "Price Evidence" in output
    assert "Volume Evidence" in output
    assert "Calendar Matches" in output


def test_markdown_format_contains_calendar_table(sample_report):
    output = format_md(sample_report)
    assert "| congress |" in output
    assert "Committee vote" in output


def test_markdown_format_details_collapsed(sample_report):
    output = format_md(sample_report)
    assert "<details>" in output
    assert "price_z" in output


def test_markdown_format_high_confidence_note(sample_report):
    output = format_md(sample_report)
    assert "High-confidence alert" in output


def test_markdown_format_low_confidence_note():
    report = AnomalyReport(
        id="rpt-low",
        agent="info_leak",
        market_id="m1",
        market_question="Q?",
        severity="low",
        anomaly_score=1.0,
        confidence=0.2,
        summary="Test",
        created_at=datetime(2026, 2, 20, 12, 0, 0, tzinfo=timezone.utc),
    )
    output = format_md(report)
    assert "false positive" in output


# -- FileSink --


@pytest.mark.asyncio
async def test_file_sink_writes_both_formats(tmp_path, sample_report):
    sink = FileSink(tmp_path / "reports")
    await sink.write(sample_report)

    json_path = tmp_path / "reports" / "rpt-001.json"
    md_path = tmp_path / "reports" / "rpt-001.md"
    assert json_path.exists()
    assert md_path.exists()

    # Verify JSON is valid
    parsed = json.loads(json_path.read_text())
    assert parsed["id"] == "rpt-001"

    # Verify Markdown has content
    md_content = md_path.read_text()
    assert "# Anomaly Report" in md_content


@pytest.mark.asyncio
async def test_file_sink_creates_directory(tmp_path, sample_report):
    deep_path = tmp_path / "a" / "b" / "c"
    sink = FileSink(deep_path)
    await sink.write(sample_report)
    assert (deep_path / "rpt-001.json").exists()


# -- StdoutSink --


@pytest.mark.asyncio
async def test_stdout_sink_markdown(sample_report, capsys):
    sink = StdoutSink(use_json=False)
    await sink.write(sample_report)
    captured = capsys.readouterr()
    assert "# Anomaly Report" in captured.out


@pytest.mark.asyncio
async def test_stdout_sink_json(sample_report, capsys):
    sink = StdoutSink(use_json=True)
    await sink.write(sample_report)
    captured = capsys.readouterr()
    parsed = json.loads(captured.out)
    assert parsed["id"] == "rpt-001"


# -- WebhookSink --


@pytest.mark.asyncio
@respx.mock
async def test_webhook_sink_success(sample_report):
    route = respx.post("https://hooks.example.com/test").mock(
        return_value=httpx.Response(200)
    )
    sink = WebhookSink("https://hooks.example.com/test")
    try:
        await sink.write(sample_report)
        assert route.called
    finally:
        await sink.close()


@pytest.mark.asyncio
@respx.mock
async def test_webhook_sink_retries_on_failure(sample_report):
    route = respx.post("https://hooks.example.com/test")
    route.side_effect = [
        httpx.Response(500),
        httpx.Response(500),
        httpx.Response(200),
    ]
    sink = WebhookSink("https://hooks.example.com/test", max_retries=3)
    try:
        await sink.write(sample_report)
        assert route.call_count == 3
    finally:
        await sink.close()


# -- CompositeSink --


@pytest.mark.asyncio
async def test_composite_sink_fans_out(tmp_path, sample_report, capsys):
    file_sink = FileSink(tmp_path / "reports")
    stdout_sink = StdoutSink(use_json=False)
    composite = CompositeSink([file_sink, stdout_sink])

    await composite.write(sample_report)

    # File was written
    assert (tmp_path / "reports" / "rpt-001.json").exists()
    # Stdout got output
    captured = capsys.readouterr()
    assert "Anomaly Report" in captured.out


@pytest.mark.asyncio
async def test_composite_sink_swallows_errors(sample_report):
    failing_sink = AsyncMock(spec=FileSink)
    failing_sink.write = AsyncMock(side_effect=RuntimeError("disk full"))
    failing_sink.close = AsyncMock()

    good_sink = AsyncMock(spec=StdoutSink)
    good_sink.write = AsyncMock()
    good_sink.close = AsyncMock()

    composite = CompositeSink([failing_sink, good_sink])
    # Should not raise even though first sink fails
    await composite.write(sample_report)
    good_sink.write.assert_called_once_with(sample_report)
