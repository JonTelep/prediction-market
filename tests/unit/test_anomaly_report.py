"""Unit tests for AnomalyReport dataclass."""

import json
from datetime import datetime, timezone


from prediction_market.reporting.anomaly_report import AnomalyReport


def _make_report(**overrides):
    defaults = {
        "id": "test-report-1",
        "agent": "info_leak",
        "market_id": "test-market-1",
        "market_question": "Will the bill pass?",
        "severity": "high",
        "anomaly_score": 5.5,
        "confidence": 0.85,
        "summary": "Unusual price spike detected before news publication.",
        "details": {"price_z": 3.2, "volume_z": 2.8},
        "price_evidence": {"before": 0.55, "after": 0.72, "change_pct": 30.9},
        "volume_evidence": {"volume_24h": 150000, "avg_24h": 50000},
        "calendar_matches": [
            {"source": "congress", "title": "Committee vote", "date": "2026-02-21"}
        ],
        "news_check": {"news_found": False, "articles": []},
        "created_at": datetime(2026, 2, 20, 12, 0, 0, tzinfo=timezone.utc),
    }
    defaults.update(overrides)
    return AnomalyReport(**defaults)


def test_creation():
    report = _make_report()
    assert report.id == "test-report-1"
    assert report.agent == "info_leak"
    assert report.severity == "high"
    assert report.confidence == 0.85


def test_severity_from_score():
    assert AnomalyReport.severity_from_score(0.1) == "low"
    assert AnomalyReport.severity_from_score(0.39) == "low"
    assert AnomalyReport.severity_from_score(0.4) == "medium"
    assert AnomalyReport.severity_from_score(0.69) == "medium"
    assert AnomalyReport.severity_from_score(0.7) == "high"
    assert AnomalyReport.severity_from_score(0.89) == "high"
    assert AnomalyReport.severity_from_score(0.9) == "critical"
    assert AnomalyReport.severity_from_score(1.0) == "critical"


def test_new_id():
    id1 = AnomalyReport.new_id()
    id2 = AnomalyReport.new_id()
    assert id1 != id2
    assert len(id1) == 36  # UUID-4 format


def test_to_dict():
    report = _make_report()
    d = report.to_dict()
    assert d["id"] == "test-report-1"
    assert d["agent"] == "info_leak"
    assert d["anomaly_score"] == 5.5
    assert isinstance(d["created_at"], str)
    assert "2026-02-20" in d["created_at"]
    assert d["price_evidence"]["before"] == 0.55
    assert d["calendar_matches"][0]["source"] == "congress"


def test_from_dict():
    report = _make_report()
    d = report.to_dict()
    restored = AnomalyReport.from_dict(d)
    assert restored.id == report.id
    assert restored.agent == report.agent
    assert restored.anomaly_score == report.anomaly_score
    assert restored.confidence == report.confidence
    assert restored.summary == report.summary
    assert isinstance(restored.created_at, datetime)


def test_to_dict_from_dict_roundtrip():
    report = _make_report()
    roundtripped = AnomalyReport.from_dict(report.to_dict())
    assert roundtripped.to_dict() == report.to_dict()


def test_to_json():
    report = _make_report()
    json_str = report.to_json()
    parsed = json.loads(json_str)
    assert parsed["id"] == "test-report-1"
    assert parsed["anomaly_score"] == 5.5
    assert parsed["details"]["price_z"] == 3.2


def test_to_json_is_valid_json():
    report = _make_report()
    json_str = report.to_json()
    # Should not raise
    parsed = json.loads(json_str)
    assert isinstance(parsed, dict)


def test_default_created_at():
    report = AnomalyReport(
        id="r1",
        agent="manipulation",
        market_id="m1",
        market_question="Q?",
        severity="low",
        anomaly_score=0.2,
        confidence=0.3,
        summary="Test",
    )
    assert report.created_at.tzinfo is not None
    # Should be recent (within last minute)
    diff = datetime.now(timezone.utc) - report.created_at
    assert diff.total_seconds() < 60


def test_from_dict_with_missing_created_at():
    d = {
        "id": "r1",
        "agent": "info_leak",
        "market_id": "m1",
        "market_question": "Q?",
        "severity": "medium",
        "anomaly_score": 0.5,
        "confidence": 0.5,
        "summary": "Test",
        "created_at": None,
    }
    report = AnomalyReport.from_dict(d)
    assert isinstance(report.created_at, datetime)
