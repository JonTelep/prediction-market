"""Tests for backtest.metrics: scoring a replay against a case's label."""

from __future__ import annotations

import pytest

from prediction_market.backtest.case_format import Case, CaseLabel
from prediction_market.backtest.metrics import evaluate_replay
from prediction_market.backtest.replay import ReplayResult
from prediction_market.data.polymarket.models import GammaMarket
from prediction_market.reporting.anomaly_report import AnomalyReport

WINDOW_START = "2026-06-02 00:00:00"
WINDOW_END = "2026-06-02 02:00:00"
EVENT_TIME = "2026-06-02 01:00:00"


def _case(*, labeled: bool = True, slug: str = "test-case") -> Case:
    label = (
        CaseLabel(
            window_start=WINDOW_START, window_end=WINDOW_END, event_time=EVENT_TIME
        )
        if labeled
        else None
    )
    return Case(
        slug=slug,
        market_id="m1",
        condition_id="c1",
        question="Will X happen?",
        archived_at="2026-06-05 00:00:00",
        notes="",
        market=GammaMarket(id="m1"),
        snapshots=[],
        trades=[],
        events=[],
        label=label,
    )


def _report(
    *, snapshot_timestamp: str, severity: str = "medium", summary: str = "anomaly"
) -> AnomalyReport:
    return AnomalyReport(
        id=AnomalyReport.new_id(),
        agent="info_leak",
        market_id="m1",
        market_question="Will X happen?",
        severity=severity,
        anomaly_score=0.75,
        confidence=0.8,
        summary=summary,
        details={"snapshot_timestamp": snapshot_timestamp},
    )


def _result(reports: list[AnomalyReport], case: Case) -> ReplayResult:
    return ReplayResult(reports=reports, steps=len(reports), case=case)


# ---------------------------------------------------------------------------
# Core scoring behavior
# ---------------------------------------------------------------------------


def test_hit_before_event_time_high_severity_detects_with_positive_lead_time():
    case = _case()
    report = _report(snapshot_timestamp="2026-06-02 00:30:00", severity="high")
    evaluation = evaluate_replay(_result([report], case), case)

    assert evaluation.labeled is True
    assert evaluation.detected is True
    assert evaluation.hits == 1
    assert evaluation.false_alarms == 0
    assert evaluation.first_hit_time == "2026-06-02 00:30:00"
    # event_time (01:00:00) - first_hit_time (00:30:00) = 30 minutes.
    assert evaluation.lead_time_minutes == pytest.approx(30.0)


def test_only_low_severity_hit_does_not_flip_detected():
    case = _case()
    report = _report(snapshot_timestamp="2026-06-02 00:30:00", severity="low")
    evaluation = evaluate_replay(_result([report], case), case)

    assert evaluation.detected is False
    assert evaluation.hits == 1
    assert evaluation.false_alarms == 0


def test_report_outside_window_is_a_false_alarm():
    case = _case()
    report = _report(snapshot_timestamp="2026-06-01 12:00:00", severity="critical")
    evaluation = evaluate_replay(_result([report], case), case)

    assert evaluation.detected is False
    assert evaluation.hits == 0
    assert evaluation.false_alarms == 1
    assert evaluation.total_reports == 1


def test_unlabeled_case_reports_labeled_false_and_not_detected():
    case = _case(labeled=False)
    report = _report(snapshot_timestamp="2026-06-01 12:00:00", severity="critical")
    evaluation = evaluate_replay(_result([report], case), case)

    assert evaluation.labeled is False
    assert evaluation.detected is False
    assert evaluation.hits == 0
    assert evaluation.false_alarms == 1
    assert evaluation.total_reports == 1
    assert evaluation.to_dict()["labeled"] is False


def test_no_reports_yields_all_zeros():
    case = _case()
    evaluation = evaluate_replay(_result([], case), case)

    assert evaluation.detected is False
    assert evaluation.hits == 0
    assert evaluation.false_alarms == 0
    assert evaluation.total_reports == 0
    assert evaluation.first_hit_time is None
    assert evaluation.lead_time_minutes is None


# ---------------------------------------------------------------------------
# Boundary: exactly at window_start and exactly at event_time both count
# ---------------------------------------------------------------------------


def test_report_exactly_at_window_start_counts_as_hit():
    case = _case()
    report = _report(snapshot_timestamp=WINDOW_START, severity="high")
    evaluation = evaluate_replay(_result([report], case), case)

    assert evaluation.hits == 1
    assert evaluation.false_alarms == 0


def test_report_exactly_at_event_time_still_detects():
    case = _case()
    report = _report(snapshot_timestamp=EVENT_TIME, severity="high")
    evaluation = evaluate_replay(_result([report], case), case)

    assert evaluation.detected is True
    assert evaluation.lead_time_minutes == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Missing snapshot_timestamp is a hard error, not a created_at fallback
# ---------------------------------------------------------------------------


def test_missing_snapshot_timestamp_raises_value_error():
    case = _case()
    report = AnomalyReport(
        id=AnomalyReport.new_id(),
        agent="info_leak",
        market_id="m1",
        market_question="Will X happen?",
        severity="high",
        anomaly_score=0.75,
        confidence=0.8,
        summary="anomaly",
        details={},
    )
    with pytest.raises(ValueError, match="snapshot_timestamp"):
        evaluate_replay(_result([report], case), case)
