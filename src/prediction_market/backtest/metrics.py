"""Scoring: turn a replayed case into a labeled-window evaluation.

Report times under replay come from ``report.details["snapshot_timestamp"]``
-- the historical TEXT timestamp the detector stamps on every report (see
``agents/info_leak_detector.py``). ``AnomalyReport.created_at`` is
wall-clock emission time (a ``datetime``, see reporting/anomaly_report.py)
and is meaningless here: it tells you when the replay engine happened to
run, not when the historical anomaly occurred. Comparisons throughout this
module are lexicographic string comparisons of ``"%Y-%m-%d %H:%M:%S"``
TEXT timestamps -- the same convention used everywhere else in the store.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from prediction_market.backtest.case_format import Case
from prediction_market.backtest.replay import ReplayResult
from prediction_market.store.models import SEVERITY_ORDER

_DETECTED_MIN_SEVERITY = SEVERITY_ORDER["medium"]


@dataclass
class ReplayEvaluation:
    """The scored outcome of replaying one labeled (or unlabeled) case.

    Attributes:
        case_slug: The replayed case's slug.
        labeled: Whether the case carried a ``[label]`` section. When
            ``False`` every other field reflects the "unlabeled" convention
            (``detected=False``, ``hits=0``, ``false_alarms=total_reports``)
            rather than a genuine miss -- callers must check this flag
            before drawing conclusions from ``detected``.
        detected: Whether at least one medium-or-higher severity report
            fell inside ``[window_start, window_end]`` at or before
            ``event_time`` (when present).
        first_hit_time: The ``snapshot_timestamp`` of the earliest report
            inside the labeled window, or ``None`` if there were none.
        lead_time_minutes: ``event_time - first_hit_time`` in minutes
            (positive means the alert came before the event). ``None``
            when there is no ``event_time`` or no hit.
        hits: Count of reports whose ``snapshot_timestamp`` falls inside
            ``[window_start, window_end]`` (any severity).
        false_alarms: Count of reports outside that window.
        total_reports: Total reports emitted during the replay.
    """

    case_slug: str
    labeled: bool
    detected: bool
    first_hit_time: str | None
    lead_time_minutes: float | None
    hits: int
    false_alarms: int
    total_reports: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_slug": self.case_slug,
            "labeled": self.labeled,
            "detected": self.detected,
            "first_hit_time": self.first_hit_time,
            "lead_time_minutes": self.lead_time_minutes,
            "hits": self.hits,
            "false_alarms": self.false_alarms,
            "total_reports": self.total_reports,
        }


def _snapshot_timestamp(report: Any) -> str:
    ts = report.details.get("snapshot_timestamp")
    if ts is None:
        raise ValueError(
            f"report {report.id!r} has no details['snapshot_timestamp'] -- "
            "this report was not produced by a Phase-2 point-in-time replay "
            "detector; falling back to created_at (wall-clock emission "
            "time) would silently misdate it."
        )
    return ts


def _minutes_between(earlier: str, later: str) -> float:
    from datetime import datetime

    fmt = "%Y-%m-%d %H:%M:%S"
    delta = datetime.strptime(later, fmt) - datetime.strptime(earlier, fmt)
    return delta.total_seconds() / 60.0


def evaluate_replay(result: ReplayResult, case: Case) -> ReplayEvaluation:
    """Score a :class:`ReplayResult` against *case*'s labeled window.

    Args:
        result: The outcome of :func:`~prediction_market.backtest.replay
            .replay_case`.
        case: The case that was replayed (carries the ``[label]`` window
            being scored against).

    Returns:
        A :class:`ReplayEvaluation`.
    """
    reports = result.reports
    total_reports = len(reports)

    if case.label is None:
        return ReplayEvaluation(
            case_slug=case.slug,
            labeled=False,
            detected=False,
            first_hit_time=None,
            lead_time_minutes=None,
            hits=0,
            false_alarms=total_reports,
            total_reports=total_reports,
        )

    label = case.label
    window_start = label.window_start
    window_end = label.window_end
    event_time = label.event_time

    hits = 0
    false_alarms = 0
    in_window_times: list[str] = []
    detected = False

    for report in reports:
        ts = _snapshot_timestamp(report)
        in_window = window_start <= ts <= window_end
        if in_window:
            hits += 1
            in_window_times.append(ts)
            severity_rank = SEVERITY_ORDER.get(report.severity, -1)
            if severity_rank >= _DETECTED_MIN_SEVERITY and ts <= event_time:
                detected = True
        else:
            false_alarms += 1

    first_hit_time = min(in_window_times) if in_window_times else None
    lead_time_minutes: float | None = None
    if first_hit_time is not None:
        lead_time_minutes = _minutes_between(first_hit_time, event_time)

    return ReplayEvaluation(
        case_slug=case.slug,
        labeled=True,
        detected=detected,
        first_hit_time=first_hit_time,
        lead_time_minutes=lead_time_minutes,
        hits=hits,
        false_alarms=false_alarms,
        total_reports=total_reports,
    )
