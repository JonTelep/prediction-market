"""PERMANENT: the Phase-2 flagship validation test.

This module is to Phase 2 what ``tests/test_pipeline.py`` is to Phase 1: it
replays two committed fixture cases through the real backtest stack --
``tests/fixtures/cases/seeded_insider/`` (a synthetic information leak that
must be caught before its event) and ``tests/fixtures/cases/benign_control/``
(a volatility- and calendar-matched twin that must stay silent) -- and
enforces the phase's central thesis in code. Any future change that misses
the insider case or flags the benign control breaks the build.

The fixtures are the source of truth. They are generated (and regenerable)
via ``scripts/generate_validation_cases.py``, but this module reads them from
disk and never regenerates them: a test that generates its own inputs can
drift in lockstep with a bug in the generator and stop testing anything.

Phase-3 note: when Phase-3 detectors are wired into the surveillance
pipeline, they must be added to this test too -- extend
``test_seeded_insider_case_flagged`` (and, if a Phase-3 detector could
plausibly fire on calendar proximity alone, ``test_benign_case_not_flagged``)
rather than creating a parallel flagship test.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from prediction_market.backtest.case_format import Case, load_case
from prediction_market.backtest.metrics import evaluate_replay
from prediction_market.backtest.replay import replay_case
from prediction_market.config import load_config

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "cases"
INSIDER_DIR = FIXTURE_DIR / "seeded_insider"
BENIGN_DIR = FIXTURE_DIR / "benign_control"

INSIDER_WALLET = "0xinsider01"
WALLET_SCORE_MIN = 0.6


def _make_config(tmp_path: Path, name: str):
    config = load_config()
    config.database.path = str(tmp_path / f"{name}.db")
    return config


def _load_insider_case() -> Case:
    return load_case(INSIDER_DIR)


def _load_benign_case() -> Case:
    return load_case(BENIGN_DIR)


# ---------------------------------------------------------------------------
# The flagship: insider caught, benign silent.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_seeded_insider_case_flagged(tmp_path):
    case = _load_insider_case()
    config = _make_config(tmp_path, "seeded_insider")

    result = await replay_case(case, config)
    evaluation = evaluate_replay(result, case)

    assert evaluation.detected is True
    assert evaluation.lead_time_minutes is not None
    assert evaluation.lead_time_minutes > 0

    label = case.label
    assert label is not None
    window_reports = [
        r
        for r in result.reports
        if label.window_start <= r.details["snapshot_timestamp"] <= label.window_end
    ]
    assert window_reports, "expected at least one report inside the labeled window"

    def _top_wallet_qualifies(report) -> bool:
        evidence = report.details.get("wallet_evidence") or []
        if not evidence:
            return False
        top = evidence[0]
        return (
            top["wallet"] == INSIDER_WALLET
            and top["score"] >= WALLET_SCORE_MIN
            and top["is_fresh"] is True
        )

    assert any(_top_wallet_qualifies(r) for r in window_reports), (
        "expected at least one window report to rank 0xinsider01 first with "
        "score >= 0.6 and is_fresh=True"
    )

    assert any(r.details.get("cusum") for r in window_reports), (
        "expected at least one window report with a truthy CUSUM alarm dict "
        "(not the always-present-but-None default)"
    )


@pytest.mark.asyncio
async def test_benign_case_not_flagged(tmp_path):
    case = _load_benign_case()
    config = _make_config(tmp_path, "benign_control")

    result = await replay_case(case, config)
    evaluation = evaluate_replay(result, case)

    # Not "no high-severity reports" -- zero reports. The control shares the
    # event calendar with the insider case, so this asserts the detector
    # does not alert on calendar proximity alone.
    assert evaluation.total_reports == 0
    assert len(result.reports) == 0


def test_validation_cases_are_committed():
    """Tripwire: a deleted or regenerated-and-broken fixture must fail loudly."""
    insider = load_case(INSIDER_DIR)
    benign = load_case(BENIGN_DIR)

    assert insider.label is not None
    assert benign.label is not None


# ---------------------------------------------------------------------------
# Honesty checks on the fixtures themselves.
# ---------------------------------------------------------------------------


def test_insider_and_benign_snapshots_identical_before_divergence():
    """Proves the control is matched, not merely similar.

    Both cases share one generated spine up to the insider case's
    divergence point (``window_start``, T-24h before the event) -- the rows
    before that point must be exactly equal, row-wise.
    """
    insider = _load_insider_case()
    benign = _load_benign_case()

    label = insider.label
    assert label is not None
    divergence_index = next(
        i
        for i, row in enumerate(insider.snapshots)
        if row.timestamp == label.window_start
    )
    assert divergence_index > 0

    assert (
        insider.snapshots[:divergence_index] == benign.snapshots[:divergence_index]
    )


def test_insider_window_logit_return_mean_exceeds_benign():
    """Proves the structure was actually injected, not just labeled."""
    from prediction_market.analysis.timeseries import clamp_probability, logit

    insider = _load_insider_case()
    benign = _load_benign_case()
    label = insider.label
    assert label is not None

    def _window_returns(case: Case) -> list[float]:
        window_rows = [
            row
            for row in case.snapshots
            if label.window_start <= row.timestamp <= label.window_end
        ]
        logits = [logit(clamp_probability(r.price_yes)) for r in window_rows]
        return [b - a for a, b in zip(logits, logits[1:])]

    insider_returns = _window_returns(insider)
    benign_returns = _window_returns(benign)

    assert insider_returns and benign_returns
    insider_mean = sum(insider_returns) / len(insider_returns)
    benign_mean = sum(benign_returns) / len(benign_returns)

    assert insider_mean > benign_mean
