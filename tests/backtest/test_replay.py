"""Tests for the point-in-time backtest replay harness.

Report comparisons throughout this file use
``(details["snapshot_timestamp"], severity, summary)`` tuples -- never
``id`` (a random uuid) and never ``created_at`` (wall-clock emission time,
which differs across runs by construction under replay).
"""

from __future__ import annotations

import dataclasses
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from prediction_market.backtest.case_format import Case, load_case, save_case
from prediction_market.backtest.replay import replay_case
from prediction_market.config import load_config
from prediction_market.data.polymarket.models import Trade
from prediction_market.reporting.anomaly_report import AnomalyReport
from prediction_market.store.database import init_database
from prediction_market.store.snapshots import save_price_snapshot

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "cases" / "minimal"

# The fixture's anomaly step: index 24 (0-indexed) in snapshots.json, at
# timestamp "2026-06-02 00:00:00" -- see design notes in case.toml.
ANOMALY_STEP_INDEX = 24


def _report_tuples(reports: list[AnomalyReport]) -> list[tuple]:
    return [(r.details["snapshot_timestamp"], r.severity, r.summary) for r in reports]


def _make_config(tmp_path: Path, name: str):
    config = load_config()
    config.database.path = str(tmp_path / f"{name}.db")
    return config


def _load_minimal_case() -> Case:
    return load_case(FIXTURE_DIR)


def _truncate_case(case: Case, n: int) -> Case:
    """Return a copy of *case* whose replay spine is only its first *n* rows."""
    return dataclasses.replace(case, snapshots=case.snapshots[:n])


# ---------------------------------------------------------------------------
# Case format: round-trip and malformed-case validation
# ---------------------------------------------------------------------------


def test_round_trip(tmp_path):
    original = _load_minimal_case()
    out_dir = tmp_path / "roundtrip"
    save_case(out_dir, original)
    reloaded = load_case(out_dir)
    assert reloaded == original


def test_missing_snapshots_json_raises(tmp_path):
    case = _load_minimal_case()
    out_dir = tmp_path / "missing_snapshots"
    save_case(out_dir, case)
    (out_dir / "snapshots.json").unlink()

    with pytest.raises(ValueError, match="snapshots.json"):
        load_case(out_dir)


def test_out_of_order_snapshots_raises(tmp_path):
    case = _load_minimal_case()
    out_dir = tmp_path / "out_of_order"
    save_case(out_dir, case)

    rows = json.loads((out_dir / "snapshots.json").read_text())
    rows[0], rows[1] = rows[1], rows[0]
    (out_dir / "snapshots.json").write_text(json.dumps(rows))

    with pytest.raises(ValueError, match="snapshots.json"):
        load_case(out_dir)


def test_market_id_mismatch_raises(tmp_path):
    case = _load_minimal_case()
    out_dir = tmp_path / "mismatch"
    save_case(out_dir, case)

    market = json.loads((out_dir / "market.json").read_text())
    market["id"] = "some-other-market-id"
    (out_dir / "market.json").write_text(json.dumps(market))

    with pytest.raises(ValueError, match="case.toml"):
        load_case(out_dir)


# ---------------------------------------------------------------------------
# Replay: basic behavior
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_replay_minimal_fixture_emits_reports(tmp_path):
    case = _load_minimal_case()
    config = _make_config(tmp_path, "basic")

    result = await replay_case(case, config)

    assert result.steps == len(case.snapshots)
    assert len(result.reports) >= 1
    for report in result.reports:
        assert report.market_id == case.market_id


@pytest.mark.asyncio
async def test_replay_is_deterministic(tmp_path):
    case = _load_minimal_case()

    config_a = _make_config(tmp_path, "det_a")
    result_a = await replay_case(case, config_a)

    config_b = _make_config(tmp_path, "det_b")
    result_b = await replay_case(case, config_b)

    assert _report_tuples(result_a.reports) == _report_tuples(result_b.reports)


# ---------------------------------------------------------------------------
# No-lookahead proofs -- the point of the engine
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_prefix_truncated_before_anomaly_emits_nothing(tmp_path):
    """Truncating to the step just before the fixture's anomaly must emit
    zero reports -- the anomaly only exists in the data at/after that step."""
    case = _load_minimal_case()
    truncated = _truncate_case(case, ANOMALY_STEP_INDEX)  # rows [0, ANOMALY_STEP_INDEX)

    config = _make_config(tmp_path, "prefix_before")
    result = await replay_case(truncated, config)

    assert result.reports == []


@pytest.mark.asyncio
async def test_prefix_property_matches_full_replay(tmp_path):
    """A truncated replay's reports must exactly equal the prefix of the
    full replay's reports up to the truncation point -- proof the engine
    never leaks data from steps beyond the one currently being processed."""
    case = _load_minimal_case()
    n = ANOMALY_STEP_INDEX + 1  # include the anomaly step itself

    full_config = _make_config(tmp_path, "prefix_full")
    full_result = await replay_case(case, full_config)

    truncated_config = _make_config(tmp_path, "prefix_truncated")
    truncated = _truncate_case(case, n)
    truncated_result = await replay_case(truncated, truncated_config)

    cutoff_ts = case.snapshots[n - 1].timestamp
    full_prefix_tuples = [
        t for t in _report_tuples(full_result.reports) if t[0] <= cutoff_ts
    ]

    assert _report_tuples(truncated_result.reports) == full_prefix_tuples
    assert len(truncated_result.reports) >= 1


@pytest.mark.asyncio
async def test_future_trade_excluded_from_anomaly_report(tmp_path):
    """A huge, fresh, one-sided trade timestamped *after* the anomaly step
    must not show up in that step's wallet_evidence -- the SQL match_time
    window on the query, not insertion order, is what enforces this."""
    case = _load_minimal_case()
    anomaly_ts = case.snapshots[ANOMALY_STEP_INDEX].timestamp
    future_ts = (
        datetime.strptime(anomaly_ts, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        + timedelta(hours=1)
    ).strftime("%Y-%m-%dT%H:%M:%SZ")

    future_trade = Trade(
        id="minimal-trade-future",
        market=case.market_id,
        asset_id="token-yes-minimal",
        side="BUY",
        size="1000000",
        price="0.99",
        status="MATCHED",
        match_time=future_ts,
        outcome="Yes",
        owner="0xownerFuture",
        proxy_wallet="0xwalletFuture",
        transaction_hash="0xtxFuture",
    )
    augmented = dataclasses.replace(case, trades=[*case.trades, future_trade])

    config = _make_config(tmp_path, "future_trade")
    result = await replay_case(augmented, config)

    anomaly_reports = [
        r for r in result.reports if r.details["snapshot_timestamp"] == anomaly_ts
    ]
    assert len(anomaly_reports) == 1
    wallets_seen = {w["wallet"] for w in anomaly_reports[0].details["wallet_evidence"]}
    # Guard against a vacuous pass: the in-window wallets must actually be
    # present, so the future wallet's absence is a real exclusion, not an
    # empty evidence list.
    assert "0xwalletA" in wallets_seen
    assert "0xwalletFuture" not in wallets_seen


# ---------------------------------------------------------------------------
# save_price_snapshot regression
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_save_price_snapshot_defaults_to_utcnow(tmp_path):
    config = _make_config(tmp_path, "default_ts")
    db = await init_database(config)
    try:
        await db.execute(
            "INSERT INTO markets (id, question) VALUES (?, ?)",
            ("m1", "test market"),
        )
        await db.commit()

        before = datetime.now(timezone.utc)
        await save_price_snapshot(db, "m1", price_yes=0.5, price_no=0.5)
        after = datetime.now(timezone.utc)

        cursor = await db.execute("SELECT timestamp FROM snapshots WHERE market_id = 'm1'")
        row = await cursor.fetchone()
        stored = datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)

        assert before - timedelta(seconds=5) <= stored <= after + timedelta(seconds=5)
    finally:
        await db.close()
