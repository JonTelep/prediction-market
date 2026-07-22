"""Tests for backtest.synthetic: null-control generation and FP-rate estimation."""

from __future__ import annotations

import copy
import math
import random
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from prediction_market.backtest.case_format import Case, load_case
from prediction_market.backtest.case_format import SnapshotRow
from prediction_market.backtest.synthetic import (
    _logit_return_sigma,
    estimate_false_positive_rate,
    generate_null_case,
)
from prediction_market.config import load_config
from prediction_market.data.external.models import ScheduledEvent
from prediction_market.data.polymarket.models import GammaMarket, Trade

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "cases" / "minimal"


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _make_template_case(n: int = 150, sigma: float = 0.05, seed: int = 7) -> Case:
    """A synthetic template with enough steps for a stable volatility estimate.

    Deliberately not the hand-built minimal fixture: that fixture's single
    huge jump dominates its logit-return sample stdev, which would make a
    25%-tolerance volatility-matching assertion flaky. This template's own
    generation uses a seeded ``random.Random`` (never the global RNG).
    """
    rng = random.Random(seed)
    ts0 = datetime(2026, 1, 1)
    logit_value = 0.0
    volume = 1000.0

    snapshots: list[SnapshotRow] = []
    for i in range(n):
        if i > 0:
            logit_value += rng.gauss(0.0, sigma)
            volume += 100.0
        price = _sigmoid(logit_value)
        ts = (ts0 + timedelta(hours=i)).strftime("%Y-%m-%d %H:%M:%S")
        snapshots.append(
            SnapshotRow(
                timestamp=ts, price_yes=price, price_no=1.0 - price, volume_total=volume
            )
        )

    trades: list[Trade] = []
    for i in range(40):
        match_dt = ts0 + timedelta(hours=i)
        # Concentrated in two wallets -- this is what the null generation
        # must break up.
        wallet = "0xwhale" if i % 3 == 0 else "0xother"
        trades.append(
            Trade(
                id=f"template-trade-{i}",
                takerOrderId=f"template-order-{i}",
                market="template-market-1",
                assetId="token-yes-template",
                side="BUY",
                size="1000",
                feeRateBps="0",
                price="0.5",
                status="MATCHED",
                matchTime=match_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                outcome="Yes",
                bucketIndex="0",
                owner=wallet,
                proxyWallet=wallet,
                transactionHash=f"0xtx{i}",
            )
        )

    events = [
        ScheduledEvent(
            source="congress",
            event_type="hearing",
            title="Template Hearing",
            description="desc",
            event_date=ts0 + timedelta(hours=10),
            url="https://example.gov/hearing/template",
            keywords=["template"],
        )
    ]

    return Case(
        slug="template",
        market_id="template-market-1",
        condition_id="0xtemplatecond1",
        question="Will the template event happen?",
        archived_at="2026-01-10 00:00:00",
        notes="synthetic test template",
        market=GammaMarket(id="template-market-1", conditionId="0xtemplatecond1"),
        snapshots=snapshots,
        trades=trades,
        events=events,
        label=None,
    )


def _load_minimal_case() -> Case:
    return load_case(FIXTURE_DIR)


# ---------------------------------------------------------------------------
# generate_null_case
# ---------------------------------------------------------------------------


def test_deterministic_for_fixed_seed():
    template = _make_template_case()
    a = generate_null_case(template, seed=42)
    b = generate_null_case(template, seed=42)
    assert a == b


def test_different_seeds_differ():
    template = _make_template_case()
    a = generate_null_case(template, seed=1)
    b = generate_null_case(template, seed=2)
    assert a != b
    assert a.snapshots != b.snapshots


def test_does_not_mutate_template():
    template = _make_template_case()
    before = copy.deepcopy(template)
    generate_null_case(template, seed=5)
    assert template == before


def test_per_step_logit_return_sigma_within_25_percent_of_template():
    template = _make_template_case()
    null_case = generate_null_case(template, seed=99)

    template_sigma = _logit_return_sigma(template.snapshots)
    null_sigma = _logit_return_sigma(null_case.snapshots)

    assert template_sigma > 0
    assert abs(null_sigma - template_sigma) <= 0.25 * template_sigma


def test_null_trades_have_at_least_10_distinct_wallets_no_concentration():
    template = _make_template_case()
    null_case = generate_null_case(template, seed=3)

    wallets = [t.proxy_wallet for t in null_case.trades]
    distinct = set(wallets)
    assert len(distinct) >= 10

    volume_by_wallet: Counter[str] = Counter()
    total_volume = 0.0
    for t in null_case.trades:
        volume_by_wallet[t.proxy_wallet] += t.volume_usd
        total_volume += t.volume_usd

    assert total_volume > 0
    for wallet_volume in volume_by_wallet.values():
        assert wallet_volume / total_volume <= 0.20


def test_null_events_identical_to_template():
    template = _make_template_case()
    null_case = generate_null_case(template, seed=11)
    assert null_case.events == template.events


# ---------------------------------------------------------------------------
# estimate_false_positive_rate -- fixed literals per spec
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_estimate_false_positive_rate_shape_and_zero_on_minimal_fixture(tmp_path):
    template = _load_minimal_case()
    config = load_config()
    config.database.path = str(tmp_path / "fp_estimate.db")

    result = await estimate_false_positive_rate(
        template, config, runs=3, base_seed=1000
    )

    assert set(result.keys()) == {
        "runs",
        "paths_with_reports",
        "reports_per_path_mean",
        "fp_path_rate",
    }
    assert result["runs"] == 3

    # Calibration finding, not a bug to tune away: on these exact literals
    # (minimal fixture, runs=3, base_seed=1000) the spec predicted a clean
    # 0.0 false-positive path rate, but the first unmodified run measured
    # 1/3 (paths_with_reports=1). Root cause, confirmed by inspecting the
    # seed=1000 null path's emitted report directly: raw_combined=2.712,
    # amplifiers=['event_proximity(x1.5, 1 event(s) within +/-24h)'],
    # combined=4.07 vs combined_score_min=4.0, cusum=None. Wallet evidence
    # cannot cause report emission -- the gate is `combined <
    # combined_score_min` on the z-composite, and the wallet path only
    # runs *after* a report is already warranted, where it can bump
    # confidence/severity but nothing more (here it bumped confidence
    # 0.508->0.608 post-emission). The actual cause: the null control
    # deliberately keeps the template's calendar events live (a null
    # market shares the public calendar), and near that event the x1.5
    # event-proximity amplifier lowered the effective raw gate from 4.0 to
    # ~2.67 -- a sigma-matched random walk crossed it by chance. This is a
    # genuine, meaningful calibration finding about the event-amplified
    # gate being crossable by a volatility-matched null walk near a
    # calendar event, not a small-sample wallet-window artifact -- per the
    # prompt's instructions this value is asserted as observed, not forced
    # to zero.
    assert result["fp_path_rate"] == pytest.approx(1 / 3)
    assert result["paths_with_reports"] == 1
