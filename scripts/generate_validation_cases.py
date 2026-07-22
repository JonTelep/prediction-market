#!/usr/bin/env python3
"""Generate the Phase-2 flagship validation case pair.

Builds two committed backtest fixtures consumed by
``tests/backtest/test_replay_validation.py``:

* ``seeded_insider`` -- a market with a synthetic information leak
  (sustained price drift, a volume spike, and a fresh, one-sided,
  high-share wallet) engineered to trip the ``InfoLeakDetector`` before a
  scheduled event.
* ``benign_control`` -- the exact same spine (identical snapshots up to
  the point the insider case diverges) with no injected structure, so it
  is a volatility- and calendar-matched twin that must stay silent.

Both cases share one hourly 240-step (10-day) logit-space Gaussian random
walk (sigma=0.05) starting at price 0.15, ~5,000 USD/step of background
volume growth with +/-20% seeded jitter, 30 small two-sided background
wallets (``0xbg00``-``0xbg29``), and a single scheduled event at step 216.
The insider case diverges only from step 192 (T-24h) onward.

The committed fixtures under ``tests/fixtures/cases/seeded_insider/`` and
``tests/fixtures/cases/benign_control/`` are the source of truth --
``test_replay_validation.py`` loads them from disk and never regenerates.
This script exists so they are regenerable and reviewable, and to pick a
seed via a real replay-and-evaluate sweep (see ``_select_seed`` below).

Usage:
    python scripts/generate_validation_cases.py
    python scripts/generate_validation_cases.py --seed 42 --output tests/fixtures/cases
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import math
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from random import Random

from prediction_market.analysis.changepoint import CusumDetector
from prediction_market.analysis.price_analyzer import PriceAnalyzer
from prediction_market.analysis.timeseries import clamp_probability, logit
from prediction_market.analysis.volume_analyzer import VolumeAnalyzer
from prediction_market.backtest.case_format import Case, CaseLabel, SnapshotRow, save_case
from prediction_market.backtest.metrics import evaluate_replay
from prediction_market.backtest.replay import replay_case
from prediction_market.config import load_config
from prediction_market.data.external.models import ScheduledEvent
from prediction_market.data.polymarket.models import GammaMarket, Trade

logger = logging.getLogger(__name__)

_TS_FMT = "%Y-%m-%d %H:%M:%S"
_TRADE_TS_FMT = "%Y-%m-%dT%H:%M:%SZ"

# --- Shared spine parameters -------------------------------------------
N_STEPS = 240
START_PRICE = 0.15
LOGIT_SIGMA = 0.05
BASE_VOLUME_INCREMENT = 5000.0
VOLUME_JITTER = 0.20
N_BACKGROUND_WALLETS = 30
BACKGROUND_TRADE_MIN = 50.0
BACKGROUND_TRADE_MAX = 250.0
TS0 = datetime(2026, 1, 5, 0, 0, 0)  # exact-hour aligned

# --- Injected structure (insider case only) -----------------------------
EVENT_STEP = 216
WINDOW_START_STEP = 192  # T-24h relative to the event
INSIDER_FIRST_TRADE_STEP = 190
DRIFT_PER_STEP = 0.09
VOLUME_MULTIPLIER = 3.0
INSIDER_WALLET = "0xinsider01"
INSIDER_TRADE_STEPS = [190, 192, 194, 196, 198, 200, 202, 204, 206, 208, 210, 212]
INSIDER_TRADE_SIZE_USD = 1500.0

# --- Seed sweep -----------------------------------------------------------
SWEEP_SIZE = 20
INSIDER_MARGIN = 1.25
BENIGN_MARGIN = 0.8


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _ts(step: int) -> str:
    return (TS0 + timedelta(hours=step)).strftime(_TS_FMT)


@dataclass
class _Spine:
    """The raw building blocks shared by both cases before divergence."""

    logit_deltas: list[float]  # length N_STEPS - 1; step i's delta from step i-1
    volume_increments: list[float]  # length N_STEPS - 1
    background_trades: list[Trade]


def _build_spine(seed: int) -> _Spine:
    rng = Random(seed)
    logit_deltas = [rng.gauss(0.0, LOGIT_SIGMA) for _ in range(N_STEPS - 1)]
    volume_increments = [
        BASE_VOLUME_INCREMENT * (1.0 + rng.uniform(-VOLUME_JITTER, VOLUME_JITTER))
        for _ in range(N_STEPS - 1)
    ]

    background_trades: list[Trade] = []
    for step in range(N_STEPS):
        for j in range(2):
            wallet_idx = rng.randrange(N_BACKGROUND_WALLETS)
            wallet = f"0xbg{wallet_idx:02d}"
            side = "BUY" if j == 0 else "SELL"
            size_usd = rng.uniform(BACKGROUND_TRADE_MIN, BACKGROUND_TRADE_MAX)
            price = 0.5
            idx = len(background_trades)
            background_trades.append(
                Trade(
                    id=f"bg-trade-{idx}",
                    takerOrderId=f"bg-order-{idx}",
                    market="",  # filled in by caller
                    assetId="token-yes",
                    side=side,
                    size=str(round(size_usd / price, 2)),
                    feeRateBps="0",
                    price=str(price),
                    status="MATCHED",
                    matchTime=(TS0 + timedelta(hours=step)).strftime(_TRADE_TS_FMT),
                    outcome="Yes",
                    bucketIndex="0",
                    owner=wallet,
                    proxyWallet=wallet,
                    transactionHash=f"0xbgtx{idx}",
                )
            )
    return _Spine(
        logit_deltas=logit_deltas,
        volume_increments=volume_increments,
        background_trades=background_trades,
    )


def _snapshots_from_deltas(
    logit_deltas: list[float], volume_increments: list[float]
) -> list[SnapshotRow]:
    logit_value = logit(START_PRICE)
    volume_total = 100_000.0
    rows = [
        SnapshotRow(
            timestamp=_ts(0),
            price_yes=clamp_probability(_sigmoid(logit_value)),
            price_no=1.0 - clamp_probability(_sigmoid(logit_value)),
            volume_total=volume_total,
        )
    ]
    for i in range(1, N_STEPS):
        logit_value += logit_deltas[i - 1]
        volume_total += volume_increments[i - 1]
        price_yes = clamp_probability(_sigmoid(logit_value))
        rows.append(
            SnapshotRow(
                timestamp=_ts(i),
                price_yes=price_yes,
                price_no=1.0 - price_yes,
                volume_total=volume_total,
            )
        )
    return rows


def _make_market(market_id: str, condition_id: str, slug: str, question: str) -> GammaMarket:
    return GammaMarket(
        id=market_id,
        question=question,
        description=f"Resolves based on the outcome announced at {_ts(EVENT_STEP)}.",
        outcomes=["Yes", "No"],
        outcomePrices=["0.15", "0.85"],
        volume=200_000.0,
        volume24hr=20_000.0,
        liquidity=50_000.0,
        active=True,
        closed=False,
        archived=True,
        createdAt=TS0.isoformat() + "Z",
        endDate=(TS0 + timedelta(hours=N_STEPS + 24)).isoformat() + "Z",
        tags=[{"label": "Politics"}],
        slug=slug,
        category="politics",
        conditionId=condition_id,
        clobTokenIds=["token-yes", "token-no"],
    )


def _make_event() -> ScheduledEvent:
    return ScheduledEvent(
        source="congress",
        event_type="vote",
        title="Scheduled Committee Vote",
        description="A scheduled committee vote shared by both validation cases.",
        event_date=datetime.strptime(_ts(EVENT_STEP), _TS_FMT),
        url="https://example.gov/vote/validation",
        keywords=["committee", "vote"],
    )


def _build_benign_case(seed: int, spine: _Spine) -> Case:
    snapshots = _snapshots_from_deltas(spine.logit_deltas, spine.volume_increments)
    market_id = "benign-control-1"
    condition_id = "0xbenigncontrol1"
    trades = [t.model_copy(update={"market": market_id}) for t in spine.background_trades]

    label = CaseLabel(
        window_start=_ts(WINDOW_START_STEP),
        window_end=_ts(EVENT_STEP),
        event_time=_ts(EVENT_STEP),
    )
    return Case(
        slug="benign_control",
        market_id=market_id,
        condition_id=condition_id,
        question="Will the committee vote favorably at the scheduled session?",
        archived_at=_ts(N_STEPS - 1),
        notes="placeholder",  # filled in after seed selection
        market=_make_market(
            market_id,
            condition_id,
            "benign-control-validation",
            "Will the committee vote favorably at the scheduled session?",
        ),
        snapshots=snapshots,
        trades=trades,
        events=[_make_event()],
        label=label,
    )


def _build_insider_case(seed: int, spine: _Spine) -> Case:
    logit_deltas = list(spine.logit_deltas)
    volume_increments = list(spine.volume_increments)

    # Inject sustained drift + a volume spike over steps
    # (WINDOW_START_STEP, EVENT_STEP] -- i.e. deltas[WINDOW_START_STEP:EVENT_STEP]
    # (0-indexed into the length-(N_STEPS-1) delta arrays, where
    # deltas[k] is the step-(k+1) delta).
    for k in range(WINDOW_START_STEP, EVENT_STEP):
        logit_deltas[k] += DRIFT_PER_STEP
        volume_increments[k] *= VOLUME_MULTIPLIER

    snapshots = _snapshots_from_deltas(logit_deltas, volume_increments)
    market_id = "seeded-insider-1"
    condition_id = "0xseededinsider1"
    trades = [t.model_copy(update={"market": market_id}) for t in spine.background_trades]

    for i, step in enumerate(INSIDER_TRADE_STEPS):
        price_yes = snapshots[step].price_yes
        trades.append(
            Trade(
                id=f"insider-trade-{i}",
                takerOrderId=f"insider-order-{i}",
                market=market_id,
                assetId="token-yes",
                side="BUY",
                size=str(round(INSIDER_TRADE_SIZE_USD / price_yes, 2)),
                feeRateBps="0",
                price=str(round(price_yes, 4)),
                status="MATCHED",
                matchTime=(TS0 + timedelta(hours=step)).strftime(_TRADE_TS_FMT),
                outcome="Yes",
                bucketIndex="0",
                owner=INSIDER_WALLET,
                proxyWallet=INSIDER_WALLET,
                transactionHash=f"0xinsidertx{i}",
            )
        )
    trades.sort(key=lambda t: t.match_time)

    label = CaseLabel(
        window_start=_ts(WINDOW_START_STEP),
        window_end=_ts(EVENT_STEP),
        event_time=_ts(EVENT_STEP),
    )
    return Case(
        slug="seeded_insider",
        market_id=market_id,
        condition_id=condition_id,
        question="Will the committee vote favorably at the scheduled session?",
        archived_at=_ts(N_STEPS - 1),
        notes="placeholder",  # filled in after seed selection
        market=_make_market(
            market_id,
            condition_id,
            "seeded-insider-validation",
            "Will the committee vote favorably at the scheduled session?",
        ),
        snapshots=snapshots,
        trades=trades,
        events=[_make_event()],
        label=label,
    )


def _peak_combined(case: Case, thresholds) -> float:
    """Peak (ungated, uncooldowned) combined anomaly score across *case*.

    Mirrors the scoring arithmetic in ``InfoLeakDetector._process_market``
    (price/volume z-composite, event-proximity amplifier, CUSUM amplifier)
    using the same analyzer classes the live detector uses, so seed
    selection reflects the real scoring path. It intentionally does not
    read this off emitted reports: the benign control's whole point is
    that it emits *zero* reports, so there is no report to peek a score
    from, and the gate (``combined < combined_score_min``) is exactly what
    we're measuring distance from.
    """
    price_analyzer = PriceAnalyzer(thresholds)
    volume_analyzer = VolumeAnalyzer(thresholds)
    cusum = CusumDetector(thresholds)
    proximity = timedelta(hours=thresholds.event_proximity_hours)
    event_dates = [e.event_date for e in case.events]

    last_volume: float | None = None
    peak = 0.0
    for row in case.snapshots:
        ts = datetime.strptime(row.timestamp, _TS_FMT)

        if last_volume is not None:
            delta = max(0.0, row.volume_total - last_volume)
            volume_analyzer.update(case.market_id, delta, ts)
        last_volume = row.volume_total

        price_analyzer.update(case.market_id, row.price_yes, ts)

        cusum.update(case.market_id, row.price_yes, ts)
        alarm = cusum.check_alarm(case.market_id)

        price_z = price_analyzer.current_z_score(case.market_id) or 0.0
        vol_z = volume_analyzer.current_z_score(case.market_id) or 0.0
        price_triggered = abs(price_z) >= thresholds.price_zscore
        volume_triggered = abs(vol_z) >= thresholds.volume_zscore
        if not (price_triggered or volume_triggered):
            continue

        combined = math.sqrt(price_z**2 + vol_z**2)
        if any(abs((ts - ed).total_seconds()) <= proximity.total_seconds() for ed in event_dates):
            combined *= thresholds.event_amplifier
        if alarm is not None:
            combined *= thresholds.cusum_amplifier

        peak = max(peak, combined)

    return peak


async def _replay_peak_and_reports(case: Case, config) -> tuple[float, int]:
    """Replay *case* for report count, and separately compute its score peak."""
    with tempfile.TemporaryDirectory() as tmp:
        run_config = config.model_copy(deep=True)
        run_config.database.path = str(Path(tmp) / f"{case.slug}.db")
        result = await replay_case(case, run_config)
        evaluate_replay(result, case)  # sanity: must not raise
        peak = _peak_combined(case, config.thresholds)
        return peak, len(result.reports)


async def _select_seed(start_seed: int) -> tuple[int, Case, Case, float, float]:
    config = load_config()
    combined_min = config.thresholds.combined_score_min

    observations: list[str] = []
    for seed in range(start_seed, start_seed + SWEEP_SIZE):
        spine = _build_spine(seed)
        insider_case = _build_insider_case(seed, spine)
        benign_case = _build_benign_case(seed, spine)

        insider_peak, insider_reports = await _replay_peak_and_reports(insider_case, config)
        benign_peak, benign_reports = await _replay_peak_and_reports(benign_case, config)

        insider_ok = insider_peak >= INSIDER_MARGIN * combined_min and insider_reports > 0
        benign_ok = benign_reports == 0 and benign_peak <= BENIGN_MARGIN * combined_min

        observations.append(
            f"seed={seed}: insider_peak={insider_peak:.3f} "
            f"insider_reports={insider_reports} benign_peak={benign_peak:.3f} "
            f"benign_reports={benign_reports} -> "
            f"{'PASS' if (insider_ok and benign_ok) else 'fail'}"
        )
        logger.info(observations[-1])

        if insider_ok and benign_ok:
            insider_margin = insider_peak / combined_min
            benign_margin = benign_peak / combined_min
            return seed, insider_case, benign_case, insider_margin, benign_margin

    raise SystemExit(
        "No seed in the sweep "
        f"[{start_seed}, {start_seed + SWEEP_SIZE - 1}] separated the seeded-insider "
        "case from the benign control at default thresholds. Per-seed observed "
        "peak scores:\n" + "\n".join(observations)
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42, help="First seed to try in the sweep")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tests/fixtures/cases"),
        help="Directory under which seeded_insider/ and benign_control/ are written",
    )
    args = parser.parse_args()

    seed, insider_case, benign_case, insider_margin, benign_margin = asyncio.run(
        _select_seed(args.seed)
    )

    note = (
        f"Generated by scripts/generate_validation_cases.py, seed={seed} "
        f"(selected from sweep {args.seed}-{args.seed + SWEEP_SIZE - 1}). "
        f"Peak combined margins vs combined_score_min: "
        f"insider={insider_margin:.3f}x (>= 1.25x required), "
        f"benign={benign_margin:.3f}x (<= 0.8x required)."
    )
    insider_case.notes = (
        "Seeded insider case: 24-step sustained logit drift "
        f"(+{DRIFT_PER_STEP}/step) and a 3x volume spike over "
        f"[{_ts(WINDOW_START_STEP)}, {_ts(EVENT_STEP)}], plus wallet "
        f"{INSIDER_WALLET} (first trade {_ts(INSIDER_FIRST_TRADE_STEP)}, "
        "12 one-sided BUY trades) trading into the scheduled event. " + note
    )
    benign_case.notes = (
        "Benign control: the shared spine, unmodified -- volatility- and "
        "calendar-matched twin of seeded_insider that must stay silent "
        "even though the same event-proximity amplifier is live. " + note
    )

    args.output.mkdir(parents=True, exist_ok=True)
    save_case(args.output / "seeded_insider", insider_case)
    save_case(args.output / "benign_control", benign_case)

    print(f"\nSelected seed={seed}")
    print(f"  insider peak-combined margin: {insider_margin:.3f}x combined_score_min")
    print(f"  benign  peak-combined margin: {benign_margin:.3f}x combined_score_min")
    print(f"Wrote {args.output / 'seeded_insider'}")
    print(f"Wrote {args.output / 'benign_control'}")


if __name__ == "__main__":
    main()
