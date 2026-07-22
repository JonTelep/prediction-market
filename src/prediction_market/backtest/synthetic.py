"""Synthetic null controls: benign volatility-matched twins of a case.

A labeled case answers "does the detector fire on this?" A null control
answers the other half of the research brief's false-positive discipline:
"does the detector stay quiet on a market that looks statistically similar
but has no real information leak?" :func:`generate_null_case` builds that
twin -- same cadence, same logit-return volatility, same public calendar,
but prices are an unconditioned random walk and trades carry no wallet
concentration. :func:`estimate_false_positive_rate` replays a batch of
these and reports how often the detector fired anyway.
"""

from __future__ import annotations

import copy
import math
import statistics
from datetime import datetime
from random import Random
from typing import Any

from prediction_market.analysis.timeseries import clamp_probability, logit
from prediction_market.backtest.case_format import Case, SnapshotRow
from prediction_market.backtest.replay import replay_case
from prediction_market.config import AppConfig
from prediction_market.data.polymarket.models import Trade

_TS_FORMAT = "%Y-%m-%d %H:%M:%S"
_TRADE_TS_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
_NULL_WALLET_COUNT = 10


def _sigmoid(x: float) -> float:
    """Inverse of :func:`~prediction_market.analysis.timeseries.logit`."""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _logit_return_sigma(snapshots: list[SnapshotRow]) -> float:
    """Sample stdev of the template's logit-returns (0.0 if under-determined)."""
    logits = [logit(clamp_probability(s.price_yes)) for s in snapshots]
    returns = [b - a for a, b in zip(logits, logits[1:])]
    if len(returns) < 2:
        return 0.0
    return statistics.stdev(returns)


def _mean_volume_increment(snapshots: list[SnapshotRow]) -> float:
    deltas = [
        b.volume_total - a.volume_total for a, b in zip(snapshots, snapshots[1:])
    ]
    if not deltas:
        return 0.0
    return statistics.mean(deltas)


def _null_snapshots(template: Case, rng: Random) -> list[SnapshotRow]:
    sigma = _logit_return_sigma(template.snapshots)
    mean_increment = _mean_volume_increment(template.snapshots)

    snapshots: list[SnapshotRow] = []
    logit_value = logit(clamp_probability(template.snapshots[0].price_yes))
    volume_total = template.snapshots[0].volume_total

    for i, row in enumerate(template.snapshots):
        if i > 0:
            logit_value += rng.gauss(0.0, sigma)
            volume_total = template.snapshots[0].volume_total + i * mean_increment
        price_yes = clamp_probability(_sigmoid(logit_value))
        snapshots.append(
            SnapshotRow(
                timestamp=row.timestamp,
                price_yes=price_yes,
                price_no=1.0 - price_yes,
                volume_total=volume_total,
            )
        )
    return snapshots


def _null_trades(template: Case, rng: Random, seed: int) -> list[Trade]:
    if not template.snapshots:
        return []

    start = datetime.strptime(template.snapshots[0].timestamp, _TS_FORMAT)
    end = datetime.strptime(template.snapshots[-1].timestamp, _TS_FORMAT)
    span = end - start
    n = len(template.trades)

    trades: list[Trade] = []
    for i, t in enumerate(template.trades):
        # Uniform spread across the timeline (n==1 lands at the midpoint).
        frac = (i + 0.5) / n if n > 1 else 0.5
        match_dt = start + span * frac
        wallet = f"0xnull{i % _NULL_WALLET_COUNT}"
        trades.append(
            Trade(
                id=f"{t.id}-null-{seed}",
                taker_order_id=f"{t.taker_order_id}-null-{seed}",
                market=t.market,
                asset_id=t.asset_id,
                side=t.side,
                size=t.size,
                fee_rate_bps=t.fee_rate_bps,
                price=t.price,
                status=t.status,
                match_time=match_dt.strftime(_TRADE_TS_FORMAT),
                outcome=t.outcome,
                bucket_index=t.bucket_index,
                owner=wallet,
                proxy_wallet=wallet,
                transaction_hash=f"{t.transaction_hash}-null-{seed}",
            )
        )
    # Keep the timeline strictly time-ascending for the replay engine.
    trades.sort(key=lambda t: t.match_time)
    return trades


def generate_null_case(template: Case, *, seed: int) -> Case:
    """Build a benign, volatility-matched synthetic twin of *template*.

    Same market metadata (slug/ids suffixed ``-null-<seed>``), same
    snapshot cadence and timestamps, prices regenerated as a logit-space
    Gaussian random walk whose per-step sigma matches the template's own
    logit-return sample stdev, volume rebuilt as a smooth cumulative
    series (no spikes), trades resampled to spread uniformly across the
    timeline with wallets replaced round-robin across
    :data:`_NULL_WALLET_COUNT` synthetic wallets, and events copied as-is
    (a null market shares the same public calendar).

    Args:
        template: The case to build a null twin of. Never mutated.
        seed: Seed for the private :class:`random.Random` instance driving
            price and trade generation. Determines the case's identity
            suffix and is fully deterministic for a fixed *template*.

    Returns:
        A new, independent :class:`Case`.
    """
    rng = Random(seed)
    suffix = f"-null-{seed}"

    market = template.market.model_copy(
        update={
            "id": f"{template.market_id}{suffix}",
            "condition_id": f"{template.condition_id}{suffix}",
            "slug": f"{template.market.slug}{suffix}",
        }
    )

    snapshots = _null_snapshots(template, rng)
    trades = _null_trades(template, rng, seed)
    events = copy.deepcopy(template.events)

    return Case(
        slug=f"{template.slug}{suffix}",
        market_id=market.id,
        condition_id=market.condition_id,
        question=template.question,
        archived_at=template.archived_at,
        notes=f"synthetic null control (seed={seed}) of {template.slug!r}",
        market=market,
        snapshots=snapshots,
        trades=trades,
        events=events,
        label=None,
    )


async def estimate_false_positive_rate(
    template: Case,
    config: AppConfig,
    *,
    runs: int,
    base_seed: int = 1000,
) -> dict[str, Any]:
    """Replay *runs* synthetic null cases and measure the false-alarm rate.

    Each run gets its own throwaway database (``config.database.path``
    suffixed per run) and its own null case (seed ``base_seed + i``).

    Args:
        template: The labeled case to build null twins of.
        config: Base application config; ``database.path`` is overridden
            per run so replays don't collide.
        runs: Number of null cases to replay.
        base_seed: First seed; run *i* uses ``base_seed + i``.

    Returns:
        ``{"runs", "paths_with_reports", "reports_per_path_mean", "fp_path_rate"}``.
    """
    paths_with_reports = 0
    total_reports = 0

    for i in range(runs):
        seed = base_seed + i
        null_case = generate_null_case(template, seed=seed)

        run_config = config.model_copy(deep=True)
        run_config.database.path = f"{config.database.path}.null-{seed}"

        result = await replay_case(null_case, run_config)
        n_reports = len(result.reports)
        total_reports += n_reports
        if n_reports > 0:
            paths_with_reports += 1

    return {
        "runs": runs,
        "paths_with_reports": paths_with_reports,
        "reports_per_path_mean": total_reports / runs if runs else 0.0,
        "fp_path_rate": paths_with_reports / runs if runs else 0.0,
    }
