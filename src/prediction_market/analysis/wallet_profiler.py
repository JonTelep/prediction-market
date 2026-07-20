"""Per-wallet anomaly feature extraction for Polymarket surveillance.

A pure, stateless computation layer: it turns a window of wallet-attributed
trade summary rows (as produced by
``prediction_market.store.queries.get_market_wallet_summary``) into ranked
per-wallet anomaly features. This is the Phase-2 adaptation of the Mitts &
Ofir composite (bet size, concentration, timing) minus the profitability
signals -- those need resolution outcomes we don't store yet and are
deferred to Phase 3.

This module must not query a database or import any storage-layer package:
callers are responsible for fetching summary rows and passing them in, the
same pattern ``PriceAnalyzer``/``VolumeAnalyzer`` use for plain values.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from prediction_market.config import ThresholdConfig

# Fixed composite weights -- not config. Three knobs nobody will tune is
# speculative surface; these are author decisions baked into the formula.
_W_SHARE = 0.40
_W_CONCENTRATION = 0.30
_W_FRESH = 0.30

_TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"


@dataclass(frozen=True)
class WalletFeatures:
    """Anomaly features computed for a single wallet within a time window.

    Attributes:
        wallet: The wallet's proxy address.
        trade_count: Number of trades within the window.
        total_volume_usd: Total notional traded within the window.
        volume_share: This wallet's share of total market volume across
            all wallets in the window, in [0, 1].
        directional_concentration: How one-sided the wallet's flow is,
            mapped from [0.5, 1.0] onto [0.0, 1.0].
        is_fresh: Whether the wallet's all-history first trade falls
            within ``thresholds.wallet_freshness_hours`` of ``window_end``.
        first_trade: The wallet's all-history first trade timestamp, as
            returned by the summary row (raw string, may be unparseable).
        score: Composite anomaly score in [0, 1].
    """

    wallet: str
    trade_count: int
    total_volume_usd: float
    volume_share: float
    directional_concentration: float
    is_fresh: bool
    first_trade: str
    score: float

    def to_dict(self) -> dict[str, Any]:
        """Return a flat, JSON-serializable representation."""
        return {
            "wallet": self.wallet,
            "trade_count": self.trade_count,
            "total_volume_usd": self.total_volume_usd,
            "volume_share": self.volume_share,
            "directional_concentration": self.directional_concentration,
            "is_fresh": self.is_fresh,
            "first_trade": self.first_trade,
            "score": self.score,
        }


def _is_fresh(first_trade: str, window_end: str, freshness_hours: int) -> bool:
    """Whether *first_trade* is within *freshness_hours* before *window_end*.

    Both timestamps are TEXT ``"%Y-%m-%d %H:%M:%S"`` values. An unparseable
    ``first_trade`` (e.g. legacy ``T``-format rows) means "not fresh" rather
    than raising -- freshness is a bonus signal, not a required field.
    """
    try:
        first_dt = datetime.strptime(first_trade, _TIMESTAMP_FORMAT)
        end_dt = datetime.strptime(window_end, _TIMESTAMP_FORMAT)
    except (ValueError, TypeError):
        return False
    return end_dt - first_dt <= timedelta(hours=freshness_hours)


def _directional_concentration(total: float, buy: float) -> float:
    """Map one-sidedness of flow from [0.5, 1.0] onto [0.0, 1.0].

    A perfectly balanced 50/50 wallet scores 0; a fully one-sided wallet
    scores 1. A wallet with zero total volume scores 0 (no signal).
    """
    if total <= 0:
        return 0.0
    sell = total - buy
    c = max(buy, sell) / total
    return (c - 0.5) * 2.0


def profile_wallets(
    summaries: list[dict],
    *,
    window_end: str,
    thresholds: ThresholdConfig,
) -> list[WalletFeatures]:
    """Compute ranked anomaly features for wallets in a trading window.

    Args:
        summaries: Rows as returned by
            ``store.queries.get_market_wallet_summary`` -- each with
            ``proxy_wallet``, ``trade_count``, ``total_volume_usd``,
            ``buy_volume_usd``, ``first_trade``, ``last_trade``.
        window_end: The end of the trading window, as a
            ``"%Y-%m-%d %H:%M:%S"`` TEXT timestamp, used to evaluate
            wallet freshness.
        thresholds: Threshold configuration providing
            ``wallet_freshness_hours`` and ``wallet_min_volume_usd``.

    Returns:
        ``WalletFeatures`` for wallets whose ``total_volume_usd`` meets
        ``thresholds.wallet_min_volume_usd``, sorted by ``score``
        descending. Dust wallets below the minimum are excluded from the
        result but still count toward the ``volume_share`` denominator.
    """
    if not summaries:
        return []

    total_market_volume = sum(float(row["total_volume_usd"]) for row in summaries)

    features: list[WalletFeatures] = []
    for row in summaries:
        total_volume = float(row["total_volume_usd"])
        if total_volume < thresholds.wallet_min_volume_usd:
            continue

        buy_volume = float(row["buy_volume_usd"])
        volume_share = (
            total_volume / total_market_volume if total_market_volume > 0 else 0.0
        )
        concentration = _directional_concentration(total_volume, buy_volume)
        fresh = _is_fresh(
            row["first_trade"], window_end, thresholds.wallet_freshness_hours
        )

        score = (
            _W_SHARE * volume_share
            + _W_CONCENTRATION * concentration
            + _W_FRESH * (1.0 if fresh else 0.0)
        )
        score = min(max(score, 0.0), 1.0)

        features.append(
            WalletFeatures(
                wallet=row["proxy_wallet"],
                trade_count=int(row["trade_count"]),
                total_volume_usd=total_volume,
                volume_share=volume_share,
                directional_concentration=concentration,
                is_fresh=fresh,
                first_trade=row["first_trade"],
                score=score,
            )
        )

    features.sort(key=lambda f: f.score, reverse=True)
    return features
