"""Order-book depth and liquidity analysis for Polymarket surveillance.

Computes depth metrics, holder concentration (HHI), and a composite
susceptibility score that indicates how vulnerable a market is to
manipulation via thin order books and concentrated holdings.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

from prediction_market.config import ThresholdConfig
from prediction_market.data.polymarket.models import MarketHolder, OrderBook

logger = logging.getLogger(__name__)


@dataclass
class LiquidityMetrics:
    """Comprehensive liquidity snapshot for a single market/token.

    Attributes:
        total_bid_depth: Total USD-equivalent depth on the bid side.
        total_ask_depth: Total USD-equivalent depth on the ask side.
        depth_1pct: Depth within 1% of the midpoint (both sides).
        depth_5pct: Depth within 5% of the midpoint (both sides).
        depth_10pct: Depth within 10% of the midpoint (both sides).
        spread_pct: Bid-ask spread as a percentage of midpoint.
        imbalance: Order-book imbalance ratio ``(bids - asks) / (bids + asks)``.
        hhi: Herfindahl-Hirschman Index measuring holder concentration (0-10000).
        susceptibility_score: Composite score (0-1) indicating manipulation
            vulnerability; higher means more susceptible.
    """

    total_bid_depth: float
    total_ask_depth: float
    depth_1pct: float
    depth_5pct: float
    depth_10pct: float
    spread_pct: float
    imbalance: float
    hhi: float
    susceptibility_score: float


class LiquidityAnalyzer:
    """Analyzes order-book depth and holder concentration.

    Combines order-book shape metrics with holder-level Herfindahl-Hirschman
    concentration to produce a single susceptibility score that downstream
    agents use to weight anomaly significance.

    Args:
        thresholds: Threshold configuration containing weights for the
            susceptibility formula.  Uses defaults if ``None``.
    """

    def __init__(self, thresholds: ThresholdConfig | None = None) -> None:
        self._thresholds = thresholds or ThresholdConfig()
        # Cache previous depths per market for drop detection.
        self._prev_depth: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def analyze(
        self,
        orderbook: OrderBook,
        holders: list[MarketHolder] | None = None,
    ) -> LiquidityMetrics:
        """Compute full liquidity metrics for an order book and holder list.

        Args:
            orderbook: A Polymarket :class:`OrderBook` snapshot.
            holders: Optional list of :class:`MarketHolder` positions.  When
                provided, the HHI concentration index is computed; otherwise
                HHI defaults to 0.

        Returns:
            A :class:`LiquidityMetrics` dataclass.
        """
        total_bid = orderbook.total_bid_depth
        total_ask = orderbook.total_ask_depth
        depth_1 = orderbook.depth_at_pct(0.01)
        depth_5 = orderbook.depth_at_pct(0.05)
        depth_10 = orderbook.depth_at_pct(0.10)
        spread = orderbook.spread_pct or 0.0
        imbalance = orderbook.imbalance

        hhi = self.compute_hhi(holders) if holders else 0.0

        metrics = LiquidityMetrics(
            total_bid_depth=total_bid,
            total_ask_depth=total_ask,
            depth_1pct=depth_1,
            depth_5pct=depth_5,
            depth_10pct=depth_10,
            spread_pct=spread,
            imbalance=imbalance,
            hhi=hhi,
            susceptibility_score=0.0,  # Filled in below.
        )
        metrics.susceptibility_score = self.compute_susceptibility(metrics, self._thresholds)

        # Store total depth for drop detection.
        market_id = orderbook.market or orderbook.asset_id
        if market_id:
            self._prev_depth[market_id] = total_bid + total_ask

        return metrics

    # ------------------------------------------------------------------
    # HHI computation
    # ------------------------------------------------------------------

    @staticmethod
    def compute_hhi(holders: list[MarketHolder]) -> float:
        """Compute the Herfindahl-Hirschman Index from holder positions.

        The HHI ranges from 0 (perfectly dispersed) to 10,000 (single holder
        owns everything).  It is the sum of squared market-share percentages.

        Args:
            holders: List of :class:`MarketHolder` objects with ``pct_supply``
                expressed as a fraction (0-1) or a percentage (0-100).

        Returns:
            HHI value on the 0-10,000 scale.
        """
        if not holders:
            return 0.0

        # Determine whether pct_supply is already a percentage or a fraction.
        # The Polymarket Data API returns values in [0, 1] range.
        total_pct = sum(h.pct_supply for h in holders)

        if total_pct == 0:
            # Fall back to computing shares from position values.
            total_value = sum(h.value for h in holders)
            if total_value == 0:
                return 0.0
            shares = [(h.value / total_value) * 100.0 for h in holders]
        elif total_pct <= 1.5:
            # Fractions in [0, 1] -- scale to percentages.
            shares = [h.pct_supply * 100.0 for h in holders]
        else:
            # Already percentages.
            shares = [h.pct_supply for h in holders]

        return sum(s * s for s in shares)

    # ------------------------------------------------------------------
    # Susceptibility score
    # ------------------------------------------------------------------

    @staticmethod
    def compute_susceptibility(metrics: LiquidityMetrics, weights: ThresholdConfig) -> float:
        """Compute a composite susceptibility score from liquidity metrics.

        The score combines four normalised signals:

        1. **Depth** -- low near-market depth (depth_5pct) increases
           susceptibility.  Normalised as ``1 - sigmoid(depth_5pct / 1000)``.
        2. **Spread** -- wide spreads indicate thinner markets.  Normalised
           as ``min(spread_pct / 0.10, 1.0)``.
        3. **Concentration** -- high HHI signals few dominant holders.
           Normalised as ``min(hhi / 5000, 1.0)``.
        4. **Imbalance** -- strongly skewed books are more susceptible.
           Uses ``abs(imbalance)``.

        Each component is multiplied by its respective weight from
        :class:`ThresholdConfig` and the total is clamped to [0, 1].

        Args:
            metrics: Pre-computed :class:`LiquidityMetrics`.
            weights: :class:`ThresholdConfig` with ``depth_weight``,
                ``spread_weight``, ``concentration_weight``, and
                ``imbalance_weight``.

        Returns:
            A float in [0, 1].
        """
        # 1. Depth component (inverse sigmoid so that low depth -> high score).
        depth_norm = 1.0 - _sigmoid(metrics.depth_5pct / 1000.0)

        # 2. Spread component.
        spread_norm = min(metrics.spread_pct / 0.10, 1.0)

        # 3. Concentration component.
        concentration_norm = min(metrics.hhi / 5000.0, 1.0)

        # 4. Imbalance component.
        imbalance_norm = abs(metrics.imbalance)

        score = (
            weights.depth_weight * depth_norm
            + weights.spread_weight * spread_norm
            + weights.concentration_weight * concentration_norm
            + weights.imbalance_weight * imbalance_norm
        )
        return max(0.0, min(1.0, score))

    # ------------------------------------------------------------------
    # Liquidity drop detection
    # ------------------------------------------------------------------

    def check_liquidity_drop(
        self,
        market_id: str,
        current_depth: float,
        previous_depth: float | None = None,
        threshold_pct: float | None = None,
    ) -> bool:
        """Detect whether liquidity has dropped significantly.

        If *previous_depth* is not supplied, the last depth recorded via
        :meth:`analyze` for *market_id* is used.

        Args:
            market_id: Polymarket market identifier.
            current_depth: Current total order-book depth (bid + ask).
            previous_depth: Previous total depth to compare against.
            threshold_pct: Fractional drop that constitutes a flag (e.g. 0.50
                means a 50% drop).  Defaults to
                :pyattr:`ThresholdConfig.liquidity_drop_pct`.

        Returns:
            ``True`` if the depth has dropped by more than the threshold.
        """
        if threshold_pct is None:
            threshold_pct = self._thresholds.liquidity_drop_pct

        if previous_depth is None:
            previous_depth = self._prev_depth.get(market_id)

        if previous_depth is None or previous_depth <= 0:
            return False

        drop_pct = (previous_depth - current_depth) / previous_depth
        if drop_pct >= threshold_pct:
            logger.warning(
                "Liquidity drop for %s: %.1f%% (prev=%.2f, curr=%.2f)",
                market_id,
                drop_pct * 100,
                previous_depth,
                current_depth,
            )
            return True
        return False

    def __repr__(self) -> str:
        return f"LiquidityAnalyzer(tracked_markets={len(self._prev_depth)})"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sigmoid(x: float) -> float:
    """Standard sigmoid function clamped to avoid overflow."""
    x = max(-500.0, min(500.0, x))
    return 1.0 / (1.0 + math.exp(-x))
