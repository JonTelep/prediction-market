"""Tests for the *live* LiquidityAnalyzer used by ManipulationGuard.

These are the first tests the live analyzer
(``agents.manipulation_guard.LiquidityAnalyzer``) has ever had -- the
previous version of this file tested a dead twin in
``analysis.liquidity_analyzer`` (deleted; see docs/CONVENTIONS.md
"Known-broken wiring" item 9 at commit f06ac00).

All assertions are exact-value / hand-computed, replacing the old
``pytest.approx(abs=100)`` tolerances -- these formulas are
deterministic.

Scale note (STEP 0 of the dedup prompt): ``tests/fixtures/holders.json``
carries ``pctSupply`` as a fraction in [0, 1] (max value 0.35), so
``concentration_score`` is exercised here with fraction-scale inputs and
no normalisation -- see the docstring on
``LiquidityAnalyzer.concentration_score``.
"""

from __future__ import annotations

import math

import pytest

from prediction_market.agents.manipulation_guard import LiquidityAnalyzer
from prediction_market.data.polymarket.models import (
    MarketHolder,
    OrderBook,
    OrderBookEntry,
)


def _book(bids: list[tuple[float, float]], asks: list[tuple[float, float]]) -> OrderBook:
    """Build an OrderBook from (price, size) tuples."""
    return OrderBook(
        market="test-market",
        asset_id="token-yes",
        bids=[OrderBookEntry(price=str(p), size=str(s)) for p, s in bids],
        asks=[OrderBookEntry(price=str(p), size=str(s)) for p, s in asks],
    )


class TestDepthScore:
    def test_at_or_below_min_depth_is_maximally_thin(self):
        # total = 5,000 (bid: price=1.0 size=5000, no asks) -> score 1.0
        book = _book(bids=[(1.0, 5000.0)], asks=[])
        assert LiquidityAnalyzer.depth_score(book) == 1.0

    def test_at_or_above_max_depth_is_fully_healthy(self):
        # total = 500,000 -> score 0.0
        book = _book(bids=[(1.0, 500_000.0)], asks=[])
        assert LiquidityAnalyzer.depth_score(book) == 0.0

    def test_midpoint_depth_is_exactly_half(self):
        # total = 50,000. Logarithmic interpolation:
        #   1 - ln(50_000 / 5_000) / ln(500_000 / 5_000)
        # = 1 - ln(10) / ln(100)
        # = 1 - ln(10) / (2 * ln(10))
        # = 1 - 0.5 = 0.5
        book = _book(bids=[(1.0, 40_000.0)], asks=[(1.0, 10_000.0)])
        expected = 1.0 - math.log(10) / math.log(100)
        assert expected == pytest.approx(0.5)
        assert LiquidityAnalyzer.depth_score(book) == pytest.approx(0.5)

    def test_zero_depth_is_maximally_thin(self):
        book = _book(bids=[], asks=[])
        assert LiquidityAnalyzer.depth_score(book) == 1.0


class TestSpreadScore:
    def test_no_spread_available_defaults_to_maximally_risky(self):
        book = _book(bids=[], asks=[])
        assert book.spread_pct is None
        assert LiquidityAnalyzer.spread_score(book) == 1.0

    def test_five_percent_spread_is_half(self):
        # mid=1.0, spread=0.05 -> spread_pct=0.05 -> min(1.0, 0.05/0.10) = 0.5
        book = _book(bids=[(0.975, 100.0)], asks=[(1.025, 100.0)])
        assert book.spread_pct == pytest.approx(0.05)
        assert LiquidityAnalyzer.spread_score(book) == pytest.approx(0.5)

    def test_ten_percent_or_wider_spread_saturates_to_one(self):
        # mid=1.0, spread=0.20 -> spread_pct=0.20 -> min(1.0, 0.20/0.10) = 1.0
        book = _book(bids=[(0.90, 100.0)], asks=[(1.10, 100.0)])
        assert book.spread_pct == pytest.approx(0.20)
        assert LiquidityAnalyzer.spread_score(book) == 1.0


class TestImbalanceScore:
    def test_passthrough_of_absolute_imbalance_positive(self):
        # bid_depth=40,000, ask_depth=10,000 -> imbalance = 30,000/50,000 = 0.6
        book = _book(bids=[(1.0, 40_000.0)], asks=[(1.0, 10_000.0)])
        assert book.imbalance == pytest.approx(0.6)
        assert LiquidityAnalyzer.imbalance_score(book) == pytest.approx(0.6)

    def test_passthrough_of_absolute_imbalance_negative(self):
        # bid_depth=10,000, ask_depth=40,000 -> imbalance = -0.6, abs -> 0.6
        book = _book(bids=[(1.0, 10_000.0)], asks=[(1.0, 40_000.0)])
        assert book.imbalance == pytest.approx(-0.6)
        assert LiquidityAnalyzer.imbalance_score(book) == pytest.approx(0.6)


class TestConcentrationScore:
    def test_single_full_holder_is_maximally_concentrated(self):
        holders = [MarketHolder(address="a", position=1000, pctSupply=1.0)]
        assert LiquidityAnalyzer.concentration_score(holders) == pytest.approx(1.0)

    def test_four_equal_holders(self):
        # HHI = 4 * (0.25 ** 2) = 4 * 0.0625 = 0.25
        holders = [
            MarketHolder(address=f"addr{i}", position=100, pctSupply=0.25)
            for i in range(4)
        ]
        assert LiquidityAnalyzer.concentration_score(holders) == pytest.approx(0.25)

    def test_empty_holder_list_is_maximally_risky(self):
        # Deliberate: unknown concentration is treated as maximally risky,
        # not zero (manipulation_guard.py concentration_score).
        assert LiquidityAnalyzer.concentration_score([]) == 1.0


class TestComputeSusceptibility:
    def test_composite_matches_hand_computed_weighted_sum(self):
        # Book engineered so that:
        #   depth_score   = 0.5   (total depth = 50,000)
        #   spread_score  = 0.5   (spread_pct = 0.05)
        #   imbalance     = 0.6   (bid=40,000 / ask=10,000)
        book = _book(bids=[(0.975, 40_000.0 / 0.975)], asks=[(1.025, 10_000.0 / 1.025)])
        assert LiquidityAnalyzer.depth_score(book) == pytest.approx(0.5)
        assert LiquidityAnalyzer.spread_score(book) == pytest.approx(0.5)
        assert LiquidityAnalyzer.imbalance_score(book) == pytest.approx(0.6)

        # concentration_score = 0.25 (four equal 25% holders)
        holders = [
            MarketHolder(address=f"addr{i}", position=100, pctSupply=0.25)
            for i in range(4)
        ]
        assert LiquidityAnalyzer.concentration_score(holders) == pytest.approx(0.25)

        # Default weights (config.py ThresholdConfig):
        #   depth=0.30, spread=0.25, concentration=0.25, imbalance=0.20
        weights = {
            "depth_weight": 0.30,
            "spread_weight": 0.25,
            "concentration_weight": 0.25,
            "imbalance_weight": 0.20,
        }
        expected = (
            0.5 * 0.30  # depth
            + 0.5 * 0.25  # spread
            + 0.25 * 0.25  # concentration
            + 0.6 * 0.20  # imbalance
        )
        assert expected == pytest.approx(0.4575)

        analyzer = LiquidityAnalyzer()
        result = analyzer.compute_susceptibility(book, holders, weights)
        assert result["depth"] == pytest.approx(0.5)
        assert result["spread"] == pytest.approx(0.5)
        assert result["concentration"] == pytest.approx(0.25)
        assert result["imbalance"] == pytest.approx(0.6)
        assert result["composite"] == pytest.approx(0.4575)

    def test_composite_uses_default_weights_when_missing(self):
        # weights dict missing keys falls back to the same defaults
        # (0.30/0.25/0.25/0.20) via dict.get() defaults in the source.
        book = _book(bids=[(0.975, 40_000.0 / 0.975)], asks=[(1.025, 10_000.0 / 1.025)])
        holders = [
            MarketHolder(address=f"addr{i}", position=100, pctSupply=0.25)
            for i in range(4)
        ]
        analyzer = LiquidityAnalyzer()
        result = analyzer.compute_susceptibility(book, holders, {})
        assert result["composite"] == pytest.approx(0.4575)
