"""Tests for liquidity analysis."""

import pytest

from prediction_market.analysis.liquidity_analyzer import LiquidityAnalyzer, LiquidityMetrics
from prediction_market.data.polymarket.models import MarketHolder, OrderBook, OrderBookEntry


class TestLiquidityAnalyzer:
    def setup_method(self):
        self.analyzer = LiquidityAnalyzer()

    def test_compute_hhi_concentrated(self):
        # HHI is on 0-10000 scale: (70)^2 + (30)^2 = 4900 + 900 = 5800
        holders = [
            MarketHolder(address="a", position=7000, pctSupply=0.70),
            MarketHolder(address="b", position=3000, pctSupply=0.30),
        ]
        hhi = self.analyzer.compute_hhi(holders)
        assert hhi == pytest.approx(5800, abs=100)

    def test_compute_hhi_dispersed(self):
        # 100 holders with 1% each: HHI = 100 * 1^2 = 100
        holders = [
            MarketHolder(address=f"addr{i}", position=100, pctSupply=0.01)
            for i in range(100)
        ]
        hhi = self.analyzer.compute_hhi(holders)
        assert hhi == pytest.approx(100, abs=10)

    def test_compute_hhi_empty(self):
        assert self.analyzer.compute_hhi([]) == 0.0

    def test_analyze_normal_book(self, sample_orderbook):
        holders = [
            MarketHolder(address="a", position=5000, pctSupply=0.20),
            MarketHolder(address="b", position=4000, pctSupply=0.16),
            MarketHolder(address="c", position=3000, pctSupply=0.12),
        ]
        metrics = self.analyzer.analyze(sample_orderbook, holders)
        assert isinstance(metrics, LiquidityMetrics)
        assert metrics.total_bid_depth > 0
        assert metrics.total_ask_depth > 0
        assert metrics.spread_pct is not None
        assert 0 <= metrics.susceptibility_score <= 1

    def test_thin_book_high_susceptibility(self, thin_orderbook):
        holders = [
            MarketHolder(address="a", position=8000, pctSupply=0.80),
            MarketHolder(address="b", position=2000, pctSupply=0.20),
        ]
        metrics = self.analyzer.analyze(thin_orderbook, holders)
        assert metrics.susceptibility_score > 0.5

    def test_liquidity_drop_detection(self):
        assert self.analyzer.check_liquidity_drop("m1", 500, 1000, 0.50) is True
        assert self.analyzer.check_liquidity_drop("m1", 600, 1000, 0.50) is False
        assert self.analyzer.check_liquidity_drop("m1", 100, 1000, 0.50) is True
