"""Tests for Pydantic models."""

import pytest

from prediction_market.data.polymarket.models import (
    GammaMarket,
    OrderBook,
    OrderBookEntry,
    Trade,
)


class TestOrderBook:
    def test_best_bid_ask(self, sample_orderbook):
        assert sample_orderbook.best_bid == 0.64
        assert sample_orderbook.best_ask == 0.66

    def test_midpoint(self, sample_orderbook):
        assert sample_orderbook.midpoint == pytest.approx(0.65)

    def test_spread(self, sample_orderbook):
        assert sample_orderbook.spread == pytest.approx(0.02)

    def test_spread_pct(self, sample_orderbook):
        expected = 0.02 / 0.65
        assert sample_orderbook.spread_pct == pytest.approx(expected)

    def test_imbalance(self, sample_orderbook):
        imb = sample_orderbook.imbalance
        assert -1 <= imb <= 1

    def test_empty_orderbook(self):
        ob = OrderBook()
        assert ob.best_bid is None
        assert ob.best_ask is None
        assert ob.midpoint is None
        assert ob.spread is None
        assert ob.imbalance == 0.0

    def test_depth_at_pct(self, sample_orderbook):
        depth = sample_orderbook.depth_at_pct(0.05)
        assert depth > 0

    def test_total_depths(self, sample_orderbook):
        assert sample_orderbook.total_bid_depth > 0
        assert sample_orderbook.total_ask_depth > 0


class TestGammaMarket:
    def test_tag_labels(self, sample_political_market):
        labels = sample_political_market.tag_labels
        assert "Politics" in labels
        assert "Legislation" in labels

    def test_empty_tags(self):
        m = GammaMarket(id="x", question="test")
        assert m.tag_labels == []


class TestTrade:
    def test_trade_properties(self):
        t = Trade(price="0.65", size="100", matchTime="2026-01-15T10:30:00Z")
        assert t.price_float == 0.65
        assert t.size_float == 100.0
        assert t.volume_usd == 65.0
        assert t.match_datetime is not None
