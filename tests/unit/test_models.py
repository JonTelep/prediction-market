"""Tests for Pydantic models."""

import pytest

from prediction_market.data.polymarket.models import (
    GammaMarket,
    OrderBook,
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


class TestGammaMarketTokenOrder:
    def test_standard_order(self):
        m = GammaMarket(
            id="x",
            question="test",
            outcomes=["Yes", "No"],
            clobTokenIds=["tok-yes", "tok-no"],
        )
        assert m.yes_token_id == "tok-yes"
        assert m.no_token_id == "tok-no"
        assert m.yes_token_id == m.clob_token_ids[0]
        assert m.no_token_id == m.clob_token_ids[1]

    def test_inverted_order(self):
        m = GammaMarket(
            id="x",
            question="test",
            outcomes=["No", "Yes"],
            clobTokenIds=["tok-a", "tok-b"],
        )
        # positional [0] would be "tok-a", but the label says it's NO
        assert m.yes_token_id == m.clob_token_ids[1]
        assert m.no_token_id == m.clob_token_ids[0]

    def test_case_insensitive_labels(self):
        m = GammaMarket(
            id="x",
            question="test",
            outcomes=["NO", "YES"],
            clobTokenIds=["tok-a", "tok-b"],
        )
        assert m.yes_token_id == "tok-b"
        assert m.no_token_id == "tok-a"

    def test_non_yes_no_labels_falls_back_to_positional(self):
        m = GammaMarket(
            id="x",
            question="test",
            outcomes=["Candidate A", "Candidate B"],
            clobTokenIds=["tok-a", "tok-b"],
        )
        assert m.yes_token_id == "tok-a"
        assert m.no_token_id == "tok-b"

    def test_empty_token_list_returns_none(self):
        m = GammaMarket(id="x", question="test", outcomes=["Yes", "No"], clobTokenIds=[])
        assert m.yes_token_id is None
        assert m.no_token_id is None

    def test_single_token_no_returns_none(self):
        m = GammaMarket(
            id="x", question="test", outcomes=["Yes", "No"], clobTokenIds=["tok-a"]
        )
        assert m.yes_token_id == "tok-a"
        assert m.no_token_id is None


class TestTrade:
    def test_trade_properties(self):
        t = Trade(price="0.65", size="100", matchTime="2026-01-15T10:30:00Z")
        assert t.price_float == 0.65
        assert t.size_float == 100.0
        assert t.volume_usd == 65.0
        assert t.match_datetime is not None

    def test_proxy_wallet_populated_from_alias(self):
        t = Trade.model_validate(
            {
                "id": "trade-1",
                "price": "0.65",
                "size": "100",
                "matchTime": "2026-01-15T10:30:00Z",
                "proxyWallet": "0xwallet1",
            }
        )
        assert t.proxy_wallet == "0xwallet1"

    def test_proxy_wallet_defaults_empty_without_key(self):
        t = Trade.model_validate(
            {
                "id": "trade-1",
                "price": "0.65",
                "size": "100",
                "matchTime": "2026-01-15T10:30:00Z",
            }
        )
        assert t.proxy_wallet == ""

    def test_proxy_wallet_from_fixture_record(self):
        import json
        from pathlib import Path

        fixtures = Path(__file__).resolve().parent.parent / "fixtures" / "trades.json"
        records = json.loads(fixtures.read_text())
        trades = [Trade.model_validate(r) for r in records]
        assert trades[0].proxy_wallet == "0xwallet1"
        assert trades[1].proxy_wallet == "0xwallet2"
        # owner remains a distinct field, not an alias for proxy_wallet
        assert trades[0].owner == "0xowner1"
