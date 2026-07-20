"""Tests for Pydantic models."""

import pytest

from prediction_market.data.polymarket.models import (
    GammaMarket,
    OrderBook,
    Trade,
    WalletActivity,
    WalletPosition,
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


class TestWalletPosition:
    def test_alias_mapping_and_properties(self):
        p = WalletPosition.model_validate(
            {
                "proxyWallet": "0xwallet1",
                "asset": "token-yes-1",
                "conditionId": "0xcondition1",
                "size": "500",
                "avgPrice": "0.60",
                "initialValue": "300",
                "currentValue": "325",
                "cashPnl": "25",
                "percentPnl": "8.33",
                "outcome": "Yes",
                "title": "Will X happen?",
            }
        )
        assert p.proxy_wallet == "0xwallet1"
        assert p.condition_id == "0xcondition1"
        assert p.cash_pnl == "25"
        assert p.cash_pnl_float == pytest.approx(25.0)
        assert p.size_float == pytest.approx(500.0)
        assert p.avg_price_float == pytest.approx(0.60)

    def test_defaults_without_keys(self):
        p = WalletPosition()
        assert p.proxy_wallet == ""
        assert p.size_float == 0.0
        assert p.cash_pnl_float == 0.0

    def test_from_fixture_record(self):
        import json
        from pathlib import Path

        fixtures = Path(__file__).resolve().parent.parent / "fixtures" / "positions.json"
        records = json.loads(fixtures.read_text())
        positions = [WalletPosition.model_validate(r) for r in records]
        assert len(positions) == 3
        assert positions[0].proxy_wallet == "0xwallet1"
        assert positions[1].cash_pnl_float == pytest.approx(-24.0)


class TestWalletActivity:
    def test_alias_mapping_and_properties(self):
        a = WalletActivity.model_validate(
            {
                "proxyWallet": "0xwallet1",
                "timestamp": "2026-02-20T10:00:00Z",
                "type": "TRADE",
                "conditionId": "0xcondition1",
                "size": "500",
                "usdcSize": "300",
                "price": "0.60",
                "side": "BUY",
                "outcome": "Yes",
                "transactionHash": "0xtx1",
            }
        )
        assert a.proxy_wallet == "0xwallet1"
        assert a.condition_id == "0xcondition1"
        assert a.usdc_size == "300"
        assert a.usdc_size_float == pytest.approx(300.0)

    def test_defaults_without_keys(self):
        a = WalletActivity()
        assert a.proxy_wallet == ""
        assert a.usdc_size_float == 0.0

    def test_from_fixture_record_mixed_types(self):
        import json
        from pathlib import Path

        fixtures = Path(__file__).resolve().parent.parent / "fixtures" / "activity.json"
        records = json.loads(fixtures.read_text())
        activities = [WalletActivity.model_validate(r) for r in records]
        assert len(activities) == 4
        types = {a.type for a in activities}
        assert types == {"TRADE", "REDEEM"}


class TestTradeNumericCoercion:
    """Live-API finding (2026-07-20): the Data API serves size/price as
    JSON numbers, not the strings the fixtures assumed."""

    def test_numeric_size_and_price_coerced_losslessly(self):
        t = Trade.model_validate(
            {"id": "t1", "size": 174.7, "price": 0.2862049227, "matchTime": "2026-01-01T00:00:00Z"}
        )
        assert t.size == "174.7"
        assert t.price == "0.2862049227"
        assert t.price_float == pytest.approx(0.2862049227)
        assert t.volume_usd == pytest.approx(174.7 * 0.2862049227)

    def test_epoch_number_match_time_stays_all_digits(self):
        t = Trade.model_validate({"id": "t1", "matchTime": 1735689600})
        assert t.match_time == "1735689600"
        t2 = Trade.model_validate({"id": "t2", "matchTime": 1735689600.0})
        assert t2.match_time == "1735689600"

    def test_string_inputs_unchanged(self):
        t = Trade.model_validate({"id": "t1", "size": "100", "price": "0.65"})
        assert t.size == "100"
        assert t.price == "0.65"


class TestTradeLiveSchema:
    """Live /trades records (2026-07-20): conditionId/asset/timestamp keys,
    no id field."""

    _RECORD = {
        "proxyWallet": "0xwallet",
        "side": "SELL",
        "asset": "123456",
        "conditionId": "0xcond",
        "size": 10.5,
        "price": 0.25,
        "timestamp": 1767442419,
        "outcome": "No",
        "transactionHash": "0xtxhash",
    }

    def test_alias_choices_map_live_keys(self):
        t = Trade.model_validate(self._RECORD)
        assert t.market == "0xcond"
        assert t.asset_id == "123456"
        assert t.match_time == "1767442419"

    def test_synthesized_id_is_deterministic_and_distinct(self):
        a = Trade.model_validate(self._RECORD)
        b = Trade.model_validate(self._RECORD)
        assert a.id == b.id != ""
        other = Trade.model_validate({**self._RECORD, "size": 11.0})
        assert other.id != a.id

    def test_explicit_id_not_overwritten(self):
        t = Trade.model_validate({**self._RECORD, "id": "given-id"})
        assert t.id == "given-id"

    def test_fixture_style_keys_still_parse(self):
        t = Trade.model_validate(
            {"id": "t1", "market": "m1", "assetId": "tok", "matchTime": "2026-01-01T00:00:00Z"}
        )
        assert t.market == "m1"
        assert t.asset_id == "tok"
        assert t.match_time == "2026-01-01T00:00:00Z"

    def test_serialization_keeps_camel_case(self):
        t = Trade.model_validate(self._RECORD)
        dumped = t.model_dump(by_alias=True)
        assert "matchTime" in dumped
        assert "assetId" in dumped
        assert "market" in dumped
