"""Tests for political market classification."""

from prediction_market.data.political_filter import PoliticalFilter
from prediction_market.data.polymarket.models import GammaMarket


def _make_market(**kwargs) -> GammaMarket:
    defaults = dict(
        id="m1",
        question="Test market",
        outcomes=["Yes", "No"],
        outcomePrices=["0.5", "0.5"],
        volume=100000.0,
        active=True,
    )
    defaults.update(kwargs)
    return GammaMarket(**defaults)


class TestPoliticalFilter:
    def setup_method(self):
        self.pf = PoliticalFilter()

    def test_political_by_tag(self, sample_political_market):
        result = self.pf.classify(sample_political_market)
        assert result.is_political
        assert result.confidence >= 0.3
        assert any("tag" in r for r in result.reasons)

    def test_nonpolitical(self, sample_nonpolitical_market):
        result = self.pf.classify(sample_nonpolitical_market)
        assert not result.is_political

    def test_political_by_keyword(self):
        market = _make_market(question="Will the president veto the defense bill?")
        result = self.pf.classify(market)
        assert result.is_political
        assert any("keyword" in r for r in result.reasons)

    def test_political_by_category(self):
        market = _make_market(
            question="Some political question",
            category="politics",
        )
        result = self.pf.classify(market)
        assert result.is_political

    def test_multiple_signals_increase_confidence(self):
        market = _make_market(
            question="Will the president sign the bill?",
            category="politics",
            tags=[{"label": "Politics"}],
        )
        result = self.pf.classify(market)
        assert result.is_political
        assert result.confidence > 0.5

    def test_filter_political_markets(self, sample_political_market, sample_nonpolitical_market):
        markets = [sample_political_market, sample_nonpolitical_market]
        filtered = self.pf.filter_political(markets)
        assert len(filtered) == 1
        assert filtered[0].id == "test-market-1"

    def test_low_volume_filtered_out(self):
        market = _make_market(
            question="Will the president resign?",
            volume=100.0,
            tags=[{"label": "Politics"}],
        )
        result = self.pf.classify(market)
        assert result.is_political
        filtered = self.pf.filter_political([market], min_volume=True)
        assert len(filtered) == 0

    def test_low_volume_not_filtered_when_disabled(self):
        market = _make_market(
            question="Will the president resign?",
            volume=100.0,
            tags=[{"label": "Politics"}],
        )
        filtered = self.pf.filter_political([market], min_volume=False)
        assert len(filtered) == 1
