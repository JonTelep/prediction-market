"""Tests for scripts/backfill_markets.py's selection logic.

This is the unification prompt's safety net: backfill_markets.py used to
carry its own inline ``classify_political`` with different volume/gate
semantics than ``PoliticalFilter.classify``. It has been deleted in favor
of the shared ``PoliticalFilter``, via the ``select_political_markets``
helper. These tests prove:

  (a) tag-matched political + high volume -> selected
  (b) tag-matched political + low volume -> excluded (volume gate survives)
  (c) non-political -> excluded
  (d) keyword-only market, exactly 2 keyword matches, no tag/category hit
      -> excluded. This is the documented BEHAVIOR CHANGE: the old
      flat-+0.3-per-match/no-confidence-gate script would have selected
      this market (confidence would've been treated as "any reasons ->
      political"); the unified PoliticalFilter computes confidence =
      min(0.3, 2 * 0.1) = 0.2, which is below the 0.3 is_political gate,
      so it is excluded. Asserting this exclusion deliberately documents
      the intended unification, not a regression.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from prediction_market.config import load_political_keywords
from prediction_market.data.political_filter import PoliticalFilter
from prediction_market.data.polymarket.models import GammaMarket

_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "backfill_markets.py"


def _load_backfill_markets_module():
    """Import scripts/backfill_markets.py as a module (it's not a package)."""
    spec = importlib.util.spec_from_file_location("backfill_markets", _SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["backfill_markets"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def backfill_markets_module():
    return _load_backfill_markets_module()


@pytest.fixture
def political_filter() -> PoliticalFilter:
    return PoliticalFilter(load_political_keywords())


def _market(**overrides) -> GammaMarket:
    defaults = dict(
        id="m",
        question="Untitled market",
        description="",
        outcomes=["Yes", "No"],
        outcomePrices=["0.5", "0.5"],
        volume=0.0,
        volume24hr=0.0,
        liquidity=0.0,
        active=True,
        closed=False,
        tags=[],
        slug="m",
        category="uncategorized",
        conditionId="0xm",
        clobTokenIds=["y", "n"],
    )
    defaults.update(overrides)
    return GammaMarket(**defaults)


class TestSelectPoliticalMarkets:
    def test_selects_only_tag_matched_high_volume_market(
        self, backfill_markets_module, political_filter
    ):
        # (a) tag-matched political + high volume -> selected
        market_a = _market(
            id="a",
            question="Will the president sign the bill?",
            tags=[{"label": "Politics"}],
            volume=500_000.0,
        )
        # (b) tag-matched political + low volume -> excluded by volume gate
        market_b = _market(
            id="b",
            question="Will the president veto the bill?",
            tags=[{"label": "Politics"}],
            volume=100.0,
        )
        # (c) non-political -> excluded
        market_c = _market(
            id="c",
            question="Will Bitcoin reach $100k?",
            tags=[{"label": "Crypto"}],
            category="crypto",
            volume=1_000_000.0,
        )
        # (d) keyword-only, exactly 2 keyword matches, no tag/category hit
        # -> excluded (confidence 0.2 < 0.3 gate). Old script would have
        # selected this one; unified PoliticalFilter does not.
        market_d = _market(
            id="d",
            question="Will the senate hold hearings before the midterm?",
            tags=[],
            category="uncategorized",
            volume=500_000.0,
        )

        # Sanity-check the classification assumptions this test relies on.
        classification_d = political_filter.classify(market_d)
        assert classification_d.confidence == pytest.approx(0.2)
        assert classification_d.is_political is False

        selected = backfill_markets_module.select_political_markets(
            [market_a, market_b, market_c, market_d], political_filter
        )
        selected_ids = {m.id for m, _ in selected}

        assert selected_ids == {"a"}
