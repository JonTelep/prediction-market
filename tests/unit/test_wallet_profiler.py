"""Tests for per-wallet anomaly feature extraction."""

from __future__ import annotations

import json
from pathlib import Path

from prediction_market.analysis.wallet_profiler import profile_wallets
from prediction_market.config import ThresholdConfig


def _row(
    wallet: str,
    trade_count: int,
    total_volume_usd: float,
    buy_volume_usd: float,
    first_trade: str,
    last_trade: str = "2026-01-02 00:00:00",
) -> dict:
    return {
        "proxy_wallet": wallet,
        "trade_count": trade_count,
        "total_volume_usd": total_volume_usd,
        "buy_volume_usd": buy_volume_usd,
        "first_trade": first_trade,
        "last_trade": last_trade,
    }


WINDOW_END = "2026-01-02 00:00:00"


class TestHandComputed:
    """Numeric proof: three constructed rows, arithmetic checked by hand."""

    def test_exact_values(self):
        # Total volume across all 3 rows = 10000.
        rows = [
            # W1: 5000/10000 share, fully one-sided buy, fresh (0h before window_end).
            _row("W1", 10, 5000.0, 5000.0, first_trade="2026-01-02 00:00:00"),
            # W2: 3000/10000 share, perfectly balanced 50/50, old (not fresh).
            _row("W2", 10, 3000.0, 1500.0, first_trade="2020-01-01 00:00:00"),
            # W3: 2000/10000 share, fully one-sided buy, old (not fresh).
            _row("W3", 5, 2000.0, 2000.0, first_trade="2020-01-01 00:00:00"),
        ]
        thresholds = ThresholdConfig(wallet_freshness_hours=48, wallet_min_volume_usd=1000.0)
        results = profile_wallets(rows, window_end=WINDOW_END, thresholds=thresholds)

        by_wallet = {f.wallet: f for f in results}

        # W1: volume_share = 5000/10000 = 0.5
        #     directional_concentration: buy=5000, sell=0, c=1.0 -> (1.0-0.5)*2 = 1.0
        #     is_fresh = True
        #     score = 0.40*0.5 + 0.30*1.0 + 0.30*1.0 = 0.2 + 0.3 + 0.3 = 0.8
        w1 = by_wallet["W1"]
        assert w1.volume_share == 0.5
        assert w1.directional_concentration == 1.0
        assert w1.is_fresh is True
        assert w1.score == 0.8

        # W2: volume_share = 3000/10000 = 0.3
        #     directional_concentration: buy=1500, sell=1500, c=0.5 -> (0.5-0.5)*2 = 0.0
        #     is_fresh = False
        #     score = 0.40*0.3 + 0.30*0.0 + 0.30*0.0 = 0.12
        w2 = by_wallet["W2"]
        assert w2.volume_share == 0.3
        assert w2.directional_concentration == 0.0
        assert w2.is_fresh is False
        assert abs(w2.score - 0.12) < 1e-9

        # W3: volume_share = 2000/10000 = 0.2
        #     directional_concentration: buy=2000, sell=0, c=1.0 -> 1.0
        #     is_fresh = False
        #     score = 0.40*0.2 + 0.30*1.0 + 0.30*0.0 = 0.08 + 0.3 = 0.38
        w3 = by_wallet["W3"]
        assert w3.volume_share == 0.2
        assert w3.directional_concentration == 1.0
        assert w3.is_fresh is False
        assert abs(w3.score - 0.38) < 1e-9

        # Ranked by score DESC: W1 (0.8) > W3 (0.38) > W2 (0.12)
        assert [f.wallet for f in results] == ["W1", "W3", "W2"]


class TestDominantInsiderShape:
    def test_dominant_fresh_one_sided_wallet_ranks_first_and_scores_high(self):
        # Dominant wallet holds 80% of window volume, fully one-sided, fresh.
        # (Note: with fixed weights 0.4/0.3/0.3, a wallet needs >= 75% volume
        # share -- not merely a bare majority -- to clear a 0.9 score even
        # with maximal concentration and freshness, since 0.4*0.6+0.3+0.3=0.84
        # < 0.9. We use 80% share here so the >=0.9 assertion is achievable.)
        rows = [
            _row("insider", 20, 8000.0, 8000.0, first_trade="2026-01-02 00:00:00"),
            _row("diffuse", 40, 2000.0, 1000.0, first_trade="2020-01-01 00:00:00"),
        ]
        thresholds = ThresholdConfig()
        results = profile_wallets(rows, window_end=WINDOW_END, thresholds=thresholds)

        assert results[0].wallet == "insider"
        assert results[0].score >= 0.9

        diffuse = next(f for f in results if f.wallet == "diffuse")
        assert diffuse.score < 0.2


class TestDustExclusion:
    def test_dust_wallet_excluded_but_deflates_others_share(self):
        rows = [
            _row("whale", 10, 3500.0, 3500.0, first_trade="2020-01-01 00:00:00"),
            _row("dust", 2, 500.0, 500.0, first_trade="2020-01-01 00:00:00"),
        ]
        thresholds = ThresholdConfig(wallet_min_volume_usd=1000.0)
        results = profile_wallets(rows, window_end=WINDOW_END, thresholds=thresholds)

        wallets = [f.wallet for f in results]
        assert "dust" not in wallets
        assert wallets == ["whale"]

        whale = results[0]
        # Denominator includes the dust wallet's volume: 3500 / (3500+500) = 0.875
        assert whale.volume_share == 0.875


class TestFreshnessBoundary:
    def test_exactly_at_threshold_is_fresh(self):
        rows = [_row("w", 1, 2000.0, 2000.0, first_trade="2025-12-31 00:00:00")]
        thresholds = ThresholdConfig(wallet_freshness_hours=48, wallet_min_volume_usd=0.0)
        results = profile_wallets(rows, window_end="2026-01-02 00:00:00", thresholds=thresholds)
        assert results[0].is_fresh is True

    def test_one_second_past_threshold_is_not_fresh(self):
        rows = [_row("w", 1, 2000.0, 2000.0, first_trade="2025-12-30 23:59:59")]
        thresholds = ThresholdConfig(wallet_freshness_hours=48, wallet_min_volume_usd=0.0)
        results = profile_wallets(rows, window_end="2026-01-02 00:00:00", thresholds=thresholds)
        assert results[0].is_fresh is False

    def test_unparseable_first_trade_is_not_fresh_no_exception(self):
        rows = [_row("w", 1, 2000.0, 2000.0, first_trade="2025-12-31T00:00:00")]
        thresholds = ThresholdConfig(wallet_min_volume_usd=0.0)
        results = profile_wallets(rows, window_end=WINDOW_END, thresholds=thresholds)
        assert results[0].is_fresh is False


class TestEmptyAndZeroVolume:
    def test_empty_input_returns_empty_list(self):
        assert profile_wallets([], window_end=WINDOW_END, thresholds=ThresholdConfig()) == []

    def test_zero_total_volume_no_zero_division(self):
        rows = [
            _row("a", 0, 0.0, 0.0, first_trade="2020-01-01 00:00:00"),
            _row("b", 0, 0.0, 0.0, first_trade="2020-01-01 00:00:00"),
        ]
        thresholds = ThresholdConfig(wallet_min_volume_usd=0.0)
        results = profile_wallets(rows, window_end=WINDOW_END, thresholds=thresholds)
        assert len(results) == 2
        for f in results:
            assert f.volume_share == 0.0
            assert f.directional_concentration == 0.0
            assert f.score == 0.0


class TestToDict:
    def test_to_dict_survives_json_roundtrip_unmodified(self):
        rows = [_row("w", 3, 2000.0, 1000.0, first_trade="2020-01-01 00:00:00")]
        thresholds = ThresholdConfig(wallet_min_volume_usd=0.0)
        results = profile_wallets(rows, window_end=WINDOW_END, thresholds=thresholds)
        d = results[0].to_dict()
        assert json.loads(json.dumps(d)) == d


class TestPurityBoundary:
    def test_no_store_or_aiosqlite_imports(self):
        """Enforce the purity rule mechanically: no import statement in the
        module may pull in ``aiosqlite`` or ``prediction_market.store``.
        (Prose mentions of these names in docstrings are fine and expected
        -- only actual `import`/`from ... import` lines are disallowed.)
        """
        import ast

        import prediction_market.analysis.wallet_profiler as mod

        path = Path(mod.__file__)
        tree = ast.parse(path.read_text())

        imported_modules: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_modules.append(node.module)

        assert not any(m == "aiosqlite" or m.startswith("aiosqlite.") for m in imported_modules)
        assert not any(
            m == "prediction_market.store" or m.startswith("prediction_market.store.")
            for m in imported_modules
        )
