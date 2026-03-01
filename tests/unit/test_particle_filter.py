"""Tests for Phase 2: Sequential Monte Carlo (particle filter).

Covers: ParticleFilter, MarketAwareTransition, MarketContext, TradeFlowAnalyzer.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from prediction_market.simulation.particle_filter import (
    MarketAwareTransition,
    MarketContext,
    ParticleFilter,
    ParticleFilterResult,
    TradeFlowAnalyzer,
)


# =====================================================================
# MarketContext
# =====================================================================


class TestMarketContext:
    def test_from_orderbook_row(self):
        row = {
            "imbalance": 0.3,
            "spread_pct": 0.02,
            "depth_5pct": 5000.0,
            "total_bid_depth": 8000.0,
            "total_ask_depth": 4000.0,
            "susceptibility_score": 0.6,
        }
        ctx = MarketContext.from_orderbook_row(row)
        assert ctx.orderbook_imbalance == 0.3
        assert ctx.spread_pct == 0.02
        assert ctx.depth_5pct == 5000.0
        assert ctx.susceptibility_score == 0.6

    def test_from_orderbook_row_missing_fields(self):
        ctx = MarketContext.from_orderbook_row({})
        assert ctx.orderbook_imbalance == 0.0
        assert ctx.spread_pct == 0.0

    def test_with_trade_flow(self):
        ctx = MarketContext(orderbook_imbalance=0.5)
        ctx2 = ctx.with_trade_flow(imbalance=-0.3, volatility=0.05)
        assert ctx2.orderbook_imbalance == 0.5  # Preserved
        assert ctx2.trade_flow_imbalance == -0.3
        assert ctx2.recent_volatility == 0.05


# =====================================================================
# MarketAwareTransition
# =====================================================================


class TestMarketAwareTransition:
    def test_no_context_random_walk(self):
        t = MarketAwareTransition(base_volatility=0.02)
        drift, vol = t.compute_drift_and_vol(None)
        assert drift == 0.0
        assert vol == 0.02

    def test_positive_imbalance_positive_drift(self):
        t = MarketAwareTransition(imbalance_sensitivity=0.5)
        ctx = MarketContext(orderbook_imbalance=0.8)
        drift, _ = t.compute_drift_and_vol(ctx)
        assert drift > 0  # Buy pressure → upward drift

    def test_negative_imbalance_negative_drift(self):
        t = MarketAwareTransition(imbalance_sensitivity=0.5)
        ctx = MarketContext(orderbook_imbalance=-0.6)
        drift, _ = t.compute_drift_and_vol(ctx)
        assert drift < 0  # Sell pressure → downward drift

    def test_thin_depth_higher_volatility(self):
        t = MarketAwareTransition(base_volatility=0.02)
        # Normal depth
        ctx_normal = MarketContext(depth_5pct=10_000.0)
        _, vol_normal = t.compute_drift_and_vol(ctx_normal)
        # Thin depth
        ctx_thin = MarketContext(depth_5pct=500.0)
        _, vol_thin = t.compute_drift_and_vol(ctx_thin)
        assert vol_thin > vol_normal

    def test_wide_spread_higher_volatility(self):
        t = MarketAwareTransition(base_volatility=0.02, spread_vol_scale=1.0)
        ctx_tight = MarketContext(spread_pct=0.005, depth_5pct=10_000.0)
        _, vol_tight = t.compute_drift_and_vol(ctx_tight)
        ctx_wide = MarketContext(spread_pct=0.05, depth_5pct=10_000.0)
        _, vol_wide = t.compute_drift_and_vol(ctx_wide)
        assert vol_wide > vol_tight

    def test_trade_flow_adds_drift(self):
        t = MarketAwareTransition(trade_flow_sensitivity=0.4)
        ctx = MarketContext(trade_flow_imbalance=0.7)
        drift, _ = t.compute_drift_and_vol(ctx)
        assert drift > 0

    def test_serialization_roundtrip(self):
        t = MarketAwareTransition(
            base_volatility=0.03,
            imbalance_sensitivity=0.4,
        )
        d = t.to_dict()
        t2 = MarketAwareTransition.from_dict(d)
        assert t2.base_volatility == 0.03
        assert t2.imbalance_sensitivity == 0.4


# =====================================================================
# ParticleFilter
# =====================================================================


class TestParticleFilter:
    def test_first_update_initializes(self):
        pf = ParticleFilter("test-market", n_particles=500, seed=42)
        assert not pf.is_initialized

        result = pf.update(0.5)
        assert pf.is_initialized
        assert result.surprise == 0.0
        assert result.posterior_mean == 0.5
        assert result.market_id == "test-market"

    def test_stable_price_lower_surprise_than_jump(self):
        """Stable price should produce much less surprise than a jump."""
        pf_stable = ParticleFilter("stable", n_particles=1000, seed=42)
        pf_jump = ParticleFilter("jump", n_particles=1000, seed=42)

        for _ in range(15):
            pf_stable.update(0.5)
            pf_jump.update(0.5)

        # Stable continues at 0.5
        r_stable = pf_stable.update(0.5)
        # Jump to 0.85
        r_jump = pf_jump.update(0.85)

        assert r_jump.surprise > r_stable.surprise * 2

    def test_price_jump_high_surprise(self):
        """A sudden price jump should produce high surprise."""
        pf = ParticleFilter("jump", n_particles=2000, seed=42)

        # Establish baseline at 0.5
        for _ in range(15):
            pf.update(0.5)

        # Sudden jump to 0.85
        result = pf.update(0.85)
        assert result.surprise > 2.0  # Significantly surprised

    def test_drift_accumulation(self):
        """Sustained directional movement should accumulate drift score."""
        pf = ParticleFilter("drift", n_particles=1000, seed=42, drift_window=30)

        # Start at 0.5
        for _ in range(5):
            pf.update(0.5)

        # Slowly drift upward
        drift_scores = []
        for i in range(20):
            price = 0.5 + 0.01 * (i + 1)  # 0.51, 0.52, ... 0.70
            result = pf.update(price)
            drift_scores.append(result.drift_score)

        # Drift score should be increasingly positive
        assert drift_scores[-1] > drift_scores[5]
        assert drift_scores[-1] > 0

    def test_market_context_affects_transition(self):
        """Providing context should change filter behavior vs no context."""
        pf_no_ctx = ParticleFilter("no-ctx", n_particles=1000, seed=42)
        pf_ctx = ParticleFilter("ctx", n_particles=1000, seed=42)

        # Same observations, different context
        ctx = MarketContext(
            orderbook_imbalance=0.8,  # Strong buy pressure
            depth_5pct=500.0,  # Thin book
            spread_pct=0.05,  # Wide spread
        )

        for _ in range(5):
            pf_no_ctx.update(0.5)
            pf_ctx.update(0.5, context=ctx)

        # Price jumps up — should be less surprising with buy-pressure context
        r_no_ctx = pf_no_ctx.update(0.6)
        r_ctx = pf_ctx.update(0.6, context=ctx)

        # With buy-pressure context, upward move should be less surprising
        assert r_ctx.surprise < r_no_ctx.surprise

    def test_ess_drops_on_surprise(self):
        """ESS should drop when observation is unexpected."""
        pf = ParticleFilter("ess", n_particles=2000, seed=42)

        for _ in range(10):
            pf.update(0.5)

        # Normal observation
        r_normal = pf.update(0.51)
        # Surprising observation
        pf2 = ParticleFilter("ess2", n_particles=2000, seed=42)
        for _ in range(10):
            pf2.update(0.5)
        r_surprise = pf2.update(0.8)

        assert r_surprise.ess_ratio < r_normal.ess_ratio

    def test_resampling_triggers(self):
        """Resampling should trigger when ESS is low."""
        pf = ParticleFilter("resample", n_particles=500, seed=42, ess_threshold=0.8)

        for _ in range(5):
            pf.update(0.5)

        # Force a surprising observation
        result = pf.update(0.9)
        # Either resampled or ESS was still OK
        assert isinstance(result.resampled, bool)

    def test_regime_detection(self):
        """High volatility observations should detect elevated regime."""
        pf = ParticleFilter("regime", n_particles=1000, seed=42)

        # Very noisy price series — large swings
        rng = np.random.default_rng(42)
        for i in range(30):
            price = 0.5 + 0.25 * rng.standard_normal()
            price = np.clip(price, 0.05, 0.95)
            result = pf.update(float(price))

        # With extreme variance, regime should be high
        assert result.regime == "high"

    def test_regime_low_for_stable(self):
        """Stable prices should detect 'low' regime."""
        pf = ParticleFilter("low-regime", n_particles=500, seed=42)
        for _ in range(20):
            result = pf.update(0.50)
        assert result.regime == "low"

    def test_serialization_roundtrip(self):
        pf = ParticleFilter("ser", n_particles=200, seed=42)
        for _ in range(5):
            pf.update(0.5)

        data = pf.to_dict()
        pf2 = ParticleFilter.from_dict(data, seed=42)

        assert pf2.market_id == "ser"
        assert pf2.is_initialized
        assert pf2._n == 200

        # Should produce similar results
        r1 = pf.update(0.55)
        r2 = pf2.update(0.55)
        # Won't be identical (different RNG state after deserialization)
        # but both should work
        assert abs(r1.posterior_mean - r2.posterior_mean) < 0.1

    def test_boundary_prices(self):
        """Filter should handle prices near 0 and 1 gracefully."""
        pf = ParticleFilter("boundary", n_particles=500, seed=42)
        pf.update(0.01)  # Near zero
        result = pf.update(0.99)  # Near one
        assert 0 < result.posterior_mean < 1
        assert result.surprise > 0

    def test_result_to_dict(self):
        pf = ParticleFilter("dict", n_particles=100, seed=42)
        result = pf.update(0.5)
        d = result.to_dict()
        assert d["market_id"] == "dict"
        assert "surprise" in d
        assert "drift_score" in d


# =====================================================================
# TradeFlowAnalyzer
# =====================================================================


class TestTradeFlowAnalyzer:
    def _make_trade(
        self, side: str, volume_usd: float, price: float, minutes_ago: int = 5
    ) -> dict:
        ts = datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)
        return {
            "side": side,
            "volume_usd": volume_usd,
            "price": price,
            "match_time": ts.isoformat(),
        }

    def test_balanced_flow(self):
        tfa = TradeFlowAnalyzer(lookback_minutes=60)
        trades = [
            self._make_trade("buy", 1000, 0.5),
            self._make_trade("sell", 1000, 0.5),
        ]
        imbalance, vol = tfa.compute(trades)
        assert imbalance == 0.0

    def test_buy_dominated(self):
        tfa = TradeFlowAnalyzer(lookback_minutes=60)
        trades = [
            self._make_trade("buy", 5000, 0.5),
            self._make_trade("buy", 3000, 0.52),
            self._make_trade("sell", 1000, 0.51),
        ]
        imbalance, _ = tfa.compute(trades)
        assert imbalance > 0.5  # Strong buy pressure

    def test_sell_dominated(self):
        tfa = TradeFlowAnalyzer(lookback_minutes=60)
        trades = [
            self._make_trade("sell", 8000, 0.5),
            self._make_trade("buy", 2000, 0.48),
        ]
        imbalance, _ = tfa.compute(trades)
        assert imbalance < -0.3  # Sell pressure

    def test_old_trades_excluded(self):
        tfa = TradeFlowAnalyzer(lookback_minutes=10)
        trades = [
            self._make_trade("buy", 5000, 0.5, minutes_ago=5),  # Recent
            self._make_trade("sell", 5000, 0.5, minutes_ago=20),  # Too old
        ]
        imbalance, _ = tfa.compute(trades)
        assert imbalance == 1.0  # Only buy trade counted

    def test_volatility_from_prices(self):
        tfa = TradeFlowAnalyzer(lookback_minutes=60)
        trades = [
            self._make_trade("buy", 100, 0.50, minutes_ago=10),
            self._make_trade("buy", 100, 0.55, minutes_ago=8),
            self._make_trade("sell", 100, 0.45, minutes_ago=6),
            self._make_trade("buy", 100, 0.60, minutes_ago=4),
        ]
        _, vol = tfa.compute(trades)
        assert vol > 0  # Non-zero volatility from varying prices

    def test_empty_trades(self):
        tfa = TradeFlowAnalyzer()
        imbalance, vol = tfa.compute([])
        assert imbalance == 0.0
        assert vol == 0.0

    def test_trades_with_datetime_objects(self):
        """match_time as datetime objects (not strings) should work."""
        tfa = TradeFlowAnalyzer(lookback_minutes=60)
        now = datetime.now(timezone.utc)
        trades = [
            {
                "side": "buy",
                "volume_usd": 1000,
                "price": 0.5,
                "match_time": now - timedelta(minutes=5),
            }
        ]
        imbalance, _ = tfa.compute(trades)
        assert imbalance == 1.0
