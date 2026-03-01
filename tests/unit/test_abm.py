"""Tests for Phase 4: Agent-Based Market Simulator.

Covers: trader agents, simulated market, ABM simulator, divergence metrics,
and calibrator.
"""

from __future__ import annotations

import numpy as np
import pytest

from prediction_market.simulation.abm.agents import (
    InformedTrader,
    MarketMaker,
    MomentumTrader,
    NoiseTrader,
    Order,
)
from prediction_market.simulation.abm.market import MarketState, SimulatedMarket
from prediction_market.simulation.abm.simulator import (
    ABMConfig,
    ABMResult,
    ABMSimulator,
    DivergenceMetrics,
)
from prediction_market.simulation.abm.calibrator import (
    Calibrator,
    CalibrationResult,
    TargetStatistics,
)


# =====================================================================
# Trader Agents
# =====================================================================


class TestNoiseTrader:
    def test_trades_randomly(self):
        rng = np.random.default_rng(42)
        trader = NoiseTrader(agent_id=0, rng=rng, trade_probability=1.0)
        state = MarketState(price=0.5)

        order = trader.decide(state)
        assert order is not None
        assert order.side in ("buy", "sell")
        assert order.size > 0
        assert order.agent_type == "noise"

    def test_sometimes_sits_out(self):
        rng = np.random.default_rng(42)
        trader = NoiseTrader(agent_id=0, rng=rng, trade_probability=0.0)
        state = MarketState(price=0.5)

        order = trader.decide(state)
        assert order is None

    def test_position_tracking(self):
        rng = np.random.default_rng(42)
        trader = NoiseTrader(agent_id=0, rng=rng)
        trader.update_position("buy", 100, 0.5)
        assert trader.position == 100
        assert trader.pnl == -50.0

        trader.update_position("sell", 100, 0.6)
        assert trader.position == 0
        assert trader.pnl == 10.0  # Bought at 0.5, sold at 0.6


class TestMarketMaker:
    def test_quotes_both_sides(self):
        rng = np.random.default_rng(42)
        mm = MarketMaker(agent_id=0, rng=rng, quote_probability=1.0)
        state = MarketState(price=0.5)

        sides = set()
        for _ in range(50):
            order = mm.decide(state)
            if order:
                sides.add(order.side)
        assert "buy" in sides and "sell" in sides

    def test_tracks_fair_value(self):
        rng = np.random.default_rng(42)
        mm = MarketMaker(agent_id=0, rng=rng, fair_value=0.5)

        # Market drifts to 0.7
        for _ in range(20):
            mm.decide(MarketState(price=0.7))

        assert mm.fair_value > 0.55  # Should have tracked upward

    def test_inventory_management(self):
        rng = np.random.default_rng(42)
        mm = MarketMaker(
            agent_id=0, rng=rng, max_position=100, quote_probability=1.0
        )
        mm.position = 90  # Near long limit

        # Should strongly prefer selling
        sell_count = 0
        for _ in range(100):
            order = mm.decide(MarketState(price=0.5))
            if order and order.side == "sell":
                sell_count += 1
        assert sell_count > 70  # Most should be sells


class TestMomentumTrader:
    def test_needs_warmup(self):
        rng = np.random.default_rng(42)
        mt = MomentumTrader(agent_id=0, rng=rng, lookback=10)

        # First few ticks: no signal
        for _ in range(5):
            order = mt.decide(MarketState(price=0.5))
        assert order is None  # Not enough history

    def test_follows_trend(self):
        rng = np.random.default_rng(42)
        mt = MomentumTrader(
            agent_id=0, rng=rng, lookback=10,
            threshold=0.001, trade_probability=1.0,
        )

        # Feed rising prices
        for i in range(15):
            mt.decide(MarketState(price=0.50 + 0.01 * i))

        order = mt.decide(MarketState(price=0.66))
        assert order is not None
        assert order.side == "buy"  # Following uptrend


class TestInformedTrader:
    def test_buys_when_underpriced(self):
        rng = np.random.default_rng(42)
        trader = InformedTrader(
            agent_id=0, rng=rng, true_value=0.8,
            edge_threshold=0.01, trade_probability=1.0,
        )

        order = trader.decide(MarketState(price=0.5))
        assert order is not None
        assert order.side == "buy"

    def test_sells_when_overpriced(self):
        rng = np.random.default_rng(42)
        trader = InformedTrader(
            agent_id=0, rng=rng, true_value=0.3,
            edge_threshold=0.01, trade_probability=1.0,
        )

        order = trader.decide(MarketState(price=0.6))
        assert order is not None
        assert order.side == "sell"

    def test_no_trade_when_fair(self):
        rng = np.random.default_rng(42)
        trader = InformedTrader(
            agent_id=0, rng=rng, true_value=0.5,
            edge_threshold=0.05, trade_probability=1.0,
        )

        order = trader.decide(MarketState(price=0.5))
        assert order is None

    def test_stealth_reduces_size(self):
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)

        loud = InformedTrader(agent_id=0, rng=rng1, true_value=0.8,
                              stealth=0.0, trade_probability=1.0)
        quiet = InformedTrader(agent_id=1, rng=rng2, true_value=0.8,
                               stealth=0.9, trade_probability=1.0)

        o1 = loud.decide(MarketState(price=0.5))
        o2 = quiet.decide(MarketState(price=0.5))

        assert o1 is not None and o2 is not None
        assert o2.aggressiveness < o1.aggressiveness


# =====================================================================
# SimulatedMarket
# =====================================================================


class TestSimulatedMarket:
    def test_initial_state(self):
        m = SimulatedMarket(initial_price=0.6)
        assert abs(m.price - 0.6) < 0.01
        assert m.state.tick == 0

    def test_buy_moves_price_up(self):
        m = SimulatedMarket(initial_price=0.5, base_liquidity=1000)
        old_price = m.price

        order = Order(agent_id=0, agent_type="test", side="buy",
                      size=100, aggressiveness=0.8)
        m.process_order(order)

        assert m.price > old_price

    def test_sell_moves_price_down(self):
        m = SimulatedMarket(initial_price=0.5, base_liquidity=1000)
        old_price = m.price

        order = Order(agent_id=0, agent_type="test", side="sell",
                      size=100, aggressiveness=0.8)
        m.process_order(order)

        assert m.price < old_price

    def test_price_stays_bounded(self):
        m = SimulatedMarket(initial_price=0.5, base_liquidity=500)

        # Massive buy
        order = Order(agent_id=0, agent_type="test", side="buy",
                      size=10000, aggressiveness=1.0)
        m.process_order(order)
        assert 0 < m.price < 1

        # Massive sell
        order = Order(agent_id=0, agent_type="test", side="sell",
                      size=20000, aggressiveness=1.0)
        m.process_order(order)
        assert 0 < m.price < 1

    def test_end_tick_records_state(self):
        m = SimulatedMarket(initial_price=0.5)
        order = Order(agent_id=0, agent_type="test", side="buy",
                      size=50, aggressiveness=0.5)
        m.process_order(order)

        state = m.end_tick()
        assert state.volume_this_tick == 50
        assert state.n_trades == 1
        assert len(m.tick_states) == 1

    def test_volume_tracking(self):
        m = SimulatedMarket(initial_price=0.5)
        m.process_order(Order(0, "t", "buy", 100, 0.5))
        m.process_order(Order(1, "t", "sell", 50, 0.5))
        state = m.end_tick()

        assert state.buy_volume == 100
        assert state.sell_volume == 50
        assert state.volume_this_tick == 150

    def test_price_history(self):
        m = SimulatedMarket(initial_price=0.5)
        for _ in range(5):
            m.process_order(Order(0, "t", "buy", 10, 0.3))
            m.end_tick()

        assert len(m.price_history) == 6  # initial + 5 ticks
        assert m.price_history[0] == 0.5

    def test_series_accessors(self):
        m = SimulatedMarket(initial_price=0.5)
        for _ in range(3):
            m.process_order(Order(0, "t", "buy", 10, 0.3))
            m.end_tick()

        assert len(m.get_volume_series()) == 3
        assert len(m.get_spread_series()) == 3
        assert len(m.get_imbalance_series()) == 3


# =====================================================================
# ABMSimulator
# =====================================================================


class TestABMSimulator:
    def test_basic_run(self):
        config = ABMConfig(n_ticks=100, seed=42)
        sim = ABMSimulator(config)
        result = sim.run()

        assert isinstance(result, ABMResult)
        assert len(result.price_series) == 101  # initial + 100 ticks
        assert result.total_volume > 0
        assert result.trade_count > 0
        assert result.elapsed_ms > 0

    def test_price_stays_bounded(self):
        config = ABMConfig(n_ticks=200, seed=42)
        result = ABMSimulator(config).run()

        for p in result.price_series:
            assert 0 < p < 1

    def test_trades_by_type(self):
        config = ABMConfig(
            n_ticks=100, seed=42,
            n_noise=10, n_market_makers=2, n_momentum=3,
        )
        result = ABMSimulator(config).run()

        assert "noise" in result.trades_by_type
        assert "market_maker" in result.trades_by_type

    def test_informed_traders_move_price(self):
        """With informed traders knowing true value = 0.8, price should drift up."""
        config_no_insider = ABMConfig(
            n_ticks=200, seed=42, initial_price=0.5,
            n_informed=0,
        )
        config_insider = ABMConfig(
            n_ticks=200, seed=42, initial_price=0.5,
            n_informed=5, informed_true_value=0.8,
        )

        r_no = ABMSimulator().run(config_no_insider)
        r_yes = ABMSimulator().run(config_insider)

        # Insider run should end with higher price
        assert r_yes.final_price > r_no.final_price

    def test_baseline_runs(self):
        config = ABMConfig(n_ticks=50, seed=42, n_noise=5, n_market_makers=1)
        sim = ABMSimulator(config)
        baselines = sim.run_baseline(n_runs=3)

        assert len(baselines) == 3
        # All should have n_informed=0
        for r in baselines:
            assert r.config.n_informed == 0

    def test_divergence_metrics_similar(self):
        """Comparing baseline against itself should show low divergence."""
        config = ABMConfig(n_ticks=100, seed=42)
        sim = ABMSimulator(config)

        baselines = sim.run_baseline(n_runs=5)
        # Use one baseline run as "observed"
        observed = baselines[0]

        metrics = sim.compare_with_observed(
            baselines[1:],
            observed_prices=np.array(observed.price_series),
            observed_volumes=np.array(observed.volume_series),
        )

        assert isinstance(metrics, DivergenceMetrics)
        # Should be relatively similar (low composite score)
        assert metrics.composite_score < 0.8

    def test_divergence_insider_vs_baseline(self):
        """Insider simulation should diverge from baseline."""
        config = ABMConfig(n_ticks=200, seed=42, base_liquidity=5000)
        sim = ABMSimulator(config)

        baselines = sim.run_baseline(n_runs=5)

        # Run with insiders
        insider_config = ABMConfig(
            n_ticks=200, seed=42, base_liquidity=5000,
            n_informed=5, informed_true_value=0.85,
            informed_stealth=0.2,  # Obvious insider
        )
        insider_result = sim.run(insider_config)

        metrics = sim.compare_with_observed(
            baselines,
            observed_prices=np.array(insider_result.price_series),
            observed_volumes=np.array(insider_result.volume_series),
        )

        # Insider run should show higher divergence than baseline-vs-baseline
        baseline_metrics = sim.compare_with_observed(
            baselines[1:],
            observed_prices=np.array(baselines[0].price_series),
            observed_volumes=np.array(baselines[0].volume_series),
        )

        assert metrics.composite_score > baseline_metrics.composite_score

    def test_result_to_dict(self):
        config = ABMConfig(n_ticks=50, seed=42)
        result = ABMSimulator(config).run()
        d = result.to_dict()
        assert "final_price" in d
        assert "trades_by_type" in d

    def test_divergence_to_dict(self):
        m = DivergenceMetrics(composite_score=0.42)
        d = m.to_dict()
        assert d["composite_score"] == 0.42


# =====================================================================
# Calibrator
# =====================================================================


class TestCalibrator:
    def test_target_from_observations(self):
        prices = np.linspace(0.45, 0.55, 100)
        volumes = np.random.default_rng(42).exponential(500, 99)

        target = TargetStatistics.from_observations(prices, volumes)

        assert 0.45 < target.mean_price < 0.55
        assert target.price_volatility > 0
        assert target.mean_volume_per_tick > 0

    def test_target_minimal(self):
        """Should work with just prices."""
        prices = np.array([0.5, 0.51, 0.49, 0.52, 0.48])
        target = TargetStatistics.from_observations(prices)
        assert target.mean_price > 0

    def test_calibration_runs(self):
        """Calibrator should complete without errors."""
        target = TargetStatistics(
            mean_price=0.5,
            price_volatility=0.02,
            mean_volume_per_tick=300,
            mean_spread=0.02,
        )

        cal = Calibrator(n_ticks=50, n_eval_runs=1, base_seed=42)
        result = cal.calibrate(target, max_iterations=5)

        assert isinstance(result, CalibrationResult)
        assert result.distance >= 0
        assert result.n_evaluations > 0
        assert result.config.n_informed == 0  # Baseline config

    def test_calibration_result_to_dict(self):
        target = TargetStatistics(mean_price=0.5, price_volatility=0.01)
        cal = Calibrator(n_ticks=30, n_eval_runs=1, base_seed=42)
        result = cal.calibrate(target, max_iterations=2)

        d = result.to_dict()
        assert "config" in d
        assert "target" in d
        assert "distance" in d
