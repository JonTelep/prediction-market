"""Tests for the simulation integration layer.

Covers: SimulationEnhancedDetector, SimulationSignals, analyze_market,
generate_report.
"""

from __future__ import annotations

import numpy as np
import pytest

from prediction_market.simulation.integration import (
    SimulationEnhancedDetector,
    SimulationSignals,
    analyze_market,
    generate_report,
    _score_to_risk,
)


class TestSimulationEnhancedDetector:
    def test_first_tick_returns_signals(self):
        det = SimulationEnhancedDetector(mc_simulations=100, n_particles=100, seed=42)
        sig = det.process_tick("m1", price=0.5, volume=100)

        assert isinstance(sig, SimulationSignals)
        assert sig.market_id == "m1"
        assert sig.pf_surprise == 0.0  # First tick = initialization

    def test_accumulates_history(self):
        det = SimulationEnhancedDetector(
            mc_simulations=100, n_particles=100, min_history=10, seed=42
        )

        for i in range(15):
            sig = det.process_tick("m1", price=0.5 + 0.001 * i)

        state = det.get_state("m1")
        assert state is not None
        assert len(state.price_history) == 15

    def test_mc_fits_after_min_history(self):
        det = SimulationEnhancedDetector(
            mc_simulations=500, n_particles=100, min_history=10, seed=42
        )

        rng = np.random.default_rng(42)
        for i in range(15):
            sig = det.process_tick("m1", price=float(rng.beta(3, 7)))

        # After 15 ticks (> min_history 10), MC should be fitted
        assert sig.mc_anomaly_score >= 0  # Has a value now
        state = det.get_state("m1")
        assert state.model_fitted

    def test_spike_produces_high_score(self):
        det = SimulationEnhancedDetector(
            mc_simulations=1000, n_particles=500, min_history=10, seed=42
        )

        # Establish baseline
        for _ in range(20):
            det.process_tick("m1", price=0.5)

        # Spike
        sig = det.process_tick("m1", price=0.85)
        assert sig.combined_sim_score > 0.1

    def test_market_context_integration(self):
        det = SimulationEnhancedDetector(
            mc_simulations=100, n_particles=100, seed=42
        )

        ob_row = {
            "imbalance": 0.6,
            "spread_pct": 0.03,
            "depth_5pct": 5000,
            "total_bid_depth": 8000,
            "total_ask_depth": 4000,
            "susceptibility_score": 0.4,
        }

        sig = det.process_tick("m1", price=0.5, orderbook_row=ob_row)
        assert sig.market_id == "m1"

    def test_tracked_markets(self):
        det = SimulationEnhancedDetector(mc_simulations=100, n_particles=100, seed=42)
        det.process_tick("a", price=0.5)
        det.process_tick("b", price=0.3)

        assert set(det.tracked_markets) == {"a", "b"}

    def test_cross_market(self):
        det = SimulationEnhancedDetector(seed=42)

        # Feed some cross-market returns
        rng = np.random.default_rng(42)
        for _ in range(60):
            r_a = rng.standard_normal() * 0.02
            r_b = r_a + rng.standard_normal() * 0.005  # Correlated
            result = det.process_cross_market("m1", "m2", r_a, r_b)

        # Should have a tail dependence estimate
        td = det.get_cross_market_dependence("m1", "m2")
        assert td is not None
        assert td.pearson > 0.5

    def test_calibrate_abm(self):
        det = SimulationEnhancedDetector(
            mc_simulations=100, n_particles=100, min_history=10, seed=42
        )

        rng = np.random.default_rng(42)
        for _ in range(60):
            det.process_tick("cal", price=float(rng.beta(3, 7)), volume=float(rng.exponential(500)))

        result = det.calibrate_abm("cal", max_iterations=3)
        assert result is not None
        assert result.distance >= 0

    def test_abm_divergence(self):
        det = SimulationEnhancedDetector(
            mc_simulations=100, n_particles=100, min_history=10, seed=42
        )

        rng = np.random.default_rng(42)
        for _ in range(60):
            det.process_tick("div", price=float(rng.beta(3, 7)), volume=float(rng.exponential(500)))

        det.calibrate_abm("div", max_iterations=3)
        metrics = det.compute_abm_divergence("div", n_baseline_runs=3)
        assert metrics is not None
        assert 0 <= metrics.composite_score <= 1

    def test_to_dict(self):
        det = SimulationEnhancedDetector(mc_simulations=100, n_particles=100, seed=42)
        det.process_tick("m1", price=0.5)
        d = det.to_dict()
        assert "states" in d
        assert "m1" in d["states"]


class TestSimulationSignals:
    def test_to_dict(self):
        sig = SimulationSignals(
            market_id="test",
            mc_anomaly_score=2.5,
            pf_surprise=3.1,
            pf_drift_score=-5.5,
        )
        d = sig.to_dict()
        assert d["market_id"] == "test"
        assert d["mc_anomaly_score"] == 2.5

    def test_summary_normal(self):
        sig = SimulationSignals(market_id="test")
        s = sig.to_summary()
        assert "NORMAL" in s

    def test_summary_with_flags(self):
        sig = SimulationSignals(
            market_id="test",
            mc_anomaly_score=3.0,
            pf_surprise=5.0,
            pf_drift_score=10.0,
            pf_regime="high",
            combined_sim_score=0.5,
        )
        s = sig.to_summary()
        assert "MC:" in s
        assert "drift=" in s
        assert "HIGH-VOL" in s


class TestAnalyzeMarket:
    def test_basic_analysis(self):
        rng = np.random.default_rng(42)
        prices = rng.beta(3, 7, size=100)

        result = analyze_market(
            prices=prices,
            market_id="test-analysis",
            mc_simulations=500,
            n_particles=200,
            seed=42,
        )

        assert result["market_id"] == "test-analysis"
        assert result["n_observations"] == 100
        assert result["current_price"] > 0
        assert result["combined_score"] >= 0
        assert result["risk_level"] is not None
        assert result["monte_carlo"] is not None
        assert result["particle_filter"] is not None

    def test_with_volumes(self):
        rng = np.random.default_rng(42)
        prices = rng.beta(3, 7, size=80)
        volumes = rng.exponential(500, size=80)

        result = analyze_market(
            prices=prices,
            volumes=volumes,
            mc_simulations=500,
            n_particles=200,
            seed=42,
        )

        assert result["n_observations"] == 80
        assert result["abm_calibration"] is not None
        assert result["abm_divergence"] is not None

    def test_report_generation(self):
        rng = np.random.default_rng(42)
        prices = rng.beta(3, 7, size=60)

        result = analyze_market(
            prices=prices,
            mc_simulations=500,
            n_particles=200,
            seed=42,
        )

        report = generate_report(result)
        assert "# Simulation Analysis" in report
        assert "Monte Carlo" in report
        assert "Particle Filter" in report
        assert "Risk Level" in report


class TestScoreToRisk:
    def test_normal(self):
        assert "NORMAL" in _score_to_risk(0.05)

    def test_low(self):
        assert "LOW" in _score_to_risk(0.2)

    def test_elevated(self):
        assert "ELEVATED" in _score_to_risk(0.4)

    def test_high(self):
        assert "HIGH" in _score_to_risk(0.6)

    def test_critical(self):
        assert "CRITICAL" in _score_to_risk(0.8)
