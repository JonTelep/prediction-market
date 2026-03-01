"""Tests for the Monte Carlo simulation engine (Phase 1).

Covers: distributions (Beta, Dirichlet), MonteCarloEngine, ImportanceSampler.
"""

from __future__ import annotations

import numpy as np
import pytest

from prediction_market.simulation.distributions import (
    BetaMarketModel,
    DirichletMarketModel,
    FitResult,
)
from prediction_market.simulation.monte_carlo import MonteCarloEngine, SimulationResult
from prediction_market.simulation.importance_sampler import ImportanceSampler, TailRiskEstimate


# =====================================================================
# BetaMarketModel
# =====================================================================


class TestBetaMarketModel:
    def test_init_defaults(self):
        m = BetaMarketModel()
        assert m.alpha == 1.0
        assert m.beta == 1.0

    def test_init_custom(self):
        m = BetaMarketModel(alpha=2.0, beta=5.0)
        assert m.alpha == 2.0
        assert m.beta == 5.0

    def test_init_invalid(self):
        with pytest.raises(ValueError):
            BetaMarketModel(alpha=0.0, beta=1.0)
        with pytest.raises(ValueError):
            BetaMarketModel(alpha=1.0, beta=-1.0)

    def test_sample_shape(self):
        m = BetaMarketModel(alpha=2.0, beta=5.0)
        samples = m.sample(1000, rng=np.random.default_rng(42))
        assert samples.shape == (1000,)
        assert np.all(samples > 0) and np.all(samples < 1)

    def test_fit_recovers_params(self):
        """Fit should recover approximately correct parameters from synthetic data."""
        rng = np.random.default_rng(42)
        true_a, true_b = 3.0, 7.0
        data = rng.beta(true_a, true_b, size=5000)

        m = BetaMarketModel()
        result = m.fit(data)

        assert isinstance(result, FitResult)
        assert abs(m.alpha - true_a) < 0.5
        assert abs(m.beta - true_b) < 1.0
        assert result.n_observations == 5000
        assert result.ks_pvalue is not None and result.ks_pvalue > 0.01

    def test_tail_probability(self):
        m = BetaMarketModel(alpha=2.0, beta=2.0)  # Symmetric around 0.5
        p_above = m.tail_probability(0.5, "above")
        p_below = m.tail_probability(0.5, "below")
        assert abs(p_above - 0.5) < 0.01
        assert abs(p_below - 0.5) < 0.01
        assert abs(p_above + p_below - 1.0) < 0.01

    def test_pdf_cdf(self):
        m = BetaMarketModel(alpha=2.0, beta=5.0)
        x = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        pdf = m.pdf(x)
        cdf = m.cdf(x)
        assert pdf.shape == (5,)
        assert cdf.shape == (5,)
        assert np.all(pdf >= 0)
        assert np.all(np.diff(cdf) >= 0)  # CDF is monotonically increasing

    def test_quantile(self):
        m = BetaMarketModel(alpha=2.0, beta=2.0)
        assert abs(m.quantile(0.5) - 0.5) < 0.01

    def test_serialization(self):
        m = BetaMarketModel(alpha=3.5, beta=7.2)
        d = m.to_dict()
        m2 = BetaMarketModel.from_dict(d)
        assert m2.alpha == 3.5
        assert m2.beta == 7.2


# =====================================================================
# DirichletMarketModel
# =====================================================================


class TestDirichletMarketModel:
    def test_init_defaults(self):
        m = DirichletMarketModel()
        assert m.k == 3

    def test_init_custom(self):
        m = DirichletMarketModel(alphas=[2.0, 3.0, 5.0, 1.0])
        assert m.k == 4

    def test_init_invalid(self):
        with pytest.raises(ValueError):
            DirichletMarketModel(alphas=[1.0, -1.0, 2.0])

    def test_sample_shape(self):
        m = DirichletMarketModel(alphas=[2.0, 3.0, 5.0])
        samples = m.sample(500, rng=np.random.default_rng(42))
        assert samples.shape == (500, 3)
        # Rows should sum to ~1
        row_sums = samples.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-10)

    def test_fit(self):
        rng = np.random.default_rng(42)
        true_alphas = np.array([5.0, 3.0, 2.0])
        data = rng.dirichlet(true_alphas, size=2000)

        m = DirichletMarketModel()
        result = m.fit(data)

        assert result.n_observations == 2000
        # Recovered alphas should be roughly proportional to true
        fitted_ratios = m.alphas / m.alphas.sum()
        true_ratios = true_alphas / true_alphas.sum()
        assert np.allclose(fitted_ratios, true_ratios, atol=0.05)

    def test_serialization(self):
        m = DirichletMarketModel(alphas=[2.0, 3.0, 5.0])
        d = m.to_dict()
        m2 = DirichletMarketModel.from_dict(d)
        assert np.allclose(m2.alphas, [2.0, 3.0, 5.0])


# =====================================================================
# MonteCarloEngine
# =====================================================================


class TestMonteCarloEngine:
    def test_fit_and_simulate(self):
        rng = np.random.default_rng(42)
        prices = rng.beta(3.0, 7.0, size=500)

        engine = MonteCarloEngine(n_simulations=5000, seed=42)
        engine.fit_model("test-market", prices, model_type="beta")

        result = engine.simulate("test-market", observed_price=0.3)

        assert isinstance(result, SimulationResult)
        assert result.market_id == "test-market"
        assert result.n_simulations == 5000
        assert 0.2 < result.mean < 0.4  # Beta(3,7) mean ≈ 0.3
        assert result.std > 0
        assert "50%" in result.percentiles
        assert result.elapsed_ms > 0
        # Observed price near the mean → low anomaly score
        assert result.anomaly_score < 2.0

    def test_simulate_anomalous(self):
        """Extreme observed price should produce high anomaly score."""
        rng = np.random.default_rng(42)
        prices = rng.beta(3.0, 7.0, size=500)

        engine = MonteCarloEngine(n_simulations=10000, seed=42)
        engine.fit_model("test-market", prices)

        result = engine.simulate("test-market", observed_price=0.85)
        assert result.anomaly_score > 3.0

    def test_simulate_missing_model(self):
        engine = MonteCarloEngine(seed=42)
        with pytest.raises(KeyError, match="No fitted model"):
            engine.simulate("nonexistent", observed_price=0.5)

    def test_probability_cone(self):
        engine = MonteCarloEngine(n_simulations=1000, seed=42)
        engine.fit_model(
            "cone-test",
            np.random.default_rng(42).beta(3, 7, size=200),
        )

        cone = engine.simulate_cone("cone-test", current_price=0.3, steps=12)
        assert cone.steps == 12
        assert cone.n_paths == 1000
        assert len(cone.mean_path) == 12
        assert "5%" in cone.percentile_bands
        assert "95%" in cone.percentile_bands
        # 5th percentile should be below 95th at each step
        for i in range(12):
            assert cone.percentile_bands["5%"][i] <= cone.percentile_bands["95%"][i]

    def test_simulate_all(self):
        rng = np.random.default_rng(42)
        engine = MonteCarloEngine(n_simulations=1000, seed=42)
        engine.fit_model("m1", rng.beta(2, 5, size=200))
        engine.fit_model("m2", rng.beta(5, 2, size=200))

        results = engine.simulate_all({"m1": 0.3, "m2": 0.7})
        assert len(results) == 2
        assert "m1" in results and "m2" in results

    def test_serialization_roundtrip(self):
        rng = np.random.default_rng(42)
        engine = MonteCarloEngine(n_simulations=5000, seed=42)
        engine.fit_model("m1", rng.beta(3, 7, size=200))

        data = engine.to_dict()
        engine2 = MonteCarloEngine.from_dict(data, seed=42)

        assert engine2.tracked_markets == ["m1"]
        model = engine2.get_model("m1")
        assert isinstance(model, BetaMarketModel)

    def test_dirichlet_simulation(self):
        rng = np.random.default_rng(42)
        data = rng.dirichlet([5, 3, 2], size=500)

        engine = MonteCarloEngine(n_simulations=5000, seed=42)
        engine.fit_model("multi-market", data, model_type="dirichlet")

        result = engine.simulate("multi-market", observed_price=0.5)
        assert result.model_type == "DirichletMarketModel"
        assert result.mean > 0


# =====================================================================
# ImportanceSampler
# =====================================================================


class TestImportanceSampler:
    def test_tail_risk_estimate(self):
        model = BetaMarketModel(alpha=3.0, beta=7.0)
        sampler = ImportanceSampler(n_samples=50_000, seed=42)

        result = sampler.estimate_tail_risk(
            "test-market", model, threshold=0.7, direction="above"
        )

        assert isinstance(result, TailRiskEstimate)
        assert result.market_id == "test-market"
        assert result.direction == "above"
        assert result.threshold == 0.7
        # Beta(3,7): P(X > 0.7) should be small (~1-2%)
        assert 0.0 < result.probability_is < 0.05
        assert result.effective_sample_size > 100  # Reasonable ESS
        assert result.n_samples == 50_000

    def test_variance_reduction(self):
        """IS should achieve meaningful variance reduction for tail events."""
        model = BetaMarketModel(alpha=3.0, beta=7.0)
        sampler = ImportanceSampler(n_samples=50_000, seed=42)

        result = sampler.estimate_tail_risk(
            "test-market", model, threshold=0.8, direction="above"
        )

        # For a genuine tail event, IS should beat naive MC
        assert result.variance_reduction > 1.0

    def test_symmetric_tails(self):
        model = BetaMarketModel(alpha=5.0, beta=5.0)  # Symmetric
        sampler = ImportanceSampler(n_samples=30_000, seed=42)

        lower, upper = sampler.estimate_symmetric_tails(
            "sym-test", model, threshold_low=0.15, threshold_high=0.85
        )

        assert lower.direction == "below"
        assert upper.direction == "above"
        # Symmetric model: tail probs should be similar
        assert abs(lower.probability_is - upper.probability_is) < 0.02

    def test_confidence_interval(self):
        model = BetaMarketModel(alpha=3.0, beta=7.0)
        sampler = ImportanceSampler(n_samples=50_000, seed=42)

        result = sampler.estimate_tail_risk(
            "ci-test", model, threshold=0.6, direction="above"
        )

        lo, hi = result.confidence_interval
        assert lo <= result.probability_is <= hi
        assert hi - lo < 0.1  # CI shouldn't be absurdly wide

    def test_fallback_for_dirichlet(self):
        """Dirichlet models should fall back to naive MC gracefully."""
        model = DirichletMarketModel(alphas=[5.0, 3.0, 2.0])
        sampler = ImportanceSampler(n_samples=10_000, seed=42)

        result = sampler.estimate_tail_risk(
            "dir-test", model, threshold=0.7, direction="above"
        )

        assert result.variance_reduction == 1.0  # No IS benefit
        assert result.probability_is >= 0

    def test_serialization(self):
        result = TailRiskEstimate(
            market_id="test",
            threshold=0.7,
            direction="above",
            probability_naive=0.01,
            probability_is=0.012,
            variance_reduction=3.5,
            effective_sample_size=15000.0,
            n_samples=50000,
            confidence_interval=(0.008, 0.016),
        )
        d = result.to_dict()
        assert d["market_id"] == "test"
        assert d["probability_is"] == 0.012
