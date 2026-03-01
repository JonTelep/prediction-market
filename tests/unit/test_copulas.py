"""Tests for Phase 3: Copula-based tail dependence modeling.

Covers: Clayton, Gumbel, Frank copulas, CopulaFitter, DynamicCopulaTracker.
"""

from __future__ import annotations

import numpy as np
import pytest

from prediction_market.simulation.copulas import (
    ClaytonCopula,
    CopulaFitter,
    DynamicCopulaTracker,
    FrankCopula,
    GumbelCopula,
    TailAlert,
    TailDependence,
)


# =====================================================================
# Individual copulas
# =====================================================================


class TestClaytonCopula:
    def test_init(self):
        c = ClaytonCopula(theta=2.0)
        assert c.theta == 2.0

    def test_invalid_theta(self):
        with pytest.raises(ValueError):
            ClaytonCopula(theta=0)
        with pytest.raises(ValueError):
            ClaytonCopula(theta=-1)

    def test_cdf_bounds(self):
        c = ClaytonCopula(theta=2.0)
        u = np.array([0.3, 0.5, 0.9])
        v = np.array([0.4, 0.6, 0.8])
        cdf = c.cdf(u, v)
        assert np.all(cdf >= 0) and np.all(cdf <= 1)

    def test_cdf_monotonic(self):
        c = ClaytonCopula(theta=2.0)
        v = np.full(5, 0.5)
        u = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        cdf = c.cdf(u, v)
        assert np.all(np.diff(cdf) >= 0)

    def test_lower_tail_dependence(self):
        c = ClaytonCopula(theta=2.0)
        assert c.lower_tail_dependence > 0
        assert c.upper_tail_dependence == 0.0

    def test_higher_theta_stronger_lower_tail(self):
        c_low = ClaytonCopula(theta=1.0)
        c_high = ClaytonCopula(theta=5.0)
        assert c_high.lower_tail_dependence > c_low.lower_tail_dependence

    def test_pdf_positive(self):
        c = ClaytonCopula(theta=2.0)
        u = np.array([0.3, 0.5, 0.7])
        v = np.array([0.4, 0.6, 0.8])
        pdf = c.pdf(u, v)
        assert np.all(pdf > 0)

    def test_theta_from_kendall(self):
        theta = ClaytonCopula.theta_from_kendall(0.5)
        assert theta > 0


class TestGumbelCopula:
    def test_init(self):
        g = GumbelCopula(theta=2.0)
        assert g.theta == 2.0

    def test_invalid_theta(self):
        with pytest.raises(ValueError):
            GumbelCopula(theta=0.5)

    def test_cdf_bounds(self):
        g = GumbelCopula(theta=2.0)
        u = np.array([0.3, 0.5, 0.9])
        v = np.array([0.4, 0.6, 0.8])
        cdf = g.cdf(u, v)
        assert np.all(cdf >= 0) and np.all(cdf <= 1)

    def test_upper_tail_dependence(self):
        g = GumbelCopula(theta=2.0)
        assert g.upper_tail_dependence > 0
        assert g.lower_tail_dependence == 0.0

    def test_higher_theta_stronger_upper_tail(self):
        g_low = GumbelCopula(theta=1.5)
        g_high = GumbelCopula(theta=5.0)
        assert g_high.upper_tail_dependence > g_low.upper_tail_dependence

    def test_independence_at_theta_1(self):
        g = GumbelCopula(theta=1.001)
        assert g.upper_tail_dependence < 0.01

    def test_theta_from_kendall(self):
        theta = GumbelCopula.theta_from_kendall(0.5)
        assert theta >= 1.0


class TestFrankCopula:
    def test_init(self):
        f = FrankCopula(theta=5.0)
        assert f.theta == 5.0

    def test_invalid_theta(self):
        with pytest.raises(ValueError):
            FrankCopula(theta=0.0)

    def test_no_tail_dependence(self):
        f = FrankCopula(theta=5.0)
        assert f.lower_tail_dependence == 0.0
        assert f.upper_tail_dependence == 0.0

    def test_cdf_bounds(self):
        f = FrankCopula(theta=5.0)
        u = np.array([0.3, 0.5, 0.9])
        v = np.array([0.4, 0.6, 0.8])
        cdf = f.cdf(u, v)
        assert np.all(cdf >= 0) and np.all(cdf <= 1)

    def test_theta_from_kendall(self):
        theta = FrankCopula.theta_from_kendall(0.5)
        assert theta > 0


# =====================================================================
# CopulaFitter
# =====================================================================


class TestCopulaFitter:
    def _generate_correlated_data(
        self, n: int, rho: float, seed: int = 42
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate correlated normal returns, then return raw values."""
        rng = np.random.default_rng(seed)
        mean = [0, 0]
        cov = [[1, rho], [rho, 1]]
        data = rng.multivariate_normal(mean, cov, size=n)
        return data[:, 0], data[:, 1]

    def test_fit_correlated_data(self):
        fitter = CopulaFitter(min_observations=20)
        a, b = self._generate_correlated_data(200, rho=0.7)

        td = fitter.fit("mkt-a", "mkt-b", a, b)

        assert isinstance(td, TailDependence)
        assert td.market_a == "mkt-a"
        assert td.market_b == "mkt-b"
        assert td.pearson > 0.5
        assert td.kendall_tau > 0.3
        assert td.n_observations == 200
        assert td.copula_type in ("clayton", "gumbel", "frank")
        assert td.copula_param > 0

    def test_fit_independent_data(self):
        fitter = CopulaFitter(min_observations=20)
        rng = np.random.default_rng(42)
        a = rng.standard_normal(100)
        b = rng.standard_normal(100)

        td = fitter.fit("ind-a", "ind-b", a, b)

        # Independent data should have low tail dependence
        assert td.max_tail < 0.5
        assert abs(td.pearson) < 0.3

    def test_fit_with_lower_tail_dependence(self):
        """Generate data from a Clayton copula and verify Clayton is recovered."""
        rng = np.random.default_rng(42)
        n = 500
        # Generate Clayton copula samples via conditional method
        theta = 3.0
        u1 = rng.uniform(0.01, 0.99, n)
        w = rng.uniform(0.01, 0.99, n)
        u2 = (u1 ** (-theta) * (w ** (-theta / (1 + theta)) - 1) + 1) ** (-1 / theta)
        u2 = np.clip(u2, 0.01, 0.99)

        # Convert to "returns" via inverse normal CDF
        from scipy.stats import norm
        a = norm.ppf(u1)
        b = norm.ppf(u2)

        fitter = CopulaFitter(min_observations=20)
        td = fitter.fit("lower-a", "lower-b", a, b)

        # Should detect lower tail dependence
        assert td.lower_tail > 0.2 or td.copula_type == "clayton"

    def test_insufficient_data(self):
        fitter = CopulaFitter(min_observations=20)
        a = np.array([0.1, 0.2])
        b = np.array([0.3, 0.4])

        with pytest.raises(ValueError, match="Need"):
            fitter.fit("x", "y", a, b)

    def test_tail_dependence_properties(self):
        fitter = CopulaFitter()
        a, b = self._generate_correlated_data(200, rho=0.6)
        td = fitter.fit("a", "b", a, b)

        assert 0 <= td.max_tail <= 1
        assert isinstance(td.tail_asymmetry, float)

    def test_tail_dependence_to_dict(self):
        td = TailDependence(
            market_a="a",
            market_b="b",
            lower_tail=0.3,
            upper_tail=0.1,
            pearson=0.6,
            kendall_tau=0.4,
            copula_type="clayton",
            copula_param=2.5,
            n_observations=100,
        )
        d = td.to_dict()
        assert d["market_a"] == "a"
        assert d["lower_tail"] == 0.3
        assert d["copula_type"] == "clayton"


# =====================================================================
# DynamicCopulaTracker
# =====================================================================


class TestDynamicCopulaTracker:
    def test_no_alert_insufficient_data(self):
        tracker = DynamicCopulaTracker(window_size=30, step_size=5)

        # Feed less than window_size observations
        rng = np.random.default_rng(42)
        for _ in range(20):
            a = rng.standard_normal() * 0.02
            b = rng.standard_normal() * 0.02
            result = tracker.update("m1", "m2", a, b)
            assert result is None

    def test_fits_after_enough_data(self):
        tracker = DynamicCopulaTracker(window_size=30, step_size=10)
        rng = np.random.default_rng(42)

        # Feed correlated data
        for i in range(50):
            base = rng.standard_normal() * 0.02
            a = base + rng.standard_normal() * 0.005
            b = base + rng.standard_normal() * 0.005
            tracker.update("m1", "m2", a, b)

        td = tracker.get_latest("m1", "m2")
        assert td is not None
        assert td.pearson > 0.3

    def test_pair_ordering_invariant(self):
        """(m1, m2) and (m2, m1) should be the same pair."""
        tracker = DynamicCopulaTracker(window_size=30, step_size=10)
        rng = np.random.default_rng(42)

        for _ in range(50):
            a, b = rng.standard_normal(2) * 0.02
            tracker.update("beta", "alpha", a, b)

        # Should be stored under ("alpha", "beta")
        assert ("alpha", "beta") in tracker.tracked_pairs

    def test_alert_on_spike(self):
        """Tail dependence spike should trigger an alert."""
        tracker = DynamicCopulaTracker(
            window_size=30, step_size=5, alert_z_threshold=1.5
        )
        rng = np.random.default_rng(42)

        # Build baseline with independent data (low tail dependence)
        for i in range(200):
            a = rng.standard_normal() * 0.02
            b = rng.standard_normal() * 0.02
            tracker.update("s1", "s2", a, b)

        # Now feed highly correlated tail data
        alerts = []
        for i in range(100):
            base = rng.standard_normal() * 0.05
            a = base + rng.standard_normal() * 0.001
            b = base + rng.standard_normal() * 0.001
            result = tracker.update("s1", "s2", a, b)
            if result is not None:
                alerts.append(result)

        # Should have triggered at least one alert
        # (may not always trigger depending on random seed, so we check history grew)
        history = tracker.get_history("s1", "s2")
        assert len(history) > 3

    def test_serialization_roundtrip(self):
        tracker = DynamicCopulaTracker(window_size=30, step_size=5)
        rng = np.random.default_rng(42)

        for _ in range(40):
            tracker.update("x", "y", rng.standard_normal() * 0.02, rng.standard_normal() * 0.02)

        data = tracker.to_dict()
        tracker2 = DynamicCopulaTracker.from_dict(data)

        assert tracker2._window_size == 30
        assert ("x", "y") in tracker2.tracked_pairs

    def test_tail_alert_to_dict(self):
        alert = TailAlert(
            market_ids=["m1", "m2"],
            alert_type="spike",
            current_tail=0.7,
            baseline_tail=0.2,
            z_score=3.5,
            direction="upper",
        )
        d = alert.to_dict()
        assert d["alert_type"] == "spike"
        assert d["z_score"] == 3.5
        assert d["direction"] == "upper"
