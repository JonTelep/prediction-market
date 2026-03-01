"""Market-specific probability distribution models.

Provides fitted distribution models for Polymarket binary and multi-outcome
markets. These feed into the Monte Carlo engine as the generative process
for simulating future price paths.

Binary markets (Yes/No) use Beta distributions — natural for bounded [0,1]
probabilities with flexible shape. Multi-outcome markets (e.g., "Who wins?")
use Dirichlet distributions — the multivariate generalization of Beta.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import stats as sp_stats


@dataclass(frozen=True)
class FitResult:
    """Result of fitting a distribution to observed data.

    Attributes:
        alpha: Shape parameter(s). Scalar for Beta, array for Dirichlet.
        beta: Second shape parameter (Beta only, None for Dirichlet).
        mean: Fitted distribution mean(s).
        variance: Fitted distribution variance(s).
        n_observations: Number of data points used for fitting.
        ks_statistic: Kolmogorov-Smirnov test statistic (Beta only).
        ks_pvalue: KS test p-value — low values suggest poor fit.
    """

    alpha: float | np.ndarray
    beta: float | None = None
    mean: float | np.ndarray = 0.0
    variance: float | np.ndarray = 0.0
    n_observations: int = 0
    ks_statistic: float | None = None
    ks_pvalue: float | None = None


class MarketModel(ABC):
    """Abstract base for market probability models."""

    @abstractmethod
    def fit(self, observations: np.ndarray) -> FitResult:
        """Fit the model to observed price data.

        Args:
            observations: Array of price observations in [0, 1].

        Returns:
            A FitResult with the estimated parameters.
        """

    @abstractmethod
    def sample(self, n: int, rng: np.random.Generator | None = None) -> np.ndarray:
        """Draw *n* samples from the fitted distribution.

        Args:
            n: Number of samples to draw.
            rng: Optional numpy random generator for reproducibility.

        Returns:
            Array of shape (n,) for binary or (n, k) for multi-outcome.
        """

    @abstractmethod
    def pdf(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the probability density at the given points.

        Args:
            x: Points at which to evaluate the PDF.

        Returns:
            Density values.
        """

    @abstractmethod
    def cdf(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the cumulative distribution function.

        Args:
            x: Points at which to evaluate the CDF.

        Returns:
            CDF values.
        """

    @abstractmethod
    def tail_probability(self, threshold: float, direction: str = "above") -> float:
        """Probability of exceeding (or falling below) a threshold.

        Args:
            threshold: The price level.
            direction: ``"above"`` for P(X > threshold), ``"below"`` for P(X < threshold).

        Returns:
            The tail probability.
        """

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Serialize the fitted model to a JSON-compatible dict."""

    @classmethod
    @abstractmethod
    def from_dict(cls, data: dict[str, Any]) -> MarketModel:
        """Reconstruct a model from serialized data."""


class BetaMarketModel(MarketModel):
    """Beta distribution model for binary (Yes/No) prediction markets.

    The Beta distribution is the natural conjugate prior for Bernoulli
    observations, making it ideal for modeling bounded [0,1] market prices.

    Parameters are estimated via method of moments from observed prices,
    with optional MLE refinement when scipy is available.

    Args:
        alpha: Beta shape parameter α (> 0). Defaults to 1 (uniform prior).
        beta: Beta shape parameter β (> 0). Defaults to 1 (uniform prior).
    """

    def __init__(self, alpha: float = 1.0, beta: float = 1.0) -> None:
        if alpha <= 0 or beta <= 0:
            raise ValueError(f"Alpha and beta must be positive, got α={alpha}, β={beta}")
        self._alpha = alpha
        self._beta = beta
        self._fit_result: FitResult | None = None

    @property
    def alpha(self) -> float:
        return self._alpha

    @property
    def beta(self) -> float:
        return self._beta

    def fit(self, observations: np.ndarray) -> FitResult:
        """Fit Beta distribution to observed market prices via MLE.

        Observations are clamped to (0.001, 0.999) to avoid degenerate
        parameter estimates at the boundaries.

        Args:
            observations: 1-D array of prices in [0, 1].

        Returns:
            FitResult with estimated α, β and goodness-of-fit statistics.
        """
        obs = np.asarray(observations, dtype=np.float64).ravel()
        if len(obs) < 2:
            raise ValueError("Need at least 2 observations to fit a Beta distribution")

        # Clamp to avoid log(0) in MLE
        obs = np.clip(obs, 1e-3, 1.0 - 1e-3)

        # MLE fit via scipy
        a, b, loc, scale = sp_stats.beta.fit(obs, floc=0, fscale=1)
        self._alpha = a
        self._beta = b

        # Goodness-of-fit
        ks_stat, ks_p = sp_stats.kstest(obs, "beta", args=(a, b))

        dist = sp_stats.beta(a, b)
        self._fit_result = FitResult(
            alpha=a,
            beta=b,
            mean=float(dist.mean()),
            variance=float(dist.var()),
            n_observations=len(obs),
            ks_statistic=float(ks_stat),
            ks_pvalue=float(ks_p),
        )
        return self._fit_result

    def sample(self, n: int, rng: np.random.Generator | None = None) -> np.ndarray:
        """Draw *n* samples from Beta(α, β).

        Args:
            n: Number of samples.
            rng: Optional numpy Generator for reproducibility.

        Returns:
            Array of shape (n,) with values in (0, 1).
        """
        if rng is None:
            rng = np.random.default_rng()
        return rng.beta(self._alpha, self._beta, size=n)

    def pdf(self, x: np.ndarray) -> np.ndarray:
        return sp_stats.beta.pdf(x, self._alpha, self._beta)

    def cdf(self, x: np.ndarray) -> np.ndarray:
        return sp_stats.beta.cdf(x, self._alpha, self._beta)

    def tail_probability(self, threshold: float, direction: str = "above") -> float:
        """Compute P(X > threshold) or P(X < threshold).

        Args:
            threshold: Price level in [0, 1].
            direction: ``"above"`` or ``"below"``.

        Returns:
            Tail probability.
        """
        cdf_val = float(sp_stats.beta.cdf(threshold, self._alpha, self._beta))
        if direction == "below":
            return cdf_val
        return 1.0 - cdf_val

    def quantile(self, q: float) -> float:
        """Return the quantile (inverse CDF) for probability *q*.

        Args:
            q: Probability in [0, 1].

        Returns:
            The price level at which CDF = q.
        """
        return float(sp_stats.beta.ppf(q, self._alpha, self._beta))

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "beta",
            "alpha": self._alpha,
            "beta": self._beta,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BetaMarketModel:
        return cls(alpha=data["alpha"], beta=data["beta"])

    def __repr__(self) -> str:
        return f"BetaMarketModel(α={self._alpha:.3f}, β={self._beta:.3f})"


class DirichletMarketModel(MarketModel):
    """Dirichlet distribution model for multi-outcome prediction markets.

    The Dirichlet is the multivariate generalization of the Beta distribution,
    modeling probability vectors that sum to 1 — perfect for markets with
    K mutually exclusive outcomes (e.g., "Who wins the election?").

    Args:
        alphas: Concentration parameters, one per outcome. Defaults to
            uniform [1, 1, 1] (3 outcomes).
    """

    def __init__(self, alphas: np.ndarray | list[float] | None = None) -> None:
        if alphas is None:
            alphas = [1.0, 1.0, 1.0]
        self._alphas = np.asarray(alphas, dtype=np.float64)
        if np.any(self._alphas <= 0):
            raise ValueError("All concentration parameters must be positive")
        self._fit_result: FitResult | None = None

    @property
    def alphas(self) -> np.ndarray:
        return self._alphas.copy()

    @property
    def k(self) -> int:
        """Number of outcomes."""
        return len(self._alphas)

    def fit(self, observations: np.ndarray) -> FitResult:
        """Fit Dirichlet parameters via method of moments.

        Each row of *observations* should be a probability vector summing to ~1.

        Args:
            observations: Array of shape (n, k) with outcome probabilities.

        Returns:
            FitResult with estimated concentration parameters.
        """
        obs = np.asarray(observations, dtype=np.float64)
        if obs.ndim != 2:
            raise ValueError(f"Expected 2-D array, got shape {obs.shape}")
        if obs.shape[0] < 2:
            raise ValueError("Need at least 2 observations")

        # Normalize rows to sum to 1
        row_sums = obs.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums > 0, row_sums, 1.0)
        obs = obs / row_sums

        # Clamp away from 0/1
        obs = np.clip(obs, 1e-6, 1.0 - 1e-6)
        obs = obs / obs.sum(axis=1, keepdims=True)

        # Method of moments estimation
        means = obs.mean(axis=0)
        variances = obs.var(axis=0)

        # Estimate precision s = sum(alphas) from first non-trivial component
        # Using: var_i = mean_i * (1 - mean_i) / (s + 1)
        # => s = mean_i * (1 - mean_i) / var_i - 1
        precisions = []
        for i in range(obs.shape[1]):
            if variances[i] > 1e-12:
                s_est = means[i] * (1.0 - means[i]) / variances[i] - 1.0
                if s_est > 0:
                    precisions.append(s_est)

        if not precisions:
            # Fallback: uniform with moderate concentration
            s = float(obs.shape[1]) * 2.0
        else:
            s = float(np.median(precisions))
            s = max(s, 1.0)  # Floor at 1

        self._alphas = means * s

        mean_vec = self._alphas / self._alphas.sum()
        s_total = self._alphas.sum()
        var_vec = mean_vec * (1 - mean_vec) / (s_total + 1)

        self._fit_result = FitResult(
            alpha=self._alphas.copy(),
            beta=None,
            mean=mean_vec,
            variance=var_vec,
            n_observations=obs.shape[0],
        )
        return self._fit_result

    def sample(self, n: int, rng: np.random.Generator | None = None) -> np.ndarray:
        """Draw *n* probability vectors from Dirichlet(α).

        Args:
            n: Number of samples.
            rng: Optional numpy Generator.

        Returns:
            Array of shape (n, k) where each row sums to 1.
        """
        if rng is None:
            rng = np.random.default_rng()
        return rng.dirichlet(self._alphas, size=n)

    def pdf(self, x: np.ndarray) -> np.ndarray:
        """Evaluate Dirichlet PDF at the given probability vectors.

        Args:
            x: Array of shape (n, k) or (k,).

        Returns:
            Density values, shape (n,) or scalar.
        """
        return sp_stats.dirichlet.pdf(x.T, self._alphas)

    def cdf(self, x: np.ndarray) -> np.ndarray:
        """Not well-defined for multivariate Dirichlet. Returns zeros."""
        # Dirichlet CDF isn't analytically tractable; use MC estimation instead
        return np.zeros(len(x))

    def tail_probability(self, threshold: float, direction: str = "above") -> float:
        """Estimate P(X_0 > threshold) via Monte Carlo for the first outcome.

        This is a convenience for the common case of checking the leading
        outcome. For full multi-outcome tail analysis, use the
        ImportanceSampler directly.

        Args:
            threshold: Probability threshold for outcome 0.
            direction: ``"above"`` or ``"below"``.

        Returns:
            Estimated tail probability.
        """
        samples = self.sample(10_000)
        outcome_0 = samples[:, 0]
        if direction == "below":
            return float(np.mean(outcome_0 < threshold))
        return float(np.mean(outcome_0 > threshold))

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "dirichlet",
            "alphas": self._alphas.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DirichletMarketModel:
        return cls(alphas=data["alphas"])

    def __repr__(self) -> str:
        return f"DirichletMarketModel(k={self.k}, α={self._alphas})"
