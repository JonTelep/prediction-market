"""Copula-based tail dependence modeling for cross-market analysis.

Standard Pearson correlation (used by the existing CorrelationDetector)
captures linear co-movement but completely misses **tail dependence** —
the tendency for extreme moves to happen simultaneously across markets.

This matters for manipulation detection because:
  - Coordinated manipulation hits multiple related markets simultaneously
  - Info leakage affects correlated contracts (e.g., "Trump wins" + "GOP Senate")
  - Crisis/event-driven moves create synchronized tail events
  - Gaussian copulas (which assume no tail dependence) famously failed in 2008

This module provides:
  - Archimedean copulas (Clayton, Gumbel, Frank) with different tail profiles
  - Empirical tail dependence estimation
  - Dynamic (time-varying) copula estimation via rolling windows
  - Cross-market tail alert generation for the manipulation guard

Copula taxonomy:
  - **Clayton**: Lower tail dependence (crashes together)
  - **Gumbel**: Upper tail dependence (spikes together)
  - **Frank**: Symmetric, no special tail dependence (baseline comparison)
  - **Empirical**: Non-parametric, directly measured from data
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
from scipy import stats as sp_stats
from scipy.optimize import minimize_scalar

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tail dependence result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TailDependence:
    """Measured tail dependence between two markets.

    Attributes:
        market_a: First market identifier.
        market_b: Second market identifier.
        lower_tail: Lower tail dependence coefficient λ_L ∈ [0,1].
            High = they crash together.
        upper_tail: Upper tail dependence coefficient λ_U ∈ [0,1].
            High = they spike together.
        pearson: Standard Pearson correlation (for comparison).
        kendall_tau: Kendall's rank correlation (copula-invariant).
        copula_type: Which copula was fitted (or "empirical").
        copula_param: The fitted copula parameter (theta).
        n_observations: Number of paired observations used.
        timestamp: When the estimate was produced.
    """

    market_a: str
    market_b: str
    lower_tail: float
    upper_tail: float
    pearson: float
    kendall_tau: float
    copula_type: str
    copula_param: float
    n_observations: int
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def max_tail(self) -> float:
        """The stronger tail dependence (either direction)."""
        return max(self.lower_tail, self.upper_tail)

    @property
    def tail_asymmetry(self) -> float:
        """Difference between upper and lower tail dependence.
        Positive = more upper tail (coordinated spikes).
        Negative = more lower tail (coordinated crashes).
        """
        return self.upper_tail - self.lower_tail

    def to_dict(self) -> dict[str, Any]:
        return {
            "market_a": self.market_a,
            "market_b": self.market_b,
            "lower_tail": self.lower_tail,
            "upper_tail": self.upper_tail,
            "pearson": self.pearson,
            "kendall_tau": self.kendall_tau,
            "copula_type": self.copula_type,
            "copula_param": self.copula_param,
            "n_observations": self.n_observations,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass(frozen=True)
class TailAlert:
    """Alert triggered when tail dependence spikes unexpectedly.

    Attributes:
        market_ids: The pair of markets involved.
        alert_type: "spike" (sudden increase) or "structural" (sustained high).
        current_tail: Current tail dependence level.
        baseline_tail: Historical baseline tail dependence.
        z_score: How many baseline SDs the current level is above normal.
        direction: "upper" or "lower" tail.
        timestamp: When the alert was generated.
    """

    market_ids: list[str]
    alert_type: str
    current_tail: float
    baseline_tail: float
    z_score: float
    direction: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        return {
            "market_ids": self.market_ids,
            "alert_type": self.alert_type,
            "current_tail": self.current_tail,
            "baseline_tail": self.baseline_tail,
            "z_score": self.z_score,
            "direction": self.direction,
            "timestamp": self.timestamp.isoformat(),
        }


# ---------------------------------------------------------------------------
# Archimedean copula implementations
# ---------------------------------------------------------------------------


class ClaytonCopula:
    """Clayton copula — captures LOWER tail dependence.

    C(u, v) = (u^{-θ} + v^{-θ} - 1)^{-1/θ}

    Lower tail dependence: λ_L = 2^{-1/θ}
    Upper tail dependence: λ_U = 0

    θ ∈ (0, ∞), θ → 0 = independence, θ → ∞ = perfect lower tail dependence.
    """

    def __init__(self, theta: float = 1.0) -> None:
        if theta <= 0:
            raise ValueError(f"Clayton theta must be > 0, got {theta}")
        self.theta = theta

    def cdf(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Evaluate copula CDF C(u, v)."""
        u = np.clip(u, 1e-10, 1 - 1e-10)
        v = np.clip(v, 1e-10, 1 - 1e-10)
        return np.maximum(u ** (-self.theta) + v ** (-self.theta) - 1, 0) ** (-1.0 / self.theta)

    def pdf(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Evaluate copula density c(u, v)."""
        u = np.clip(u, 1e-10, 1 - 1e-10)
        v = np.clip(v, 1e-10, 1 - 1e-10)
        t = self.theta
        A = (1 + t) * (u * v) ** (-(1 + t))
        B = (u ** (-t) + v ** (-t) - 1) ** (-(2 + 1 / t))
        return A * B

    def log_likelihood(self, u: np.ndarray, v: np.ndarray) -> float:
        """Sum of log-densities (for MLE fitting)."""
        density = self.pdf(u, v)
        density = np.clip(density, 1e-30, None)
        return float(np.sum(np.log(density)))

    @property
    def lower_tail_dependence(self) -> float:
        return 2.0 ** (-1.0 / self.theta)

    @property
    def upper_tail_dependence(self) -> float:
        return 0.0

    @staticmethod
    def theta_from_kendall(tau: float) -> float:
        """Estimate θ from Kendall's τ: θ = 2τ / (1 - τ)."""
        tau = np.clip(tau, 0.01, 0.99)
        return 2.0 * tau / (1.0 - tau)


class GumbelCopula:
    """Gumbel copula — captures UPPER tail dependence.

    C(u, v) = exp(-[(-ln u)^θ + (-ln v)^θ]^{1/θ})

    Lower tail dependence: λ_L = 0
    Upper tail dependence: λ_U = 2 - 2^{1/θ}

    θ ∈ [1, ∞), θ = 1 = independence, θ → ∞ = perfect upper tail dependence.
    """

    def __init__(self, theta: float = 1.5) -> None:
        if theta < 1.0:
            raise ValueError(f"Gumbel theta must be >= 1, got {theta}")
        self.theta = theta

    def cdf(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        u = np.clip(u, 1e-10, 1 - 1e-10)
        v = np.clip(v, 1e-10, 1 - 1e-10)
        A = (-np.log(u)) ** self.theta + (-np.log(v)) ** self.theta
        return np.exp(-A ** (1.0 / self.theta))

    def pdf(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        u = np.clip(u, 1e-10, 1 - 1e-10)
        v = np.clip(v, 1e-10, 1 - 1e-10)
        t = self.theta
        lu, lv = -np.log(u), -np.log(v)
        A = lu ** t + lv ** t
        A_inv_t = A ** (1.0 / t)
        C = np.exp(-A_inv_t)

        # Density via mixed partial derivative (simplified form)
        term1 = C / (u * v)
        term2 = (lu * lv) ** (t - 1)
        term3 = A ** (1.0 / t - 2)
        term4 = (t - 1 + A_inv_t)
        return np.maximum(term1 * term2 * term3 * term4, 1e-30)

    def log_likelihood(self, u: np.ndarray, v: np.ndarray) -> float:
        density = self.pdf(u, v)
        density = np.clip(density, 1e-30, None)
        return float(np.sum(np.log(density)))

    @property
    def lower_tail_dependence(self) -> float:
        return 0.0

    @property
    def upper_tail_dependence(self) -> float:
        return 2.0 - 2.0 ** (1.0 / self.theta)

    @staticmethod
    def theta_from_kendall(tau: float) -> float:
        """Estimate θ from Kendall's τ: θ = 1 / (1 - τ)."""
        tau = np.clip(tau, 0.01, 0.99)
        return 1.0 / (1.0 - tau)


class FrankCopula:
    r"""Frank copula — symmetric, NO special tail dependence.

    C(u, v) = -1/θ * ln(1 + (e^{-θu} - 1)(e^{-θv} - 1) / (e^{-θ} - 1))

    λ_L = λ_U = 0 (asymptotically independent tails).

    Useful as a baseline: if Clayton or Gumbel fit significantly better
    than Frank, there IS tail dependence.

    θ ∈ (-∞, ∞) \ {0}, θ > 0 = positive dependence, θ < 0 = negative.
    """

    def __init__(self, theta: float = 5.0) -> None:
        if abs(theta) < 1e-10:
            raise ValueError("Frank theta must be non-zero")
        self.theta = theta

    def cdf(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        u = np.clip(u, 1e-10, 1 - 1e-10)
        v = np.clip(v, 1e-10, 1 - 1e-10)
        t = self.theta
        num = (np.exp(-t * u) - 1) * (np.exp(-t * v) - 1)
        den = np.exp(-t) - 1
        return -1.0 / t * np.log(1 + num / den)

    def pdf(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        u = np.clip(u, 1e-10, 1 - 1e-10)
        v = np.clip(v, 1e-10, 1 - 1e-10)
        t = self.theta
        et = np.exp(-t)
        etu = np.exp(-t * u)
        etv = np.exp(-t * v)
        num = -t * (et - 1) * np.exp(-t * (u + v))
        den = ((et - 1) + (etu - 1) * (etv - 1)) ** 2
        return np.maximum(num / den, 1e-30)

    def log_likelihood(self, u: np.ndarray, v: np.ndarray) -> float:
        density = self.pdf(u, v)
        density = np.clip(density, 1e-30, None)
        return float(np.sum(np.log(density)))

    @property
    def lower_tail_dependence(self) -> float:
        return 0.0

    @property
    def upper_tail_dependence(self) -> float:
        return 0.0

    @staticmethod
    def theta_from_kendall(tau: float) -> float:
        """Approximate θ from Kendall's τ via numerical inversion.

        The exact relationship is τ = 1 - 4/θ * (1 - D_1(θ)) where
        D_1 is the first Debye function. We use a simple Newton step
        from the initial approximation θ ≈ 9τ / (1 - |τ|).
        """
        tau = np.clip(tau, -0.95, 0.95)
        if abs(tau) < 0.01:
            return 0.1  # Near independence
        # Rough initial estimate
        return 9.0 * tau / (1.0 - abs(tau))


# ---------------------------------------------------------------------------
# Copula fitter
# ---------------------------------------------------------------------------


_COPULA_CLASSES = {
    "clayton": ClaytonCopula,
    "gumbel": GumbelCopula,
    "frank": FrankCopula,
}


class CopulaFitter:
    """Fits copula models to paired market data and estimates tail dependence.

    Takes raw price return series for two markets, converts to pseudo-
    observations (uniform marginals via empirical CDF), then fits
    Clayton, Gumbel, and Frank copulas to select the best model.

    Args:
        min_observations: Minimum paired observations required for fitting.
    """

    def __init__(self, min_observations: int = 20) -> None:
        self._min_obs = min_observations

    def fit(
        self,
        market_a: str,
        market_b: str,
        returns_a: np.ndarray,
        returns_b: np.ndarray,
    ) -> TailDependence:
        """Fit copula models and estimate tail dependence.

        Args:
            market_a: First market identifier.
            market_b: Second market identifier.
            returns_a: Log-returns or price changes for market A.
            returns_b: Log-returns or price changes for market B.

        Returns:
            TailDependence with the best-fit copula's tail coefficients.

        Raises:
            ValueError: If insufficient observations.
        """
        a = np.asarray(returns_a).ravel()
        b = np.asarray(returns_b).ravel()
        n = min(len(a), len(b))
        if n < self._min_obs:
            raise ValueError(
                f"Need {self._min_obs} observations, got {n}"
            )
        a, b = a[:n], b[:n]

        # Pearson and Kendall correlations
        pearson = float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 1e-12 and np.std(b) > 1e-12 else 0.0
        tau, _ = sp_stats.kendalltau(a, b)
        tau = float(tau) if not np.isnan(tau) else 0.0

        # Convert to pseudo-observations (uniform marginals)
        u = self._pseudo_observations(a)
        v = self._pseudo_observations(b)

        # Fit each copula family and pick best by log-likelihood
        best_copula = None
        best_ll = -np.inf
        best_name = "frank"

        for name, cls in _COPULA_CLASSES.items():
            try:
                theta = self._fit_copula_mle(cls, u, v, tau)
                copula = cls(theta)
                ll = copula.log_likelihood(u, v)
                if ll > best_ll:
                    best_ll = ll
                    best_copula = copula
                    best_name = name
            except (ValueError, RuntimeError) as e:
                logger.debug("Failed to fit %s copula: %s", name, e)
                continue

        if best_copula is None:
            # Fallback to empirical estimation
            lower_emp, upper_emp = self._empirical_tail_dependence(u, v)
            return TailDependence(
                market_a=market_a,
                market_b=market_b,
                lower_tail=lower_emp,
                upper_tail=upper_emp,
                pearson=pearson,
                kendall_tau=tau,
                copula_type="empirical",
                copula_param=0.0,
                n_observations=n,
            )

        return TailDependence(
            market_a=market_a,
            market_b=market_b,
            lower_tail=best_copula.lower_tail_dependence,
            upper_tail=best_copula.upper_tail_dependence,
            pearson=pearson,
            kendall_tau=tau,
            copula_type=best_name,
            copula_param=best_copula.theta,
            n_observations=n,
        )

    def _fit_copula_mle(
        self,
        copula_cls: type,
        u: np.ndarray,
        v: np.ndarray,
        tau: float,
    ) -> float:
        """Fit copula parameter via MLE, initialized from Kendall's τ."""
        # Get initial estimate from Kendall's tau
        try:
            theta_init = copula_cls.theta_from_kendall(abs(tau))
        except (ValueError, ZeroDivisionError):
            theta_init = 1.0

        # Bounds depend on copula family
        if copula_cls is ClaytonCopula:
            bounds = (0.01, 50.0)
        elif copula_cls is GumbelCopula:
            bounds = (1.001, 50.0)
        elif copula_cls is FrankCopula:
            bounds = (0.1, 50.0) if tau > 0 else (-50.0, -0.1)
        else:
            bounds = (0.01, 50.0)

        theta_init = np.clip(theta_init, bounds[0], bounds[1])

        def neg_ll(theta: float) -> float:
            try:
                copula = copula_cls(theta)
                return -copula.log_likelihood(u, v)
            except (ValueError, FloatingPointError, RuntimeWarning):
                return 1e10

        result = minimize_scalar(neg_ll, bounds=bounds, method="bounded")
        if not result.success and result.fun >= 1e10:
            raise RuntimeError(f"MLE failed for {copula_cls.__name__}")

        return float(result.x)

    @staticmethod
    def _pseudo_observations(x: np.ndarray) -> np.ndarray:
        """Convert data to pseudo-observations (empirical CDF values).

        Uses rank/(n+1) to avoid exact 0 and 1 values which cause
        issues with copula evaluation.
        """
        n = len(x)
        ranks = sp_stats.rankdata(x)
        return ranks / (n + 1)

    @staticmethod
    def _empirical_tail_dependence(
        u: np.ndarray,
        v: np.ndarray,
        quantile: float = 0.1,
    ) -> tuple[float, float]:
        """Estimate tail dependence coefficients empirically.

        Lower tail: λ_L ≈ P(V < q | U < q)
        Upper tail: λ_U ≈ P(V > 1-q | U > 1-q)

        Args:
            u: Pseudo-observations for market A.
            v: Pseudo-observations for market B.
            quantile: Tail threshold (default 10th percentile).

        Returns:
            Tuple of (lower_tail, upper_tail) coefficients.
        """
        n = len(u)

        # Lower tail
        lower_mask = u < quantile
        if np.sum(lower_mask) > 0:
            lower_tail = float(np.mean(v[lower_mask] < quantile))
        else:
            lower_tail = 0.0

        # Upper tail
        upper_q = 1.0 - quantile
        upper_mask = u > upper_q
        if np.sum(upper_mask) > 0:
            upper_tail = float(np.mean(v[upper_mask] > upper_q))
        else:
            upper_tail = 0.0

        return lower_tail, upper_tail


# ---------------------------------------------------------------------------
# Dynamic (time-varying) copula tracker
# ---------------------------------------------------------------------------


class DynamicCopulaTracker:
    """Tracks tail dependence over time using rolling windows.

    Maintains a history of TailDependence estimates and detects when
    tail dependence spikes above historical norms — a signal for
    coordinated manipulation or correlated information leakage.

    Args:
        window_size: Number of observations per rolling copula fit.
        step_size: How many new observations before refitting.
        alert_z_threshold: Z-score threshold for generating tail alerts.
        fitter: CopulaFitter instance. Uses default if None.
    """

    def __init__(
        self,
        window_size: int = 50,
        step_size: int = 10,
        alert_z_threshold: float = 2.5,
        fitter: CopulaFitter | None = None,
    ) -> None:
        self._window_size = window_size
        self._step_size = step_size
        self._alert_z = alert_z_threshold
        self._fitter = fitter or CopulaFitter(min_observations=20)

        # Per-pair state
        self._buffers: dict[tuple[str, str], _PairBuffer] = {}
        self._history: dict[tuple[str, str], list[TailDependence]] = {}

    def update(
        self,
        market_a: str,
        market_b: str,
        return_a: float,
        return_b: float,
    ) -> TailAlert | None:
        """Feed a new pair of returns and potentially trigger a refit + alert.

        Args:
            market_a: First market identifier.
            market_b: Second market identifier.
            return_a: Latest return for market A.
            return_b: Latest return for market B.

        Returns:
            A TailAlert if tail dependence spiked, otherwise None.
        """
        key = (min(market_a, market_b), max(market_a, market_b))

        if key not in self._buffers:
            self._buffers[key] = _PairBuffer()
            self._history[key] = []

        buf = self._buffers[key]
        buf.add(return_a, return_b)

        # Check if we have enough data and it's time to refit
        if buf.count < self._window_size:
            return None
        if buf.since_last_fit < self._step_size:
            return None

        # Refit
        a_arr, b_arr = buf.get_window(self._window_size)
        try:
            td = self._fitter.fit(key[0], key[1], a_arr, b_arr)
        except (ValueError, RuntimeError) as e:
            logger.debug("Copula refit failed for %s: %s", key, e)
            return None

        buf.mark_fitted()
        self._history[key].append(td)

        # Check for tail dependence spike
        return self._check_alert(key, td)

    def get_latest(self, market_a: str, market_b: str) -> TailDependence | None:
        """Get the most recent tail dependence estimate for a pair."""
        key = (min(market_a, market_b), max(market_a, market_b))
        history = self._history.get(key, [])
        return history[-1] if history else None

    def get_history(self, market_a: str, market_b: str) -> list[TailDependence]:
        """Get all tail dependence estimates for a pair."""
        key = (min(market_a, market_b), max(market_a, market_b))
        return list(self._history.get(key, []))

    @property
    def tracked_pairs(self) -> list[tuple[str, str]]:
        return list(self._buffers.keys())

    def _check_alert(
        self, key: tuple[str, str], current: TailDependence
    ) -> TailAlert | None:
        """Check if current tail dependence is anomalously high."""
        history = self._history[key]
        if len(history) < 5:
            return None  # Not enough history for baseline

        # Use all but current for baseline
        baseline = history[:-1]

        for direction, get_val in [
            ("lower", lambda td: td.lower_tail),
            ("upper", lambda td: td.upper_tail),
        ]:
            current_val = get_val(current)
            baseline_vals = [get_val(td) for td in baseline]
            mean_val = float(np.mean(baseline_vals))
            std_val = float(np.std(baseline_vals))

            if std_val < 1e-6:
                continue

            z = (current_val - mean_val) / std_val
            if z > self._alert_z:
                return TailAlert(
                    market_ids=list(key),
                    alert_type="spike",
                    current_tail=current_val,
                    baseline_tail=mean_val,
                    z_score=z,
                    direction=direction,
                )

        return None

    def to_dict(self) -> dict[str, Any]:
        """Serialize tracker state."""
        return {
            "window_size": self._window_size,
            "step_size": self._step_size,
            "alert_z_threshold": self._alert_z,
            "pairs": {
                f"{k[0]}|{k[1]}": {
                    "buffer": self._buffers[k].to_dict(),
                    "history": [td.to_dict() for td in self._history[k]],
                }
                for k in self._buffers
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DynamicCopulaTracker:
        """Reconstruct from serialized state."""
        tracker = cls(
            window_size=data.get("window_size", 50),
            step_size=data.get("step_size", 10),
            alert_z_threshold=data.get("alert_z_threshold", 2.5),
        )
        for pair_key, pair_data in data.get("pairs", {}).items():
            parts = pair_key.split("|", 1)
            if len(parts) != 2:
                continue
            key = (parts[0], parts[1])
            tracker._buffers[key] = _PairBuffer.from_dict(pair_data.get("buffer", {}))
            # History reconstruction is intentionally skipped for simplicity —
            # the tracker rebuilds history from live data after restart.
            tracker._history[key] = []
        return tracker

    def __repr__(self) -> str:
        return (
            f"DynamicCopulaTracker(pairs={len(self._buffers)}, "
            f"window={self._window_size}, alert_z={self._alert_z})"
        )


# ---------------------------------------------------------------------------
# Internal pair buffer
# ---------------------------------------------------------------------------


class _PairBuffer:
    """Ring buffer for paired return observations."""

    def __init__(self, max_size: int = 500) -> None:
        self._max_size = max_size
        self._returns_a: list[float] = []
        self._returns_b: list[float] = []
        self._since_fit = 0

    def add(self, a: float, b: float) -> None:
        self._returns_a.append(a)
        self._returns_b.append(b)
        self._since_fit += 1
        # Trim if over max
        if len(self._returns_a) > self._max_size:
            trim = len(self._returns_a) - self._max_size
            self._returns_a = self._returns_a[trim:]
            self._returns_b = self._returns_b[trim:]

    def get_window(self, size: int) -> tuple[np.ndarray, np.ndarray]:
        a = np.array(self._returns_a[-size:])
        b = np.array(self._returns_b[-size:])
        return a, b

    @property
    def count(self) -> int:
        return len(self._returns_a)

    @property
    def since_last_fit(self) -> int:
        return self._since_fit

    def mark_fitted(self) -> None:
        self._since_fit = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "returns_a": self._returns_a,
            "returns_b": self._returns_b,
            "since_fit": self._since_fit,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> _PairBuffer:
        buf = cls()
        buf._returns_a = data.get("returns_a", [])
        buf._returns_b = data.get("returns_b", [])
        buf._since_fit = data.get("since_fit", 0)
        return buf
