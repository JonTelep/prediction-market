"""Importance sampling for rare-event estimation in prediction markets.

Standard Monte Carlo is inefficient for estimating probabilities of rare
events (e.g., P(price crash > 30%) or P(correlated manipulation across
markets)). Importance sampling shifts the sampling distribution toward
the region of interest, then corrects via likelihood ratios.

This module provides:
  - Tail risk estimation with dramatically reduced variance vs naive MC
  - Effective sample size (ESS) diagnostics to detect weight degeneracy
  - Self-normalized importance sampling for robustness
  - Integration with the existing anomaly detection pipeline
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
from scipy import stats as sp_stats

from prediction_market.simulation.distributions import BetaMarketModel, MarketModel

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TailRiskEstimate:
    """Result of an importance-sampled tail risk estimation.

    Attributes:
        market_id: Market identifier.
        threshold: The price threshold being evaluated.
        direction: ``"above"`` or ``"below"``.
        probability_naive: Naive MC estimate (for comparison).
        probability_is: Importance-sampled estimate.
        variance_reduction: Ratio of naive variance to IS variance
            (> 1 means IS was more efficient).
        effective_sample_size: ESS — measures how many "effective" independent
            samples the IS weights represent. Low ESS = poor proposal.
        n_samples: Total samples drawn.
        confidence_interval: 95% CI for the IS estimate.
        elapsed_ms: Computation time in milliseconds.
        timestamp: When the estimate was produced.
    """

    market_id: str
    threshold: float
    direction: str
    probability_naive: float
    probability_is: float
    variance_reduction: float
    effective_sample_size: float
    n_samples: int
    confidence_interval: tuple[float, float] = (0.0, 0.0)
    elapsed_ms: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        return {
            "market_id": self.market_id,
            "threshold": self.threshold,
            "direction": self.direction,
            "probability_naive": self.probability_naive,
            "probability_is": self.probability_is,
            "variance_reduction": self.variance_reduction,
            "effective_sample_size": self.effective_sample_size,
            "n_samples": self.n_samples,
            "confidence_interval": list(self.confidence_interval),
            "elapsed_ms": self.elapsed_ms,
            "timestamp": self.timestamp.isoformat(),
        }


class ImportanceSampler:
    """Importance sampler for tail risk estimation in prediction markets.

    Given a fitted "nominal" model (the target distribution), constructs
    a shifted proposal distribution that oversamples the tail region of
    interest, then reweights samples to get an unbiased estimate of the
    tail probability.

    For Beta-distributed binary markets, the proposal is another Beta
    distribution with shifted parameters to concentrate mass in the tail.

    Args:
        n_samples: Number of importance samples to draw.
        seed: Random seed for reproducibility.
        shift_strength: How aggressively to shift the proposal toward the
            tail. Higher = more aggressive (default 2.0).
    """

    def __init__(
        self,
        n_samples: int = 50_000,
        seed: int | None = None,
        shift_strength: float = 2.0,
    ) -> None:
        self._n_samples = n_samples
        self._rng = np.random.default_rng(seed)
        self._shift_strength = shift_strength

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def estimate_tail_risk(
        self,
        market_id: str,
        model: MarketModel,
        threshold: float,
        direction: str = "above",
        n: int | None = None,
    ) -> TailRiskEstimate:
        """Estimate the probability of a tail event using importance sampling.

        Args:
            market_id: Market identifier (for labeling).
            model: The fitted nominal model (target distribution).
            threshold: Price threshold defining the tail event.
            direction: ``"above"`` for P(X > threshold), ``"below"`` for
                P(X < threshold).
            n: Override sample count.

        Returns:
            TailRiskEstimate with the IS estimate and diagnostics.
        """
        if not isinstance(model, BetaMarketModel):
            # Fallback to naive MC for non-Beta models
            return self._naive_estimate(market_id, model, threshold, direction, n)

        n = n or self._n_samples
        t0 = time.monotonic()

        # --- 1. Build proposal distribution ---
        proposal = self._build_beta_proposal(model, threshold, direction)

        # --- 2. Draw samples from proposal ---
        samples = proposal.sample(n, rng=self._rng)

        # --- 3. Compute importance weights ---
        # w(x) = p_target(x) / q_proposal(x)
        log_target = sp_stats.beta.logpdf(samples, model.alpha, model.beta)
        log_proposal = sp_stats.beta.logpdf(samples, proposal.alpha, proposal.beta)
        log_weights = log_target - log_proposal

        # Stabilize: subtract max for numerical safety
        log_weights -= np.max(log_weights)
        weights = np.exp(log_weights)

        # --- 4. Compute indicator for tail event ---
        if direction == "above":
            indicator = (samples > threshold).astype(float)
        else:
            indicator = (samples < threshold).astype(float)

        # --- 5. Self-normalized importance sampling estimate ---
        weight_sum = np.sum(weights)
        if weight_sum < 1e-30:
            logger.warning("Weight sum near zero for market %s — degenerate proposal", market_id)
            return self._naive_estimate(market_id, model, threshold, direction, n)

        prob_is = float(np.sum(weights * indicator) / weight_sum)

        # --- 6. Effective sample size ---
        normalized_weights = weights / weight_sum
        ess = 1.0 / float(np.sum(normalized_weights**2))

        # --- 7. Naive MC estimate for comparison ---
        naive_samples = model.sample(n, rng=self._rng)
        if direction == "above":
            prob_naive = float(np.mean(naive_samples > threshold))
        else:
            prob_naive = float(np.mean(naive_samples < threshold))

        # --- 8. Variance reduction ratio ---
        # Var_naive / Var_IS (approximate)
        is_var = self._is_variance(weights, indicator, weight_sum)
        naive_var = prob_naive * (1 - prob_naive) / n if n > 0 else float("inf")
        var_reduction = naive_var / is_var if is_var > 1e-30 else float("inf")

        # --- 9. Confidence interval (Wilson score-style for IS) ---
        ci = self._confidence_interval(prob_is, ess)

        elapsed = (time.monotonic() - t0) * 1000

        result = TailRiskEstimate(
            market_id=market_id,
            threshold=threshold,
            direction=direction,
            probability_naive=prob_naive,
            probability_is=prob_is,
            variance_reduction=var_reduction,
            effective_sample_size=ess,
            n_samples=n,
            confidence_interval=ci,
            elapsed_ms=elapsed,
        )

        logger.debug(
            "IS estimate for %s: P(X %s %.3f) = %.6f (naive=%.6f, VR=%.1fx, ESS=%.0f)",
            market_id,
            ">" if direction == "above" else "<",
            threshold,
            prob_is,
            prob_naive,
            var_reduction,
            ess,
        )
        return result

    # ------------------------------------------------------------------
    # Batch operations
    # ------------------------------------------------------------------

    def estimate_symmetric_tails(
        self,
        market_id: str,
        model: MarketModel,
        threshold_low: float,
        threshold_high: float,
        n: int | None = None,
    ) -> tuple[TailRiskEstimate, TailRiskEstimate]:
        """Estimate both tail probabilities for a market.

        Args:
            market_id: Market identifier.
            model: Fitted model.
            threshold_low: Lower tail threshold.
            threshold_high: Upper tail threshold.
            n: Override sample count.

        Returns:
            Tuple of (lower_tail_estimate, upper_tail_estimate).
        """
        lower = self.estimate_tail_risk(market_id, model, threshold_low, "below", n)
        upper = self.estimate_tail_risk(market_id, model, threshold_high, "above", n)
        return lower, upper

    # ------------------------------------------------------------------
    # Proposal construction
    # ------------------------------------------------------------------

    def _build_beta_proposal(
        self,
        target: BetaMarketModel,
        threshold: float,
        direction: str,
    ) -> BetaMarketModel:
        """Build a shifted Beta proposal that oversamples the tail region.

        For ``direction="above"``, the proposal shifts mass toward higher
        values by increasing α relative to β. For ``"below"``, the reverse.

        The shift strength controls how aggressively the proposal departs
        from the target — too aggressive causes weight degeneracy (low ESS),
        too conservative yields minimal variance reduction.
        """
        a, b = target.alpha, target.beta
        s = self._shift_strength

        if direction == "above":
            # Shift mean upward: increase α, decrease β
            prop_alpha = a * s
            prop_beta = max(b / s, 0.5)
        else:
            # Shift mean downward: decrease α, increase β
            prop_alpha = max(a / s, 0.5)
            prop_beta = b * s

        return BetaMarketModel(alpha=prop_alpha, beta=prop_beta)

    # ------------------------------------------------------------------
    # Variance estimation
    # ------------------------------------------------------------------

    @staticmethod
    def _is_variance(
        weights: np.ndarray,
        indicator: np.ndarray,
        weight_sum: float,
    ) -> float:
        """Estimate the variance of the self-normalized IS estimator."""
        n = len(weights)
        if n < 2 or weight_sum < 1e-30:
            return float("inf")

        normalized = weights / weight_sum
        estimate = float(np.sum(normalized * indicator))
        residuals = indicator - estimate
        return float(np.sum(normalized**2 * residuals**2))

    @staticmethod
    def _confidence_interval(
        estimate: float,
        ess: float,
        z: float = 1.96,
    ) -> tuple[float, float]:
        """Compute approximate 95% CI using effective sample size.

        Uses the normal approximation with ESS as the effective n.
        """
        if ess < 2:
            return (0.0, 1.0)

        se = (estimate * (1 - estimate) / ess) ** 0.5
        lo = max(0.0, estimate - z * se)
        hi = min(1.0, estimate + z * se)
        return (lo, hi)

    # ------------------------------------------------------------------
    # Fallback
    # ------------------------------------------------------------------

    def _naive_estimate(
        self,
        market_id: str,
        model: MarketModel,
        threshold: float,
        direction: str,
        n: int | None = None,
    ) -> TailRiskEstimate:
        """Fallback to naive Monte Carlo when IS isn't applicable."""
        n = n or self._n_samples
        t0 = time.monotonic()

        samples = model.sample(n, rng=self._rng)
        if samples.ndim == 2:
            samples = samples[:, 0]

        if direction == "above":
            prob = float(np.mean(samples > threshold))
        else:
            prob = float(np.mean(samples < threshold))

        elapsed = (time.monotonic() - t0) * 1000

        return TailRiskEstimate(
            market_id=market_id,
            threshold=threshold,
            direction=direction,
            probability_naive=prob,
            probability_is=prob,
            variance_reduction=1.0,
            effective_sample_size=float(n),
            n_samples=n,
            confidence_interval=self._confidence_interval(prob, float(n)),
            elapsed_ms=elapsed,
        )

    def __repr__(self) -> str:
        return (
            f"ImportanceSampler(n={self._n_samples}, "
            f"shift={self._shift_strength})"
        )
