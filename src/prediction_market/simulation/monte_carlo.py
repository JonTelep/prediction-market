"""Monte Carlo simulation engine for prediction market probability estimation.

Runs configurable numbers of simulations to produce probability distributions,
confidence intervals, and probability cones for market price trajectories.
Integrates with the existing PriceAnalyzer to augment deterministic z-score
detection with stochastic modeling.

Key capabilities:
  - Single-step probability estimation (what's the chance price exceeds X?)
  - Multi-step path simulation (probability cones over N future periods)
  - Anomaly scoring via simulation (how unlikely is the observed price under
    the fitted model?)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np

from prediction_market.simulation.distributions import (
    BetaMarketModel,
    DirichletMarketModel,
    MarketModel,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SimulationResult:
    """Result of a Monte Carlo simulation run.

    Attributes:
        market_id: The market this simulation was run for.
        n_simulations: Number of simulations executed.
        mean: Mean of simulated outcomes.
        std: Standard deviation of simulated outcomes.
        median: Median simulated outcome.
        percentiles: Dict mapping percentile labels to values
            (e.g., {"5%": 0.32, "95%": 0.78}).
        probability_above: P(X > current_price) from simulations.
        probability_below: P(X < current_price) from simulations.
        tail_probability: P(X in extreme tail) — either direction.
        anomaly_score: How many simulated SDs the observed value is from
            the simulated mean. Higher = more anomalous.
        observed_price: The actual observed price being evaluated.
        model_type: Name of the distribution model used.
        elapsed_ms: Wall-clock time for the simulation in milliseconds.
        timestamp: When the simulation was run.
    """

    market_id: str
    n_simulations: int
    mean: float
    std: float
    median: float
    percentiles: dict[str, float] = field(default_factory=dict)
    probability_above: float = 0.0
    probability_below: float = 0.0
    tail_probability: float = 0.0
    anomaly_score: float = 0.0
    observed_price: float = 0.0
    model_type: str = ""
    elapsed_ms: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        return {
            "market_id": self.market_id,
            "n_simulations": self.n_simulations,
            "mean": self.mean,
            "std": self.std,
            "median": self.median,
            "percentiles": self.percentiles,
            "probability_above": self.probability_above,
            "probability_below": self.probability_below,
            "tail_probability": self.tail_probability,
            "anomaly_score": self.anomaly_score,
            "observed_price": self.observed_price,
            "model_type": self.model_type,
            "elapsed_ms": self.elapsed_ms,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass(frozen=True)
class ProbabilityCone:
    """Multi-step simulation producing a probability cone (fan chart).

    Attributes:
        market_id: The market this cone was generated for.
        steps: Number of forward time steps simulated.
        n_paths: Number of simulated paths.
        percentile_bands: Dict mapping band labels to arrays of shape (steps,).
            E.g., {"5%": [...], "25%": [...], "50%": [...], "75%": [...], "95%": [...]}.
        mean_path: Array of shape (steps,) with the mean at each step.
        timestamp: When the cone was generated.
    """

    market_id: str
    steps: int
    n_paths: int
    percentile_bands: dict[str, list[float]]
    mean_path: list[float]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        return {
            "market_id": self.market_id,
            "steps": self.steps,
            "n_paths": self.n_paths,
            "percentile_bands": self.percentile_bands,
            "mean_path": self.mean_path,
            "timestamp": self.timestamp.isoformat(),
        }


class MonteCarloEngine:
    """Monte Carlo simulation engine for prediction markets.

    Runs stochastic simulations against fitted market models to produce
    probability estimates, confidence intervals, and anomaly scores.

    Args:
        n_simulations: Default number of simulations per run.
        seed: Random seed for reproducibility. None = non-deterministic.
        tail_percentile: Percentile threshold for defining "tail" events
            (default 5% — i.e., events in the bottom/top 5%).
    """

    def __init__(
        self,
        n_simulations: int = 10_000,
        seed: int | None = None,
        tail_percentile: float = 5.0,
    ) -> None:
        self._n_simulations = n_simulations
        self._rng = np.random.default_rng(seed)
        self._tail_pct = tail_percentile
        self._models: dict[str, MarketModel] = {}

    # ------------------------------------------------------------------
    # Model management
    # ------------------------------------------------------------------

    def fit_model(
        self,
        market_id: str,
        prices: np.ndarray,
        model_type: str = "beta",
    ) -> MarketModel:
        """Fit a distribution model to historical prices for a market.

        Args:
            market_id: Market identifier.
            prices: Historical price observations.
            model_type: ``"beta"`` for binary markets, ``"dirichlet"`` for
                multi-outcome (in which case prices should be 2-D).

        Returns:
            The fitted MarketModel.
        """
        if model_type == "beta":
            model = BetaMarketModel()
        elif model_type == "dirichlet":
            model = DirichletMarketModel()
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        model.fit(np.asarray(prices))
        self._models[market_id] = model
        logger.info("Fitted %s model for market %s", model_type, market_id)
        return model

    def get_model(self, market_id: str) -> MarketModel | None:
        """Retrieve the fitted model for a market, if any."""
        return self._models.get(market_id)

    def set_model(self, market_id: str, model: MarketModel) -> None:
        """Manually set a model for a market (e.g., from deserialized state)."""
        self._models[market_id] = model

    # ------------------------------------------------------------------
    # Core simulation
    # ------------------------------------------------------------------

    def simulate(
        self,
        market_id: str,
        observed_price: float,
        n: int | None = None,
    ) -> SimulationResult:
        """Run a Monte Carlo simulation for a single market.

        Draws *n* samples from the fitted model, then computes statistics
        and an anomaly score for the observed price.

        Args:
            market_id: Market identifier (must have a fitted model).
            observed_price: The current/latest observed price.
            n: Number of simulations (overrides default).

        Returns:
            SimulationResult with full statistics.

        Raises:
            KeyError: If no model has been fitted for *market_id*.
        """
        model = self._models.get(market_id)
        if model is None:
            raise KeyError(f"No fitted model for market '{market_id}'. Call fit_model() first.")

        n = n or self._n_simulations
        t0 = time.monotonic()

        samples = model.sample(n, rng=self._rng)

        # For Dirichlet, take the first outcome (Yes price equivalent)
        if samples.ndim == 2:
            samples = samples[:, 0]

        mean = float(np.mean(samples))
        std = float(np.std(samples))
        median = float(np.median(samples))

        # Percentiles
        pct_keys = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        pct_values = np.percentile(samples, pct_keys)
        percentiles = {f"{k}%": float(v) for k, v in zip(pct_keys, pct_values)}

        # Directional probabilities
        prob_above = float(np.mean(samples > observed_price))
        prob_below = float(np.mean(samples < observed_price))

        # Tail probability: P(X in bottom or top tail_pct%)
        lower_tail = float(np.percentile(samples, self._tail_pct))
        upper_tail = float(np.percentile(samples, 100 - self._tail_pct))
        tail_prob = float(np.mean((samples <= lower_tail) | (samples >= upper_tail)))

        # Anomaly score: how many simulated SDs is the observed price from the mean
        anomaly_score = abs(observed_price - mean) / std if std > 1e-12 else 0.0

        elapsed = (time.monotonic() - t0) * 1000

        result = SimulationResult(
            market_id=market_id,
            n_simulations=n,
            mean=mean,
            std=std,
            median=median,
            percentiles=percentiles,
            probability_above=prob_above,
            probability_below=prob_below,
            tail_probability=tail_prob,
            anomaly_score=anomaly_score,
            observed_price=observed_price,
            model_type=type(model).__name__,
            elapsed_ms=elapsed,
        )

        logger.debug(
            "MC simulation for %s: mean=%.4f std=%.4f anomaly=%.2f (%dms)",
            market_id,
            mean,
            std,
            anomaly_score,
            elapsed,
        )
        return result

    # ------------------------------------------------------------------
    # Probability cones (multi-step)
    # ------------------------------------------------------------------

    def simulate_cone(
        self,
        market_id: str,
        current_price: float,
        steps: int = 24,
        n_paths: int | None = None,
        volatility: float | None = None,
    ) -> ProbabilityCone:
        """Simulate forward price paths to produce a probability cone.

        Uses geometric Brownian motion anchored to the fitted model's
        parameters. Each path starts at *current_price* and evolves for
        *steps* periods.

        Args:
            market_id: Market identifier (must have a fitted model).
            current_price: Starting price.
            steps: Number of forward time steps.
            n_paths: Number of paths to simulate (defaults to engine default).
            volatility: Override volatility. If None, estimated from the
                fitted model's standard deviation.

        Returns:
            A ProbabilityCone with percentile bands at each step.
        """
        model = self._models.get(market_id)
        if model is None:
            raise KeyError(f"No fitted model for market '{market_id}'")

        n = n_paths or self._n_simulations

        # Estimate volatility from model
        if volatility is None:
            test_samples = model.sample(5000, rng=self._rng)
            if test_samples.ndim == 2:
                test_samples = test_samples[:, 0]
            volatility = float(np.std(np.log(np.clip(test_samples, 1e-6, 1.0))))
            volatility = max(volatility, 0.01)  # Floor

        # Simulate paths via GBM with reflecting boundaries at [0.01, 0.99]
        paths = np.zeros((n, steps))
        paths[:, 0] = current_price

        for t in range(1, steps):
            noise = self._rng.standard_normal(n)
            log_returns = -0.5 * volatility**2 + volatility * noise
            paths[:, t] = paths[:, t - 1] * np.exp(log_returns)
            # Reflect at boundaries (prediction market prices are in [0, 1])
            paths[:, t] = np.clip(paths[:, t], 0.01, 0.99)

        # Extract percentile bands
        bands = {}
        for pct in [5, 25, 50, 75, 95]:
            bands[f"{pct}%"] = np.percentile(paths, pct, axis=0).tolist()

        mean_path = np.mean(paths, axis=0).tolist()

        return ProbabilityCone(
            market_id=market_id,
            steps=steps,
            n_paths=n,
            percentile_bands=bands,
            mean_path=mean_path,
        )

    # ------------------------------------------------------------------
    # Batch operations
    # ------------------------------------------------------------------

    def simulate_all(
        self,
        observed_prices: dict[str, float],
        n: int | None = None,
    ) -> dict[str, SimulationResult]:
        """Run simulations for all fitted markets.

        Args:
            observed_prices: Dict mapping market_id → current price.
            n: Override simulation count.

        Returns:
            Dict mapping market_id → SimulationResult.
        """
        results = {}
        for market_id, model in self._models.items():
            price = observed_prices.get(market_id)
            if price is None:
                logger.warning("No observed price for market %s — skipping", market_id)
                continue
            results[market_id] = self.simulate(market_id, price, n=n)
        return results

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize engine state (models only — RNG state is not preserved)."""
        return {
            "n_simulations": self._n_simulations,
            "tail_percentile": self._tail_pct,
            "models": {
                mid: model.to_dict() for mid, model in self._models.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], seed: int | None = None) -> MonteCarloEngine:
        """Reconstruct an engine from serialized state.

        Args:
            data: Dict from to_dict().
            seed: Optional new random seed.

        Returns:
            Restored MonteCarloEngine.
        """
        engine = cls(
            n_simulations=data.get("n_simulations", 10_000),
            seed=seed,
            tail_percentile=data.get("tail_percentile", 5.0),
        )
        for mid, model_data in data.get("models", {}).items():
            model_type = model_data.get("type", "beta")
            if model_type == "beta":
                engine._models[mid] = BetaMarketModel.from_dict(model_data)
            elif model_type == "dirichlet":
                engine._models[mid] = DirichletMarketModel.from_dict(model_data)
        return engine

    @property
    def tracked_markets(self) -> list[str]:
        """Market IDs with fitted models."""
        return list(self._models.keys())

    def __repr__(self) -> str:
        return (
            f"MonteCarloEngine(n={self._n_simulations}, "
            f"markets={len(self._models)}, tail={self._tail_pct}%)"
        )
