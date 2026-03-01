"""Calibrator: fits ABM agent parameters to observed real market data.

The calibrator solves the inverse problem: given observed market
microstructure (price volatility, volume patterns, spread, order flow
imbalance), find the agent population parameters that best reproduce
those statistics in simulation.

This is essential because the detection power depends on the baseline
simulation matching the real market's "normal" behavior. A poorly
calibrated baseline produces false positives.

Calibration approach:
  1. Extract target statistics from observed data
  2. Run ABM with candidate parameters
  3. Compute distance between simulated and target statistics
  4. Iterate via grid search (fast) or Nelder-Mead (refined)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.optimize import minimize

from prediction_market.simulation.abm.simulator import ABMConfig, ABMResult, ABMSimulator

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TargetStatistics:
    """Target statistics extracted from observed market data.

    These are the moments/features we want the ABM to match.

    Attributes:
        mean_price: Average price over observation period.
        price_volatility: Standard deviation of log-returns.
        mean_volume_per_tick: Average volume per observation interval.
        volume_volatility: Std dev of per-tick volume.
        mean_spread: Average bid-ask spread.
        mean_imbalance_abs: Average absolute order flow imbalance.
        return_autocorrelation: Lag-1 return autocorrelation.
        volume_price_correlation: Correlation between volume and |return|.
    """

    mean_price: float = 0.5
    price_volatility: float = 0.01
    mean_volume_per_tick: float = 500.0
    volume_volatility: float = 200.0
    mean_spread: float = 0.02
    mean_imbalance_abs: float = 0.3
    return_autocorrelation: float = 0.0
    volume_price_correlation: float = 0.3

    @classmethod
    def from_observations(
        cls,
        prices: np.ndarray,
        volumes: np.ndarray | None = None,
        spreads: np.ndarray | None = None,
        imbalances: np.ndarray | None = None,
    ) -> TargetStatistics:
        """Extract target statistics from observed market data.

        Args:
            prices: Price time series.
            volumes: Per-period volume (optional).
            spreads: Bid-ask spreads (optional).
            imbalances: Order flow imbalances (optional).

        Returns:
            TargetStatistics instance.
        """
        prices = np.asarray(prices)
        mean_price = float(np.mean(prices))

        log_prices = np.log(np.clip(prices, 1e-6, 1 - 1e-6))
        returns = np.diff(log_prices)
        price_vol = float(np.std(returns)) if len(returns) > 1 else 0.01

        mean_vol = 500.0
        vol_vol = 200.0
        if volumes is not None and len(volumes) > 0:
            mean_vol = float(np.mean(volumes))
            vol_vol = float(np.std(volumes))

        mean_spread = 0.02
        if spreads is not None and len(spreads) > 0:
            mean_spread = float(np.mean(spreads))

        mean_imb = 0.3
        if imbalances is not None and len(imbalances) > 0:
            mean_imb = float(np.mean(np.abs(imbalances)))

        autocorr = 0.0
        if len(returns) > 2:
            mean_r = np.mean(returns)
            var_r = np.var(returns)
            if var_r > 1e-15:
                autocov = np.mean((returns[1:] - mean_r) * (returns[:-1] - mean_r))
                autocorr = float(autocov / var_r)

        vol_price_corr = 0.3
        if volumes is not None and len(volumes) > 1 and len(returns) > 0:
            min_len = min(len(volumes), len(returns))
            abs_returns = np.abs(returns[:min_len])
            vols = np.array(volumes[:min_len], dtype=float)
            if np.std(abs_returns) > 1e-10 and np.std(vols) > 1e-10:
                vol_price_corr = float(np.corrcoef(abs_returns, vols)[0, 1])
                if np.isnan(vol_price_corr):
                    vol_price_corr = 0.3

        return cls(
            mean_price=mean_price,
            price_volatility=price_vol,
            mean_volume_per_tick=mean_vol,
            volume_volatility=vol_vol,
            mean_spread=mean_spread,
            mean_imbalance_abs=mean_imb,
            return_autocorrelation=autocorr,
            volume_price_correlation=vol_price_corr,
        )

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__


@dataclass(frozen=True)
class CalibrationResult:
    """Result of the calibration process.

    Attributes:
        config: The best-fit ABM configuration.
        target: The target statistics we were matching.
        achieved: Statistics from the calibrated simulation.
        distance: Final distance metric (lower = better fit).
        n_evaluations: Number of simulation runs during calibration.
        elapsed_ms: Total calibration wall-clock time.
    """

    config: ABMConfig
    target: TargetStatistics
    achieved: TargetStatistics
    distance: float
    n_evaluations: int
    elapsed_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "target": self.target.to_dict(),
            "achieved": self.achieved.to_dict(),
            "distance": self.distance,
            "n_evaluations": self.n_evaluations,
            "elapsed_ms": self.elapsed_ms,
        }


class Calibrator:
    """Calibrates ABM parameters to match observed market microstructure.

    Two-phase approach:
      1. **Grid search**: Coarse sweep over key parameters to find a
         reasonable starting point.
      2. **Nelder-Mead**: Local optimization to refine the fit.

    The calibrated parameters can then be used to create a realistic
    baseline simulation for divergence detection.

    Args:
        n_ticks: Simulation length for each calibration run.
        n_eval_runs: Number of runs to average per parameter evaluation
            (reduces simulation noise).
        base_seed: Starting seed for reproducibility.
    """

    def __init__(
        self,
        n_ticks: int = 200,
        n_eval_runs: int = 3,
        base_seed: int = 42,
    ) -> None:
        self._n_ticks = n_ticks
        self._n_eval_runs = n_eval_runs
        self._base_seed = base_seed
        self._eval_count = 0

    def calibrate(
        self,
        target: TargetStatistics,
        initial_price: float | None = None,
        max_iterations: int = 50,
    ) -> CalibrationResult:
        """Run full calibration pipeline.

        Args:
            target: Target statistics from observed data.
            initial_price: Override starting price (defaults to target mean).
            max_iterations: Max Nelder-Mead iterations.

        Returns:
            CalibrationResult with the best-fit configuration.
        """
        t0 = time.monotonic()
        self._eval_count = 0

        if initial_price is None:
            initial_price = target.mean_price

        # Phase 1: Grid search for coarse parameters
        best_params, best_distance = self._grid_search(target, initial_price)
        logger.info(
            "Grid search complete: distance=%.4f after %d evaluations",
            best_distance, self._eval_count,
        )

        # Phase 2: Nelder-Mead refinement
        refined_params, refined_distance = self._refine(
            target, initial_price, best_params, max_iterations
        )
        logger.info(
            "Refinement complete: distance=%.4f after %d total evaluations",
            refined_distance, self._eval_count,
        )

        # Build final config
        final_config = self._params_to_config(
            refined_params, initial_price, self._base_seed
        )

        # Run final evaluation to get achieved statistics
        achieved = self._evaluate_statistics(final_config)

        elapsed = (time.monotonic() - t0) * 1000

        return CalibrationResult(
            config=final_config,
            target=target,
            achieved=achieved,
            distance=refined_distance,
            n_evaluations=self._eval_count,
            elapsed_ms=elapsed,
        )

    # ------------------------------------------------------------------
    # Grid search
    # ------------------------------------------------------------------

    def _grid_search(
        self, target: TargetStatistics, initial_price: float
    ) -> tuple[np.ndarray, float]:
        """Coarse grid search over key parameters.

        Parameters searched:
          [0] noise_trade_prob (0.1 - 0.5)
          [1] noise_mean_size (50 - 500)
          [2] base_liquidity (1000 - 50000)
          [3] mm_spread (0.005 - 0.05)
        """
        best_params = np.array([0.3, 100.0, 10000.0, 0.02])
        best_dist = float("inf")

        # Coarse grid
        for ntp in [0.15, 0.3, 0.5]:
            for nms in [50.0, 150.0, 400.0]:
                for liq in [2000.0, 10000.0, 30000.0]:
                    for spread in [0.01, 0.02, 0.04]:
                        params = np.array([ntp, nms, liq, spread])
                        dist = self._objective(params, target, initial_price)
                        if dist < best_dist:
                            best_dist = dist
                            best_params = params.copy()

        return best_params, best_dist

    # ------------------------------------------------------------------
    # Nelder-Mead refinement
    # ------------------------------------------------------------------

    def _refine(
        self,
        target: TargetStatistics,
        initial_price: float,
        x0: np.ndarray,
        max_iter: int,
    ) -> tuple[np.ndarray, float]:
        """Refine parameters via Nelder-Mead optimization."""
        result = minimize(
            self._objective,
            x0,
            args=(target, initial_price),
            method="Nelder-Mead",
            options={
                "maxiter": max_iter,
                "xatol": 0.01,
                "fatol": 0.001,
                "adaptive": True,
            },
        )
        return result.x, float(result.fun)

    # ------------------------------------------------------------------
    # Objective function
    # ------------------------------------------------------------------

    def _objective(
        self,
        params: np.ndarray,
        target: TargetStatistics,
        initial_price: float,
    ) -> float:
        """Compute distance between simulation output and target.

        Lower is better.
        """
        self._eval_count += 1

        # Enforce bounds
        noise_prob = np.clip(params[0], 0.05, 0.8)
        noise_size = np.clip(params[1], 10.0, 1000.0)
        liquidity = np.clip(params[2], 500.0, 100_000.0)
        spread = np.clip(params[3], 0.002, 0.1)

        config = ABMConfig(
            n_ticks=self._n_ticks,
            initial_price=initial_price,
            base_liquidity=liquidity,
            n_noise=20,
            n_market_makers=3,
            n_momentum=5,
            n_informed=0,
            noise_trade_prob=noise_prob,
            noise_mean_size=noise_size,
            mm_spread=spread,
            seed=self._base_seed,
        )

        achieved = self._evaluate_statistics(config)

        # Weighted distance across target dimensions
        distance = 0.0

        # Price volatility (most important for detection)
        if target.price_volatility > 1e-10:
            distance += 3.0 * (
                (achieved.price_volatility - target.price_volatility)
                / target.price_volatility
            ) ** 2

        # Volume
        if target.mean_volume_per_tick > 1e-10:
            distance += 1.0 * (
                (achieved.mean_volume_per_tick - target.mean_volume_per_tick)
                / target.mean_volume_per_tick
            ) ** 2

        # Spread
        if target.mean_spread > 1e-10:
            distance += 2.0 * (
                (achieved.mean_spread - target.mean_spread) / target.mean_spread
            ) ** 2

        # Imbalance
        if target.mean_imbalance_abs > 1e-10:
            distance += 1.0 * (
                (achieved.mean_imbalance_abs - target.mean_imbalance_abs)
                / target.mean_imbalance_abs
            ) ** 2

        # Autocorrelation
        distance += 0.5 * (
            achieved.return_autocorrelation - target.return_autocorrelation
        ) ** 2

        return distance

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _evaluate_statistics(self, config: ABMConfig) -> TargetStatistics:
        """Run simulation(s) and extract statistics."""
        simulator = ABMSimulator(config)
        results: list[ABMResult] = []

        for i in range(self._n_eval_runs):
            cfg = ABMConfig(
                **{k: v for k, v in config.__dict__.items() if k != "seed"},
                seed=(config.seed or 42) + i,
            )
            results.append(simulator.run(cfg))

        # Average across runs
        vols = [r.realized_volatility for r in results]
        volumes = [r.total_volume / max(r.config.n_ticks, 1) for r in results]
        spreads = [r.mean_spread for r in results]
        autocorrs = [r.return_autocorrelation for r in results]

        # Imbalance: average across all tick states
        all_imbalances = []
        for r in results:
            all_imbalances.extend([abs(x) for x in r.imbalance_series])
        mean_imb = float(np.mean(all_imbalances)) if all_imbalances else 0.3

        return TargetStatistics(
            mean_price=float(np.mean([r.final_price for r in results])),
            price_volatility=float(np.mean(vols)),
            mean_volume_per_tick=float(np.mean(volumes)),
            volume_volatility=float(np.std(volumes)),
            mean_spread=float(np.mean(spreads)),
            mean_imbalance_abs=mean_imb,
            return_autocorrelation=float(np.mean(autocorrs)),
        )

    @staticmethod
    def _params_to_config(
        params: np.ndarray, initial_price: float, seed: int
    ) -> ABMConfig:
        """Convert optimization parameters to ABMConfig."""
        return ABMConfig(
            n_ticks=500,
            initial_price=initial_price,
            base_liquidity=float(np.clip(params[2], 500, 100_000)),
            n_noise=20,
            n_market_makers=3,
            n_momentum=5,
            n_informed=0,
            noise_trade_prob=float(np.clip(params[0], 0.05, 0.8)),
            noise_mean_size=float(np.clip(params[1], 10, 1000)),
            mm_spread=float(np.clip(params[3], 0.002, 0.1)),
            seed=seed,
        )

    def __repr__(self) -> str:
        return (
            f"Calibrator(n_ticks={self._n_ticks}, "
            f"n_eval_runs={self._n_eval_runs})"
        )
