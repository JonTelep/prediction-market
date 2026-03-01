"""Integration layer: wires the simulation engine into the live detection pipeline.

Provides a SimulationEnhancedDetector that augments the existing z-score
pipeline with:
  - Monte Carlo anomaly scores (Phase 1)
  - Particle filter surprise + drift scores (Phase 2)
  - Copula-based cross-market tail alerts (Phase 3)
  - ABM divergence scoring (Phase 4)

Also provides standalone analysis functions for the CLI.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np

from prediction_market.simulation.distributions import BetaMarketModel
from prediction_market.simulation.monte_carlo import MonteCarloEngine, SimulationResult
from prediction_market.simulation.importance_sampler import ImportanceSampler, TailRiskEstimate
from prediction_market.simulation.particle_filter import (
    MarketAwareTransition,
    MarketContext,
    ParticleFilter,
    ParticleFilterResult,
    TradeFlowAnalyzer,
)
from prediction_market.simulation.copulas import (
    CopulaFitter,
    DynamicCopulaTracker,
    TailAlert,
    TailDependence,
)
from prediction_market.simulation.abm import (
    ABMConfig,
    ABMResult,
    ABMSimulator,
    Calibrator,
    CalibrationResult,
    DivergenceMetrics,
    TargetStatistics,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-market simulation state
# ---------------------------------------------------------------------------


@dataclass
class MarketSimulationState:
    """Holds all simulation components for a single market.

    Created lazily when a market first receives enough data for fitting.
    """

    market_id: str
    particle_filter: ParticleFilter
    mc_engine: MonteCarloEngine
    importance_sampler: ImportanceSampler
    trade_flow_analyzer: TradeFlowAnalyzer
    model_fitted: bool = False
    last_mc_result: SimulationResult | None = None
    last_pf_result: ParticleFilterResult | None = None
    last_tail_risk: TailRiskEstimate | None = None
    abm_calibrated: bool = False
    abm_config: ABMConfig | None = None
    abm_divergence: DivergenceMetrics | None = None
    price_history: list[float] = field(default_factory=list)
    volume_history: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "market_id": self.market_id,
            "particle_filter": self.particle_filter.to_dict(),
            "model_fitted": self.model_fitted,
            "mc_engine": self.mc_engine.to_dict(),
            "last_mc_result": self.last_mc_result.to_dict() if self.last_mc_result else None,
            "last_pf_result": self.last_pf_result.to_dict() if self.last_pf_result else None,
            "abm_calibrated": self.abm_calibrated,
            "price_history": self.price_history[-500:],
            "volume_history": self.volume_history[-500:],
        }


# ---------------------------------------------------------------------------
# Enhanced detector
# ---------------------------------------------------------------------------


class SimulationEnhancedDetector:
    """Wraps the full simulation engine for integration with InfoLeakDetector.

    Call ``process_tick()`` on each orchestrator tick with the market's
    current state. Returns a ``SimulationSignals`` object containing all
    simulation-derived scores that can augment the existing z-score pipeline.

    Args:
        mc_simulations: Number of Monte Carlo simulations per tick.
        is_samples: Number of importance samples for tail estimation.
        n_particles: Particles per particle filter.
        min_history: Minimum observations before fitting models.
        seed: Base random seed.
    """

    def __init__(
        self,
        mc_simulations: int = 5_000,
        is_samples: int = 20_000,
        n_particles: int = 2_000,
        min_history: int = 30,
        seed: int = 42,
    ) -> None:
        self._mc_n = mc_simulations
        self._is_n = is_samples
        self._n_particles = n_particles
        self._min_history = min_history
        self._seed = seed

        self._states: dict[str, MarketSimulationState] = {}
        self._copula_tracker = DynamicCopulaTracker(
            window_size=50, step_size=10, alert_z_threshold=2.5
        )
        self._trade_flow = TradeFlowAnalyzer(lookback_minutes=30)

    def process_tick(
        self,
        market_id: str,
        price: float,
        volume: float = 0.0,
        timestamp: datetime | None = None,
        orderbook_row: dict[str, Any] | None = None,
        recent_trades: list[dict[str, Any]] | None = None,
    ) -> SimulationSignals:
        """Process one tick for a market through all simulation layers.

        Args:
            market_id: Market identifier.
            price: Current price (0-1).
            volume: Current tick volume.
            timestamp: Observation time.
            orderbook_row: Dict from orderbook_snapshots table.
            recent_trades: List of trade dicts for trade flow analysis.

        Returns:
            SimulationSignals with all derived scores.
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        state = self._get_or_create(market_id)
        state.price_history.append(price)
        state.volume_history.append(volume)

        # Trim histories
        if len(state.price_history) > 1000:
            state.price_history = state.price_history[-1000:]
        if len(state.volume_history) > 1000:
            state.volume_history = state.volume_history[-1000:]

        signals = SimulationSignals(market_id=market_id, timestamp=timestamp)

        # --- Build market context for particle filter ---
        context = None
        if orderbook_row:
            context = MarketContext.from_orderbook_row(orderbook_row)
            if recent_trades:
                flow_imb, real_vol = self._trade_flow.compute(recent_trades)
                context = context.with_trade_flow(flow_imb, real_vol)

        # --- Phase 2: Particle filter (runs every tick) ---
        pf_result = state.particle_filter.update(price, timestamp, context)
        state.last_pf_result = pf_result
        signals.pf_surprise = pf_result.surprise
        signals.pf_drift_score = pf_result.drift_score
        signals.pf_ess_ratio = pf_result.ess_ratio
        signals.pf_regime = pf_result.regime
        signals.pf_posterior_mean = pf_result.posterior_mean

        # --- Phase 1: Monte Carlo (after enough history) ---
        if len(state.price_history) >= self._min_history:
            if not state.model_fitted:
                try:
                    state.mc_engine.fit_model(
                        market_id,
                        np.array(state.price_history),
                        model_type="beta",
                    )
                    state.model_fitted = True
                except Exception as e:
                    logger.debug("MC model fit failed for %s: %s", market_id, e)

            if state.model_fitted:
                mc_result = state.mc_engine.simulate(market_id, price)
                state.last_mc_result = mc_result
                signals.mc_anomaly_score = mc_result.anomaly_score
                signals.mc_probability_above = mc_result.probability_above
                signals.mc_probability_below = mc_result.probability_below
                signals.mc_mean = mc_result.mean
                signals.mc_std = mc_result.std

                # Importance sampling for tail risk
                model = state.mc_engine.get_model(market_id)
                if model is not None:
                    try:
                        tail_above = state.importance_sampler.estimate_tail_risk(
                            market_id, model, threshold=price + 0.15, direction="above"
                        )
                        tail_below = state.importance_sampler.estimate_tail_risk(
                            market_id, model, threshold=price - 0.15, direction="below"
                        )
                        state.last_tail_risk = tail_above
                        signals.tail_risk_above = tail_above.probability_is
                        signals.tail_risk_below = tail_below.probability_is
                    except Exception as e:
                        logger.debug("IS failed for %s: %s", market_id, e)

        # --- Compute combined simulation score ---
        signals.combined_sim_score = self._compute_combined_score(signals)

        return signals

    def process_cross_market(
        self,
        market_a: str,
        market_b: str,
        return_a: float,
        return_b: float,
    ) -> TailAlert | None:
        """Feed cross-market returns into the copula tracker.

        Call this for every pair of related markets on each tick.

        Returns:
            TailAlert if tail dependence spiked, else None.
        """
        return self._copula_tracker.update(market_a, market_b, return_a, return_b)

    def get_cross_market_dependence(
        self, market_a: str, market_b: str
    ) -> TailDependence | None:
        """Get latest tail dependence estimate for a market pair."""
        return self._copula_tracker.get_latest(market_a, market_b)

    def calibrate_abm(
        self,
        market_id: str,
        max_iterations: int = 30,
    ) -> CalibrationResult | None:
        """Calibrate ABM parameters for a market using its history.

        Returns None if insufficient data.
        """
        state = self._states.get(market_id)
        if state is None or len(state.price_history) < 50:
            return None

        target = TargetStatistics.from_observations(
            prices=np.array(state.price_history),
            volumes=np.array(state.volume_history) if state.volume_history else None,
        )

        calibrator = Calibrator(n_ticks=min(len(state.price_history), 200), n_eval_runs=2)
        result = calibrator.calibrate(target, max_iterations=max_iterations)

        state.abm_calibrated = True
        state.abm_config = result.config
        return result

    def compute_abm_divergence(
        self,
        market_id: str,
        n_baseline_runs: int = 5,
    ) -> DivergenceMetrics | None:
        """Run ABM baseline and compare against real market data.

        Must call calibrate_abm() first.
        """
        state = self._states.get(market_id)
        if state is None or not state.abm_calibrated or state.abm_config is None:
            return None

        simulator = ABMSimulator(state.abm_config)
        baselines = simulator.run_baseline(n_runs=n_baseline_runs)

        metrics = simulator.compare_with_observed(
            baselines,
            observed_prices=np.array(state.price_history),
            observed_volumes=np.array(state.volume_history) if state.volume_history else None,
        )

        state.abm_divergence = metrics
        return metrics

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def get_state(self, market_id: str) -> MarketSimulationState | None:
        return self._states.get(market_id)

    @property
    def tracked_markets(self) -> list[str]:
        return list(self._states.keys())

    @property
    def copula_tracker(self) -> DynamicCopulaTracker:
        return self._copula_tracker

    def to_dict(self) -> dict[str, Any]:
        return {
            "states": {
                mid: state.to_dict() for mid, state in self._states.items()
            },
            "copula_tracker": self._copula_tracker.to_dict(),
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_or_create(self, market_id: str) -> MarketSimulationState:
        if market_id not in self._states:
            self._states[market_id] = MarketSimulationState(
                market_id=market_id,
                particle_filter=ParticleFilter(
                    market_id, n_particles=self._n_particles, seed=self._seed
                ),
                mc_engine=MonteCarloEngine(
                    n_simulations=self._mc_n, seed=self._seed
                ),
                importance_sampler=ImportanceSampler(
                    n_samples=self._is_n, seed=self._seed
                ),
                trade_flow_analyzer=TradeFlowAnalyzer(),
            )
        return self._states[market_id]

    @staticmethod
    def _compute_combined_score(signals: SimulationSignals) -> float:
        """Combine all simulation signals into a single 0-1 score.

        Weights:
          - MC anomaly: 25% (how unlikely is this price?)
          - PF surprise: 25% (how unexpected given recent dynamics?)
          - PF drift: 30% (is there sustained directional pressure?)
          - Tail risk: 20% (how extreme could this go?)
        """
        # Normalize MC anomaly (typically 0-5+, map to 0-1)
        mc_norm = min(signals.mc_anomaly_score / 4.0, 1.0)

        # Normalize PF surprise (typically 0-15, map to 0-1)
        pf_surprise_norm = min(signals.pf_surprise / 10.0, 1.0)

        # Normalize drift (abs value, typically 0-50+, map to 0-1)
        drift_norm = min(abs(signals.pf_drift_score) / 20.0, 1.0)

        # Tail risk: max of above/below (already 0-1)
        tail_norm = max(signals.tail_risk_above, signals.tail_risk_below)

        score = (
            0.25 * mc_norm
            + 0.25 * pf_surprise_norm
            + 0.30 * drift_norm
            + 0.20 * tail_norm
        )
        return float(np.clip(score, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Signal container
# ---------------------------------------------------------------------------


@dataclass
class SimulationSignals:
    """All simulation-derived signals for a single tick of a single market.

    These augment the existing z-score signals in InfoLeakDetector.
    """

    market_id: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Phase 1: Monte Carlo
    mc_anomaly_score: float = 0.0
    mc_probability_above: float = 0.5
    mc_probability_below: float = 0.5
    mc_mean: float = 0.0
    mc_std: float = 0.0

    # Phase 1: Importance Sampling (tail risk)
    tail_risk_above: float = 0.0
    tail_risk_below: float = 0.0

    # Phase 2: Particle Filter
    pf_surprise: float = 0.0
    pf_drift_score: float = 0.0
    pf_ess_ratio: float = 1.0
    pf_regime: str = "normal"
    pf_posterior_mean: float = 0.0

    # Combined
    combined_sim_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "market_id": self.market_id,
            "timestamp": self.timestamp.isoformat(),
            "mc_anomaly_score": self.mc_anomaly_score,
            "mc_probability_above": self.mc_probability_above,
            "mc_probability_below": self.mc_probability_below,
            "mc_mean": self.mc_mean,
            "mc_std": self.mc_std,
            "tail_risk_above": self.tail_risk_above,
            "tail_risk_below": self.tail_risk_below,
            "pf_surprise": self.pf_surprise,
            "pf_drift_score": self.pf_drift_score,
            "pf_ess_ratio": self.pf_ess_ratio,
            "pf_regime": self.pf_regime,
            "pf_posterior_mean": self.pf_posterior_mean,
            "combined_sim_score": self.combined_sim_score,
        }

    def to_summary(self) -> str:
        """Human-readable one-line summary."""
        flags = []
        if self.mc_anomaly_score > 2.0:
            flags.append(f"MC:{self.mc_anomaly_score:.1f}σ")
        if self.pf_surprise > 3.0:
            flags.append(f"PF:surprise={self.pf_surprise:.1f}")
        if abs(self.pf_drift_score) > 5.0:
            flags.append(f"drift={self.pf_drift_score:+.1f}")
        if self.pf_regime == "high":
            flags.append("HIGH-VOL")
        if max(self.tail_risk_above, self.tail_risk_below) > 0.05:
            flags.append(f"tail={max(self.tail_risk_above, self.tail_risk_below):.1%}")

        if not flags:
            return f"{self.market_id}: NORMAL (score={self.combined_sim_score:.3f})"

        return (
            f"{self.market_id}: {'⚠️ ' if self.combined_sim_score > 0.3 else ''}"
            f"score={self.combined_sim_score:.3f} [{', '.join(flags)}]"
        )


# ---------------------------------------------------------------------------
# Standalone analysis function (for CLI)
# ---------------------------------------------------------------------------


def analyze_market(
    prices: np.ndarray,
    volumes: np.ndarray | None = None,
    orderbook_rows: list[dict[str, Any]] | None = None,
    trades: list[dict[str, Any]] | None = None,
    market_id: str = "analysis",
    mc_simulations: int = 10_000,
    n_particles: int = 2_000,
    seed: int = 42,
) -> dict[str, Any]:
    """Run full simulation analysis on a market's historical data.

    Standalone function for CLI and one-shot analysis — doesn't require
    a live orchestrator.

    Args:
        prices: Price time series.
        volumes: Per-tick volume series (optional).
        orderbook_rows: List of orderbook snapshot dicts (optional).
        trades: List of trade dicts (optional).
        market_id: Label for the analysis.
        mc_simulations: Number of MC simulations.
        n_particles: Particle filter particles.
        seed: Random seed.

    Returns:
        Dict with full analysis results.
    """
    t0 = time.monotonic()
    prices = np.asarray(prices)
    n = len(prices)

    detector = SimulationEnhancedDetector(
        mc_simulations=mc_simulations,
        n_particles=n_particles,
        seed=seed,
    )

    # Feed all prices through the detector
    signals_history: list[SimulationSignals] = []
    for i in range(n):
        vol = float(volumes[i]) if volumes is not None and i < len(volumes) else 0.0
        ob = orderbook_rows[i] if orderbook_rows and i < len(orderbook_rows) else None
        tr = trades if trades and i == n - 1 else None  # trades only on last tick

        sig = detector.process_tick(
            market_id=market_id,
            price=float(prices[i]),
            volume=vol,
            orderbook_row=ob,
            recent_trades=tr,
        )
        signals_history.append(sig)

    latest = signals_history[-1] if signals_history else None

    # MC simulation details
    state = detector.get_state(market_id)
    mc_result = state.last_mc_result if state else None
    pf_result = state.last_pf_result if state else None

    # ABM calibration + divergence (if enough data)
    calibration = None
    divergence = None
    if n >= 50:
        try:
            calibration = detector.calibrate_abm(market_id, max_iterations=20)
            divergence = detector.compute_abm_divergence(market_id, n_baseline_runs=5)
        except Exception as e:
            logger.warning("ABM analysis failed: %s", e)

    elapsed = (time.monotonic() - t0) * 1000

    result = {
        "market_id": market_id,
        "n_observations": n,
        "current_price": float(prices[-1]),
        "elapsed_ms": elapsed,
        "latest_signals": latest.to_dict() if latest else {},
        "signals_summary": latest.to_summary() if latest else "No data",
        "monte_carlo": mc_result.to_dict() if mc_result else None,
        "particle_filter": {
            "surprise": pf_result.surprise if pf_result else 0,
            "drift_score": pf_result.drift_score if pf_result else 0,
            "ess_ratio": pf_result.ess_ratio if pf_result else 1,
            "regime": pf_result.regime if pf_result else "unknown",
            "posterior_mean": pf_result.posterior_mean if pf_result else 0,
        } if pf_result else None,
        "abm_calibration": calibration.to_dict() if calibration else None,
        "abm_divergence": divergence.to_dict() if divergence else None,
        "combined_score": latest.combined_sim_score if latest else 0,
        "risk_level": _score_to_risk(latest.combined_sim_score if latest else 0),
    }

    return result


def generate_report(analysis: dict[str, Any]) -> str:
    """Generate a markdown report from analysis results."""
    lines = []
    mid = analysis["market_id"]
    score = analysis.get("combined_score", 0)
    risk = analysis.get("risk_level", "unknown")

    lines.append(f"# Simulation Analysis: {mid}")
    lines.append("")
    lines.append(f"**Risk Level:** {risk}")
    lines.append(f"**Combined Score:** {score:.4f}")
    lines.append(f"**Observations:** {analysis['n_observations']}")
    lines.append(f"**Current Price:** {analysis['current_price']:.4f}")
    lines.append(f"**Analysis Time:** {analysis['elapsed_ms']:.0f}ms")
    lines.append("")

    # Signals summary
    lines.append("## Signal Summary")
    lines.append(f"```\n{analysis.get('signals_summary', 'N/A')}\n```")
    lines.append("")

    # Monte Carlo
    mc = analysis.get("monte_carlo")
    if mc:
        lines.append("## Monte Carlo Analysis")
        lines.append(f"- **Anomaly Score:** {mc['anomaly_score']:.2f}σ")
        lines.append(f"- **Simulated Mean:** {mc['mean']:.4f}")
        lines.append(f"- **Simulated Std:** {mc['std']:.4f}")
        lines.append(f"- **P(above current):** {mc['probability_above']:.1%}")
        lines.append(f"- **P(below current):** {mc['probability_below']:.1%}")
        pcts = mc.get("percentiles", {})
        if pcts:
            lines.append(f"- **5th percentile:** {pcts.get('5%', 'N/A')}")
            lines.append(f"- **95th percentile:** {pcts.get('95%', 'N/A')}")
        lines.append("")

    # Particle filter
    pf = analysis.get("particle_filter")
    if pf:
        lines.append("## Particle Filter (Sequential Monte Carlo)")
        lines.append(f"- **Surprise:** {pf['surprise']:.3f}")
        lines.append(f"- **Drift Score:** {pf['drift_score']:.3f}")
        lines.append(f"- **ESS Ratio:** {pf['ess_ratio']:.1%}")
        lines.append(f"- **Regime:** {pf['regime']}")
        lines.append(f"- **Posterior Mean:** {pf['posterior_mean']:.4f}")
        lines.append("")

    # Tail risk
    signals = analysis.get("latest_signals", {})
    if signals.get("tail_risk_above", 0) > 0 or signals.get("tail_risk_below", 0) > 0:
        lines.append("## Tail Risk (Importance Sampling)")
        lines.append(f"- **P(+15% spike):** {signals.get('tail_risk_above', 0):.4%}")
        lines.append(f"- **P(-15% crash):** {signals.get('tail_risk_below', 0):.4%}")
        lines.append("")

    # ABM
    div = analysis.get("abm_divergence")
    if div:
        lines.append("## Agent-Based Model Divergence")
        lines.append(f"- **Composite Score:** {div['composite_score']:.4f}")
        lines.append(f"- **Price Distribution (KS):** stat={div['price_ks_statistic']:.3f}, p={div['price_ks_pvalue']:.4f}")
        lines.append(f"- **Volatility Ratio:** {div['volatility_ratio']:.2f}x")
        lines.append(f"- **Spread Ratio:** {div['spread_mean_ratio']:.2f}x")
        lines.append(f"- **Autocorrelation Gap:** {div['autocorrelation_divergence']:.4f}")
        lines.append("")

    cal = analysis.get("abm_calibration")
    if cal:
        lines.append("## ABM Calibration")
        lines.append(f"- **Fit Distance:** {cal['distance']:.4f}")
        lines.append(f"- **Evaluations:** {cal['n_evaluations']}")
        lines.append(f"- **Calibration Time:** {cal['elapsed_ms']:.0f}ms")
        lines.append("")

    return "\n".join(lines)


def _score_to_risk(score: float) -> str:
    if score >= 0.7:
        return "🔴 CRITICAL"
    if score >= 0.5:
        return "🟠 HIGH"
    if score >= 0.3:
        return "🟡 ELEVATED"
    if score >= 0.15:
        return "🔵 LOW"
    return "🟢 NORMAL"
