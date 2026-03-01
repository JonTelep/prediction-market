"""ABM simulator: orchestrates agents and market, produces baseline statistics.

Runs configurable multi-agent simulations to produce synthetic market data.
The key output is a statistical fingerprint of "normal" market behavior
(no insiders) that can be compared against real observations.

Two modes:
  1. **Baseline mode**: No informed traders — produces the "expected" market
  2. **Insider mode**: Adds informed traders — shows what insider activity looks like

The divergence between real data and baseline simulation is the detection signal.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import stats as sp_stats

from prediction_market.simulation.abm.agents import (
    InformedTrader,
    MarketMaker,
    MomentumTrader,
    NoiseTrader,
    TraderAgent,
)
from prediction_market.simulation.abm.market import SimulatedMarket, MarketState

logger = logging.getLogger(__name__)


@dataclass
class ABMConfig:
    """Configuration for an agent-based simulation run.

    Attributes:
        n_ticks: Number of simulation ticks.
        initial_price: Starting market price.
        base_liquidity: Market liquidity depth.
        n_noise: Number of noise traders.
        n_market_makers: Number of market makers.
        n_momentum: Number of momentum traders.
        n_informed: Number of informed traders (0 for baseline).
        informed_true_value: Private signal for informed traders.
        informed_stealth: How stealthy informed traders are (0-1).
        noise_trade_prob: Per-tick trade probability for noise traders.
        noise_mean_size: Average noise trader order size.
        mm_spread: Market maker half-spread.
        mm_max_position: Market maker position limit.
        momentum_lookback: Momentum signal lookback.
        momentum_threshold: Minimum momentum to trigger trade.
        seed: Random seed for reproducibility.
    """

    n_ticks: int = 500
    initial_price: float = 0.5
    base_liquidity: float = 10_000.0
    n_noise: int = 20
    n_market_makers: int = 3
    n_momentum: int = 5
    n_informed: int = 0
    informed_true_value: float = 0.7
    informed_stealth: float = 0.5
    noise_trade_prob: float = 0.3
    noise_mean_size: float = 100.0
    mm_spread: float = 0.02
    mm_max_position: float = 1000.0
    momentum_lookback: int = 10
    momentum_threshold: float = 0.005
    seed: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


@dataclass(frozen=True)
class DivergenceMetrics:
    """Statistical divergence between simulated and observed market data.

    Higher values indicate the real market is behaving differently from
    the simulated baseline — potential signal of informed participation.

    Attributes:
        price_ks_statistic: KS test statistic on price return distributions.
        price_ks_pvalue: KS test p-value (low = different distributions).
        volume_ks_statistic: KS test on volume distributions.
        volume_ks_pvalue: KS test p-value for volume.
        imbalance_ks_statistic: KS test on order flow imbalance.
        imbalance_ks_pvalue: KS p-value for imbalance.
        spread_mean_ratio: Ratio of observed to simulated mean spread.
        volatility_ratio: Ratio of observed to simulated volatility.
        autocorrelation_divergence: Difference in return autocorrelation
            (informed trading creates predictable patterns).
        composite_score: Weighted combination of all metrics (0-1).
            Higher = more likely informed participation.
    """

    price_ks_statistic: float = 0.0
    price_ks_pvalue: float = 1.0
    volume_ks_statistic: float = 0.0
    volume_ks_pvalue: float = 1.0
    imbalance_ks_statistic: float = 0.0
    imbalance_ks_pvalue: float = 1.0
    spread_mean_ratio: float = 1.0
    volatility_ratio: float = 1.0
    autocorrelation_divergence: float = 0.0
    composite_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "price_ks_statistic": self.price_ks_statistic,
            "price_ks_pvalue": self.price_ks_pvalue,
            "volume_ks_statistic": self.volume_ks_statistic,
            "volume_ks_pvalue": self.volume_ks_pvalue,
            "imbalance_ks_statistic": self.imbalance_ks_statistic,
            "imbalance_ks_pvalue": self.imbalance_ks_pvalue,
            "spread_mean_ratio": self.spread_mean_ratio,
            "volatility_ratio": self.volatility_ratio,
            "autocorrelation_divergence": self.autocorrelation_divergence,
            "composite_score": self.composite_score,
        }


@dataclass(frozen=True)
class ABMResult:
    """Result of a single ABM simulation run.

    Attributes:
        config: The configuration used.
        price_series: Simulated price trajectory.
        volume_series: Per-tick volume.
        spread_series: Per-tick spread.
        imbalance_series: Per-tick order flow imbalance.
        final_price: Price at end of simulation.
        total_volume: Total volume traded.
        realized_volatility: Realized volatility of the simulation.
        mean_spread: Average spread across all ticks.
        price_return_std: Std dev of log-returns.
        return_autocorrelation: Lag-1 autocorrelation of returns.
        trade_count: Total number of trades.
        trades_by_type: Dict mapping agent_type → trade count.
        volume_by_type: Dict mapping agent_type → total volume.
        elapsed_ms: Simulation wall-clock time.
    """

    config: ABMConfig
    price_series: list[float]
    volume_series: list[float]
    spread_series: list[float]
    imbalance_series: list[float]
    final_price: float
    total_volume: float
    realized_volatility: float
    mean_spread: float
    price_return_std: float
    return_autocorrelation: float
    trade_count: int
    trades_by_type: dict[str, int]
    volume_by_type: dict[str, float]
    elapsed_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "final_price": self.final_price,
            "total_volume": self.total_volume,
            "realized_volatility": self.realized_volatility,
            "mean_spread": self.mean_spread,
            "price_return_std": self.price_return_std,
            "return_autocorrelation": self.return_autocorrelation,
            "trade_count": self.trade_count,
            "trades_by_type": self.trades_by_type,
            "volume_by_type": self.volume_by_type,
            "elapsed_ms": self.elapsed_ms,
        }


class ABMSimulator:
    """Agent-Based Market Simulator.

    Orchestrates agent creation, market initialization, and simulation
    execution. Provides methods for running baseline (no-insider) and
    comparison (with-insider) simulations, and computing divergence
    metrics against observed data.

    Args:
        config: Simulation configuration.
    """

    def __init__(self, config: ABMConfig | None = None) -> None:
        self._config = config or ABMConfig()

    def run(self, config: ABMConfig | None = None) -> ABMResult:
        """Execute a single simulation run.

        Args:
            config: Override configuration. Uses default if None.

        Returns:
            ABMResult with full simulation output.
        """
        cfg = config or self._config
        rng = np.random.default_rng(cfg.seed)
        t0 = time.monotonic()

        # Create market
        market = SimulatedMarket(
            initial_price=cfg.initial_price,
            base_liquidity=cfg.base_liquidity,
            base_spread=cfg.mm_spread,
        )

        # Create agents
        agents = self._create_agents(cfg, rng)

        # Run simulation
        trade_counts: dict[str, int] = {}
        volume_by_type: dict[str, float] = {}

        for tick in range(cfg.n_ticks):
            state = market.state

            # Shuffle agent order each tick
            order = rng.permutation(len(agents))

            for idx in order:
                agent = agents[idx]
                decision = agent.decide(state)
                if decision is not None:
                    trade = market.process_order(decision)
                    if trade is not None:
                        agent.update_position(trade.side, trade.size, trade.price)
                        trade_counts[trade.agent_type] = (
                            trade_counts.get(trade.agent_type, 0) + 1
                        )
                        volume_by_type[trade.agent_type] = (
                            volume_by_type.get(trade.agent_type, 0) + trade.size
                        )
                        # Refresh state after each trade for subsequent agents
                        state = market.state

            market.end_tick()

        elapsed = (time.monotonic() - t0) * 1000

        # Compute statistics
        prices = market.get_price_series()
        volumes = market.get_volume_series()
        spreads = market.get_spread_series()
        imbalances = market.get_imbalance_series()

        log_returns = np.diff(np.log(np.clip(prices, 1e-6, 1 - 1e-6)))
        vol = float(np.std(log_returns)) if len(log_returns) > 1 else 0.0
        ret_std = vol
        autocorr = self._autocorrelation(log_returns)

        return ABMResult(
            config=cfg,
            price_series=prices.tolist(),
            volume_series=volumes.tolist() if len(volumes) > 0 else [],
            spread_series=spreads.tolist() if len(spreads) > 0 else [],
            imbalance_series=imbalances.tolist() if len(imbalances) > 0 else [],
            final_price=float(prices[-1]),
            total_volume=float(np.sum(volumes)) if len(volumes) > 0 else 0.0,
            realized_volatility=vol,
            mean_spread=float(np.mean(spreads)) if len(spreads) > 0 else 0.0,
            price_return_std=ret_std,
            return_autocorrelation=autocorr,
            trade_count=sum(trade_counts.values()),
            trades_by_type=trade_counts,
            volume_by_type=volume_by_type,
            elapsed_ms=elapsed,
        )

    def run_baseline(self, n_runs: int = 10) -> list[ABMResult]:
        """Run multiple no-insider simulations for a robust baseline.

        Creates slightly different seeds for each run to capture
        natural simulation variance.

        Args:
            n_runs: Number of independent baseline runs.

        Returns:
            List of ABMResult instances.
        """
        results = []
        base_seed = self._config.seed or 42
        for i in range(n_runs):
            cfg = ABMConfig(
                **{
                    k: v
                    for k, v in self._config.__dict__.items()
                    if k != "seed"
                },
                seed=base_seed + i,
            )
            cfg = ABMConfig(**{**cfg.__dict__, "n_informed": 0})
            results.append(self.run(cfg))
        return results

    def compare_with_observed(
        self,
        baseline_results: list[ABMResult],
        observed_prices: np.ndarray,
        observed_volumes: np.ndarray | None = None,
        observed_imbalances: np.ndarray | None = None,
        observed_spreads: np.ndarray | None = None,
    ) -> DivergenceMetrics:
        """Compare observed market data against simulated baseline.

        Computes distribution-level divergence metrics between the real
        market and the ensemble of baseline simulations.

        Args:
            baseline_results: List of baseline (no-insider) simulation results.
            observed_prices: Real price series.
            observed_volumes: Real volume series (optional).
            observed_imbalances: Real imbalance series (optional).
            observed_spreads: Real spread series (optional).

        Returns:
            DivergenceMetrics quantifying how different the real market is
            from the simulated baseline.
        """
        observed_prices = np.asarray(observed_prices)

        # Pool simulated returns from all baseline runs
        sim_returns = []
        sim_volumes = []
        sim_imbalances = []
        sim_spreads_flat = []
        sim_vols = []
        sim_autocorrs = []

        for r in baseline_results:
            p = np.array(r.price_series)
            p = np.clip(p, 1e-6, 1 - 1e-6)
            lr = np.diff(np.log(p))
            sim_returns.extend(lr.tolist())
            sim_volumes.extend(r.volume_series)
            sim_imbalances.extend(r.imbalance_series)
            sim_spreads_flat.extend(r.spread_series)
            sim_vols.append(r.realized_volatility)
            sim_autocorrs.append(r.return_autocorrelation)

        sim_returns = np.array(sim_returns)
        obs_returns = np.diff(np.log(np.clip(observed_prices, 1e-6, 1 - 1e-6)))

        # --- Price return distribution (KS test) ---
        if len(obs_returns) >= 3 and len(sim_returns) >= 3:
            price_ks, price_p = sp_stats.ks_2samp(obs_returns, sim_returns)
        else:
            price_ks, price_p = 0.0, 1.0

        # --- Volume distribution ---
        vol_ks, vol_p = 0.0, 1.0
        if observed_volumes is not None and len(sim_volumes) >= 3:
            obs_v = np.asarray(observed_volumes)
            if len(obs_v) >= 3:
                vol_ks, vol_p = sp_stats.ks_2samp(obs_v, np.array(sim_volumes))

        # --- Imbalance distribution ---
        imb_ks, imb_p = 0.0, 1.0
        if observed_imbalances is not None and len(sim_imbalances) >= 3:
            obs_i = np.asarray(observed_imbalances)
            if len(obs_i) >= 3:
                imb_ks, imb_p = sp_stats.ks_2samp(obs_i, np.array(sim_imbalances))

        # --- Spread ratio ---
        spread_ratio = 1.0
        if observed_spreads is not None and len(sim_spreads_flat) > 0:
            obs_spread_mean = float(np.mean(observed_spreads))
            sim_spread_mean = float(np.mean(sim_spreads_flat))
            if sim_spread_mean > 1e-10:
                spread_ratio = obs_spread_mean / sim_spread_mean

        # --- Volatility ratio ---
        vol_ratio = 1.0
        if sim_vols:
            obs_vol = float(np.std(obs_returns)) if len(obs_returns) > 1 else 0.0
            sim_vol_mean = float(np.mean(sim_vols))
            if sim_vol_mean > 1e-10:
                vol_ratio = obs_vol / sim_vol_mean

        # --- Autocorrelation divergence ---
        obs_autocorr = self._autocorrelation(obs_returns)
        sim_autocorr_mean = float(np.mean(sim_autocorrs)) if sim_autocorrs else 0.0
        autocorr_div = abs(obs_autocorr - sim_autocorr_mean)

        # --- Composite score ---
        # Weight components into a 0-1 score
        # Low p-values and high ratios both contribute
        composite = self._compute_composite(
            price_p, vol_p, imb_p, spread_ratio, vol_ratio, autocorr_div
        )

        return DivergenceMetrics(
            price_ks_statistic=float(price_ks),
            price_ks_pvalue=float(price_p),
            volume_ks_statistic=float(vol_ks),
            volume_ks_pvalue=float(vol_p),
            imbalance_ks_statistic=float(imb_ks),
            imbalance_ks_pvalue=float(imb_p),
            spread_mean_ratio=spread_ratio,
            volatility_ratio=vol_ratio,
            autocorrelation_divergence=autocorr_div,
            composite_score=composite,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _create_agents(
        self, cfg: ABMConfig, rng: np.random.Generator
    ) -> list[TraderAgent]:
        """Instantiate all trader agents from config."""
        agents: list[TraderAgent] = []
        agent_id = 0

        for _ in range(cfg.n_noise):
            agents.append(
                NoiseTrader(
                    agent_id=agent_id,
                    rng=rng,
                    trade_probability=cfg.noise_trade_prob,
                    mean_size=cfg.noise_mean_size,
                )
            )
            agent_id += 1

        for _ in range(cfg.n_market_makers):
            agents.append(
                MarketMaker(
                    agent_id=agent_id,
                    rng=rng,
                    fair_value=cfg.initial_price,
                    spread=cfg.mm_spread,
                    max_position=cfg.mm_max_position,
                )
            )
            agent_id += 1

        for _ in range(cfg.n_momentum):
            agents.append(
                MomentumTrader(
                    agent_id=agent_id,
                    rng=rng,
                    lookback=cfg.momentum_lookback,
                    threshold=cfg.momentum_threshold,
                )
            )
            agent_id += 1

        for _ in range(cfg.n_informed):
            agents.append(
                InformedTrader(
                    agent_id=agent_id,
                    rng=rng,
                    true_value=cfg.informed_true_value,
                    stealth=cfg.informed_stealth,
                )
            )
            agent_id += 1

        return agents

    @staticmethod
    def _autocorrelation(returns: np.ndarray, lag: int = 1) -> float:
        """Compute lag-1 autocorrelation of a return series."""
        if len(returns) < lag + 2:
            return 0.0
        r = np.array(returns)
        mean = np.mean(r)
        var = np.var(r)
        if var < 1e-15:
            return 0.0
        autocov = np.mean((r[lag:] - mean) * (r[:-lag] - mean))
        return float(autocov / var)

    @staticmethod
    def _compute_composite(
        price_p: float,
        vol_p: float,
        imb_p: float,
        spread_ratio: float,
        vol_ratio: float,
        autocorr_div: float,
    ) -> float:
        """Compute a 0-1 composite divergence score.

        Uses a weighted combination where:
        - Low p-values → high score (distributions differ)
        - Ratio far from 1 → high score (structural difference)
        - High autocorrelation divergence → high score
        """
        # Convert p-values to signal strength: -log10(p), capped
        def p_to_signal(p: float) -> float:
            p = max(p, 1e-10)
            return min(-np.log10(p) / 5.0, 1.0)  # -log10(1e-5)=5 → score 1.0

        price_signal = p_to_signal(price_p)
        vol_signal = p_to_signal(vol_p)
        imb_signal = p_to_signal(imb_p)

        # Ratio divergence: |log(ratio)| normalized
        spread_signal = min(abs(np.log(max(spread_ratio, 0.01))) / 2.0, 1.0)
        vol_r_signal = min(abs(np.log(max(vol_ratio, 0.01))) / 2.0, 1.0)

        # Autocorrelation: directly as signal (capped at 1)
        autocorr_signal = min(autocorr_div * 5.0, 1.0)

        # Weighted average
        composite = (
            0.30 * price_signal
            + 0.20 * vol_signal
            + 0.15 * imb_signal
            + 0.10 * spread_signal
            + 0.15 * vol_r_signal
            + 0.10 * autocorr_signal
        )

        return float(np.clip(composite, 0.0, 1.0))

    def __repr__(self) -> str:
        return f"ABMSimulator(config={self._config})"
