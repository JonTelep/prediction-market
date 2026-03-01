"""Sequential Monte Carlo (particle filter) for real-time market state estimation.

Maintains a cloud of weighted particles, each representing a hypothesis about
the market's latent state (true price + volatility regime). On each new
observation, particles are propagated via a market-aware transition model,
reweighted against the observation likelihood, and resampled when effective
sample size drops too low.

The key output is **surprise** — how unexpected the observation is under the
current belief. Sustained surprise accumulates into a **drift score** that
catches slow information leakage invisible to single-tick z-scores.

Market-aware transition model (Option 2 — aggressive):
  - Order book imbalance biases the drift direction
  - Recent trade flow momentum adjusts expected volatility
  - Depth thinning widens the transition variance (thin books → more volatile)
  - Spread widening signals regime uncertainty
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Market context fed into the transition model
# ---------------------------------------------------------------------------


@dataclass
class MarketContext:
    """Observable market microstructure features that inform state transitions.

    All fields are optional — the filter degrades gracefully to a random walk
    when context is unavailable.

    Attributes:
        orderbook_imbalance: (bids - asks) / (bids + asks), in [-1, 1].
            Positive = buy pressure, negative = sell pressure.
        spread_pct: Bid-ask spread as fraction of midpoint.
        depth_5pct: Total depth within 5% of midpoint (USD).
        total_bid_depth: Total bid-side depth (USD).
        total_ask_depth: Total ask-side depth (USD).
        trade_flow_imbalance: Net buy volume - net sell volume over recent
            window, normalized to [-1, 1].
        recent_volatility: Realized volatility from recent returns.
        susceptibility_score: Composite manipulation susceptibility (0-1).
    """

    orderbook_imbalance: float = 0.0
    spread_pct: float = 0.0
    depth_5pct: float = 0.0
    total_bid_depth: float = 0.0
    total_ask_depth: float = 0.0
    trade_flow_imbalance: float = 0.0
    recent_volatility: float = 0.0
    susceptibility_score: float = 0.0

    @classmethod
    def from_orderbook_row(cls, row: dict[str, Any]) -> MarketContext:
        """Build context from a database orderbook_snapshots row."""
        return cls(
            orderbook_imbalance=row.get("imbalance", 0.0) or 0.0,
            spread_pct=row.get("spread_pct", 0.0) or 0.0,
            depth_5pct=row.get("depth_5pct", 0.0) or 0.0,
            total_bid_depth=row.get("total_bid_depth", 0.0) or 0.0,
            total_ask_depth=row.get("total_ask_depth", 0.0) or 0.0,
            susceptibility_score=row.get("susceptibility_score", 0.0) or 0.0,
        )

    def with_trade_flow(self, imbalance: float, volatility: float) -> MarketContext:
        """Return a copy with trade flow fields populated."""
        return MarketContext(
            orderbook_imbalance=self.orderbook_imbalance,
            spread_pct=self.spread_pct,
            depth_5pct=self.depth_5pct,
            total_bid_depth=self.total_bid_depth,
            total_ask_depth=self.total_ask_depth,
            trade_flow_imbalance=imbalance,
            recent_volatility=volatility,
            susceptibility_score=self.susceptibility_score,
        )


# ---------------------------------------------------------------------------
# Particle filter result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParticleFilterResult:
    """Output of a single particle filter update step.

    Attributes:
        market_id: Market identifier.
        surprise: Log-likelihood surprise of the observation. Higher = more
            unexpected under current belief. Normalized to be comparable
            across markets (in units of "nats").
        drift_score: Cumulative directional surprise over a sliding window.
            Catches slow, sustained leakage that individual ticks miss.
        ess: Effective sample size after reweighting (before resampling).
            Low ESS = observation was very unexpected.
        ess_ratio: ESS / n_particles — fraction of "useful" particles.
        posterior_mean: Weighted mean of particle positions after update.
        posterior_std: Weighted std of particle positions after update.
        observed_price: The actual observation that was processed.
        regime: Detected volatility regime: "low", "normal", or "high".
        transition_drift: The drift applied by the transition model
            (from market context).
        transition_vol: The volatility used in the transition model.
        resampled: Whether systematic resampling was triggered this step.
        timestamp: When this update was performed.
    """

    market_id: str
    surprise: float
    drift_score: float
    ess: float
    ess_ratio: float
    posterior_mean: float
    posterior_std: float
    observed_price: float
    regime: str = "normal"
    transition_drift: float = 0.0
    transition_vol: float = 0.0
    resampled: bool = False
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        return {
            "market_id": self.market_id,
            "surprise": self.surprise,
            "drift_score": self.drift_score,
            "ess": self.ess,
            "ess_ratio": self.ess_ratio,
            "posterior_mean": self.posterior_mean,
            "posterior_std": self.posterior_std,
            "observed_price": self.observed_price,
            "regime": self.regime,
            "transition_drift": self.transition_drift,
            "transition_vol": self.transition_vol,
            "resampled": self.resampled,
            "timestamp": self.timestamp.isoformat(),
        }


# ---------------------------------------------------------------------------
# Transition model
# ---------------------------------------------------------------------------


class MarketAwareTransition:
    """Market-microstructure-informed state transition model.

    Converts order book imbalance, trade flow, depth, and spread into
    drift and volatility parameters for particle propagation.

    The transition operates in logit space to respect [0,1] bounds naturally:
        logit(p) = log(p / (1-p))
        p = sigmoid(logit_p)

    Args:
        base_volatility: Baseline per-tick volatility in logit space.
        imbalance_sensitivity: How strongly order book imbalance affects drift.
        trade_flow_sensitivity: How strongly net trade flow affects drift.
        depth_vol_scale: How much thin depth amplifies volatility.
        spread_vol_scale: How much wide spreads amplify volatility.
    """

    def __init__(
        self,
        base_volatility: float = 0.02,
        imbalance_sensitivity: float = 0.3,
        trade_flow_sensitivity: float = 0.2,
        depth_vol_scale: float = 1.5,
        spread_vol_scale: float = 1.0,
    ) -> None:
        self.base_volatility = base_volatility
        self.imbalance_sensitivity = imbalance_sensitivity
        self.trade_flow_sensitivity = trade_flow_sensitivity
        self.depth_vol_scale = depth_vol_scale
        self.spread_vol_scale = spread_vol_scale

    def compute_drift_and_vol(
        self, ctx: MarketContext | None
    ) -> tuple[float, float]:
        """Compute the transition drift (in logit space) and volatility.

        Args:
            ctx: Current market microstructure context. None = random walk.

        Returns:
            Tuple of (drift, volatility) in logit space.
        """
        if ctx is None:
            return 0.0, self.base_volatility

        # --- Drift from order flow ---
        # Positive imbalance (more bids) → positive drift (price up)
        ob_drift = self.imbalance_sensitivity * ctx.orderbook_imbalance
        tf_drift = self.trade_flow_sensitivity * ctx.trade_flow_imbalance
        drift = ob_drift + tf_drift

        # --- Volatility from market conditions ---
        vol = self.base_volatility

        # Thin depth → higher volatility
        if ctx.depth_5pct > 0:
            # Normalize: $10k depth = normal, lower = amplified
            depth_factor = min(10_000.0 / max(ctx.depth_5pct, 100.0), self.depth_vol_scale)
            vol *= depth_factor

        # Wide spread → higher volatility
        if ctx.spread_pct > 0:
            # Normalize: 1% spread = normal, wider = amplified
            spread_factor = min(1.0 + ctx.spread_pct * self.spread_vol_scale * 10, 2.0)
            vol *= spread_factor

        # High susceptibility → additional volatility boost
        if ctx.susceptibility_score > 0.5:
            vol *= 1.0 + (ctx.susceptibility_score - 0.5)

        # Use recent realized volatility as a floor if available
        if ctx.recent_volatility > vol:
            vol = 0.7 * ctx.recent_volatility + 0.3 * vol

        return drift, vol

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_volatility": self.base_volatility,
            "imbalance_sensitivity": self.imbalance_sensitivity,
            "trade_flow_sensitivity": self.trade_flow_sensitivity,
            "depth_vol_scale": self.depth_vol_scale,
            "spread_vol_scale": self.spread_vol_scale,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MarketAwareTransition:
        return cls(**{k: data[k] for k in data if k in cls.__init__.__code__.co_varnames})


# ---------------------------------------------------------------------------
# Core particle filter
# ---------------------------------------------------------------------------


class ParticleFilter:
    """Sequential Monte Carlo particle filter for a single market.

    Each particle represents a hypothesis about the market's latent "true"
    price. The filter maintains these in logit space for numerical stability
    and natural [0,1] bounding.

    Lifecycle per tick:
        1. **Predict**: propagate particles via MarketAwareTransition
        2. **Update**: reweight by observation likelihood
        3. **Measure surprise**: quantify how unexpected the observation was
        4. **Resample**: if ESS drops below threshold, systematic resample
        5. **Accumulate drift**: track cumulative directional surprise

    Args:
        market_id: Market identifier.
        n_particles: Number of particles (more = more accurate, slower).
        observation_noise: Std dev of the observation model (how noisy are
            price observations relative to "true" value).
        ess_threshold: Fraction of n_particles below which resampling triggers.
        drift_window: Number of ticks over which to accumulate drift score.
        transition: Transition model. Uses default if None.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        market_id: str,
        n_particles: int = 2000,
        observation_noise: float = 0.01,
        ess_threshold: float = 0.5,
        drift_window: int = 60,
        transition: MarketAwareTransition | None = None,
        seed: int | None = None,
    ) -> None:
        self.market_id = market_id
        self._n = n_particles
        self._obs_noise = observation_noise
        self._ess_threshold = ess_threshold
        self._drift_window = drift_window
        self._transition = transition or MarketAwareTransition()
        self._rng = np.random.default_rng(seed)

        # Particle state: positions in logit space, uniform weights
        self._particles: np.ndarray | None = None  # shape (n,)
        self._weights: np.ndarray = np.ones(n_particles) / n_particles
        self._initialized = False

        # Drift accumulator: ring buffer of signed surprises
        self._surprise_history: list[float] = []
        # Observation history for realized volatility / regime detection
        self._price_history: list[float] = []
        self._regime = "normal"

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def initialize(self, price: float, spread: float = 0.05) -> None:
        """Initialize particles around an observed price.

        Particles are spread in logit space centered on the observation,
        with width proportional to *spread*.

        Args:
            price: Initial observed price in (0, 1).
            spread: Initial spread of particle cloud in probability space.
        """
        price = np.clip(price, 0.01, 0.99)
        logit_center = _logit(price)
        # Spread in logit space (wider near boundaries due to logit transform)
        logit_spread = spread / (price * (1 - price))
        self._particles = self._rng.normal(
            logit_center, logit_spread, size=self._n
        )
        self._weights = np.ones(self._n) / self._n
        self._initialized = True
        logger.debug(
            "Initialized particle filter for %s: center=%.4f, %d particles",
            self.market_id, price, self._n,
        )

    # ------------------------------------------------------------------
    # Core update
    # ------------------------------------------------------------------

    def update(
        self,
        observed_price: float,
        timestamp: datetime | None = None,
        context: MarketContext | None = None,
    ) -> ParticleFilterResult:
        """Process one new price observation.

        This is the main entry point called on each orchestrator tick.

        Args:
            observed_price: Latest market price in (0, 1).
            timestamp: Observation time.
            context: Market microstructure context for transition model.

        Returns:
            ParticleFilterResult with surprise, drift score, and diagnostics.
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        observed_price = np.clip(observed_price, 0.01, 0.99)

        if not self._initialized:
            self.initialize(observed_price)
            return ParticleFilterResult(
                market_id=self.market_id,
                surprise=0.0,
                drift_score=0.0,
                ess=float(self._n),
                ess_ratio=1.0,
                posterior_mean=observed_price,
                posterior_std=0.0,
                observed_price=observed_price,
                timestamp=timestamp,
            )

        # --- 1. PREDICT: propagate particles ---
        drift, vol = self._transition.compute_drift_and_vol(context)
        noise = self._rng.standard_normal(self._n)
        self._particles = self._particles + drift + vol * noise

        # --- 2. UPDATE: compute observation likelihood & reweight ---
        # Observation model: P(obs | particle) ~ Normal(sigmoid(particle), obs_noise)
        # Scale observation noise by transition volatility so particles and
        # observation model stay in proportion.
        effective_noise = max(self._obs_noise, vol * 0.5)
        particle_prices = _sigmoid_array(self._particles)
        log_likelihood = -0.5 * ((observed_price - particle_prices) / effective_noise) ** 2

        # Stabilize
        log_likelihood -= np.max(log_likelihood)
        likelihood = np.exp(log_likelihood)

        # Update weights
        new_weights = self._weights * likelihood
        weight_sum = np.sum(new_weights)

        if weight_sum < 1e-30:
            # Complete particle depletion — extreme surprise
            logger.warning(
                "Particle depletion for %s at price %.4f — reinitializing",
                self.market_id, observed_price,
            )
            self.initialize(observed_price)
            surprise = 10.0  # Maximum surprise
        else:
            self._weights = new_weights / weight_sum

            # --- 3. MEASURE SURPRISE ---
            # Surprise = negative log marginal likelihood (higher = more unexpected)
            # Approximated as -log(mean likelihood under prior weights)
            surprise = -math.log(max(weight_sum / self._n, 1e-30))

        # --- 4. EFFECTIVE SAMPLE SIZE ---
        ess = 1.0 / float(np.sum(self._weights ** 2))
        ess_ratio = ess / self._n

        # --- 5. RESAMPLE if ESS too low ---
        resampled = False
        if ess_ratio < self._ess_threshold:
            self._systematic_resample()
            resampled = True

        # --- 6. POSTERIOR STATISTICS ---
        posterior_prices = _sigmoid_array(self._particles)
        posterior_mean = float(np.average(posterior_prices, weights=self._weights))
        posterior_std = float(
            np.sqrt(np.average((posterior_prices - posterior_mean) ** 2, weights=self._weights))
        )

        # --- 7. SIGNED SURPRISE & DRIFT ACCUMULATION ---
        # Direction: positive if price moved above posterior expectation
        sign = 1.0 if observed_price > posterior_mean else -1.0
        signed_surprise = sign * surprise
        self._surprise_history.append(signed_surprise)

        # Trim to window
        if len(self._surprise_history) > self._drift_window:
            self._surprise_history = self._surprise_history[-self._drift_window:]

        # Drift score: sum of signed surprises (catches sustained directional pressure)
        drift_score = sum(self._surprise_history)

        # --- 8. REGIME DETECTION (from realized observation volatility) ---
        self._price_history.append(observed_price)
        if len(self._price_history) > self._drift_window:
            self._price_history = self._price_history[-self._drift_window:]
        self._regime = self._detect_regime_from_history(self._price_history)

        result = ParticleFilterResult(
            market_id=self.market_id,
            surprise=surprise,
            drift_score=drift_score,
            ess=ess,
            ess_ratio=ess_ratio,
            posterior_mean=posterior_mean,
            posterior_std=posterior_std,
            observed_price=observed_price,
            regime=self._regime,
            transition_drift=drift,
            transition_vol=vol,
            resampled=resampled,
            timestamp=timestamp,
        )

        logger.debug(
            "PF update %s: surprise=%.3f drift=%.3f ess=%.0f regime=%s",
            self.market_id, surprise, drift_score, ess, self._regime,
        )
        return result

    # ------------------------------------------------------------------
    # Resampling
    # ------------------------------------------------------------------

    def _systematic_resample(self) -> None:
        """Systematic resampling — low variance, O(n)."""
        n = self._n
        positions = (self._rng.random() + np.arange(n)) / n
        cumsum = np.cumsum(self._weights)
        indices = np.searchsorted(cumsum, positions)
        indices = np.clip(indices, 0, n - 1)

        self._particles = self._particles[indices].copy()
        self._weights = np.ones(n) / n

    # ------------------------------------------------------------------
    # Regime detection
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_regime_from_history(prices: list[float]) -> str:
        """Classify volatility regime from realized observation volatility.

        Uses the standard deviation of recent log-returns. Thresholds are
        calibrated for ~1-minute tick intervals on Polymarket binary markets.
        """
        if len(prices) < 5:
            return "normal"
        arr = np.array(prices)
        arr = np.clip(arr, 1e-6, 1.0 - 1e-6)
        log_returns = np.diff(np.log(arr))
        realized_vol = float(np.std(log_returns))
        if realized_vol < 0.005:
            return "low"
        elif realized_vol > 0.03:
            return "high"
        return "normal"

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize filter state for crash resilience."""
        return {
            "market_id": self.market_id,
            "n_particles": self._n,
            "observation_noise": self._obs_noise,
            "ess_threshold": self._ess_threshold,
            "drift_window": self._drift_window,
            "transition": self._transition.to_dict(),
            "initialized": self._initialized,
            "particles": self._particles.tolist() if self._particles is not None else None,
            "weights": self._weights.tolist(),
            "surprise_history": self._surprise_history,
            "price_history": self._price_history,
            "regime": self._regime,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], seed: int | None = None) -> ParticleFilter:
        """Reconstruct from serialized state."""
        transition = MarketAwareTransition.from_dict(data.get("transition", {}))
        pf = cls(
            market_id=data["market_id"],
            n_particles=data.get("n_particles", 2000),
            observation_noise=data.get("observation_noise", 0.01),
            ess_threshold=data.get("ess_threshold", 0.5),
            drift_window=data.get("drift_window", 60),
            transition=transition,
            seed=seed,
        )
        if data.get("initialized") and data.get("particles") is not None:
            pf._particles = np.array(data["particles"])
            pf._weights = np.array(data["weights"])
            pf._initialized = True
            pf._surprise_history = data.get("surprise_history", [])
            pf._price_history = data.get("price_history", [])
            pf._regime = data.get("regime", "normal")
        return pf

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    def __repr__(self) -> str:
        return (
            f"ParticleFilter(market={self.market_id}, n={self._n}, "
            f"initialized={self._initialized}, regime={self._regime})"
        )


# ---------------------------------------------------------------------------
# Trade flow analyzer (computes MarketContext.trade_flow_imbalance)
# ---------------------------------------------------------------------------


class TradeFlowAnalyzer:
    """Computes trade flow imbalance and realized volatility from trade history.

    Designed to be called with recent trades from the database to populate
    the ``trade_flow_imbalance`` and ``recent_volatility`` fields of
    :class:`MarketContext`.

    Args:
        lookback_minutes: How far back to consider trades for flow calculation.
    """

    def __init__(self, lookback_minutes: int = 30) -> None:
        self._lookback = timedelta(minutes=lookback_minutes)

    def compute(
        self,
        trades: list[dict[str, Any]],
        reference_time: datetime | None = None,
    ) -> tuple[float, float]:
        """Compute trade flow imbalance and realized volatility.

        Args:
            trades: List of trade dicts from the database (must have 'side',
                'volume_usd', 'price', 'match_time' keys).
            reference_time: Current time for lookback window. Defaults to
                utcnow.

        Returns:
            Tuple of (flow_imbalance in [-1,1], realized_volatility).
        """
        if not trades:
            return 0.0, 0.0

        if reference_time is None:
            reference_time = datetime.now(timezone.utc)

        cutoff = reference_time - self._lookback

        buy_volume = 0.0
        sell_volume = 0.0
        prices: list[float] = []

        for t in trades:
            # Parse match_time
            mt = t.get("match_time", "")
            if isinstance(mt, str) and mt:
                try:
                    trade_time = datetime.fromisoformat(mt.replace("Z", "+00:00"))
                    if trade_time.tzinfo is None:
                        trade_time = trade_time.replace(tzinfo=timezone.utc)
                except (ValueError, TypeError):
                    continue
            elif isinstance(mt, datetime):
                trade_time = mt
            else:
                continue

            if trade_time < cutoff:
                continue

            vol = float(t.get("volume_usd", 0) or 0)
            side = t.get("side", "").lower()
            price = float(t.get("price", 0) or 0)

            if side == "buy":
                buy_volume += vol
            elif side == "sell":
                sell_volume += vol

            if price > 0:
                prices.append(price)

        # Flow imbalance: normalized to [-1, 1]
        total = buy_volume + sell_volume
        if total > 0:
            flow_imbalance = (buy_volume - sell_volume) / total
        else:
            flow_imbalance = 0.0

        # Realized volatility from trade prices
        realized_vol = 0.0
        if len(prices) >= 3:
            log_returns = np.diff(np.log(np.clip(prices, 1e-6, None)))
            realized_vol = float(np.std(log_returns))

        return flow_imbalance, realized_vol

    def __repr__(self) -> str:
        return f"TradeFlowAnalyzer(lookback={self._lookback})"


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def _logit(p: float) -> float:
    """Logit transform: log(p / (1-p))."""
    p = max(min(p, 1 - 1e-7), 1e-7)
    return math.log(p / (1.0 - p))


def _sigmoid(x: float) -> float:
    """Sigmoid (inverse logit)."""
    if x > 500:
        return 1.0
    if x < -500:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


def _sigmoid_array(x: np.ndarray) -> np.ndarray:
    """Vectorized sigmoid with overflow protection."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
