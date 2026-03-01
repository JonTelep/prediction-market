"""Monte Carlo simulation engine for prediction market analysis.

Provides stochastic modeling, importance sampling for rare-event detection,
probability distribution fitting, sequential Monte Carlo (particle filter)
for real-time Bayesian state estimation, and copula-based tail dependence
modeling for cross-market correlation analysis.
"""

from prediction_market.simulation.distributions import (
    BetaMarketModel,
    DirichletMarketModel,
    MarketModel,
)
from prediction_market.simulation.monte_carlo import MonteCarloEngine, SimulationResult
from prediction_market.simulation.importance_sampler import (
    ImportanceSampler,
    TailRiskEstimate,
)
from prediction_market.simulation.particle_filter import (
    MarketAwareTransition,
    MarketContext,
    ParticleFilter,
    ParticleFilterResult,
    TradeFlowAnalyzer,
)
from prediction_market.simulation.copulas import (
    ClaytonCopula,
    CopulaFitter,
    DynamicCopulaTracker,
    FrankCopula,
    GumbelCopula,
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
    InformedTrader,
    MarketMaker,
    MomentumTrader,
    NoiseTrader,
    SimulatedMarket,
    TraderAgent,
)

__all__ = [
    "BetaMarketModel",
    "DirichletMarketModel",
    "MarketModel",
    "MonteCarloEngine",
    "SimulationResult",
    "ImportanceSampler",
    "TailRiskEstimate",
    "MarketAwareTransition",
    "MarketContext",
    "ParticleFilter",
    "ParticleFilterResult",
    "TradeFlowAnalyzer",
    "ClaytonCopula",
    "CopulaFitter",
    "DynamicCopulaTracker",
    "FrankCopula",
    "GumbelCopula",
    "TailAlert",
    "TailDependence",
    "ABMConfig",
    "ABMResult",
    "ABMSimulator",
    "Calibrator",
    "CalibrationResult",
    "DivergenceMetrics",
    "InformedTrader",
    "MarketMaker",
    "MomentumTrader",
    "NoiseTrader",
    "SimulatedMarket",
    "TraderAgent",
]
