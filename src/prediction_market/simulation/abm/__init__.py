"""Agent-Based Market Simulator for prediction market surveillance.

Simulates heterogeneous trader populations interacting through a simplified
order book to produce a synthetic "no-insider" baseline. Comparing real
market microstructure against this baseline reveals statistical divergence
that signals potential information leakage or manipulation.

Trader archetypes:
  - NoiseTrader: Random activity, provides baseline liquidity
  - MarketMaker: Quotes both sides, mean-reverts around fair value
  - MomentumTrader: Follows recent trends, amplifies moves
  - InformedTrader: Trades on private signal (the "insider" we're detecting)

The simulator produces synthetic price/volume/spread series that can be
compared against real observations via divergence metrics.
"""

from prediction_market.simulation.abm.agents import (
    InformedTrader,
    MarketMaker,
    MomentumTrader,
    NoiseTrader,
    TraderAgent,
)
from prediction_market.simulation.abm.market import SimulatedMarket, MarketState
from prediction_market.simulation.abm.simulator import (
    ABMSimulator,
    ABMConfig,
    ABMResult,
    DivergenceMetrics,
)
from prediction_market.simulation.abm.calibrator import Calibrator, CalibrationResult, TargetStatistics

__all__ = [
    "InformedTrader",
    "MarketMaker",
    "MomentumTrader",
    "NoiseTrader",
    "TraderAgent",
    "SimulatedMarket",
    "MarketState",
    "ABMSimulator",
    "ABMConfig",
    "ABMResult",
    "DivergenceMetrics",
    "Calibrator",
    "CalibrationResult",
    "TargetStatistics",
]
