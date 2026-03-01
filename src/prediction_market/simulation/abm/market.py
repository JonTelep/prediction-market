"""Simulated market mechanism for the agent-based model.

Implements a simplified Automated Market Maker (AMM) inspired by
Polymarket's CLOB but abstracted for simulation efficiency. Processes
agent orders, updates prices via a constant-product-like mechanism,
and tracks microstructure metrics (spread, depth, volume, trade flow).

The market maintains a full tick-by-tick history for comparison against
real market data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np

from prediction_market.simulation.abm.agents import Order


@dataclass
class MarketState:
    """Snapshot of the simulated market at a single point in time.

    This is what agents observe when making decisions.

    Attributes:
        tick: The simulation step number.
        price: Current market price (0-1).
        spread: Current bid-ask spread.
        volume_this_tick: Volume traded in the current tick.
        cumulative_volume: Total volume traded so far.
        buy_volume: Buy-side volume this tick.
        sell_volume: Sell-side volume this tick.
        n_trades: Number of trades this tick.
        bid_depth: Total resting bid liquidity.
        ask_depth: Total resting ask liquidity.
        imbalance: Order flow imbalance (-1 to 1).
        volatility: Realized volatility (rolling).
    """

    tick: int = 0
    price: float = 0.5
    spread: float = 0.02
    volume_this_tick: float = 0.0
    cumulative_volume: float = 0.0
    buy_volume: float = 0.0
    sell_volume: float = 0.0
    n_trades: int = 0
    bid_depth: float = 0.0
    ask_depth: float = 0.0
    imbalance: float = 0.0
    volatility: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "tick": self.tick,
            "price": self.price,
            "spread": self.spread,
            "volume_this_tick": self.volume_this_tick,
            "cumulative_volume": self.cumulative_volume,
            "buy_volume": self.buy_volume,
            "sell_volume": self.sell_volume,
            "n_trades": self.n_trades,
            "bid_depth": self.bid_depth,
            "ask_depth": self.ask_depth,
            "imbalance": self.imbalance,
            "volatility": self.volatility,
        }


@dataclass
class Trade:
    """A single executed trade in the simulation."""

    tick: int
    agent_id: int
    agent_type: str
    side: str
    size: float
    price: float
    price_impact: float  # How much this trade moved the price


class SimulatedMarket:
    """Simplified AMM for agent-based simulation.

    Uses a logistic price impact model: each trade moves the price in
    logit space proportional to signed volume divided by liquidity depth.
    This naturally keeps prices in (0, 1) and produces realistic
    microstructure (larger orders have more impact in thin markets).

    Args:
        initial_price: Starting price (0-1).
        base_liquidity: Liquidity depth parameter — higher = less price
            impact per unit of volume.
        base_spread: Base bid-ask spread.
        price_impact_scale: Scaling factor for price impact.
        volatility_window: Number of ticks for realized vol calculation.
    """

    def __init__(
        self,
        initial_price: float = 0.5,
        base_liquidity: float = 10_000.0,
        base_spread: float = 0.02,
        price_impact_scale: float = 1.0,
        volatility_window: int = 20,
    ) -> None:
        self._price = np.clip(initial_price, 0.01, 0.99)
        self._logit_price = _logit(self._price)
        self._base_liquidity = base_liquidity
        self._base_spread = base_spread
        self._impact_scale = price_impact_scale
        self._vol_window = volatility_window

        # State tracking
        self._tick = 0
        self._cumulative_volume = 0.0
        self._price_history: list[float] = [self._price]
        self._trade_history: list[Trade] = []
        self._tick_states: list[MarketState] = []

        # Per-tick accumulators
        self._tick_buy_vol = 0.0
        self._tick_sell_vol = 0.0
        self._tick_trades: list[Trade] = []

        # Dynamic liquidity: adjusts based on recent activity
        self._effective_liquidity = base_liquidity

    @property
    def price(self) -> float:
        return self._price

    @property
    def state(self) -> MarketState:
        """Current market state snapshot."""
        vol = self._realized_volatility()
        total_tick_vol = self._tick_buy_vol + self._tick_sell_vol
        imbalance = 0.0
        if total_tick_vol > 0:
            imbalance = (self._tick_buy_vol - self._tick_sell_vol) / total_tick_vol

        return MarketState(
            tick=self._tick,
            price=self._price,
            spread=self._current_spread(),
            volume_this_tick=total_tick_vol,
            cumulative_volume=self._cumulative_volume,
            buy_volume=self._tick_buy_vol,
            sell_volume=self._tick_sell_vol,
            n_trades=len(self._tick_trades),
            bid_depth=self._effective_liquidity * 0.5,
            ask_depth=self._effective_liquidity * 0.5,
            imbalance=imbalance,
            volatility=vol,
        )

    def process_order(self, order: Order) -> Trade | None:
        """Process a single order and return the resulting trade.

        Args:
            order: The order to execute.

        Returns:
            A Trade if the order was filled, None if rejected.
        """
        if order.size <= 0:
            return None

        # Price impact in logit space
        signed_volume = order.size if order.side == "buy" else -order.size
        impact = (
            self._impact_scale
            * signed_volume
            / self._effective_liquidity
            * order.aggressiveness
        )

        # Apply impact
        old_price = self._price
        self._logit_price += impact
        # Clamp logit to prevent extreme prices
        self._logit_price = np.clip(self._logit_price, -6.0, 6.0)
        self._price = _sigmoid(self._logit_price)

        price_impact = self._price - old_price

        # Execute at midpoint between old and new price (average fill)
        fill_price = (old_price + self._price) / 2.0

        trade = Trade(
            tick=self._tick,
            agent_id=order.agent_id,
            agent_type=order.agent_type,
            side=order.side,
            size=order.size,
            price=fill_price,
            price_impact=price_impact,
        )

        # Update accumulators
        self._cumulative_volume += order.size
        if order.side == "buy":
            self._tick_buy_vol += order.size
        else:
            self._tick_sell_vol += order.size
        self._tick_trades.append(trade)
        self._trade_history.append(trade)

        return trade

    def end_tick(self) -> MarketState:
        """Finalize the current tick and prepare for the next one.

        Returns:
            The MarketState snapshot for the completed tick.
        """
        state = self.state
        self._tick_states.append(state)
        self._price_history.append(self._price)

        # Update effective liquidity: mean-revert toward base,
        # but reduce during high-volume periods (liquidity consumed)
        tick_vol = self._tick_buy_vol + self._tick_sell_vol
        vol_pressure = min(tick_vol / self._base_liquidity, 0.5)
        self._effective_liquidity = (
            0.95 * self._effective_liquidity
            + 0.05 * self._base_liquidity
            - self._base_liquidity * vol_pressure * 0.1
        )
        self._effective_liquidity = max(
            self._effective_liquidity, self._base_liquidity * 0.2
        )

        # Reset per-tick accumulators
        self._tick += 1
        self._tick_buy_vol = 0.0
        self._tick_sell_vol = 0.0
        self._tick_trades = []

        return state

    # ------------------------------------------------------------------
    # History access
    # ------------------------------------------------------------------

    @property
    def price_history(self) -> list[float]:
        return list(self._price_history)

    @property
    def trade_history(self) -> list[Trade]:
        return list(self._trade_history)

    @property
    def tick_states(self) -> list[MarketState]:
        return list(self._tick_states)

    def get_price_series(self) -> np.ndarray:
        """Return price history as numpy array."""
        return np.array(self._price_history)

    def get_volume_series(self) -> np.ndarray:
        """Return per-tick volume as numpy array."""
        return np.array([s.volume_this_tick for s in self._tick_states])

    def get_spread_series(self) -> np.ndarray:
        """Return per-tick spread as numpy array."""
        return np.array([s.spread for s in self._tick_states])

    def get_imbalance_series(self) -> np.ndarray:
        """Return per-tick order flow imbalance as numpy array."""
        return np.array([s.imbalance for s in self._tick_states])

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _current_spread(self) -> float:
        """Dynamic spread: widens with volatility and thin liquidity."""
        vol = self._realized_volatility()
        liq_ratio = self._base_liquidity / max(self._effective_liquidity, 1.0)
        return self._base_spread * (1.0 + vol * 10) * min(liq_ratio, 3.0)

    def _realized_volatility(self) -> float:
        """Realized volatility from recent price history."""
        if len(self._price_history) < 3:
            return 0.0
        recent = self._price_history[-self._vol_window:]
        if len(recent) < 3:
            return 0.0
        arr = np.array(recent)
        arr = np.clip(arr, 1e-6, 1 - 1e-6)
        log_returns = np.diff(np.log(arr))
        return float(np.std(log_returns))

    def __repr__(self) -> str:
        return (
            f"SimulatedMarket(price={self._price:.4f}, tick={self._tick}, "
            f"volume={self._cumulative_volume:.0f})"
        )


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def _logit(p: float) -> float:
    p = max(min(p, 1 - 1e-7), 1e-7)
    return math.log(p / (1.0 - p))


def _sigmoid(x: float) -> float:
    if x > 500:
        return 1.0
    if x < -500:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))
