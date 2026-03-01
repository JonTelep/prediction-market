"""Trader agent archetypes for the agent-based market simulator.

Each agent type encapsulates a distinct trading strategy and decision process.
On each simulation tick, agents observe the market state and submit orders
(buy/sell with size and aggressiveness).

The key insight for surveillance: a market WITHOUT InformedTraders produces
a characteristic microstructure signature. When real market data diverges
from this baseline, it suggests informed participation.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class Order:
    """An order submitted by a trader agent.

    Attributes:
        agent_id: Which agent placed the order.
        agent_type: Agent archetype name.
        side: "buy" or "sell".
        size: Order size in units.
        aggressiveness: How far from midpoint the agent is willing to trade.
            0 = passive (at best bid/ask), 1 = aggressive (market order).
        urgency: How time-sensitive the order is (0-1). Informed traders
            have high urgency.
    """

    agent_id: int
    agent_type: str
    side: str
    size: float
    aggressiveness: float = 0.5
    urgency: float = 0.5


class TraderAgent(ABC):
    """Abstract base class for trader agents."""

    agent_type: str = "base"

    def __init__(self, agent_id: int, rng: np.random.Generator) -> None:
        self.agent_id = agent_id
        self._rng = rng
        self.position: float = 0.0  # Net position
        self.pnl: float = 0.0

    @abstractmethod
    def decide(self, state: Any) -> Order | None:
        """Observe market state and optionally submit an order.

        Args:
            state: Current MarketState snapshot.

        Returns:
            An Order to submit, or None to sit out this tick.
        """

    def update_position(self, side: str, size: float, price: float) -> None:
        """Update position and PnL after a fill."""
        if side == "buy":
            self.position += size
            self.pnl -= size * price
        else:
            self.position -= size
            self.pnl += size * price

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "position": self.position,
            "pnl": self.pnl,
        }


class NoiseTrader(TraderAgent):
    """Random trader providing baseline market activity.

    Trades randomly with configurable frequency and size. Represents
    uninformed retail participants whose activity creates "normal"
    market noise.

    Args:
        agent_id: Unique identifier.
        rng: Random generator.
        trade_probability: Probability of trading on any given tick (0-1).
        mean_size: Average order size.
        size_std: Order size standard deviation.
    """

    agent_type = "noise"

    def __init__(
        self,
        agent_id: int,
        rng: np.random.Generator,
        trade_probability: float = 0.3,
        mean_size: float = 100.0,
        size_std: float = 50.0,
    ) -> None:
        super().__init__(agent_id, rng)
        self._trade_prob = trade_probability
        self._mean_size = mean_size
        self._size_std = size_std

    def decide(self, state: Any) -> Order | None:
        if self._rng.random() > self._trade_prob:
            return None

        side = "buy" if self._rng.random() < 0.5 else "sell"
        size = max(1.0, self._rng.normal(self._mean_size, self._size_std))

        return Order(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            side=side,
            size=size,
            aggressiveness=self._rng.uniform(0.2, 0.8),
            urgency=0.2,
        )


class MarketMaker(TraderAgent):
    """Liquidity provider that quotes both sides around fair value.

    Mean-reverts toward a perceived fair value, providing tighter spreads
    when confident and wider spreads when uncertain. Adjusts quotes based
    on inventory to avoid accumulating large directional positions.

    Args:
        agent_id: Unique identifier.
        rng: Random generator.
        fair_value: Initial estimate of fair value (0-1).
        spread: Half-spread the MM quotes (in price units).
        max_position: Position limit — MM widens spread when near limit.
        inventory_skew: How much to skew quotes per unit of inventory.
        quote_probability: Probability of quoting on any tick.
    """

    agent_type = "market_maker"

    def __init__(
        self,
        agent_id: int,
        rng: np.random.Generator,
        fair_value: float = 0.5,
        spread: float = 0.02,
        max_position: float = 1000.0,
        inventory_skew: float = 0.0001,
        quote_probability: float = 0.8,
    ) -> None:
        super().__init__(agent_id, rng)
        self.fair_value = fair_value
        self._spread = spread
        self._max_pos = max_position
        self._inv_skew = inventory_skew
        self._quote_prob = quote_probability

    def decide(self, state: Any) -> Order | None:
        if self._rng.random() > self._quote_prob:
            return None

        # Update fair value estimate: slowly track the market price
        if hasattr(state, "price") and state.price > 0:
            self.fair_value = 0.95 * self.fair_value + 0.05 * state.price

        # Inventory-adjusted side selection
        # If long, prefer to sell; if short, prefer to buy
        inventory_bias = -self.position * self._inv_skew
        buy_prob = 0.5 + inventory_bias

        # Near position limit: strongly prefer reducing
        if abs(self.position) > self._max_pos * 0.8:
            if self.position > 0:
                buy_prob = 0.1
            else:
                buy_prob = 0.9

        side = "buy" if self._rng.random() < buy_prob else "sell"

        # Size: smaller when near position limit
        position_utilization = abs(self.position) / self._max_pos
        size_factor = max(0.2, 1.0 - position_utilization)
        size = max(1.0, self._rng.exponential(150.0) * size_factor)

        return Order(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            side=side,
            size=size,
            aggressiveness=0.3,  # Passive — provides liquidity
            urgency=0.3,
        )


class MomentumTrader(TraderAgent):
    """Trend-following trader that amplifies price moves.

    Buys when price is rising, sells when falling. Uses a simple
    moving average crossover signal. Represents algorithmic trend
    followers and retail FOMO.

    Args:
        agent_id: Unique identifier.
        rng: Random generator.
        lookback: Number of ticks for the momentum signal.
        threshold: Minimum price change to trigger a trade.
        trade_probability: Base probability of acting on a signal.
        mean_size: Average order size.
    """

    agent_type = "momentum"

    def __init__(
        self,
        agent_id: int,
        rng: np.random.Generator,
        lookback: int = 10,
        threshold: float = 0.005,
        trade_probability: float = 0.4,
        mean_size: float = 200.0,
    ) -> None:
        super().__init__(agent_id, rng)
        self._lookback = lookback
        self._threshold = threshold
        self._trade_prob = trade_probability
        self._mean_size = mean_size
        self._price_history: list[float] = []

    def decide(self, state: Any) -> Order | None:
        price = getattr(state, "price", 0.5)
        self._price_history.append(price)
        if len(self._price_history) > self._lookback * 2:
            self._price_history = self._price_history[-self._lookback * 2:]

        if len(self._price_history) < self._lookback:
            return None

        # Momentum signal: recent average vs older average
        recent = np.mean(self._price_history[-self._lookback // 2:])
        older = np.mean(self._price_history[-self._lookback:])
        momentum = recent - older

        if abs(momentum) < self._threshold:
            return None

        if self._rng.random() > self._trade_prob:
            return None

        side = "buy" if momentum > 0 else "sell"
        # Size proportional to signal strength
        signal_strength = min(abs(momentum) / self._threshold, 3.0)
        size = max(1.0, self._rng.exponential(self._mean_size * signal_strength))

        return Order(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            side=side,
            size=size,
            aggressiveness=0.6 + 0.3 * min(signal_strength / 3, 1),
            urgency=0.6,
        )


class InformedTrader(TraderAgent):
    """Trader with a private information signal.

    This is the archetype we're trying to DETECT in real markets.
    In the simulation, informed traders know the "true" outcome
    probability and trade toward it with high urgency when the
    market price diverges from their signal.

    The simulation uses this to produce "with-insider" scenarios
    that can be compared against "no-insider" baselines.

    Args:
        agent_id: Unique identifier.
        rng: Random generator.
        true_value: The insider's private estimate of true probability.
        edge_threshold: Minimum |market_price - true_value| before trading.
        trade_probability: Probability of acting when edge exists.
        mean_size: Average order size (informed traders tend to trade larger).
        stealth: How much to disguise order flow (0 = obvious, 1 = stealthy).
            Stealthy insiders split orders and reduce aggressiveness.
    """

    agent_type = "informed"

    def __init__(
        self,
        agent_id: int,
        rng: np.random.Generator,
        true_value: float = 0.7,
        edge_threshold: float = 0.03,
        trade_probability: float = 0.6,
        mean_size: float = 500.0,
        stealth: float = 0.5,
    ) -> None:
        super().__init__(agent_id, rng)
        self.true_value = true_value
        self._edge_threshold = edge_threshold
        self._trade_prob = trade_probability
        self._mean_size = mean_size
        self._stealth = stealth

    def decide(self, state: Any) -> Order | None:
        price = getattr(state, "price", 0.5)

        edge = self.true_value - price
        if abs(edge) < self._edge_threshold:
            return None

        if self._rng.random() > self._trade_prob:
            return None

        side = "buy" if edge > 0 else "sell"

        # Size: proportional to edge, reduced by stealth
        edge_factor = min(abs(edge) / self._edge_threshold, 3.0)
        stealth_factor = 1.0 - self._stealth * 0.7
        size = max(1.0, self._rng.exponential(self._mean_size * edge_factor * stealth_factor))

        # Aggressiveness: high urgency but stealthy insiders are more passive
        aggressiveness = 0.8 - self._stealth * 0.4
        urgency = 0.9 - self._stealth * 0.3

        return Order(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            side=side,
            size=size,
            aggressiveness=aggressiveness,
            urgency=urgency,
        )
