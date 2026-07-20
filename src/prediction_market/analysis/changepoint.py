"""CUSUM change-point detection over logit-return series.

A two-sided CUSUM (cumulative sum) detector accumulates standardized
logit-returns and flags a sustained regime shift once the running sum
crosses a decision threshold. This complements
:class:`~prediction_market.analysis.price_analyzer.PriceAnalyzer`'s rolling
z-score: a run of moderate moves (e.g. six consecutive +1.5 sigma returns)
never individually crosses a 2.5 sigma z-score threshold, but accumulates in
CUSUM within a handful of observations. See docs/RESEARCH-BRIEF.md for the
motivating discussion.

Detector wiring into the live agent pipeline is a later prompt -- this
module is standalone.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from prediction_market.analysis.timeseries import RollingStats, clamp_probability, logit
from prediction_market.config import ThresholdConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CusumAlarm:
    """Result object emitted when a CUSUM statistic crosses its threshold.

    Attributes:
        market_id: The Polymarket condition/market identifier.
        direction: ``"up"`` if the positive CUSUM statistic (S+) crossed the
            threshold, ``"down"`` if the negative statistic (S-) did.
        statistic: The accumulated CUSUM value (S+ or S-, matching
            *direction*) at the moment it crossed *threshold*.
        threshold: The decision threshold (``cusum_h``) that was crossed.
        observations_since_reset: Number of return observations accumulated
            since the last reset (alarm or detector creation).
        timestamp: When the triggering observation was recorded.
    """

    market_id: str
    direction: str
    statistic: float
    threshold: float
    observations_since_reset: int
    timestamp: datetime


@dataclass
class _MarketCusumState:
    """Internal per-market tracking state."""

    baseline: RollingStats
    s_pos: float = 0.0
    s_neg: float = 0.0
    observations_since_reset: int = 0
    last_price: float | None = None
    last_timestamp: datetime | None = None


class CusumDetector:
    """Detects sustained regime shifts using two-sided CUSUM over logit-returns.

    For each market, the detector maintains a
    :class:`~prediction_market.analysis.timeseries.RollingStats` baseline of
    logit-returns (mirroring
    :class:`~prediction_market.analysis.price_analyzer.PriceAnalyzer`) and
    two cumulative sums, ``S+`` and ``S-``, that accumulate standardized
    returns net of a slack parameter ``k``. An alarm fires when either sum
    crosses the decision threshold ``h``.

    The baseline includes every observed return, including the ones that
    trip the alarm -- excluding post-alarm returns from the baseline
    ("contamination control") is a Phase-3 refinement, not implemented here.

    Args:
        thresholds: Threshold configuration. If ``None``, defaults are used
            (``cusum_k=0.5``, ``cusum_h=5.0``, ``cusum_min_observations=5``,
            7-day rolling baseline window).
    """

    def __init__(self, thresholds: ThresholdConfig | None = None) -> None:
        self._thresholds = thresholds or ThresholdConfig()
        self._k = self._thresholds.cusum_k
        self._h = self._thresholds.cusum_h
        self._min_observations = self._thresholds.cusum_min_observations
        self._window = timedelta(days=self._thresholds.rolling_window_days)
        self._states: dict[str, _MarketCusumState] = {}

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def update(self, market_id: str, price: float, timestamp: datetime | None = None) -> None:
        """Record a price observation for a market.

        Computes a logit-return from the previous price exactly as
        :class:`~prediction_market.analysis.price_analyzer.PriceAnalyzer`
        does, feeds it into the per-market baseline, and -- once the
        baseline has at least ``cusum_min_observations`` returns --
        standardizes the return and accumulates it into the CUSUM
        statistics.

        Args:
            market_id: Polymarket market identifier.
            price: Current price (typically between 0 and 1 for binary markets).
            timestamp: Observation time. Defaults to ``datetime.now(timezone.utc)``.
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        if market_id not in self._states:
            self._states[market_id] = _MarketCusumState(baseline=RollingStats(window=self._window))

        state = self._states[market_id]

        if state.last_price is not None:
            logit_return = logit(clamp_probability(price)) - logit(clamp_probability(state.last_price))
            state.baseline.add(logit_return, timestamp)

            if state.baseline.count >= self._min_observations:
                mean = state.baseline.mean
                std = state.baseline.std
                z = 0.0 if std < 1e-12 else (logit_return - mean) / std

                state.s_pos = max(0.0, state.s_pos + z - self._k)
                state.s_neg = max(0.0, state.s_neg - z - self._k)
                state.observations_since_reset += 1

        state.last_price = price
        state.last_timestamp = timestamp

    def check_alarm(self, market_id: str) -> CusumAlarm | None:
        """Check whether the accumulated CUSUM statistics have crossed the threshold.

        Returns ``None`` during warm-up (fewer than ``cusum_min_observations``
        return observations accumulated). When an alarm fires, both CUSUM
        statistics and the observation counter are reset to zero -- one
        alarm per excursion, not one per tick.

        Args:
            market_id: Polymarket market identifier.

        Returns:
            A :class:`CusumAlarm` if a threshold crossing occurred, otherwise ``None``.
        """
        state = self._states.get(market_id)
        if state is None:
            return None

        if state.observations_since_reset < 1:
            return None

        ts = state.last_timestamp or datetime.now(timezone.utc)

        if state.s_pos >= self._h:
            alarm = CusumAlarm(
                market_id=market_id,
                direction="up",
                statistic=state.s_pos,
                threshold=self._h,
                observations_since_reset=state.observations_since_reset,
                timestamp=ts,
            )
            state.s_pos = 0.0
            state.s_neg = 0.0
            state.observations_since_reset = 0
            logger.info(
                "CUSUM alarm for %s: direction=up statistic=%.2f threshold=%.2f",
                market_id,
                alarm.statistic,
                self._h,
            )
            return alarm

        if state.s_neg >= self._h:
            alarm = CusumAlarm(
                market_id=market_id,
                direction="down",
                statistic=state.s_neg,
                threshold=self._h,
                observations_since_reset=state.observations_since_reset,
                timestamp=ts,
            )
            state.s_pos = 0.0
            state.s_neg = 0.0
            state.observations_since_reset = 0
            logger.info(
                "CUSUM alarm for %s: direction=down statistic=%.2f threshold=%.2f",
                market_id,
                alarm.statistic,
                self._h,
            )
            return alarm

        return None

    def current_statistics(self, market_id: str) -> tuple[float, float] | None:
        """Return the current (S+, S-) CUSUM statistics for diagnostics.

        Args:
            market_id: Polymarket market identifier.

        Returns:
            A ``(s_pos, s_neg)`` tuple, or ``None`` if the market is untracked.
        """
        state = self._states.get(market_id)
        if state is None:
            return None
        return (state.s_pos, state.s_neg)

    @property
    def tracked_markets(self) -> list[str]:
        """Market IDs currently being tracked."""
        return list(self._states.keys())

    # ------------------------------------------------------------------
    # Serialization (for persistence in rolling_stats table)
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize the full detector state to a JSON-compatible dict."""
        markets: dict[str, Any] = {}
        for mid, state in self._states.items():
            markets[mid] = {
                "baseline": state.baseline.to_dict(),
                "s_pos": state.s_pos,
                "s_neg": state.s_neg,
                "observations_since_reset": state.observations_since_reset,
                "last_price": state.last_price,
                "last_timestamp": (
                    state.last_timestamp.isoformat() if state.last_timestamp else None
                ),
            }
        return {
            "k": self._k,
            "h": self._h,
            "min_observations": self._min_observations,
            "window_days": self._thresholds.rolling_window_days,
            "markets": markets,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], thresholds: ThresholdConfig | None = None) -> CusumDetector:
        """Reconstruct a ``CusumDetector`` from a serialized dict.

        Args:
            data: Dict produced by :meth:`to_dict`.
            thresholds: Optional threshold config override.

        Returns:
            A restored ``CusumDetector`` instance.
        """
        detector = cls(thresholds=thresholds)
        for mid, state_data in data.get("markets", {}).items():
            ts_raw = state_data.get("last_timestamp")
            last_ts = datetime.fromisoformat(ts_raw) if ts_raw else None
            detector._states[mid] = _MarketCusumState(
                baseline=RollingStats.from_dict(state_data["baseline"]),
                s_pos=state_data.get("s_pos", 0.0),
                s_neg=state_data.get("s_neg", 0.0),
                observations_since_reset=state_data.get("observations_since_reset", 0),
                last_price=state_data.get("last_price"),
                last_timestamp=last_ts,
            )
        return detector

    def __repr__(self) -> str:
        return (
            f"CusumDetector(markets={len(self._states)}, "
            f"k={self._k}, h={self._h}, min_observations={self._min_observations})"
        )
