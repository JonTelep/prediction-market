"""Price move anomaly detection for Polymarket surveillance.

Tracks market prices using both a rolling-window approach and an exponentially
weighted moving average.  Anomalies are flagged when the z-score of a price
return exceeds the configured threshold.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from prediction_market.analysis.timeseries import EWMA, RollingStats, compute_z_score
from prediction_market.config import ThresholdConfig

logger = logging.getLogger(__name__)


@dataclass
class PriceAnomaly:
    """Result object emitted when a price move is anomalous.

    Attributes:
        market_id: The Polymarket condition/market identifier.
        z_score: How many standard deviations the price return is from normal.
        current_price: The price observation that triggered the anomaly.
        baseline_price: The EWMA baseline price at the time of detection.
        price_return: The log-return of the current observation relative to
            the previous observation.
        timestamp: When the anomalous observation was recorded.
    """

    market_id: str
    z_score: float
    current_price: float
    baseline_price: float
    price_return: float
    timestamp: datetime


@dataclass
class _MarketPriceState:
    """Internal per-market tracking state."""

    ewma: EWMA
    return_stats: RollingStats
    last_price: float | None = None
    last_timestamp: datetime | None = None


class PriceAnalyzer:
    """Detects anomalous price moves using EWMA baselines and rolling z-scores.

    For each market, the analyzer maintains:

    * An :class:`~prediction_market.analysis.timeseries.EWMA` to track the
      baseline price level.
    * A :class:`~prediction_market.analysis.timeseries.RollingStats` over
      log-returns to detect sudden moves.

    A price anomaly is flagged when the z-score of the latest log-return
    exceeds :pyattr:`ThresholdConfig.price_zscore`.

    Args:
        thresholds: Threshold configuration.  If ``None``, defaults are used
            (price z-score of 2.5, 7-day rolling window).
        ewma_span: Span for the EWMA smoother.  Defaults to
            ``rolling_window_days * 24`` (hourly observations over the window).
    """

    def __init__(
        self,
        thresholds: ThresholdConfig | None = None,
        ewma_span: float | None = None,
    ) -> None:
        self._thresholds = thresholds or ThresholdConfig()
        self._window = timedelta(days=self._thresholds.rolling_window_days)
        self._z_threshold = self._thresholds.price_zscore
        self._ewma_span = ewma_span or float(self._thresholds.rolling_window_days * 24)
        self._states: dict[str, _MarketPriceState] = {}

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def update(self, market_id: str, price: float, timestamp: datetime | None = None) -> None:
        """Record a price observation for a market.

        The first observation initialises the EWMA baseline.  Subsequent
        observations compute a log-return and feed it into both the EWMA
        and the rolling-return tracker.

        Args:
            market_id: Polymarket market identifier.
            price: Current price (typically between 0 and 1 for binary markets).
            timestamp: Observation time.  Defaults to ``datetime.now(timezone.utc)``.
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        if market_id not in self._states:
            self._states[market_id] = _MarketPriceState(
                ewma=EWMA(span=self._ewma_span),
                return_stats=RollingStats(window=self._window),
            )

        state = self._states[market_id]

        # Compute log-return if we have a previous price.
        if state.last_price is not None and state.last_price > 0 and price > 0:
            log_return = math.log(price / state.last_price)
            state.return_stats.add(log_return, timestamp)

        # Update EWMA baseline with the raw price.
        state.ewma.update(price)
        state.last_price = price
        state.last_timestamp = timestamp

    def check_anomaly(self, market_id: str) -> PriceAnomaly | None:
        """Check whether the latest price observation for *market_id* is anomalous.

        An anomaly is flagged when the z-score of the most recent log-return
        exceeds :pyattr:`ThresholdConfig.price_zscore`.

        Args:
            market_id: Polymarket market identifier.

        Returns:
            A :class:`PriceAnomaly` if the threshold is exceeded, otherwise ``None``.
        """
        state = self._states.get(market_id)
        if state is None:
            return None

        rs = state.return_stats
        if rs.count < 3:
            # Not enough return observations to judge.
            return None

        latest_return = rs.latest
        if latest_return is None:
            return None

        z = rs.z_score(latest_return)
        if abs(z) < self._z_threshold:
            return None

        ts = state.last_timestamp or datetime.now(timezone.utc)
        anomaly = PriceAnomaly(
            market_id=market_id,
            z_score=z,
            current_price=state.last_price or 0.0,
            baseline_price=state.ewma.value,
            price_return=latest_return,
            timestamp=ts,
        )
        logger.info(
            "Price anomaly detected for %s: z=%.2f price=%.4f baseline=%.4f return=%.4f",
            market_id,
            z,
            anomaly.current_price,
            anomaly.baseline_price,
            latest_return,
        )
        return anomaly

    # ------------------------------------------------------------------
    # Bulk helpers
    # ------------------------------------------------------------------

    def check_all_anomalies(self) -> list[PriceAnomaly]:
        """Run anomaly detection across all tracked markets.

        Returns:
            A list of :class:`PriceAnomaly` instances (may be empty).
        """
        anomalies: list[PriceAnomaly] = []
        for market_id in list(self._states):
            result = self.check_anomaly(market_id)
            if result is not None:
                anomalies.append(result)
        return anomalies

    @property
    def tracked_markets(self) -> list[str]:
        """Market IDs currently being tracked."""
        return list(self._states.keys())

    # ------------------------------------------------------------------
    # Serialization (for persistence in rolling_stats table)
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize the full analyzer state to a JSON-compatible dict."""
        markets: dict[str, Any] = {}
        for mid, state in self._states.items():
            markets[mid] = {
                "ewma": state.ewma.to_dict(),
                "return_stats": state.return_stats.to_dict(),
                "last_price": state.last_price,
                "last_timestamp": (
                    state.last_timestamp.isoformat() if state.last_timestamp else None
                ),
            }
        return {
            "z_threshold": self._z_threshold,
            "ewma_span": self._ewma_span,
            "window_days": self._thresholds.rolling_window_days,
            "markets": markets,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], thresholds: ThresholdConfig | None = None) -> PriceAnalyzer:
        """Reconstruct a ``PriceAnalyzer`` from a serialized dict.

        Args:
            data: Dict produced by :meth:`to_dict`.
            thresholds: Optional threshold config override.

        Returns:
            A restored ``PriceAnalyzer`` instance.
        """
        ewma_span = data.get("ewma_span")
        analyzer = cls(thresholds=thresholds, ewma_span=ewma_span)
        for mid, state_data in data.get("markets", {}).items():
            ts_raw = state_data.get("last_timestamp")
            last_ts = datetime.fromisoformat(ts_raw) if ts_raw else None
            analyzer._states[mid] = _MarketPriceState(
                ewma=EWMA.from_dict(state_data["ewma"]),
                return_stats=RollingStats.from_dict(state_data["return_stats"]),
                last_price=state_data.get("last_price"),
                last_timestamp=last_ts,
            )
        return analyzer

    def __repr__(self) -> str:
        return (
            f"PriceAnalyzer(markets={len(self._states)}, "
            f"z_threshold={self._z_threshold}, ewma_span={self._ewma_span})"
        )
