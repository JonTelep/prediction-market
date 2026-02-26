"""Volume anomaly detection for Polymarket surveillance.

Tracks hourly volume per market using a rolling window and flags observations
whose z-score exceeds the configured threshold.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from prediction_market.analysis.timeseries import RollingStats
from prediction_market.config import ThresholdConfig

logger = logging.getLogger(__name__)

# Default rolling window pulled from ThresholdConfig.rolling_window_days (7).
_DEFAULT_WINDOW_DAYS = 7


@dataclass
class VolumeAnomaly:
    """Result object emitted when a volume spike is detected.

    Attributes:
        market_id: The Polymarket condition/market identifier.
        z_score: How many standard deviations the current volume is from the mean.
        current_volume: The hourly volume observation that triggered the anomaly.
        mean_volume: Rolling mean volume over the window.
        std_volume: Rolling standard deviation of volume over the window.
        timestamp: When the anomalous observation was recorded.
    """

    market_id: str
    z_score: float
    current_volume: float
    mean_volume: float
    std_volume: float
    timestamp: datetime


class VolumeAnalyzer:
    """Detects anomalous volume spikes using rolling-window z-scores.

    Maintains one :class:`~prediction_market.analysis.timeseries.RollingStats`
    instance per market.  Each call to :meth:`update` appends an hourly volume
    observation; :meth:`check_anomaly` then compares the latest observation to
    the rolling distribution.

    Args:
        thresholds: Threshold configuration.  If ``None``, default values are
            used (volume z-score threshold of 3.0, 7-day window).
    """

    def __init__(self, thresholds: ThresholdConfig | None = None) -> None:
        self._thresholds = thresholds or ThresholdConfig()
        self._window = timedelta(days=self._thresholds.rolling_window_days)
        self._z_threshold = self._thresholds.volume_zscore
        self._stats: dict[str, RollingStats] = {}

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def update(self, market_id: str, hourly_volume: float, timestamp: datetime | None = None) -> None:
        """Record an hourly volume observation for a market.

        Args:
            market_id: Polymarket market identifier.
            hourly_volume: Volume (in USD or contract units) for the hour.
            timestamp: Observation time.  Defaults to ``datetime.now(timezone.utc)``.
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
        if market_id not in self._stats:
            self._stats[market_id] = RollingStats(window=self._window)
        self._stats[market_id].add(hourly_volume, timestamp)

    def check_anomaly(self, market_id: str) -> VolumeAnomaly | None:
        """Check whether the most recent volume observation for *market_id* is anomalous.

        An anomaly is flagged when the z-score of the latest hourly volume
        exceeds :pyattr:`ThresholdConfig.volume_zscore`.

        Args:
            market_id: Polymarket market identifier.

        Returns:
            A :class:`VolumeAnomaly` if the threshold is exceeded, otherwise ``None``.
        """
        stats = self._stats.get(market_id)
        if stats is None or stats.count < 3:
            # Not enough history to judge.
            return None

        latest = stats.latest
        if latest is None:
            return None

        z = stats.z_score(latest)
        if abs(z) < self._z_threshold:
            return None

        now = datetime.now(timezone.utc)
        # Use the timestamp of the last entry if available.
        ts = stats._values[-1].timestamp if stats._values else now

        anomaly = VolumeAnomaly(
            market_id=market_id,
            z_score=z,
            current_volume=latest,
            mean_volume=stats.mean,
            std_volume=stats.std,
            timestamp=ts,
        )
        logger.info(
            "Volume anomaly detected for %s: z=%.2f vol=%.2f mean=%.2f",
            market_id,
            z,
            latest,
            stats.mean,
        )
        return anomaly

    # ------------------------------------------------------------------
    # Bulk helpers
    # ------------------------------------------------------------------

    def check_all_anomalies(self) -> list[VolumeAnomaly]:
        """Run anomaly detection across all tracked markets.

        Returns:
            A list of :class:`VolumeAnomaly` instances (may be empty).
        """
        anomalies: list[VolumeAnomaly] = []
        for market_id in list(self._stats):
            result = self.check_anomaly(market_id)
            if result is not None:
                anomalies.append(result)
        return anomalies

    @property
    def tracked_markets(self) -> list[str]:
        """Market IDs currently being tracked."""
        return list(self._stats.keys())

    # ------------------------------------------------------------------
    # Serialization (for persistence in rolling_stats table)
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize the full analyzer state to a JSON-compatible dict."""
        return {
            "z_threshold": self._z_threshold,
            "window_days": self._thresholds.rolling_window_days,
            "markets": {
                mid: rs.to_dict() for mid, rs in self._stats.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], thresholds: ThresholdConfig | None = None) -> VolumeAnalyzer:
        """Reconstruct a ``VolumeAnalyzer`` from a serialized dict.

        Args:
            data: Dict produced by :meth:`to_dict`.
            thresholds: Optional threshold config override.

        Returns:
            A restored ``VolumeAnalyzer`` instance.
        """
        analyzer = cls(thresholds=thresholds)
        for mid, rs_data in data.get("markets", {}).items():
            analyzer._stats[mid] = RollingStats.from_dict(rs_data)
        return analyzer

    def __repr__(self) -> str:
        return (
            f"VolumeAnalyzer(markets={len(self._stats)}, "
            f"z_threshold={self._z_threshold}, window={self._window})"
        )
