"""Core statistical primitives for time-series analysis.

Provides rolling-window statistics and exponentially weighted moving averages
used by the volume, price, and liquidity analyzers.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any


def compute_z_score(value: float, mean: float, std: float) -> float:
    """Compute the z-score of a value given a mean and standard deviation.

    Returns 0.0 when the standard deviation is zero or near-zero to avoid
    division-by-zero artifacts.

    Args:
        value: The observed value.
        mean: The population/rolling mean.
        std: The population/rolling standard deviation.

    Returns:
        The z-score, or 0.0 if std is effectively zero.
    """
    if std < 1e-12:
        return 0.0
    return (value - mean) / std


@dataclass
class _TimestampedValue:
    """Internal pair of (value, timestamp) kept in the rolling window."""

    value: float
    timestamp: datetime


class RollingStats:
    """Maintains a rolling window of numeric values and provides summary statistics.

    The window is defined by a time duration (default 7 days).  Values older
    than the window are automatically evicted when new values are added or
    when statistics are queried.

    Args:
        window: The duration of the rolling window.
    """

    def __init__(self, window: timedelta = timedelta(days=7)) -> None:
        self.window = window
        self._values: deque[_TimestampedValue] = deque()

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def add(self, value: float, timestamp: datetime | None = None) -> None:
        """Append a value and evict entries older than the window.

        Args:
            value: The numeric observation.
            timestamp: When the observation occurred.  Defaults to ``datetime.now(timezone.utc)``.
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
        self._values.append(_TimestampedValue(value=value, timestamp=timestamp))
        self._evict(timestamp)

    @property
    def count(self) -> int:
        """Number of observations currently in the window."""
        return len(self._values)

    @property
    def mean(self) -> float:
        """Arithmetic mean of values in the window.

        Returns 0.0 when the window is empty.
        """
        if not self._values:
            return 0.0
        return sum(v.value for v in self._values) / len(self._values)

    @property
    def std(self) -> float:
        """Population standard deviation of values in the window.

        Returns 0.0 when fewer than 2 observations are present.
        """
        n = len(self._values)
        if n < 2:
            return 0.0
        m = self.mean
        variance = sum((v.value - m) ** 2 for v in self._values) / n
        return math.sqrt(variance)

    def z_score(self, value: float) -> float:
        """Compute the z-score of *value* relative to the current window.

        Args:
            value: The observation to score.

        Returns:
            The z-score, or 0.0 if insufficient data.
        """
        return compute_z_score(value, self.mean, self.std)

    @property
    def values(self) -> list[float]:
        """Return a copy of the raw values in chronological order."""
        return [v.value for v in self._values]

    @property
    def latest(self) -> float | None:
        """The most recently added value, or ``None`` if empty."""
        if not self._values:
            return None
        return self._values[-1].value

    # ------------------------------------------------------------------
    # Serialization (for persistence in rolling_stats table)
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize the rolling window to a JSON-compatible dict.

        Returns:
            Dictionary with ``window_seconds`` and ``entries``.
        """
        return {
            "window_seconds": self.window.total_seconds(),
            "entries": [
                {"value": v.value, "timestamp": v.timestamp.isoformat()}
                for v in self._values
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RollingStats:
        """Reconstruct a ``RollingStats`` instance from a dict produced by :meth:`to_dict`.

        Args:
            data: Serialized representation.

        Returns:
            A new ``RollingStats`` instance with the restored state.
        """
        window = timedelta(seconds=data.get("window_seconds", 7 * 86400))
        stats = cls(window=window)
        for entry in data.get("entries", []):
            ts_raw = entry["timestamp"]
            if isinstance(ts_raw, str):
                ts = datetime.fromisoformat(ts_raw)
            else:
                ts = ts_raw
            stats._values.append(_TimestampedValue(value=entry["value"], timestamp=ts))
        # Evict stale entries relative to the latest timestamp
        if stats._values:
            stats._evict(stats._values[-1].timestamp)
        return stats

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _evict(self, reference: datetime) -> None:
        """Remove entries older than ``reference - window``."""
        cutoff = reference - self.window
        while self._values and self._values[0].timestamp < cutoff:
            self._values.popleft()

    def __repr__(self) -> str:
        return (
            f"RollingStats(window={self.window}, count={self.count}, "
            f"mean={self.mean:.4f}, std={self.std:.4f})"
        )


class EWMA:
    """Exponentially weighted moving average with z-score support.

    Tracks both the EWMA of values and the EWMA of squared values so that a
    running estimate of variance (and therefore z-scores) can be computed
    without storing raw observations.

    Args:
        span: The effective window length.  ``alpha = 2 / (span + 1)``.
            Mutually exclusive with *alpha*.
        alpha: The smoothing factor directly (0 < alpha <= 1).
            Mutually exclusive with *span*.
    """

    def __init__(
        self,
        span: float | None = None,
        alpha: float | None = None,
    ) -> None:
        if span is not None and alpha is not None:
            raise ValueError("Specify either span or alpha, not both")
        if span is not None:
            if span <= 0:
                raise ValueError("span must be positive")
            self._alpha = 2.0 / (span + 1.0)
        elif alpha is not None:
            if not (0.0 < alpha <= 1.0):
                raise ValueError("alpha must be in (0, 1]")
            self._alpha = alpha
        else:
            # Default span mirrors rolling_window_days * 24 hourly observations
            self._alpha = 2.0 / (7 * 24 + 1)

        self._mean: float | None = None
        self._var: float | None = None
        self._count: int = 0

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def update(self, value: float) -> None:
        """Incorporate a new observation.

        Args:
            value: The numeric observation.
        """
        if self._mean is None:
            self._mean = value
            self._var = 0.0
        else:
            diff = value - self._mean
            self._mean += self._alpha * diff
            self._var = (1.0 - self._alpha) * (self._var + self._alpha * diff * diff)  # type: ignore[operator]
        self._count += 1

    @property
    def value(self) -> float:
        """Current EWMA value.

        Returns 0.0 if no observations have been recorded.
        """
        if self._mean is None:
            return 0.0
        return self._mean

    @property
    def std(self) -> float:
        """Current EWMA-estimated standard deviation."""
        if self._var is None or self._var < 0:
            return 0.0
        return math.sqrt(self._var)

    @property
    def count(self) -> int:
        """Number of observations processed (lifetime, not windowed)."""
        return self._count

    def z_score(self, value: float) -> float:
        """Compute the z-score of *value* relative to the EWMA state.

        Args:
            value: The observation to score.

        Returns:
            The z-score, or 0.0 if insufficient data.
        """
        return compute_z_score(value, self.value, self.std)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize the EWMA state to a JSON-compatible dict."""
        return {
            "alpha": self._alpha,
            "mean": self._mean,
            "var": self._var,
            "count": self._count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EWMA:
        """Reconstruct an ``EWMA`` instance from a dict produced by :meth:`to_dict`.

        Args:
            data: Serialized representation.

        Returns:
            A new ``EWMA`` instance with the restored state.
        """
        instance = cls(alpha=data["alpha"])
        instance._mean = data.get("mean")
        instance._var = data.get("var")
        instance._count = data.get("count", 0)
        return instance

    def __repr__(self) -> str:
        return (
            f"EWMA(alpha={self._alpha:.4f}, value={self.value:.4f}, "
            f"std={self.std:.4f}, count={self._count})"
        )
