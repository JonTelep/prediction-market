"""Cross-market correlation detection for Polymarket surveillance.

Identifies groups of markets whose prices move in tandem within a short
time window -- a potential signal for coordinated activity or information
leakage across related contracts.
"""

from __future__ import annotations

import itertools
import logging
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import numpy as np

logger = logging.getLogger(__name__)

# Minimum number of overlapping observations required to compute a
# meaningful correlation coefficient.
_MIN_OVERLAP = 5

# Default correlation coefficient threshold above which a pair of markets
# is flagged as correlated.
_CORRELATION_THRESHOLD = 0.80


@dataclass
class CorrelatedMove:
    """A detected correlated price movement across two or more markets.

    Attributes:
        market_ids: The set of market identifiers involved.
        correlation_coefficient: Pearson correlation between the aligned
            price-change series of the pair with the highest correlation.
        time_window: The lookback window (in minutes) used for detection.
        price_changes: Mapping from ``market_id`` to the cumulative
            price change over the window.
    """

    market_ids: list[str]
    correlation_coefficient: float
    time_window: int  # minutes
    price_changes: dict[str, float]


@dataclass
class _PriceEntry:
    """Internal timestamped price observation."""

    price: float
    timestamp: datetime


class CorrelationDetector:
    """Detects synchronised price moves across multiple markets.

    Maintains a rolling buffer of price observations per market and, on
    demand, computes pairwise Pearson correlations over a configurable
    window.

    Args:
        history_window: How far back to retain price observations.
            Defaults to 2 hours so that the default 60-minute correlation
            window always has enough runway.
        correlation_threshold: Minimum absolute Pearson *r* to flag a pair.
    """

    def __init__(
        self,
        history_window: timedelta = timedelta(hours=2),
        correlation_threshold: float = _CORRELATION_THRESHOLD,
    ) -> None:
        self._history_window = history_window
        self._correlation_threshold = correlation_threshold
        self._prices: dict[str, deque[_PriceEntry]] = defaultdict(deque)

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def update(self, market_id: str, price: float, timestamp: datetime | None = None) -> None:
        """Record a price observation for a market.

        Args:
            market_id: Polymarket market identifier.
            price: The observed price.
            timestamp: Observation time.  Defaults to ``datetime.now(timezone.utc)``.
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
        self._prices[market_id].append(_PriceEntry(price=price, timestamp=timestamp))
        self._evict(market_id, timestamp)

    def find_correlated_moves(
        self,
        window_minutes: int = 60,
    ) -> list[CorrelatedMove]:
        """Scan all tracked markets for correlated price moves.

        For every pair of markets with sufficient overlapping observations
        within the window, the Pearson correlation of their price-change
        series is computed.  Pairs exceeding the threshold are returned.

        Connected pairs are *not* merged into larger groups -- each pair
        is reported individually for maximum transparency.

        Args:
            window_minutes: The lookback window in minutes.

        Returns:
            A list of :class:`CorrelatedMove` instances (may be empty).
        """
        window = timedelta(minutes=window_minutes)
        now = datetime.now(timezone.utc)
        cutoff = now - window

        # Build per-market change series within the window.
        changes: dict[str, list[tuple[datetime, float]]] = {}
        for mid, entries in self._prices.items():
            series = self._price_changes_in_window(entries, cutoff)
            if len(series) >= _MIN_OVERLAP:
                changes[mid] = series

        if len(changes) < 2:
            return []

        results: list[CorrelatedMove] = []
        market_ids = list(changes.keys())

        for mid_a, mid_b in itertools.combinations(market_ids, 2):
            series_a = changes[mid_a]
            series_b = changes[mid_b]

            aligned_a, aligned_b = self._align_series(series_a, series_b)
            if len(aligned_a) < _MIN_OVERLAP:
                continue

            corr = self._pearson(aligned_a, aligned_b)
            if abs(corr) >= self._correlation_threshold:
                # Compute cumulative price changes for reporting.
                cum_a = sum(v for _, v in series_a)
                cum_b = sum(v for _, v in series_b)
                results.append(
                    CorrelatedMove(
                        market_ids=[mid_a, mid_b],
                        correlation_coefficient=corr,
                        time_window=window_minutes,
                        price_changes={mid_a: cum_a, mid_b: cum_b},
                    )
                )
                logger.info(
                    "Correlated move: %s <-> %s  r=%.3f  window=%dmin",
                    mid_a,
                    mid_b,
                    corr,
                    window_minutes,
                )

        return results

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def tracked_markets(self) -> list[str]:
        """Market IDs currently being tracked."""
        return list(self._prices.keys())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _evict(self, market_id: str, reference: datetime) -> None:
        """Remove entries older than ``reference - history_window``."""
        cutoff = reference - self._history_window
        buf = self._prices[market_id]
        while buf and buf[0].timestamp < cutoff:
            buf.popleft()

    @staticmethod
    def _price_changes_in_window(
        entries: deque[_PriceEntry],
        cutoff: datetime,
    ) -> list[tuple[datetime, float]]:
        """Extract (timestamp, price_change) pairs since *cutoff*.

        Returns consecutive differences for entries after the cutoff.
        """
        filtered = [e for e in entries if e.timestamp >= cutoff]
        if len(filtered) < 2:
            return []
        return [
            (filtered[i].timestamp, filtered[i].price - filtered[i - 1].price)
            for i in range(1, len(filtered))
        ]

    @staticmethod
    def _align_series(
        series_a: list[tuple[datetime, float]],
        series_b: list[tuple[datetime, float]],
        tolerance: timedelta = timedelta(seconds=120),
    ) -> tuple[list[float], list[float]]:
        """Align two timestamped change series by nearest-timestamp matching.

        For each entry in *series_a*, the closest entry in *series_b*
        (within *tolerance*) is matched.  Unmatched entries are dropped.

        Args:
            series_a: First change series.
            series_b: Second change series.
            tolerance: Maximum time difference for a match.

        Returns:
            Two aligned lists of change values.
        """
        aligned_a: list[float] = []
        aligned_b: list[float] = []
        b_idx = 0

        for ts_a, val_a in series_a:
            # Advance b_idx to the nearest entry.
            best_diff: timedelta | None = None
            best_val: float | None = None
            best_j = b_idx

            j = b_idx
            while j < len(series_b):
                ts_b, val_b = series_b[j]
                diff = abs(ts_a - ts_b)
                if diff > tolerance and ts_b > ts_a:
                    break
                if best_diff is None or diff < best_diff:
                    best_diff = diff
                    best_val = val_b
                    best_j = j
                j += 1

            if best_diff is not None and best_diff <= tolerance and best_val is not None:
                aligned_a.append(val_a)
                aligned_b.append(best_val)
                b_idx = best_j + 1

        return aligned_a, aligned_b

    @staticmethod
    def _pearson(x: list[float], y: list[float]) -> float:
        """Compute Pearson correlation coefficient between two aligned lists.

        Uses numpy for numerical stability.

        Returns:
            The Pearson *r* coefficient, or 0.0 if computation fails.
        """
        if len(x) < 2 or len(x) != len(y):
            return 0.0
        try:
            arr_x = np.array(x)
            arr_y = np.array(y)
            # Degenerate case: constant series.
            if np.std(arr_x) < 1e-12 or np.std(arr_y) < 1e-12:
                return 0.0
            corr_matrix = np.corrcoef(arr_x, arr_y)
            return float(corr_matrix[0, 1])
        except (ValueError, FloatingPointError):
            return 0.0

    def __repr__(self) -> str:
        return (
            f"CorrelationDetector(markets={len(self._prices)}, "
            f"threshold={self._correlation_threshold})"
        )
