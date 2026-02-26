"""Unit tests for cross-market correlation detection."""

from datetime import datetime, timedelta, timezone


from prediction_market.analysis.correlation import CorrelationDetector


def _ts(minutes_ago: int) -> datetime:
    """Utility: return a timestamp `minutes_ago` minutes before a reference."""
    return datetime(2026, 2, 20, 12, 0, 0, tzinfo=timezone.utc) - timedelta(minutes=minutes_ago)


def test_update_tracks_markets():
    detector = CorrelationDetector()
    detector.update("m1", 0.50, _ts(10))
    detector.update("m2", 0.60, _ts(10))
    assert sorted(detector.tracked_markets) == ["m1", "m2"]


def test_find_correlated_moves_insufficient_data():
    detector = CorrelationDetector()
    # Only 2 observations per market -- below MIN_OVERLAP (5)
    detector.update("m1", 0.50, _ts(10))
    detector.update("m1", 0.55, _ts(5))
    detector.update("m2", 0.60, _ts(10))
    detector.update("m2", 0.65, _ts(5))
    results = detector.find_correlated_moves(window_minutes=15)
    assert results == []


def test_find_correlated_moves_single_market():
    detector = CorrelationDetector()
    for i in range(10):
        detector.update("m1", 0.50 + i * 0.01, _ts(10 - i))
    # Only one market -- nothing to correlate
    results = detector.find_correlated_moves()
    assert results == []


def test_find_correlated_perfectly_correlated():
    detector = CorrelationDetector()
    # Two markets with correlated but varying price changes
    now = datetime.now(timezone.utc)
    # Use non-constant deltas so the change series has nonzero variance
    prices_m1 = [0.50, 0.52, 0.51, 0.54, 0.53, 0.56, 0.55, 0.58, 0.57, 0.60]
    prices_m2 = [0.60, 0.62, 0.61, 0.64, 0.63, 0.66, 0.65, 0.68, 0.67, 0.70]
    for i in range(10):
        ts = now - timedelta(minutes=10 - i)
        detector.update("m1", prices_m1[i], ts)
        detector.update("m2", prices_m2[i], ts)

    results = detector.find_correlated_moves(window_minutes=15)
    assert len(results) == 1
    assert results[0].correlation_coefficient > 0.95
    assert sorted(results[0].market_ids) == ["m1", "m2"]


def test_find_correlated_uncorrelated():
    detector = CorrelationDetector()
    now = datetime.now(timezone.utc)
    # m1 goes up, m2 oscillates randomly-ish
    prices_m2 = [0.60, 0.55, 0.65, 0.50, 0.70, 0.45, 0.62, 0.58, 0.52, 0.68]
    for i in range(10):
        ts = now - timedelta(minutes=10 - i)
        detector.update("m1", 0.50 + i * 0.01, ts)
        detector.update("m2", prices_m2[i], ts)

    results = detector.find_correlated_moves(window_minutes=15)
    # Should not detect correlation (oscillating vs monotonic)
    assert len(results) == 0


def test_find_correlated_negatively_correlated():
    detector = CorrelationDetector()
    now = datetime.now(timezone.utc)
    # Varying changes that are mirror-images of each other
    prices_m1 = [0.50, 0.52, 0.51, 0.54, 0.53, 0.56, 0.55, 0.58, 0.57, 0.60]
    prices_m2 = [0.70, 0.68, 0.69, 0.66, 0.67, 0.64, 0.65, 0.62, 0.63, 0.60]
    for i in range(10):
        ts = now - timedelta(minutes=10 - i)
        detector.update("m1", prices_m1[i], ts)
        detector.update("m2", prices_m2[i], ts)

    results = detector.find_correlated_moves(window_minutes=15)
    # Negative correlation should also be detected (abs >= threshold)
    assert len(results) == 1
    assert results[0].correlation_coefficient < -0.95


def test_eviction_removes_old_data():
    detector = CorrelationDetector(history_window=timedelta(minutes=5))
    now = datetime.now(timezone.utc)
    # Add old data
    for i in range(10):
        ts = now - timedelta(minutes=60 - i)
        detector.update("m1", 0.50 + i * 0.01, ts)

    # Add recent data (triggers eviction)
    detector.update("m1", 0.65, now)

    # Only the recent observation should remain
    assert len(detector._prices["m1"]) <= 2


def test_identical_price_series():
    """Constant prices should yield 0 correlation (degenerate case)."""
    detector = CorrelationDetector()
    now = datetime.now(timezone.utc)
    for i in range(10):
        ts = now - timedelta(minutes=10 - i)
        detector.update("m1", 0.50, ts)  # constant
        detector.update("m2", 0.60, ts)  # constant

    results = detector.find_correlated_moves(window_minutes=15)
    assert results == []  # std=0 → correlation returns 0.0


def test_correlated_move_price_changes():
    detector = CorrelationDetector()
    now = datetime.now(timezone.utc)
    # Varying deltas so change series has nonzero variance
    prices_m1 = [0.50, 0.54, 0.52, 0.58, 0.56, 0.62, 0.60, 0.66, 0.64, 0.70]
    prices_m2 = [0.30, 0.36, 0.33, 0.42, 0.39, 0.48, 0.45, 0.54, 0.51, 0.60]
    for i in range(10):
        ts = now - timedelta(minutes=10 - i)
        detector.update("m1", prices_m1[i], ts)
        detector.update("m2", prices_m2[i], ts)

    results = detector.find_correlated_moves(window_minutes=15)
    assert len(results) == 1
    assert "m1" in results[0].price_changes
    assert "m2" in results[0].price_changes
    assert results[0].time_window == 15
