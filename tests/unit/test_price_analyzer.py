"""Tests for price move anomaly detection."""

from datetime import datetime, timedelta, timezone

import pytest

from prediction_market.analysis.price_analyzer import PriceAnalyzer, _clamp, _logit
from prediction_market.config import ThresholdConfig


class TestPriceAnalyzer:
    def test_no_anomaly_stable_price(self):
        pa = PriceAnalyzer(thresholds=ThresholdConfig(price_zscore=2.5))
        now = datetime.now(timezone.utc)
        for i in range(168):
            pa.update("m1", 0.65, now + timedelta(hours=i))
        anomaly = pa.check_anomaly("m1")
        assert anomaly is None

    def test_anomaly_on_spike(self):
        pa = PriceAnalyzer(thresholds=ThresholdConfig(price_zscore=2.5))
        now = datetime.now(timezone.utc)
        for i in range(168):
            pa.update("m1", 0.65, now + timedelta(hours=i))
        pa.update("m1", 0.85, now + timedelta(hours=169))
        anomaly = pa.check_anomaly("m1")
        assert anomaly is not None
        assert abs(anomaly.z_score) > 2.5

    def test_anomaly_on_crash(self):
        pa = PriceAnalyzer(thresholds=ThresholdConfig(price_zscore=2.5))
        now = datetime.now(timezone.utc)
        for i in range(168):
            pa.update("m1", 0.65, now + timedelta(hours=i))
        pa.update("m1", 0.35, now + timedelta(hours=169))
        anomaly = pa.check_anomaly("m1")
        assert anomaly is not None
        assert abs(anomaly.z_score) > 2.5

    def test_unknown_market(self):
        pa = PriceAnalyzer()
        assert pa.check_anomaly("unknown") is None

    def test_serialization(self):
        pa = PriceAnalyzer()
        now = datetime.now(timezone.utc)
        for i in range(10):
            pa.update("m1", 0.65 + i * 0.001, now + timedelta(hours=i))
        data = pa.to_dict()
        restored = PriceAnalyzer.from_dict(data)

        assert restored.tracked_markets == pa.tracked_markets
        assert restored.current_z_score("m1") == pytest.approx(pa.current_z_score("m1"))


class TestLogitHelpers:
    def test_logit_of_half_is_zero(self):
        assert _logit(0.5) == 0.0

    def test_logit_at_clamp_bound(self):
        # ln(0.995 / 0.005) = ln(199) = 5.293304824724491
        assert _logit(0.995) == pytest.approx(5.2933048)

    def test_clamp_caps_resolved_market_price(self):
        assert _clamp(1.0) == 0.995
        assert _clamp(0.0) == 0.005

    def test_resolved_market_prices_produce_finite_returns(self):
        """A market that snaps to 0.0/1.0 on resolution must not blow up."""
        pa = PriceAnalyzer(thresholds=ThresholdConfig(price_zscore=2.5))
        now = datetime.now(timezone.utc)
        prices = [0.60, 0.62, 0.58, 0.0, 1.0, 1.0, 0.0]
        for i, p in enumerate(prices):
            pa.update("m1", p, now + timedelta(minutes=i))
        # No exception raised above; every stored return must be finite.
        for v in pa._states["m1"].return_stats.values:
            assert v == pytest.approx(v)  # NaN check: NaN != NaN
            assert abs(v) < float("inf")


class TestLogitBoundaryCompression:
    def test_equal_absolute_move_scores_larger_near_boundary(self):
        """The core rationale for the logit switch: identical +0.02 absolute
        jumps must produce a larger logit-return near a probability boundary
        (0.95 -> 0.97) than in the middle of the range (0.50 -> 0.52).

        Hand-derived deltas (ln(p/(1-p)) applied by hand):
          logit(0.52) - logit(0.50) = 0.0800427077  (~0.080, per spec)
          logit(0.97) - logit(0.95) = 0.5316597107  (~0.532, per spec)

        Under the OLD log-return formula (log(price) - log(last_price)) this
        would invert: log(0.52) - log(0.50) = 0.0392207132, while
        log(0.97) - log(0.95) = 0.0208340869 -- the boundary jump would score
        *smaller* than the mid-range jump, the exact mis-scaling this prompt
        fixes. (Verified by inspection of the old formula, not executed --
        the old code path no longer exists in this file.)
        """
        now = datetime.now(timezone.utc)

        mid = PriceAnalyzer(thresholds=ThresholdConfig(price_zscore=2.5))
        mid_prices = [0.50, 0.501, 0.499, 0.500, 0.52]
        for i, p in enumerate(mid_prices):
            mid.update("m1", p, now + timedelta(minutes=i))
        mid_return = mid._states["m1"].return_stats.latest

        boundary = PriceAnalyzer(thresholds=ThresholdConfig(price_zscore=2.5))
        boundary_prices = [0.95, 0.951, 0.949, 0.950, 0.97]
        for i, p in enumerate(boundary_prices):
            boundary.update("m1", p, now + timedelta(minutes=i))
        boundary_return = boundary._states["m1"].return_stats.latest

        assert mid_return == pytest.approx(0.0800427077, abs=1e-6)
        assert boundary_return == pytest.approx(0.5316597107, abs=1e-6)
        assert boundary_return > mid_return
