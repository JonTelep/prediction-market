"""Tests for price move anomaly detection."""

from datetime import datetime, timedelta, timezone


from prediction_market.analysis.price_analyzer import PriceAnalyzer
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
        assert data is not None
