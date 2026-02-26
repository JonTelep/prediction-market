"""Tests for volume anomaly detection."""

from datetime import datetime, timedelta, timezone


from prediction_market.analysis.volume_analyzer import VolumeAnalyzer
from prediction_market.config import ThresholdConfig


class TestVolumeAnalyzer:
    def test_no_anomaly_with_stable_volume(self):
        va = VolumeAnalyzer(thresholds=ThresholdConfig(volume_zscore=3.0))
        now = datetime.now(timezone.utc)
        for i in range(168):
            va.update("m1", 1000.0, now + timedelta(hours=i))
        anomaly = va.check_anomaly("m1")
        assert anomaly is None

    def test_anomaly_with_spike(self):
        va = VolumeAnalyzer(thresholds=ThresholdConfig(volume_zscore=3.0))
        now = datetime.now(timezone.utc)
        for i in range(168):
            va.update("m1", 1000.0, now + timedelta(hours=i))
        va.update("m1", 50000.0, now + timedelta(hours=169))
        anomaly = va.check_anomaly("m1")
        assert anomaly is not None
        assert anomaly.z_score > 3.0
        assert anomaly.market_id == "m1"

    def test_unknown_market_returns_none(self):
        va = VolumeAnalyzer()
        assert va.check_anomaly("unknown") is None

    def test_serialization(self):
        va = VolumeAnalyzer()
        now = datetime.now(timezone.utc)
        for i in range(10):
            va.update("m1", 1000.0 + i, now + timedelta(hours=i))
        data = va.to_dict()
        assert data is not None
        va2 = VolumeAnalyzer.from_dict(data)
        assert va2.check_anomaly("m1") is None
