"""Tests for timeseries analysis primitives."""

from datetime import datetime, timedelta, timezone

import pytest

from prediction_market.analysis.timeseries import EWMA, RollingStats, compute_z_score


class TestComputeZScore:
    def test_basic(self):
        assert compute_z_score(10, 5, 2) == pytest.approx(2.5)

    def test_zero_std(self):
        assert compute_z_score(10, 5, 0) == 0.0

    def test_negative(self):
        assert compute_z_score(3, 5, 2) == pytest.approx(-1.0)


class TestRollingStats:
    def test_add_and_mean(self):
        rs = RollingStats(window=timedelta(days=7))
        now = datetime.now(timezone.utc)
        for i in range(10):
            rs.add(float(i), now + timedelta(seconds=i))
        assert rs.mean == pytest.approx(4.5)

    def test_std(self):
        rs = RollingStats(window=timedelta(days=7))
        now = datetime.now(timezone.utc)
        for v in [10, 10, 10, 10]:
            rs.add(v, now)
        assert rs.std == pytest.approx(0.0)

    def test_z_score(self):
        rs = RollingStats(window=timedelta(days=7))
        now = datetime.now(timezone.utc)
        # Add values with some variance so std > 0
        for i in range(50):
            rs.add(10.0 + (i % 5) * 0.1, now + timedelta(seconds=i))
        z = rs.z_score(20.0)
        assert z > 0

    def test_window_expiry(self):
        rs = RollingStats(window=timedelta(seconds=5))
        now = datetime.now(timezone.utc)
        for i in range(10):
            rs.add(float(i), now + timedelta(seconds=i))
        assert rs.count <= 6  # Only values within 5s window

    def test_serialization(self):
        rs = RollingStats(window=timedelta(days=7))
        now = datetime.now(timezone.utc)
        for i in range(5):
            rs.add(float(i), now + timedelta(seconds=i))
        data = rs.to_dict()
        rs2 = RollingStats.from_dict(data)
        assert rs2.mean == pytest.approx(rs.mean)
        assert rs2.count == rs.count

    def test_empty_stats(self):
        rs = RollingStats(window=timedelta(days=7))
        assert rs.mean == 0.0
        assert rs.std == 0.0
        assert rs.z_score(5.0) == 0.0


class TestEWMA:
    def test_converges_to_constant(self):
        ewma = EWMA(span=10)
        for _ in range(100):
            ewma.update(5.0)
        assert ewma.value == pytest.approx(5.0, abs=0.01)

    def test_tracks_trend(self):
        ewma = EWMA(span=5)
        for i in range(20):
            ewma.update(float(i))
        assert ewma.value > 10

    def test_z_score(self):
        ewma = EWMA(span=20)
        for _ in range(50):
            ewma.update(10.0)
        z = ewma.z_score(15.0)
        assert isinstance(z, float)

    def test_serialization(self):
        ewma = EWMA(span=10)
        for i in range(10):
            ewma.update(float(i))
        data = ewma.to_dict()
        ewma2 = EWMA.from_dict(data)
        assert ewma2.value == pytest.approx(ewma.value)
