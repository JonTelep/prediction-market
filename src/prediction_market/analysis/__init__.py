"""Phase 3 analysis engine -- statistical anomaly detection and correlation.

This package provides the core analysis primitives for the Polymarket
surveillance system:

- **timeseries** -- Rolling-window statistics and EWMA primitives.
- **volume_analyzer** -- Volume spike anomaly detection.
- **price_analyzer** -- Price move anomaly detection.
- **correlation** -- Cross-market correlation detection.

Order-book depth and holder concentration analysis (susceptibility
scoring) lives in ``agents.manipulation_guard.LiquidityAnalyzer`` -- the
live implementation used by ManipulationGuard, not a module here.
"""

from prediction_market.analysis.correlation import CorrelatedMove, CorrelationDetector
from prediction_market.analysis.price_analyzer import PriceAnalyzer, PriceAnomaly
from prediction_market.analysis.timeseries import EWMA, RollingStats, compute_z_score
from prediction_market.analysis.volume_analyzer import VolumeAnalyzer, VolumeAnomaly

__all__ = [
    "EWMA",
    "CorrelatedMove",
    "CorrelationDetector",
    "PriceAnalyzer",
    "PriceAnomaly",
    "RollingStats",
    "VolumeAnalyzer",
    "VolumeAnomaly",
    "compute_z_score",
]
