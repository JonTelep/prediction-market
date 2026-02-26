"""Phase 3 analysis engine -- statistical anomaly detection and correlation.

This package provides the core analysis primitives for the Polymarket
surveillance system:

- **timeseries** -- Rolling-window statistics and EWMA primitives.
- **volume_analyzer** -- Volume spike anomaly detection.
- **price_analyzer** -- Price move anomaly detection.
- **liquidity_analyzer** -- Order-book depth and holder concentration analysis.
- **correlation** -- Cross-market correlation detection.
"""

from prediction_market.analysis.correlation import CorrelatedMove, CorrelationDetector
from prediction_market.analysis.liquidity_analyzer import LiquidityAnalyzer, LiquidityMetrics
from prediction_market.analysis.price_analyzer import PriceAnalyzer, PriceAnomaly
from prediction_market.analysis.timeseries import EWMA, RollingStats, compute_z_score
from prediction_market.analysis.volume_analyzer import VolumeAnalyzer, VolumeAnomaly

__all__ = [
    "EWMA",
    "CorrelatedMove",
    "CorrelationDetector",
    "LiquidityAnalyzer",
    "LiquidityMetrics",
    "PriceAnalyzer",
    "PriceAnomaly",
    "RollingStats",
    "VolumeAnalyzer",
    "VolumeAnomaly",
    "compute_z_score",
]
