"""Phase 3 analysis engine -- statistical anomaly detection and correlation.

This package provides the core analysis primitives for the Polymarket
surveillance system:

- **timeseries** -- Rolling-window statistics and EWMA primitives.
- **volume_analyzer** -- Volume spike anomaly detection.
- **price_analyzer** -- Price move anomaly detection.
- **changepoint** -- CUSUM change-point detection over logit-returns (Phase 2).
- **correlation** -- Cross-market correlation detection.
- **wallet_profiler** -- Per-wallet anomaly feature extraction (Phase 2).

Order-book depth and holder concentration analysis (susceptibility
scoring) lives in ``agents.manipulation_guard.LiquidityAnalyzer`` -- the
live implementation used by ManipulationGuard, not a module here.
"""

from prediction_market.analysis.changepoint import CusumAlarm, CusumDetector
from prediction_market.analysis.correlation import CorrelatedMove, CorrelationDetector
from prediction_market.analysis.price_analyzer import PriceAnalyzer, PriceAnomaly
from prediction_market.analysis.timeseries import (
    EWMA,
    RollingStats,
    clamp_probability,
    compute_z_score,
    logit,
)
from prediction_market.analysis.volume_analyzer import VolumeAnalyzer, VolumeAnomaly
from prediction_market.analysis.wallet_profiler import WalletFeatures, profile_wallets

__all__ = [
    "EWMA",
    "CorrelatedMove",
    "CorrelationDetector",
    "CusumAlarm",
    "CusumDetector",
    "PriceAnalyzer",
    "PriceAnomaly",
    "RollingStats",
    "VolumeAnalyzer",
    "VolumeAnomaly",
    "WalletFeatures",
    "clamp_probability",
    "compute_z_score",
    "logit",
    "profile_wallets",
]
