"""Schema-related helpers and constants for the database store."""

from __future__ import annotations

# Table names as constants
MARKETS_TABLE = "markets"
SNAPSHOTS_TABLE = "snapshots"
ORDERBOOK_SNAPSHOTS_TABLE = "orderbook_snapshots"
TRADES_TABLE = "trades"
SCHEDULED_EVENTS_TABLE = "scheduled_events"
ANOMALY_REPORTS_TABLE = "anomaly_reports"
ROLLING_STATS_TABLE = "rolling_stats"

# Severity levels
SEVERITY_LOW = "low"
SEVERITY_MEDIUM = "medium"
SEVERITY_HIGH = "high"
SEVERITY_CRITICAL = "critical"

SEVERITY_ORDER = {
    SEVERITY_LOW: 0,
    SEVERITY_MEDIUM: 1,
    SEVERITY_HIGH: 2,
    SEVERITY_CRITICAL: 3,
}
