"""Prebuilt analytical queries for market surveillance data."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

import aiosqlite

logger = logging.getLogger(__name__)


def _hours_ago(hours: int) -> str:
    """Return an ISO-8601 timestamp for `hours` ago in UTC.

    SQLite stores timestamps as text, so we compare as strings in
    'YYYY-MM-DD HH:MM:SS' format which sorts lexicographically.
    """
    from datetime import timedelta

    dt = datetime.now(timezone.utc) - timedelta(hours=hours)
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _utcnow() -> str:
    """Return current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


async def _fetch_all_dicts(
    db: aiosqlite.Connection,
    sql: str,
    params: tuple[Any, ...] = (),
) -> list[dict[str, Any]]:
    """Execute a query and return results as a list of dicts."""
    db.row_factory = aiosqlite.Row
    cursor = await db.execute(sql, params)
    rows = await cursor.fetchall()
    return [dict(row) for row in rows]


async def _fetch_one_dict(
    db: aiosqlite.Connection,
    sql: str,
    params: tuple[Any, ...] = (),
) -> dict[str, Any] | None:
    """Execute a query and return one result as a dict, or None."""
    db.row_factory = aiosqlite.Row
    cursor = await db.execute(sql, params)
    row = await cursor.fetchone()
    if row is None:
        return None
    return dict(row)


# ---------------------------------------------------------------------------
# Price & Volume Snapshots
# ---------------------------------------------------------------------------


async def get_recent_snapshots(
    db: aiosqlite.Connection,
    market_id: str,
    hours: int = 24,
) -> list[dict[str, Any]]:
    """Get recent price/volume snapshots for a market.

    Args:
        db: An open aiosqlite connection.
        market_id: The market to query.
        hours: How many hours of history to retrieve.

    Returns:
        List of snapshot dicts ordered by timestamp ascending.
    """
    cutoff = _hours_ago(hours)
    return await _fetch_all_dicts(
        db,
        """
        SELECT id, market_id, timestamp, price_yes, price_no,
               volume_24hr, volume_total, liquidity, num_trades_1hr
        FROM snapshots
        WHERE market_id = ? AND timestamp >= ?
        ORDER BY timestamp ASC
        """,
        (market_id, cutoff),
    )


async def get_price_history(
    db: aiosqlite.Connection,
    market_id: str,
    hours: int = 24,
) -> list[dict[str, Any]]:
    """Get price time series for a market.

    Returns only timestamp and price columns for lightweight charting.

    Args:
        db: An open aiosqlite connection.
        market_id: The market to query.
        hours: How many hours of history to retrieve.

    Returns:
        List of dicts with timestamp, price_yes, price_no.
    """
    cutoff = _hours_ago(hours)
    return await _fetch_all_dicts(
        db,
        """
        SELECT timestamp, price_yes, price_no
        FROM snapshots
        WHERE market_id = ? AND timestamp >= ?
        ORDER BY timestamp ASC
        """,
        (market_id, cutoff),
    )


async def get_volume_history(
    db: aiosqlite.Connection,
    market_id: str,
    hours: int = 24,
) -> list[dict[str, Any]]:
    """Get volume time series for a market.

    Args:
        db: An open aiosqlite connection.
        market_id: The market to query.
        hours: How many hours of history to retrieve.

    Returns:
        List of dicts with timestamp, volume_24hr, volume_total, liquidity.
    """
    cutoff = _hours_ago(hours)
    return await _fetch_all_dicts(
        db,
        """
        SELECT timestamp, volume_24hr, volume_total, liquidity
        FROM snapshots
        WHERE market_id = ? AND timestamp >= ?
        ORDER BY timestamp ASC
        """,
        (market_id, cutoff),
    )


# ---------------------------------------------------------------------------
# Order Book Snapshots
# ---------------------------------------------------------------------------


async def get_recent_orderbooks(
    db: aiosqlite.Connection,
    market_id: str,
    token_id: str,
    hours: int = 24,
) -> list[dict[str, Any]]:
    """Get recent order book snapshots for a market and token.

    Args:
        db: An open aiosqlite connection.
        market_id: The market to query.
        token_id: The CLOB token ID for a specific outcome.
        hours: How many hours of history to retrieve.

    Returns:
        List of orderbook snapshot dicts ordered by timestamp ascending.
    """
    cutoff = _hours_ago(hours)
    return await _fetch_all_dicts(
        db,
        """
        SELECT id, market_id, token_id, timestamp,
               best_bid, best_ask, midpoint, spread, spread_pct,
               total_bid_depth, total_ask_depth,
               depth_1pct, depth_5pct, depth_10pct,
               imbalance, hhi, susceptibility_score
        FROM orderbook_snapshots
        WHERE market_id = ? AND token_id = ? AND timestamp >= ?
        ORDER BY timestamp ASC
        """,
        (market_id, token_id, cutoff),
    )


# ---------------------------------------------------------------------------
# Trades
# ---------------------------------------------------------------------------


async def get_market_trades(
    db: aiosqlite.Connection,
    market_id: str,
    hours: int = 24,
) -> list[dict[str, Any]]:
    """Get recent trades for a market.

    Args:
        db: An open aiosqlite connection.
        market_id: The market to query.
        hours: How many hours of trade history.

    Returns:
        List of trade dicts ordered by match_time ascending.
    """
    cutoff = _hours_ago(hours)
    return await _fetch_all_dicts(
        db,
        """
        SELECT id, market_id, asset_id, side, size, price,
               volume_usd, outcome, owner, match_time, transaction_hash
        FROM trades
        WHERE market_id = ? AND match_time >= ?
        ORDER BY match_time ASC
        """,
        (market_id, cutoff),
    )


# ---------------------------------------------------------------------------
# Anomaly Reports
# ---------------------------------------------------------------------------


async def get_anomaly_reports(
    db: aiosqlite.Connection,
    severity: str | None = None,
    agent: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Query anomaly reports with optional filters.

    Args:
        db: An open aiosqlite connection.
        severity: Filter by severity level ('low', 'medium', 'high', 'critical').
        agent: Filter by detecting agent ('info_leak', 'manipulation').
        limit: Maximum number of reports to return.

    Returns:
        List of anomaly report dicts ordered by creation time descending.
    """
    conditions: list[str] = []
    params: list[Any] = []

    if severity is not None:
        conditions.append("severity = ?")
        params.append(severity)
    if agent is not None:
        conditions.append("agent = ?")
        params.append(agent)

    where_clause = ""
    if conditions:
        where_clause = "WHERE " + " AND ".join(conditions)

    params.append(limit)

    return await _fetch_all_dicts(
        db,
        f"""
        SELECT id, agent, market_id, severity, anomaly_score, confidence,
               summary, details, price_evidence, volume_evidence,
               calendar_matches, news_check, created_at
        FROM anomaly_reports
        {where_clause}
        ORDER BY created_at DESC
        LIMIT ?
        """,
        tuple(params),
    )


async def save_anomaly_report(
    db: aiosqlite.Connection,
    report_dict: dict[str, Any],
) -> int:
    """Save a new anomaly report and return its ID.

    Args:
        db: An open aiosqlite connection.
        report_dict: Dict with keys matching anomaly_reports columns.
            Required: agent, market_id, summary.
            Optional: severity, anomaly_score, confidence, details,
                price_evidence, volume_evidence, calendar_matches, news_check.

    Returns:
        The auto-generated row ID of the inserted report.
    """
    now = _utcnow()

    cursor = await db.execute(
        """
        INSERT INTO anomaly_reports (
            agent, market_id, severity, anomaly_score, confidence,
            summary, details, price_evidence, volume_evidence,
            calendar_matches, news_check, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            report_dict["agent"],
            report_dict["market_id"],
            report_dict.get("severity", "medium"),
            report_dict.get("anomaly_score", 0.0),
            report_dict.get("confidence", 0.0),
            report_dict["summary"],
            json.dumps(report_dict.get("details", {})),
            json.dumps(report_dict.get("price_evidence", {})),
            json.dumps(report_dict.get("volume_evidence", {})),
            json.dumps(report_dict.get("calendar_matches", [])),
            json.dumps(report_dict.get("news_check", {})),
            now,
        ),
    )
    await db.commit()

    report_id = cursor.lastrowid or 0
    logger.info(
        "Saved anomaly report #%d: agent=%s market=%s severity=%s",
        report_id,
        report_dict["agent"],
        report_dict["market_id"],
        report_dict.get("severity", "medium"),
    )
    return report_id


# ---------------------------------------------------------------------------
# Rolling Statistics
# ---------------------------------------------------------------------------


async def get_rolling_stats(
    db: aiosqlite.Connection,
    market_id: str,
    stat_type: str,
) -> dict[str, Any] | None:
    """Retrieve the latest rolling statistics for a market.

    Args:
        db: An open aiosqlite connection.
        market_id: The market to query.
        stat_type: The type of statistic ('price', 'volume', 'depth').

    Returns:
        Dict with serialized_data (parsed from JSON), window_days,
        and updated_at, or None if no stats exist.
    """
    row = await _fetch_one_dict(
        db,
        """
        SELECT id, market_id, stat_type, window_days,
               serialized_data, updated_at
        FROM rolling_stats
        WHERE market_id = ? AND stat_type = ?
        ORDER BY updated_at DESC
        LIMIT 1
        """,
        (market_id, stat_type),
    )
    if row is None:
        return None

    # Parse the JSON blob back into a dict
    try:
        row["serialized_data"] = json.loads(row["serialized_data"])
    except (json.JSONDecodeError, TypeError):
        row["serialized_data"] = {}

    return row


async def save_rolling_stats(
    db: aiosqlite.Connection,
    market_id: str,
    stat_type: str,
    window_days: int,
    data: dict[str, Any],
) -> None:
    """Upsert rolling statistics for a market.

    Args:
        db: An open aiosqlite connection.
        market_id: The market to store stats for.
        stat_type: The type of statistic ('price', 'volume', 'depth').
        window_days: The rolling window size in days.
        data: Arbitrary stats data to serialize as JSON.
    """
    now = _utcnow()
    serialized = json.dumps(data)

    await db.execute(
        """
        INSERT INTO rolling_stats (
            market_id, stat_type, window_days, serialized_data, updated_at
        ) VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(market_id, stat_type, window_days) DO UPDATE SET
            serialized_data = excluded.serialized_data,
            updated_at = excluded.updated_at
        """,
        (market_id, stat_type, window_days, serialized, now),
    )
    await db.commit()
    logger.debug(
        "Saved rolling stats for market %s type=%s window=%dd",
        market_id,
        stat_type,
        window_days,
    )
