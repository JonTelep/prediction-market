"""Periodic snapshot writers for markets, prices, order books, and trades."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

import aiosqlite

from prediction_market.data.polymarket.models import GammaMarket, OrderBook, Trade

logger = logging.getLogger(__name__)


def _utcnow() -> str:
    """Return current UTC time as an ISO-8601 string for SQLite."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


async def save_market(
    db: aiosqlite.Connection,
    market: GammaMarket,
    political_classification: dict | None = None,
) -> None:
    """Upsert a market record from Gamma API data.

    Inserts a new row if the market doesn't exist, or updates metadata
    and volume/liquidity figures if it does.

    Args:
        db: An open aiosqlite connection.
        market: The GammaMarket model from the Gamma API.
        political_classification: Optional dict with 'confidence' (float)
            and 'reasons' (list[str]) from the political classifier.
    """
    confidence = 0.0
    reasons: list[str] = []
    if political_classification:
        confidence = political_classification.get("confidence", 0.0)
        reasons = political_classification.get("reasons", [])

    tags_json = json.dumps(market.tag_labels)
    clob_ids_json = json.dumps(market.clob_token_ids)
    reasons_json = json.dumps(reasons)
    now = _utcnow()

    await db.execute(
        """
        INSERT INTO markets (
            id, question, description, category, tags, slug,
            condition_id, clob_token_ids, volume, liquidity,
            active, closed, political_confidence, political_reasons,
            created_at, end_date, first_seen, last_updated
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            question = excluded.question,
            description = excluded.description,
            category = excluded.category,
            tags = excluded.tags,
            slug = excluded.slug,
            condition_id = excluded.condition_id,
            clob_token_ids = excluded.clob_token_ids,
            volume = excluded.volume,
            liquidity = excluded.liquidity,
            active = excluded.active,
            closed = excluded.closed,
            political_confidence = excluded.political_confidence,
            political_reasons = excluded.political_reasons,
            end_date = excluded.end_date,
            last_updated = excluded.last_updated
        """,
        (
            market.id,
            market.question,
            market.description,
            market.category,
            tags_json,
            market.slug,
            market.condition_id,
            clob_ids_json,
            market.volume,
            market.liquidity,
            int(market.active),
            int(market.closed),
            confidence,
            reasons_json,
            market.created_at,
            market.end_date,
            now,
            now,
        ),
    )
    await db.commit()
    logger.debug("Upserted market %s: %s", market.id, market.question[:60])


async def save_price_snapshot(
    db: aiosqlite.Connection,
    market_id: str,
    price_yes: float | None = None,
    price_no: float | None = None,
    volume_24hr: float | None = None,
    volume_total: float | None = None,
    liquidity: float | None = None,
) -> None:
    """Save a point-in-time price/volume snapshot for a market.

    Args:
        db: An open aiosqlite connection.
        market_id: The market ID (foreign key to markets table).
        price_yes: Current YES outcome price (0-1).
        price_no: Current NO outcome price (0-1).
        volume_24hr: Trading volume in the last 24 hours (USD).
        volume_total: All-time total trading volume (USD).
        liquidity: Current liquidity available (USD).
    """
    now = _utcnow()

    await db.execute(
        """
        INSERT INTO snapshots (
            market_id, timestamp, price_yes, price_no,
            volume_24hr, volume_total, liquidity
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(market_id, timestamp) DO UPDATE SET
            price_yes = excluded.price_yes,
            price_no = excluded.price_no,
            volume_24hr = excluded.volume_24hr,
            volume_total = excluded.volume_total,
            liquidity = excluded.liquidity
        """,
        (market_id, now, price_yes, price_no, volume_24hr, volume_total, liquidity),
    )
    await db.commit()
    logger.debug("Saved price snapshot for market %s", market_id)


async def save_orderbook_snapshot(
    db: aiosqlite.Connection,
    market_id: str,
    token_id: str,
    orderbook: OrderBook,
    hhi: float | None = None,
    susceptibility: float | None = None,
) -> None:
    """Save an order book snapshot with derived metrics.

    Args:
        db: An open aiosqlite connection.
        market_id: The market ID (foreign key to markets table).
        token_id: The CLOB token ID for this outcome side.
        orderbook: The OrderBook model with bids/asks.
        hhi: Herfindahl-Hirschman Index measuring order concentration.
        susceptibility: Susceptibility score (0-1) indicating ease of
            price manipulation based on depth and concentration.
    """
    now = _utcnow()

    await db.execute(
        """
        INSERT INTO orderbook_snapshots (
            market_id, token_id, timestamp,
            best_bid, best_ask, midpoint, spread, spread_pct,
            total_bid_depth, total_ask_depth,
            depth_1pct, depth_5pct, depth_10pct,
            imbalance, hhi, susceptibility_score
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(market_id, token_id, timestamp) DO UPDATE SET
            best_bid = excluded.best_bid,
            best_ask = excluded.best_ask,
            midpoint = excluded.midpoint,
            spread = excluded.spread,
            spread_pct = excluded.spread_pct,
            total_bid_depth = excluded.total_bid_depth,
            total_ask_depth = excluded.total_ask_depth,
            depth_1pct = excluded.depth_1pct,
            depth_5pct = excluded.depth_5pct,
            depth_10pct = excluded.depth_10pct,
            imbalance = excluded.imbalance,
            hhi = excluded.hhi,
            susceptibility_score = excluded.susceptibility_score
        """,
        (
            market_id,
            token_id,
            now,
            orderbook.best_bid,
            orderbook.best_ask,
            orderbook.midpoint,
            orderbook.spread,
            orderbook.spread_pct,
            orderbook.total_bid_depth,
            orderbook.total_ask_depth,
            orderbook.depth_at_pct(0.01),
            orderbook.depth_at_pct(0.05),
            orderbook.depth_at_pct(0.10),
            orderbook.imbalance,
            hhi,
            susceptibility,
        ),
    )
    await db.commit()
    logger.debug("Saved orderbook snapshot for market %s token %s", market_id, token_id)


async def save_trade(
    db: aiosqlite.Connection,
    trade: Trade,
    market_id: str,
) -> None:
    """Save a single trade, ignoring duplicates.

    Uses INSERT OR IGNORE since trade IDs are unique primary keys,
    making this safe for replay or overlapping fetches.

    Args:
        db: An open aiosqlite connection.
        trade: The Trade model from the Data API.
        market_id: The market ID to associate the trade with.
    """
    await db.execute(
        """
        INSERT OR IGNORE INTO trades (
            id, market_id, asset_id, side, size, price,
            volume_usd, outcome, owner, match_time, transaction_hash
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            trade.id,
            market_id,
            trade.asset_id,
            trade.side,
            trade.size_float,
            trade.price_float,
            trade.volume_usd,
            trade.outcome,
            trade.owner,
            trade.match_time,
            trade.transaction_hash,
        ),
    )
    await db.commit()
    logger.debug("Saved trade %s for market %s", trade.id, market_id)


async def save_trades_batch(
    db: aiosqlite.Connection,
    trades: list[Trade],
    market_id: str,
) -> int:
    """Save a batch of trades efficiently, ignoring duplicates.

    Args:
        db: An open aiosqlite connection.
        trades: List of Trade models from the Data API.
        market_id: The market ID to associate trades with.

    Returns:
        Number of new trades inserted.
    """
    if not trades:
        return 0

    rows = [
        (
            t.id,
            market_id,
            t.asset_id,
            t.side,
            t.size_float,
            t.price_float,
            t.volume_usd,
            t.outcome,
            t.owner,
            t.match_time,
            t.transaction_hash,
        )
        for t in trades
    ]

    cursor = await db.executemany(
        """
        INSERT OR IGNORE INTO trades (
            id, market_id, asset_id, side, size, price,
            volume_usd, outcome, owner, match_time, transaction_hash
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    await db.commit()
    inserted = cursor.rowcount if cursor.rowcount >= 0 else len(trades)
    logger.debug("Batch inserted %d trades for market %s", inserted, market_id)
    return inserted
