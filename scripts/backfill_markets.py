#!/usr/bin/env python3
"""Backfill political markets with price history and trades from Polymarket APIs.

Discovers all political markets via the Gamma API, then fetches historical
price data (CLOB API) and trade history (Data API) for each, storing
everything in the local SQLite database.

Usage:
    python scripts/backfill_markets.py --days 30
    python scripts/backfill_markets.py --days 7 --config config/custom.toml
    python scripts/backfill_markets.py --days 30 --dry-run
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import click

from prediction_market.config import load_config, load_political_keywords
from prediction_market.data.polymarket.clob_client import ClobClient
from prediction_market.data.polymarket.data_client import DataClient
from prediction_market.data.polymarket.gamma_client import GammaClient
from prediction_market.data.polymarket.models import GammaMarket
from prediction_market.store.database import init_database
from prediction_market.store.snapshots import (
    save_market,
    save_price_snapshot,
    save_trades_batch,
)

logger = logging.getLogger(__name__)


def classify_political(
    market: GammaMarket,
    keywords_config: dict,
) -> dict | None:
    """Classify whether a market is political based on tags and title keywords.

    Returns a classification dict with confidence and reasons, or None
    if the market does not appear to be political.
    """
    classification = keywords_config.get("classification", {})
    political_tags = {t.lower() for t in classification.get("political_tags", [])}
    title_keywords = classification.get("title_keywords", [])
    political_categories = {c.lower() for c in classification.get("political_categories", [])}
    min_volume = classification.get("min_volume_usd", 10000)

    reasons: list[str] = []
    confidence = 0.0

    # Check tags
    market_tags = {t.lower() for t in market.tag_labels}
    matching_tags = market_tags & political_tags
    if matching_tags:
        reasons.append(f"tags: {', '.join(matching_tags)}")
        confidence += 0.4

    # Check category
    if market.category.lower() in political_categories:
        reasons.append(f"category: {market.category}")
        confidence += 0.3

    # Check title keywords
    title_lower = market.question.lower()
    desc_lower = market.description.lower()
    matched_keywords: list[str] = []
    for kw in title_keywords:
        if kw in title_lower or kw in desc_lower:
            matched_keywords.append(kw)
    if matched_keywords:
        reasons.append(f"keywords: {', '.join(matched_keywords[:5])}")
        confidence += 0.3

    if not reasons:
        return None

    # Volume filter
    if market.volume < min_volume:
        return None

    confidence = min(confidence, 1.0)
    return {"confidence": confidence, "reasons": reasons}


async def backfill_market(
    market: GammaMarket,
    political_class: dict,
    clob: ClobClient,
    data: DataClient,
    db,
    days: int,
) -> dict[str, int]:
    """Backfill a single market with price history and trades.

    Returns a summary dict with counts of data points inserted.
    """
    stats = {"price_points": 0, "trades": 0}

    # Save/update the market record
    await save_market(db, market, political_class)

    # Determine time range
    now_ts = int(time.time())
    start_ts = now_ts - (days * 86400)

    # --- Price history from CLOB API ---
    for token_id in market.clob_token_ids:
        if not token_id:
            continue
        try:
            history = await clob.get_price_history(
                token_id=token_id,
                start_ts=start_ts,
                end_ts=now_ts,
                interval="max",
                fidelity=min(days * 24, 10000),  # ~hourly for the period
            )
            for point in history.history:
                # Determine YES/NO from token position
                is_yes = market.clob_token_ids.index(token_id) == 0
                await save_price_snapshot(
                    db,
                    market_id=market.id,
                    price_yes=point.p if is_yes else None,
                    price_no=point.p if not is_yes else None,
                    volume_24hr=market.volume_24hr,
                    volume_total=market.volume,
                    liquidity=market.liquidity,
                )
                stats["price_points"] += 1
        except Exception:
            logger.warning(
                "Failed to fetch price history for token %s (market %s)",
                token_id,
                market.id,
                exc_info=True,
            )

    # --- Trade history from Data API ---
    try:
        trades = await data.get_all_trades(
            condition_id=market.condition_id,
            max_pages=20,
            limit=100,
        )
        # Filter trades to the backfill window
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        recent_trades = []
        for t in trades:
            match_dt = t.match_datetime
            if match_dt is not None and match_dt >= cutoff:
                recent_trades.append(t)
            elif match_dt is None:
                # Include trades with unparseable timestamps
                recent_trades.append(t)

        if recent_trades:
            inserted = await save_trades_batch(db, recent_trades, market.id)
            stats["trades"] = inserted
    except Exception:
        logger.warning(
            "Failed to fetch trades for market %s",
            market.id,
            exc_info=True,
        )

    return stats


async def run_backfill(days: int, config_path: str | None, dry_run: bool) -> None:
    """Main backfill orchestration."""
    cfg_path = Path(config_path) if config_path else None
    config = load_config(cfg_path)
    keywords_config = load_political_keywords()

    # Initialize clients
    gamma = GammaClient(config)
    clob = ClobClient(config)
    data = DataClient(config)
    db = await init_database(config)

    try:
        # Step 1: Discover all markets (active + recently closed)
        logger.info("Discovering markets from Gamma API...")
        active_markets = await gamma.get_all_markets(active=True, max_pages=100)
        closed_markets = await gamma.get_all_markets(active=False, closed=True, max_pages=20)
        all_markets = active_markets + closed_markets
        logger.info("Found %d total markets", len(all_markets))

        # Step 2: Filter to political markets
        political_markets: list[tuple[GammaMarket, dict]] = []
        for market in all_markets:
            classification = classify_political(market, keywords_config)
            if classification is not None:
                political_markets.append((market, classification))

        logger.info(
            "Identified %d political markets (out of %d total)",
            len(political_markets),
            len(all_markets),
        )

        if dry_run:
            for market, cls in political_markets:
                print(
                    f"  [{cls['confidence']:.1f}] {market.question[:80]}"
                    f"  (vol=${market.volume:,.0f})"
                )
            print(f"\nTotal: {len(political_markets)} political markets")
            return

        # Step 3: Backfill each market
        total_prices = 0
        total_trades = 0

        for i, (market, classification) in enumerate(political_markets, 1):
            logger.info(
                "[%d/%d] Backfilling: %s",
                i,
                len(political_markets),
                market.question[:70],
            )
            try:
                stats = await backfill_market(
                    market, classification, clob, data, db, days
                )
                total_prices += stats["price_points"]
                total_trades += stats["trades"]
                logger.info(
                    "  -> %d price points, %d trades",
                    stats["price_points"],
                    stats["trades"],
                )
            except Exception:
                logger.error(
                    "Failed to backfill market %s: %s",
                    market.id,
                    market.question[:50],
                    exc_info=True,
                )

        logger.info(
            "Backfill complete: %d markets, %d price points, %d trades",
            len(political_markets),
            total_prices,
            total_trades,
        )

    finally:
        await gamma.close()
        await clob.close()
        await data.close()
        await db.close()


@click.command()
@click.option(
    "--days",
    default=30,
    show_default=True,
    help="Number of days of history to backfill.",
)
@click.option(
    "--config",
    "config_path",
    default=None,
    type=click.Path(exists=True),
    help="Path to a custom TOML config file.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="List political markets without fetching data.",
)
@click.option(
    "--verbose",
    is_flag=True,
    default=False,
    help="Enable debug logging.",
)
def main(days: int, config_path: str | None, dry_run: bool, verbose: bool) -> None:
    """Backfill political prediction markets from Polymarket APIs."""
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    asyncio.run(run_backfill(days, config_path, dry_run))


if __name__ == "__main__":
    main()
