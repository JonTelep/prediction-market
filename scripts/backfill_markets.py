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

from prediction_market.config import load_config
from prediction_market.data.political_filter import PoliticalClassification, PoliticalFilter
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


def select_political_markets(
    markets: list[GammaMarket],
    political_filter: PoliticalFilter,
) -> list[tuple[GammaMarket, PoliticalClassification]]:
    """Select markets to backfill using the shared :class:`PoliticalFilter`.

    A market is selected iff ``classification.is_political`` is True AND
    ``market.volume >= political_filter.min_volume``.

    This uses the same classifier as ``monitor``/``scan`` -- one source of
    truth for political classification. It is NOT equivalent to the old
    inline ``classify_political`` this replaced: that scored any keyword
    match a flat +0.3 and treated any non-empty reason list as political
    (no confidence gate), while ``PoliticalFilter.classify`` scores
    keywords incrementally (``min(0.3, matches * 0.1)``) and gates on
    ``confidence >= 0.3``. Consequence: a market matching only 1-2
    keywords (confidence 0.1-0.2, no tag/category hit) that used to be
    backfilled will no longer be selected. This is the intended
    unification, not a regression.
    """
    selected: list[tuple[GammaMarket, PoliticalClassification]] = []
    for market in markets:
        classification = political_filter.classify(market)
        if not classification.is_political:
            continue
        if market.volume < political_filter.min_volume:
            continue
        selected.append((market, classification))
    return selected


async def backfill_market(
    market: GammaMarket,
    classification: PoliticalClassification,
    clob: ClobClient,
    data: DataClient,
    db,
    days: int,
) -> dict[str, int]:
    """Backfill a single market with price history and trades.

    Returns a summary dict with counts of data points inserted.
    """
    stats = {"price_points": 0, "trades": 0}

    # Save/update the market record. save_market() expects a plain dict
    # with "confidence"/"reasons" keys, not the PoliticalClassification
    # dataclass.
    await save_market(
        db,
        market,
        {"confidence": classification.confidence, "reasons": classification.reasons},
    )

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
    political_filter = PoliticalFilter()

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

        # Step 2: Filter to political markets via the shared PoliticalFilter
        political_markets = select_political_markets(all_markets, political_filter)

        logger.info(
            "Identified %d political markets (out of %d total)",
            len(political_markets),
            len(all_markets),
        )

        if dry_run:
            for market, cls in political_markets:
                print(
                    f"  [{cls.confidence:.1f}] {market.question[:80]}"
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
