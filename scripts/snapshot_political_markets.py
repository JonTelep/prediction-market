"""One-shot snapshot of all political markets.

Usage:
    uv run python scripts/snapshot_political_markets.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from datetime import datetime, timezone

from prediction_market.config import load_config
from prediction_market.data.political_filter import PoliticalFilter
from prediction_market.data.polymarket.gamma_client import GammaClient
from prediction_market.store.database import init_database

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


async def main() -> None:
    config = load_config()
    db = await init_database(config)
    gamma = GammaClient(config)
    pf = PoliticalFilter()

    try:
        logger.info("Fetching all active markets from Gamma API...")
        all_markets = await gamma.get_all_markets(active=True)
        logger.info("Found %d total active markets", len(all_markets))

        political = pf.filter_political(all_markets)
        logger.info("Identified %d political markets", len(political))

        for market in political:
            classification = pf.classify(market)

            # Upsert market into DB
            await db.execute(
                """INSERT OR REPLACE INTO markets
                   (id, question, description, category, tags, slug, condition_id,
                    clob_token_ids, volume, liquidity, active, closed,
                    political_confidence, political_reasons, created_at, end_date, last_updated)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))""",
                (
                    market.id,
                    market.question,
                    market.description,
                    market.category,
                    json.dumps(market.tag_labels),
                    market.slug,
                    market.condition_id,
                    json.dumps(market.clob_token_ids),
                    market.volume,
                    market.liquidity,
                    int(market.active),
                    int(market.closed),
                    classification.confidence,
                    json.dumps(classification.reasons),
                    market.created_at,
                    market.end_date,
                ),
            )

            # Save a price snapshot
            prices = market.outcome_prices
            price_yes = float(prices[0]) if len(prices) > 0 else None
            price_no = float(prices[1]) if len(prices) > 1 else None

            await db.execute(
                """INSERT INTO snapshots (market_id, price_yes, price_no, volume_24hr, volume_total, liquidity)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (market.id, price_yes, price_no, market.volume_24hr, market.volume, market.liquidity),
            )

        await db.commit()
        logger.info("Snapshot complete. %d political markets stored.", len(political))

        # Print summary
        print(f"\n{'='*80}")
        print(f"Political Market Snapshot — {datetime.now(timezone.utc).isoformat()}")
        print(f"{'='*80}\n")
        for m in sorted(political, key=lambda x: x.volume, reverse=True)[:20]:
            prices = m.outcome_prices
            yes_price = float(prices[0]) if prices else 0
            print(f"  [{yes_price:.0%} YES]  ${m.volume:>12,.0f} vol  {m.question[:70]}")
        print(f"\nTotal: {len(political)} political markets tracked")

    finally:
        await gamma.close()
        await db.close()


if __name__ == "__main__":
    asyncio.run(main())
