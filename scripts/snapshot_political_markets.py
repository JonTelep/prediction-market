"""One-shot snapshot of all political markets.

Usage:
    uv run python scripts/snapshot_political_markets.py
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from prediction_market.config import load_config
from prediction_market.data.political_filter import PoliticalFilter
from prediction_market.data.polymarket.gamma_client import GammaClient
from prediction_market.store.database import init_database
from prediction_market.store.snapshots import save_market, save_price_snapshot

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

            # Upsert market into DB. save_market() is an ON CONFLICT upsert
            # that does NOT clobber first_seen on an existing row -- unlike
            # the raw "INSERT OR REPLACE" this replaces, which reset
            # first_seen on every run. This is a deliberate improvement.
            await save_market(
                db,
                market,
                {"confidence": classification.confidence, "reasons": classification.reasons},
            )

            # Save a price snapshot
            prices = market.outcome_prices
            price_yes = float(prices[0]) if len(prices) > 0 else None
            price_no = float(prices[1]) if len(prices) > 1 else None

            await save_price_snapshot(
                db,
                market_id=market.id,
                price_yes=price_yes,
                price_no=price_no,
                volume_24hr=market.volume_24hr,
                volume_total=market.volume,
                liquidity=market.liquidity,
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
