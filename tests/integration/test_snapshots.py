"""Integration tests for snapshot writers against real SQLite."""

import pytest
import pytest_asyncio

from prediction_market.config import load_config
from prediction_market.data.polymarket.models import GammaMarket, OrderBook, OrderBookEntry, Trade
from prediction_market.store.database import init_database
from prediction_market.store.snapshots import (
    save_market,
    save_orderbook_snapshot,
    save_price_snapshot,
    save_trade,
    save_trades_batch,
)


@pytest.fixture
def db_config(tmp_path):
    config = load_config()
    config.database.path = str(tmp_path / "test.db")
    return config


@pytest_asyncio.fixture
async def db(db_config):
    conn = await init_database(db_config)
    yield conn
    await conn.close()


@pytest.fixture
def sample_market():
    return GammaMarket(
        id="snap-market-1",
        question="Will the bill pass?",
        description="Test market",
        outcomes=["Yes", "No"],
        outcomePrices=["0.65", "0.35"],
        volume=500000.0,
        volume24hr=25000.0,
        liquidity=100000.0,
        active=True,
        closed=False,
        tags=[{"label": "Politics"}],
        slug="bill-pass",
        category="politics",
        conditionId="0xabc",
        clobTokenIds=["tok-yes", "tok-no"],
    )


@pytest.fixture
def sample_orderbook():
    return OrderBook(
        market="snap-market-1",
        asset_id="tok-yes",
        bids=[
            OrderBookEntry(price="0.64", size="1000"),
            OrderBookEntry(price="0.63", size="2000"),
        ],
        asks=[
            OrderBookEntry(price="0.66", size="800"),
            OrderBookEntry(price="0.67", size="1500"),
        ],
    )


@pytest.fixture
def sample_trades():
    return [
        Trade(
            id=f"trade-snap-{i}",
            market="snap-market-1",
            assetId="tok-yes",
            side="BUY",
            size=str(100 * (i + 1)),
            price="0.65",
            matchTime="2026-02-20T10:00:00Z",
            outcome="Yes",
            owner=f"0xowner{i}",
            transactionHash=f"0xtx{i}",
        )
        for i in range(5)
    ]


@pytest.mark.asyncio
async def test_save_market(db, sample_market):
    await save_market(db, sample_market, {"confidence": 0.8, "reasons": ["politics tag"]})

    cursor = await db.execute("SELECT id, question, volume FROM markets WHERE id = ?", ("snap-market-1",))
    row = await cursor.fetchone()
    assert row[0] == "snap-market-1"
    assert row[1] == "Will the bill pass?"
    assert row[2] == 500000.0


@pytest.mark.asyncio
async def test_save_market_upsert(db, sample_market):
    await save_market(db, sample_market)
    # Update volume and re-save
    sample_market.volume = 600000.0
    await save_market(db, sample_market)

    cursor = await db.execute("SELECT volume FROM markets WHERE id = ?", ("snap-market-1",))
    row = await cursor.fetchone()
    assert row[0] == 600000.0


@pytest.mark.asyncio
async def test_save_market_no_classification(db, sample_market):
    await save_market(db, sample_market, political_classification=None)
    cursor = await db.execute(
        "SELECT political_confidence FROM markets WHERE id = ?", ("snap-market-1",)
    )
    row = await cursor.fetchone()
    assert row[0] == 0.0


@pytest.mark.asyncio
async def test_save_price_snapshot(db, sample_market):
    await save_market(db, sample_market)
    await save_price_snapshot(
        db, "snap-market-1",
        price_yes=0.65, price_no=0.35,
        volume_24hr=25000, volume_total=500000, liquidity=100000,
    )

    cursor = await db.execute(
        "SELECT price_yes, price_no, volume_24hr FROM snapshots WHERE market_id = ?",
        ("snap-market-1",),
    )
    row = await cursor.fetchone()
    assert row[0] == 0.65
    assert row[1] == 0.35
    assert row[2] == 25000


@pytest.mark.asyncio
async def test_save_orderbook_snapshot(db, sample_market, sample_orderbook):
    await save_market(db, sample_market)
    await save_orderbook_snapshot(
        db, "snap-market-1", "tok-yes", sample_orderbook,
        hhi=2500.0, susceptibility=0.6,
    )

    cursor = await db.execute(
        "SELECT best_bid, best_ask, midpoint, hhi, susceptibility_score "
        "FROM orderbook_snapshots WHERE market_id = ?",
        ("snap-market-1",),
    )
    row = await cursor.fetchone()
    assert row[0] == 0.64
    assert row[1] == 0.66
    assert row[3] == 2500.0
    assert row[4] == 0.6


@pytest.mark.asyncio
async def test_save_trade(db, sample_market, sample_trades):
    await save_market(db, sample_market)
    await save_trade(db, sample_trades[0], "snap-market-1")

    cursor = await db.execute(
        "SELECT id, side, price FROM trades WHERE market_id = ?",
        ("snap-market-1",),
    )
    row = await cursor.fetchone()
    assert row[0] == "trade-snap-0"
    assert row[1] == "BUY"


@pytest.mark.asyncio
async def test_save_trade_ignores_duplicates(db, sample_market, sample_trades):
    await save_market(db, sample_market)
    await save_trade(db, sample_trades[0], "snap-market-1")
    # Insert same trade again -- should not raise
    await save_trade(db, sample_trades[0], "snap-market-1")

    cursor = await db.execute("SELECT COUNT(*) FROM trades WHERE market_id = ?", ("snap-market-1",))
    row = await cursor.fetchone()
    assert row[0] == 1


@pytest.mark.asyncio
async def test_save_trades_batch(db, sample_market, sample_trades):
    await save_market(db, sample_market)
    inserted = await save_trades_batch(db, sample_trades, "snap-market-1")
    assert inserted >= 0  # rowcount may vary by driver

    cursor = await db.execute("SELECT COUNT(*) FROM trades WHERE market_id = ?", ("snap-market-1",))
    row = await cursor.fetchone()
    assert row[0] == 5


@pytest.mark.asyncio
async def test_save_trades_batch_empty(db):
    result = await save_trades_batch(db, [], "m1")
    assert result == 0
