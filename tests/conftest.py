"""Shared test fixtures."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
import pytest_asyncio

from prediction_market.config import AppConfig, load_config
from prediction_market.data.polymarket.models import GammaMarket, OrderBook, OrderBookEntry


@pytest.fixture
def app_config(tmp_path: Path) -> AppConfig:
    """Create a test config with temporary database."""
    config = load_config()
    config.database.path = str(tmp_path / "test.db")
    config.reporting.output_dir = str(tmp_path / "reports")
    return config


@pytest.fixture
def sample_political_market() -> GammaMarket:
    return GammaMarket(
        id="test-market-1",
        question="Will the president sign the infrastructure bill by March 2026?",
        description="Resolves YES if the president signs the bill into law.",
        outcomes=["Yes", "No"],
        outcomePrices=["0.65", "0.35"],
        volume=500000.0,
        volume24hr=25000.0,
        liquidity=100000.0,
        active=True,
        closed=False,
        tags=[{"label": "Politics"}, {"label": "Legislation"}],
        slug="president-sign-infrastructure-bill",
        category="politics",
        conditionId="0xabc123",
        clobTokenIds=["token-yes-1", "token-no-1"],
    )


@pytest.fixture
def sample_nonpolitical_market() -> GammaMarket:
    return GammaMarket(
        id="test-market-2",
        question="Will Bitcoin reach $100k by end of 2026?",
        description="Resolves YES if BTC price exceeds $100,000.",
        outcomes=["Yes", "No"],
        outcomePrices=["0.45", "0.55"],
        volume=1000000.0,
        volume24hr=50000.0,
        liquidity=200000.0,
        active=True,
        closed=False,
        tags=[{"label": "Crypto"}],
        slug="bitcoin-100k-2026",
        category="crypto",
        conditionId="0xdef456",
        clobTokenIds=["token-yes-2", "token-no-2"],
    )


@pytest.fixture
def sample_orderbook() -> OrderBook:
    return OrderBook(
        market="test-market-1",
        asset_id="token-yes-1",
        bids=[
            OrderBookEntry(price="0.64", size="1000"),
            OrderBookEntry(price="0.63", size="2000"),
            OrderBookEntry(price="0.60", size="5000"),
            OrderBookEntry(price="0.55", size="3000"),
            OrderBookEntry(price="0.50", size="4000"),
        ],
        asks=[
            OrderBookEntry(price="0.66", size="800"),
            OrderBookEntry(price="0.67", size="1500"),
            OrderBookEntry(price="0.70", size="3000"),
            OrderBookEntry(price="0.75", size="2000"),
            OrderBookEntry(price="0.80", size="5000"),
        ],
    )


@pytest.fixture
def thin_orderbook() -> OrderBook:
    """Order book with very thin liquidity."""
    return OrderBook(
        market="test-market-1",
        asset_id="token-yes-1",
        bids=[
            OrderBookEntry(price="0.60", size="50"),
            OrderBookEntry(price="0.55", size="100"),
        ],
        asks=[
            OrderBookEntry(price="0.70", size="30"),
            OrderBookEntry(price="0.80", size="80"),
        ],
    )
