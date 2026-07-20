"""Pydantic models for Polymarket API responses."""

from __future__ import annotations

import json
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import AliasChoices, BaseModel, Field, field_validator, model_validator


class MarketOutcome(str, Enum):
    YES = "Yes"
    NO = "No"


def _parse_json_list(v: Any) -> list:
    """Parse a value that may be a JSON-encoded string or already a list."""
    if isinstance(v, str):
        try:
            parsed = json.loads(v)
            if isinstance(parsed, list):
                return parsed
        except (json.JSONDecodeError, TypeError):
            pass
        return []
    if isinstance(v, list):
        return v
    return []


class GammaMarket(BaseModel):
    """Market from Gamma API (discovery/metadata)."""

    id: str = ""
    question: str = ""
    description: str = ""
    outcomes: list[str] = Field(default_factory=list)
    outcome_prices: list[str] = Field(default_factory=list, alias="outcomePrices")
    volume: float = 0.0
    volume_24hr: float = Field(0.0, alias="volume24hr")
    liquidity: float = 0.0
    active: bool = True
    closed: bool = False
    archived: bool = False
    created_at: str = Field("", alias="createdAt")
    end_date: str = Field("", alias="endDate")
    tags: list[dict] | None = None
    slug: str = ""
    category: str = ""
    condition_id: str = Field("", alias="conditionId")
    clob_token_ids: list[str] = Field(default_factory=list, alias="clobTokenIds")
    rewards: dict | None = None
    enable_order_book: bool = Field(True, alias="enableOrderBook")

    model_config = {"populate_by_name": True}

    @field_validator("outcomes", "outcome_prices", "clob_token_ids", mode="before")
    @classmethod
    def _parse_json_strings(cls, v: Any) -> list:
        return _parse_json_list(v)

    @property
    def tag_labels(self) -> list[str]:
        if not self.tags:
            return []
        return [t.get("label", "") for t in self.tags if isinstance(t, dict)]

    def _label_mapped_token(self, label: str) -> str | None:
        """Map a Yes/No label to its token id via ``outcomes``, if possible.

        Only trusted when ``outcomes`` has exactly two entries both labeled
        (case-insensitively) "yes"/"no" and ``clob_token_ids`` has at least
        two entries -- otherwise returns None so callers fall back to the
        positional convention.
        """
        if len(self.outcomes) != 2 or len(self.clob_token_ids) < 2:
            return None
        lowered = [o.strip().lower() for o in self.outcomes]
        if set(lowered) != {"yes", "no"}:
            return None
        index = lowered.index(label)
        return self.clob_token_ids[index]

    @property
    def yes_token_id(self) -> str | None:
        """The CLOB token id for the YES outcome.

        Prefers mapping by the ``outcomes`` labels (catches inverted
        Yes/No listings); falls back to the standard positional
        convention (``clob_token_ids[0]``) when labels aren't a clean
        Yes/No pair (e.g. candidate-named outcomes) or lengths mismatch.
        Returns None if no token exists at all.
        """
        mapped = self._label_mapped_token("yes")
        if mapped is not None:
            return mapped
        return self.clob_token_ids[0] if self.clob_token_ids else None

    @property
    def no_token_id(self) -> str | None:
        """The CLOB token id for the NO outcome.

        See ``yes_token_id`` for the label-mapping/positional-fallback
        logic. Returns None if fewer than two tokens exist.
        """
        mapped = self._label_mapped_token("no")
        if mapped is not None:
            return mapped
        return self.clob_token_ids[1] if len(self.clob_token_ids) > 1 else None


class OrderBookEntry(BaseModel):
    """Single entry in an order book."""

    price: str
    size: str

    @property
    def price_float(self) -> float:
        return float(self.price)

    @property
    def size_float(self) -> float:
        return float(self.size)


class OrderBook(BaseModel):
    """Order book from CLOB API."""

    market: str = ""
    asset_id: str = ""
    bids: list[OrderBookEntry] = Field(default_factory=list)
    asks: list[OrderBookEntry] = Field(default_factory=list)
    hash: str = ""
    timestamp: str = ""

    @property
    def best_bid(self) -> float | None:
        if not self.bids:
            return None
        return max(b.price_float for b in self.bids)

    @property
    def best_ask(self) -> float | None:
        if not self.asks:
            return None
        return min(a.price_float for a in self.asks)

    @property
    def midpoint(self) -> float | None:
        bid, ask = self.best_bid, self.best_ask
        if bid is None or ask is None:
            return None
        return (bid + ask) / 2

    @property
    def spread(self) -> float | None:
        bid, ask = self.best_bid, self.best_ask
        if bid is None or ask is None:
            return None
        return ask - bid

    @property
    def spread_pct(self) -> float | None:
        mid = self.midpoint
        spr = self.spread
        if mid is None or spr is None or mid == 0:
            return None
        return spr / mid

    def depth_at_pct(self, pct: float, side: str = "both") -> float:
        """Total size within pct% of midpoint."""
        mid = self.midpoint
        if mid is None:
            return 0.0
        total = 0.0
        if side in ("both", "bid"):
            threshold = mid * (1 - pct)
            for b in self.bids:
                if b.price_float >= threshold:
                    total += b.size_float * b.price_float
        if side in ("both", "ask"):
            threshold = mid * (1 + pct)
            for a in self.asks:
                if a.price_float <= threshold:
                    total += a.size_float * a.price_float
        return total

    @property
    def total_bid_depth(self) -> float:
        return sum(b.size_float * b.price_float for b in self.bids)

    @property
    def total_ask_depth(self) -> float:
        return sum(a.size_float * a.price_float for a in self.asks)

    @property
    def imbalance(self) -> float:
        """Order book imbalance: (bids - asks) / (bids + asks)."""
        total_bids = self.total_bid_depth
        total_asks = self.total_ask_depth
        total = total_bids + total_asks
        if total == 0:
            return 0.0
        return (total_bids - total_asks) / total


class ClobPrice(BaseModel):
    """Price point from CLOB API."""

    t: int = 0  # timestamp
    p: float = 0.0  # price


class ClobPriceHistory(BaseModel):
    """Price history from CLOB API."""

    history: list[ClobPrice] = Field(default_factory=list)


class Trade(BaseModel):
    """Individual trade from Data API.

    Live-API note (verified 2026-07-20 during the Phase-2 validation
    session): the real ``/trades`` records differ from this repo's
    original fixture assumptions -- the market is keyed ``conditionId``
    (not ``market``), the token is ``asset`` (not ``assetId``), the trade
    time is an epoch number under ``timestamp`` (not ``matchTime``), and
    there is **no** ``id`` field. The alias choices below accept both
    shapes, and a validator synthesizes a deterministic ``id`` from the
    trade's identifying fields when the API provides none (the trades
    table uses ``id`` as its primary key for INSERT OR IGNORE dedup).
    """

    id: str = ""
    taker_order_id: str = Field("", alias="takerOrderId")
    market: str = Field(
        "",
        validation_alias=AliasChoices("market", "conditionId"),
        serialization_alias="market",
    )
    asset_id: str = Field(
        "",
        validation_alias=AliasChoices("assetId", "asset"),
        serialization_alias="assetId",
    )
    side: str = ""
    size: str = ""
    fee_rate_bps: str = Field("", alias="feeRateBps")
    price: str = ""
    status: str = ""
    match_time: str = Field(
        "",
        validation_alias=AliasChoices("matchTime", "timestamp"),
        serialization_alias="matchTime",
    )
    outcome: str = ""
    bucket_index: str = Field("", alias="bucketIndex")
    owner: str = ""
    proxy_wallet: str = Field("", alias="proxyWallet")
    transaction_hash: str = Field("", alias="transactionHash")

    model_config = {"populate_by_name": True}

    @model_validator(mode="after")
    def _synthesize_id(self) -> "Trade":
        if not self.id and self.transaction_hash:
            self.id = (
                f"{self.transaction_hash}-{self.asset_id[:16]}-{self.match_time}"
                f"-{self.side}-{self.size}-{self.price}"
            )
        return self

    @field_validator(
        "size", "price", "fee_rate_bps", "bucket_index", "match_time", mode="before"
    )
    @classmethod
    def _coerce_numeric_to_str(cls, v: Any) -> Any:
        """Accept JSON numbers for the string-typed fields.

        Live-API finding (2026-07-20 validation session): the Data API
        serves ``size``/``price`` as JSON numbers, not the strings this
        repo's fixtures assumed. ``match_time`` can likewise arrive as an
        epoch number; the store's write-time normalization already handles
        the all-digits string form.
        """
        if isinstance(v, bool):
            return v
        if isinstance(v, float):
            # str() is lossless (repr round-trips); integral floats drop
            # the trailing ".0" so epoch timestamps stay all-digits for
            # the store's normalization.
            return str(int(v)) if v.is_integer() else str(v)
        if isinstance(v, int):
            return str(v)
        return v

    @property
    def price_float(self) -> float:
        return float(self.price) if self.price else 0.0

    @property
    def size_float(self) -> float:
        return float(self.size) if self.size else 0.0

    @property
    def volume_usd(self) -> float:
        return self.price_float * self.size_float

    @property
    def match_datetime(self) -> datetime | None:
        if not self.match_time:
            return None
        try:
            return datetime.fromisoformat(self.match_time.replace("Z", "+00:00"))
        except ValueError:
            return None


class MarketHolder(BaseModel):
    """Holder position from Data API."""

    address: str = ""
    position: float = 0.0
    value: float = 0.0
    pct_supply: float = Field(0.0, alias="pctSupply")

    model_config = {"populate_by_name": True}


class OpenInterest(BaseModel):
    """Open interest data."""

    asset_id: str = Field("", alias="assetId")
    open_interest: float = Field(0.0, alias="openInterest")

    model_config = {"populate_by_name": True}


class WalletPosition(BaseModel):
    """A wallet's open position in a market, from Data API /positions."""

    proxy_wallet: str = Field("", alias="proxyWallet")
    asset: str = Field("", alias="asset")
    condition_id: str = Field("", alias="conditionId")
    size: str = ""
    avg_price: str = Field("", alias="avgPrice")
    initial_value: str = Field("", alias="initialValue")
    current_value: str = Field("", alias="currentValue")
    cash_pnl: str = Field("", alias="cashPnl")
    percent_pnl: str = Field("", alias="percentPnl")
    outcome: str = ""
    title: str = ""

    model_config = {"populate_by_name": True}

    @property
    def size_float(self) -> float:
        return float(self.size) if self.size else 0.0

    @property
    def avg_price_float(self) -> float:
        return float(self.avg_price) if self.avg_price else 0.0

    @property
    def cash_pnl_float(self) -> float:
        return float(self.cash_pnl) if self.cash_pnl else 0.0


class WalletActivity(BaseModel):
    """A single activity event for a wallet, from Data API /activity."""

    proxy_wallet: str = Field("", alias="proxyWallet")
    timestamp: str = ""
    type: str = ""
    condition_id: str = Field("", alias="conditionId")
    size: str = ""
    usdc_size: str = Field("", alias="usdcSize")
    price: str = ""
    side: str = ""
    outcome: str = ""
    transaction_hash: str = Field("", alias="transactionHash")

    model_config = {"populate_by_name": True}

    @property
    def usdc_size_float(self) -> float:
        return float(self.usdc_size) if self.usdc_size else 0.0
