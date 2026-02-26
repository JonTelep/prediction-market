"""Agent 2: Manipulation Guard.

Monitors order-book depth, holder concentration, and liquidity
conditions to identify markets that are susceptible to price
manipulation.  Emits an :class:`AnomalyReport` whenever the composite
susceptibility score exceeds the configured threshold.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

import aiosqlite
import httpx

from prediction_market.agents.base import BaseAgent
from prediction_market.config import AppConfig
from prediction_market.data.polymarket.models import MarketHolder, OrderBook
from prediction_market.data.polymarket.rate_limiter import TokenBucketRateLimiter
from prediction_market.reporting.anomaly_report import AnomalyReport
from prediction_market.reporting.sink import ReportSink

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Liquidity analysis helpers
# ---------------------------------------------------------------------------

class LiquidityAnalyzer:
    """Stateless analyser that scores an order book for manipulation risk.

    Each metric returns a value in [0, 1] where higher means more
    susceptible.
    """

    @staticmethod
    def depth_score(book: OrderBook) -> float:
        """Score based on total depth relative to a healthy baseline.

        Thin books are trivially manipulated.  We consider total depth
        under $5 000 as maximally thin (score 1.0) and depth above
        $500 000 as fully healthy (score 0.0), with a logarithmic
        scale between.
        """
        import math

        total = book.total_bid_depth + book.total_ask_depth
        if total <= 0:
            return 1.0
        min_depth = 5_000.0
        max_depth = 500_000.0
        if total <= min_depth:
            return 1.0
        if total >= max_depth:
            return 0.0
        # Logarithmic interpolation
        log_range = math.log(max_depth / min_depth)
        return 1.0 - math.log(total / min_depth) / log_range

    @staticmethod
    def spread_score(book: OrderBook) -> float:
        """Wide spreads signal low market-maker participation."""
        spread_pct = book.spread_pct
        if spread_pct is None:
            return 1.0
        # Spread > 10% of midpoint -> score 1.0
        return min(1.0, spread_pct / 0.10)

    @staticmethod
    def imbalance_score(book: OrderBook) -> float:
        """Extreme order-book imbalance suggests directional pressure."""
        return abs(book.imbalance)

    @staticmethod
    def concentration_score(holders: list[MarketHolder]) -> float:
        """Score based on HHI (Herfindahl-Hirschman Index) of holder pct.

        An HHI of 1.0 means one holder owns everything.
        """
        if not holders:
            return 1.0
        shares = [h.pct_supply for h in holders if h.pct_supply > 0]
        if not shares:
            return 1.0
        hhi = sum(s ** 2 for s in shares)
        return min(1.0, hhi)

    def compute_susceptibility(
        self,
        book: OrderBook,
        holders: list[MarketHolder],
        weights: dict[str, float],
    ) -> dict[str, float]:
        """Return individual scores and weighted composite.

        *weights* is expected to contain keys:
        ``depth_weight``, ``spread_weight``, ``concentration_weight``,
        ``imbalance_weight``.
        """
        d = self.depth_score(book)
        s = self.spread_score(book)
        c = self.concentration_score(holders)
        i = self.imbalance_score(book)

        composite = (
            d * weights.get("depth_weight", 0.30)
            + s * weights.get("spread_weight", 0.25)
            + c * weights.get("concentration_weight", 0.25)
            + i * weights.get("imbalance_weight", 0.20)
        )

        return {
            "depth": d,
            "spread": s,
            "concentration": c,
            "imbalance": i,
            "composite": composite,
        }


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class ManipulationGuard(BaseAgent):
    """Detects markets vulnerable to price manipulation.

    On each tick the agent iterates over every tracked market, fetches
    the current order book and holder list, scores liquidity with
    :class:`LiquidityAnalyzer`, and emits an :class:`AnomalyReport` if
    the composite susceptibility exceeds the configured threshold.

    It also compares the current order-book depth against the most
    recent stored snapshot to detect sudden liquidity withdrawal.
    """

    def __init__(
        self,
        config: AppConfig,
        db: aiosqlite.Connection,
        sinks: list[ReportSink] | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        super().__init__(config, db, sinks)
        self._http = http_client or httpx.AsyncClient(timeout=30.0)
        self._owns_http = http_client is None
        self._analyzer = LiquidityAnalyzer()
        self._clob_limiter = TokenBucketRateLimiter(
            max_tokens=config.rate_limits.clob_requests_per_window,
            window_seconds=config.rate_limits.clob_window_seconds,
        )
        self._data_limiter = TokenBucketRateLimiter(
            max_tokens=config.rate_limits.data_requests_per_window,
            window_seconds=config.rate_limits.data_window_seconds,
        )
        # Cache: market_id -> most recent liquidity metrics
        self._liquidity_cache: dict[str, dict[str, Any]] = {}

    # -- BaseAgent properties -------------------------------------------

    @property
    def name(self) -> str:
        return "manipulation_guard"

    @property
    def tick_interval_seconds(self) -> int:
        return self.config.polling.orderbook_interval_seconds  # default 300s

    # -- Public accessor for cross-agent queries ------------------------

    def get_liquidity_metrics(self, market_id: str) -> dict[str, Any] | None:
        """Return the most recent liquidity metrics for *market_id*.

        Called by Agent 1 (Info-Leak Detector) to cross-reference
        manipulation risk when scoring anomalous price movements.
        """
        return self._liquidity_cache.get(market_id)

    # -- API helpers ----------------------------------------------------

    async def _fetch_orderbook(self, token_id: str) -> OrderBook | None:
        """Fetch current order book from the CLOB API."""
        await self._clob_limiter.acquire()
        url = f"{self.config.apis.clob_base_url}/book"
        try:
            resp = await self._http.get(url, params={"token_id": token_id})
            resp.raise_for_status()
            data = resp.json()
            return OrderBook.model_validate(data)
        except Exception:
            logger.exception("Failed to fetch orderbook for token %s", token_id)
            return None

    async def _fetch_holders(self, condition_id: str) -> list[MarketHolder]:
        """Fetch top holders from the Data API."""
        await self._data_limiter.acquire()
        url = f"{self.config.apis.data_base_url}/positions"
        try:
            resp = await self._http.get(
                url, params={"condition_id": condition_id, "limit": 50}
            )
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, list):
                return [MarketHolder.model_validate(h) for h in data]
            return []
        except Exception:
            logger.exception("Failed to fetch holders for condition %s", condition_id)
            return []

    # -- Snapshot persistence -------------------------------------------

    async def _get_previous_depth(self, market_id: str, token_id: str) -> float | None:
        """Load the most recent total depth from the orderbook_snapshots table."""
        cursor = await self.db.execute(
            """
            SELECT total_bid_depth, total_ask_depth
            FROM orderbook_snapshots
            WHERE market_id = ? AND token_id = ?
            ORDER BY timestamp DESC
            LIMIT 1
            """,
            (market_id, token_id),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        bid_depth, ask_depth = row
        if bid_depth is None or ask_depth is None:
            return None
        return bid_depth + ask_depth

    async def _store_orderbook_snapshot(
        self,
        market_id: str,
        token_id: str,
        book: OrderBook,
        susceptibility_score: float,
    ) -> None:
        """Persist the current order-book metrics."""
        now = datetime.now(timezone.utc).isoformat()
        try:
            await self.db.execute(
                """
                INSERT OR REPLACE INTO orderbook_snapshots
                    (market_id, token_id, timestamp,
                     best_bid, best_ask, midpoint, spread, spread_pct,
                     total_bid_depth, total_ask_depth,
                     depth_1pct, depth_5pct, depth_10pct,
                     imbalance, hhi, susceptibility_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    market_id,
                    token_id,
                    now,
                    book.best_bid,
                    book.best_ask,
                    book.midpoint,
                    book.spread,
                    book.spread_pct,
                    book.total_bid_depth,
                    book.total_ask_depth,
                    book.depth_at_pct(0.01),
                    book.depth_at_pct(0.05),
                    book.depth_at_pct(0.10),
                    book.imbalance,
                    None,  # HHI computed separately from holders
                    susceptibility_score,
                ),
            )
            await self.db.commit()
        except Exception:
            logger.exception(
                "Failed to store orderbook snapshot for %s/%s", market_id, token_id
            )

    # -- Tracked markets ------------------------------------------------

    async def _get_tracked_markets(self) -> list[dict[str, Any]]:
        """Load active political markets from the database."""
        cursor = await self.db.execute(
            """
            SELECT id, question, condition_id, clob_token_ids
            FROM markets
            WHERE active = 1 AND closed = 0
            ORDER BY volume DESC
            """
        )
        rows = await cursor.fetchall()
        markets: list[dict[str, Any]] = []
        for row in rows:
            market_id, question, condition_id, token_ids_raw = row
            try:
                token_ids = json.loads(token_ids_raw) if token_ids_raw else []
            except (json.JSONDecodeError, TypeError):
                token_ids = []
            markets.append(
                {
                    "id": market_id,
                    "question": question,
                    "condition_id": condition_id,
                    "token_ids": token_ids,
                }
            )
        return markets

    # -- Core tick ------------------------------------------------------

    async def tick(self) -> None:
        """Run one surveillance cycle across all tracked markets."""
        markets = await self._get_tracked_markets()
        if not markets:
            logger.debug("ManipulationGuard: no tracked markets")
            return

        threshold = self.config.thresholds.susceptibility_threshold
        drop_pct = self.config.thresholds.liquidity_drop_pct

        weights = {
            "depth_weight": self.config.thresholds.depth_weight,
            "spread_weight": self.config.thresholds.spread_weight,
            "concentration_weight": self.config.thresholds.concentration_weight,
            "imbalance_weight": self.config.thresholds.imbalance_weight,
        }

        for market in markets:
            market_id: str = market["id"]
            question: str = market["question"]
            condition_id: str = market["condition_id"]
            token_ids: list[str] = market["token_ids"]

            if not token_ids:
                continue

            # Use the first token (YES side) as primary
            primary_token = token_ids[0]

            # (a) Fetch order book
            book = await self._fetch_orderbook(primary_token)
            if book is None:
                continue

            # (b) Fetch holders
            holders = await self._fetch_holders(condition_id) if condition_id else []

            # (c) Run LiquidityAnalyzer
            scores = self._analyzer.compute_susceptibility(book, holders, weights)
            composite = scores["composite"]

            # Update cache for cross-agent use
            self._liquidity_cache[market_id] = {
                "scores": scores,
                "total_depth": book.total_bid_depth + book.total_ask_depth,
                "spread_pct": book.spread_pct,
                "imbalance": book.imbalance,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            # (f) Check for sudden liquidity withdrawal
            previous_depth = await self._get_previous_depth(market_id, primary_token)
            current_depth = book.total_bid_depth + book.total_ask_depth
            liquidity_withdrawal = False
            depth_drop_ratio = 0.0

            if previous_depth is not None and previous_depth > 0:
                depth_drop_ratio = 1.0 - (current_depth / previous_depth)
                if depth_drop_ratio >= drop_pct:
                    liquidity_withdrawal = True
                    logger.warning(
                        "Sudden liquidity drop in %s: %.1f%% (%.0f -> %.0f)",
                        market_id,
                        depth_drop_ratio * 100,
                        previous_depth,
                        current_depth,
                    )

            # Store snapshot for future comparisons
            await self._store_orderbook_snapshot(
                market_id, primary_token, book, composite
            )

            # (d)/(e) Emit report if susceptibility exceeds threshold
            should_report = composite >= threshold or liquidity_withdrawal

            if not should_report:
                continue

            # Build report
            effective_score = max(composite, 0.9 if liquidity_withdrawal else composite)
            severity = AnomalyReport.severity_from_score(effective_score)

            summary_parts: list[str] = []
            if composite >= threshold:
                summary_parts.append(
                    f"Market susceptibility score {composite:.3f} exceeds "
                    f"threshold {threshold:.2f}."
                )
            if liquidity_withdrawal:
                summary_parts.append(
                    f"Sudden liquidity withdrawal detected: "
                    f"{depth_drop_ratio:.1%} depth drop."
                )

            # Confidence: higher when both signals agree
            confidence = min(1.0, composite)
            if liquidity_withdrawal and composite >= threshold:
                confidence = min(1.0, confidence + 0.15)

            report = AnomalyReport(
                id=AnomalyReport.new_id(),
                agent="manipulation",
                market_id=market_id,
                market_question=question,
                severity=severity,
                anomaly_score=effective_score,
                confidence=confidence,
                summary=" ".join(summary_parts),
                details={
                    "scores": scores,
                    "weights": weights,
                    "threshold": threshold,
                    "liquidity_withdrawal": liquidity_withdrawal,
                    "depth_drop_ratio": depth_drop_ratio,
                    "holder_count": len(holders),
                },
                price_evidence={
                    "best_bid": book.best_bid,
                    "best_ask": book.best_ask,
                    "midpoint": book.midpoint,
                    "spread": book.spread,
                    "spread_pct": book.spread_pct,
                },
                volume_evidence={
                    "total_bid_depth": book.total_bid_depth,
                    "total_ask_depth": book.total_ask_depth,
                    "current_total_depth": current_depth,
                    "previous_total_depth": previous_depth,
                    "depth_1pct": book.depth_at_pct(0.01),
                    "depth_5pct": book.depth_at_pct(0.05),
                    "depth_10pct": book.depth_at_pct(0.10),
                },
                calendar_matches=[],
                news_check={},
                created_at=datetime.now(timezone.utc),
            )

            await self.emit(report)

    # -- Cleanup --------------------------------------------------------

    async def stop(self) -> None:
        """Stop the agent and close owned HTTP resources."""
        await super().stop()
        if self._owns_http:
            await self._http.aclose()
