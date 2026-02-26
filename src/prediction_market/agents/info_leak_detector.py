"""Info-Leak Detector agent (Agent 1).

Monitors political prediction markets for anomalous price/volume movements
that may indicate information leakage before public announcements.

Runs on a configurable tick interval (default 60s) per tracked market.
On each tick it:
  1. Computes price-return z-score against a 7-day EWMA baseline.
  2. Computes volume z-score against 7-day rolling hourly volume.
  3. If either z-score exceeds its threshold, computes a combined anomaly
     score and applies contextual amplifiers/dampeners (scheduled political
     events, pre-existing public news).
  4. Cross-references Agent 2 liquidity data for thin-market annotation.
  5. Emits an AnomalyReport via configured sinks when the final score
     exceeds the combined-score threshold.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
from datetime import datetime, timedelta, timezone
from typing import Any

import aiosqlite

from prediction_market.agents.base import BaseAgent
from prediction_market.analysis.price_analyzer import PriceAnalyzer, PriceAnomaly
from prediction_market.analysis.volume_analyzer import VolumeAnalyzer, VolumeAnomaly
from prediction_market.analysis.timeseries import RollingStats
from prediction_market.config import AppConfig
from prediction_market.data.external.news_checker import NewsChecker, NewsCheckResult
from prediction_market.reporting.anomaly_report import AnomalyReport
from prediction_market.store import queries

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal dataclass-like containers for intermediate results
# ---------------------------------------------------------------------------


class _MarketState:
    """Per-market mutable state held in memory between ticks."""

    __slots__ = ("market_id", "price_analyzer", "volume_analyzer", "rolling_stats")

    def __init__(self, market_id: str, window_days: int) -> None:
        self.market_id = market_id
        self.price_analyzer = PriceAnalyzer(window_days=window_days)
        self.volume_analyzer = VolumeAnalyzer(window_days=window_days)
        self.rolling_stats = RollingStats(window_days=window_days)


class _AnomalyContext:
    """Collects all evidence about a single anomaly before report creation."""

    __slots__ = (
        "market_id",
        "timestamp",
        "price_zscore",
        "volume_zscore",
        "raw_combined_score",
        "final_score",
        "price_anomaly",
        "volume_anomaly",
        "nearby_events",
        "news_result",
        "thin_liquidity",
        "amplifiers_applied",
        "dampeners_applied",
    )

    def __init__(self, market_id: str) -> None:
        self.market_id = market_id
        self.timestamp: datetime = datetime.now(timezone.utc)
        self.price_zscore: float = 0.0
        self.volume_zscore: float = 0.0
        self.raw_combined_score: float = 0.0
        self.final_score: float = 0.0
        self.price_anomaly: PriceAnomaly | None = None
        self.volume_anomaly: VolumeAnomaly | None = None
        self.nearby_events: list[dict[str, Any]] = []
        self.news_result: NewsCheckResult | None = None
        self.thin_liquidity: bool = False
        self.amplifiers_applied: list[str] = []
        self.dampeners_applied: list[str] = []


# ---------------------------------------------------------------------------
# Agent implementation
# ---------------------------------------------------------------------------


class InfoLeakDetector(BaseAgent):
    """Detects potential information leakage in political prediction markets.

    The detector maintains per-market rolling statistics (EWMA price baseline,
    hourly volume distribution) and fires when both the magnitude and context
    of a move suggest non-public information may be driving the market.

    Parameters
    ----------
    config : AppConfig
        Application-wide configuration.
    db : aiosqlite.Connection
        Shared database connection.
    news_checker : NewsChecker | None
        Optional NewsChecker instance. When *None* a new one is created from
        *config*.
    """

    name: str = "info_leak"

    def __init__(
        self,
        config: AppConfig,
        db: aiosqlite.Connection,
        news_checker: NewsChecker | None = None,
    ) -> None:
        super().__init__(
            name=self.name,
            tick_interval=config.polling.snapshot_interval_seconds,
            config=config,
        )
        self._config = config
        self._db = db
        self._thresholds = config.thresholds
        self._news_checker = news_checker or NewsChecker(config)

        # Per-market state keyed by market_id.
        self._states: dict[str, _MarketState] = {}

    # ------------------------------------------------------------------
    # BaseAgent lifecycle hooks
    # ------------------------------------------------------------------

    async def on_start(self) -> None:
        """Restore rolling stats from the database for restart resilience."""
        logger.info("InfoLeakDetector starting — restoring rolling stats")
        try:
            rows = await queries.get_rolling_stats(
                self._db, stat_types=["price", "volume"]
            )
            for row in rows:
                market_id: str = row["market_id"]
                state = self._get_or_create_state(market_id)
                serialized: str = row["serialized_data"]
                try:
                    data = json.loads(serialized)
                except (json.JSONDecodeError, TypeError):
                    logger.warning(
                        "Corrupt rolling_stats blob for market %s, type %s — skipping",
                        market_id,
                        row["stat_type"],
                    )
                    continue

                if row["stat_type"] == "price":
                    state.price_analyzer.restore(data)
                elif row["stat_type"] == "volume":
                    state.volume_analyzer.restore(data)

            logger.info(
                "Restored rolling stats for %d markets", len(self._states)
            )
        except Exception:
            logger.exception("Failed to restore rolling stats — starting fresh")

    async def on_stop(self) -> None:
        """Persist rolling stats to the database on shutdown."""
        logger.info("InfoLeakDetector stopping — persisting rolling stats")
        await self._persist_all_stats()

    async def tick(self) -> None:
        """Execute one full detection cycle across every tracked market."""
        market_ids = await self._get_tracked_market_ids()
        if not market_ids:
            logger.debug("No tracked markets — skipping tick")
            return

        for market_id in market_ids:
            try:
                await self._process_market(market_id)
            except Exception:
                logger.exception(
                    "Error processing market %s — continuing with remaining markets",
                    market_id,
                )

        # Periodically flush stats to DB for crash resilience.
        await self._persist_all_stats()

    # ------------------------------------------------------------------
    # Per-market processing
    # ------------------------------------------------------------------

    async def _process_market(self, market_id: str) -> None:
        """Run one detection cycle for a single market."""
        snapshot = await queries.get_latest_snapshot(self._db, market_id)
        if snapshot is None:
            logger.debug("No snapshot available for market %s", market_id)
            return

        price: float = snapshot["price_yes"]
        volume: float = snapshot.get("volume_24hr", 0.0) or 0.0
        timestamp_str: str = snapshot.get("timestamp", "")
        timestamp = self._parse_timestamp(timestamp_str)

        state = self._get_or_create_state(market_id)

        # --- 1. Update analyzers and compute z-scores -------------------
        price_anomaly: PriceAnomaly | None = state.price_analyzer.update(
            price, timestamp
        )
        volume_anomaly: VolumeAnomaly | None = state.volume_analyzer.update(
            volume, timestamp
        )

        price_z = price_anomaly.zscore if price_anomaly else 0.0
        volume_z = volume_anomaly.zscore if volume_anomaly else 0.0

        # --- 2. Check thresholds ----------------------------------------
        price_triggered = abs(price_z) >= self._thresholds.price_zscore
        volume_triggered = abs(volume_z) >= self._thresholds.volume_zscore

        if not (price_triggered or volume_triggered):
            return  # Nothing unusual — move on.

        # --- 3. Compute combined anomaly score --------------------------
        combined_raw = math.sqrt(price_z**2 + volume_z**2)

        ctx = _AnomalyContext(market_id)
        ctx.timestamp = timestamp
        ctx.price_zscore = price_z
        ctx.volume_zscore = volume_z
        ctx.raw_combined_score = combined_raw
        ctx.price_anomaly = price_anomaly
        ctx.volume_anomaly = volume_anomaly

        # --- 4. Apply contextual amplifiers / dampeners -----------------
        score = combined_raw
        score = await self._apply_event_amplifier(ctx, score, timestamp)
        score = await self._apply_news_dampener(ctx, score, market_id, timestamp)
        ctx.final_score = score

        # --- 5. Cross-reference Agent 2 liquidity data ------------------
        await self._annotate_liquidity(ctx, market_id)

        # --- 6. Emit if final score exceeds threshold -------------------
        if ctx.final_score >= self._thresholds.combined_score_min:
            await self._emit_anomaly(ctx)

    # ------------------------------------------------------------------
    # Contextual adjustments
    # ------------------------------------------------------------------

    async def _apply_event_amplifier(
        self, ctx: _AnomalyContext, score: float, timestamp: datetime
    ) -> float:
        """Amplify score if a scheduled political event is within +/-24 h."""
        proximity_hours = self._thresholds.event_proximity_hours
        window_start = timestamp - timedelta(hours=proximity_hours)
        window_end = timestamp + timedelta(hours=proximity_hours)

        try:
            events = await queries.get_scheduled_events_in_range(
                self._db,
                start=window_start.isoformat(),
                end=window_end.isoformat(),
            )
        except Exception:
            logger.warning(
                "Could not query scheduled_events for market %s",
                ctx.market_id,
                exc_info=True,
            )
            events = []

        if events:
            ctx.nearby_events = [
                {
                    "source": e["source"],
                    "title": e["title"],
                    "event_date": e["event_date"],
                }
                for e in events
            ]
            score *= self._thresholds.event_amplifier
            ctx.amplifiers_applied.append(
                f"event_proximity(x{self._thresholds.event_amplifier},"
                f" {len(events)} event(s) within ±{proximity_hours}h)"
            )
            logger.info(
                "Market %s: amplifying score %.2f -> %.2f (%d nearby events)",
                ctx.market_id,
                ctx.raw_combined_score,
                score,
                len(events),
            )

        return score

    async def _apply_news_dampener(
        self,
        ctx: _AnomalyContext,
        score: float,
        market_id: str,
        timestamp: datetime,
    ) -> float:
        """Dampen score if public news already existed at time of the move."""
        try:
            market_row = await queries.get_market(self._db, market_id)
            question = market_row["question"] if market_row else market_id
            news_result: NewsCheckResult = await self._news_checker.check(
                query=question,
                before=timestamp,
            )
            ctx.news_result = news_result
        except Exception:
            logger.warning(
                "News check failed for market %s — skipping dampener",
                market_id,
                exc_info=True,
            )
            return score

        if news_result.has_prior_news:
            dampener = 1.0 - self._thresholds.news_dampener
            previous_score = score
            score *= dampener
            ctx.dampeners_applied.append(
                f"prior_news(x{dampener:.2f}, "
                f"{news_result.article_count} article(s) found)"
            )
            logger.info(
                "Market %s: dampening score %.2f -> %.2f (prior news found)",
                market_id,
                previous_score,
                score,
            )

        return score

    async def _annotate_liquidity(
        self, ctx: _AnomalyContext, market_id: str
    ) -> None:
        """Check Agent 2's liquidity/orderbook data for thin-market flag."""
        try:
            ob_snap = await queries.get_latest_orderbook_snapshot(
                self._db, market_id
            )
            if ob_snap is None:
                return

            susceptibility: float = ob_snap.get("susceptibility_score", 0.0) or 0.0
            if susceptibility >= self._thresholds.susceptibility_threshold:
                ctx.thin_liquidity = True
                logger.info(
                    "Market %s flagged as thin liquidity (susceptibility=%.2f)",
                    market_id,
                    susceptibility,
                )
        except Exception:
            logger.debug(
                "Could not fetch orderbook data for market %s",
                market_id,
                exc_info=True,
            )

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    async def _emit_anomaly(self, ctx: _AnomalyContext) -> None:
        """Build an AnomalyReport and send it through configured sinks."""
        severity = self._classify_severity(ctx.final_score)

        summary_parts: list[str] = [
            f"Potential info-leak detected in market {ctx.market_id}.",
            f"Combined score {ctx.final_score:.2f}"
            f" (price z={ctx.price_zscore:.2f}, volume z={ctx.volume_zscore:.2f}).",
        ]
        if ctx.thin_liquidity:
            summary_parts.append("NOTE: market has thin liquidity.")
        if ctx.nearby_events:
            summary_parts.append(
                f"{len(ctx.nearby_events)} scheduled event(s) within proximity window."
            )
        if ctx.news_result and ctx.news_result.has_prior_news:
            summary_parts.append(
                f"Prior news detected ({ctx.news_result.article_count} article(s)) "
                "— score dampened."
            )
        summary = " ".join(summary_parts)

        details: dict[str, Any] = {
            "price_zscore": ctx.price_zscore,
            "volume_zscore": ctx.volume_zscore,
            "raw_combined_score": ctx.raw_combined_score,
            "final_score": ctx.final_score,
            "amplifiers": ctx.amplifiers_applied,
            "dampeners": ctx.dampeners_applied,
            "thin_liquidity": ctx.thin_liquidity,
        }

        price_evidence: dict[str, Any] = {}
        if ctx.price_anomaly is not None:
            price_evidence = {
                "zscore": ctx.price_anomaly.zscore,
                "current_price": ctx.price_anomaly.current_price,
                "ewma_baseline": ctx.price_anomaly.ewma_baseline,
                "return_pct": ctx.price_anomaly.return_pct,
            }

        volume_evidence: dict[str, Any] = {}
        if ctx.volume_anomaly is not None:
            volume_evidence = {
                "zscore": ctx.volume_anomaly.zscore,
                "current_volume": ctx.volume_anomaly.current_volume,
                "expected_volume": ctx.volume_anomaly.expected_volume,
            }

        calendar_matches: list[dict[str, Any]] = ctx.nearby_events

        news_check: dict[str, Any] = {}
        if ctx.news_result is not None:
            news_check = {
                "has_prior_news": ctx.news_result.has_prior_news,
                "article_count": ctx.news_result.article_count,
                "earliest_article": ctx.news_result.earliest_article,
                "sources": ctx.news_result.sources,
            }

        report = AnomalyReport(
            agent=self.name,
            market_id=ctx.market_id,
            severity=severity,
            anomaly_score=ctx.final_score,
            confidence=self._score_to_confidence(ctx.final_score),
            summary=summary,
            details=details,
            price_evidence=price_evidence,
            volume_evidence=volume_evidence,
            calendar_matches=calendar_matches,
            news_check=news_check,
        )

        # Persist to database
        try:
            await queries.insert_anomaly_report(self._db, report)
        except Exception:
            logger.exception(
                "Failed to persist anomaly report for market %s", ctx.market_id
            )

        # Emit through configured sinks (webhook, log, etc.)
        try:
            await self.emit(report)
        except Exception:
            logger.exception(
                "Failed to emit anomaly report for market %s", ctx.market_id
            )

        logger.warning(
            "ANOMALY [%s] market=%s score=%.2f severity=%s — %s",
            self.name,
            ctx.market_id,
            ctx.final_score,
            severity,
            summary,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_or_create_state(self, market_id: str) -> _MarketState:
        """Return (or create) the in-memory state for *market_id*."""
        if market_id not in self._states:
            self._states[market_id] = _MarketState(
                market_id, window_days=self._thresholds.rolling_window_days
            )
        return self._states[market_id]

    async def _get_tracked_market_ids(self) -> list[str]:
        """Fetch the set of active political markets to monitor."""
        try:
            rows = await queries.get_active_political_markets(self._db)
            return [r["id"] for r in rows]
        except Exception:
            logger.exception("Could not fetch tracked market list")
            return []

    async def _persist_all_stats(self) -> None:
        """Serialize every market's rolling stats to the database."""
        for market_id, state in self._states.items():
            try:
                price_data = state.price_analyzer.serialize()
                await queries.upsert_rolling_stats(
                    self._db,
                    market_id=market_id,
                    stat_type="price",
                    window_days=self._thresholds.rolling_window_days,
                    serialized_data=json.dumps(price_data),
                )

                volume_data = state.volume_analyzer.serialize()
                await queries.upsert_rolling_stats(
                    self._db,
                    market_id=market_id,
                    stat_type="volume",
                    window_days=self._thresholds.rolling_window_days,
                    serialized_data=json.dumps(volume_data),
                )
            except Exception:
                logger.exception(
                    "Failed to persist rolling stats for market %s", market_id
                )

    @staticmethod
    def _classify_severity(score: float) -> str:
        """Map a combined anomaly score to a severity label."""
        if score >= 8.0:
            return "critical"
        if score >= 6.0:
            return "high"
        if score >= 4.0:
            return "medium"
        return "low"

    @staticmethod
    def _score_to_confidence(score: float) -> float:
        """Convert the final anomaly score to a 0.0-1.0 confidence value.

        Uses a logistic-style mapping: confidence approaches 1.0 as the
        score grows, with 0.5 at the minimum threshold (4.0).
        """
        # Logistic: 1 / (1 + exp(-k*(x - x0)))  where x0=4.0, k=0.5
        try:
            return 1.0 / (1.0 + math.exp(-0.5 * (score - 4.0)))
        except OverflowError:
            return 1.0 if score > 4.0 else 0.0

    @staticmethod
    def _parse_timestamp(ts: str) -> datetime:
        """Parse an ISO-format timestamp string to a UTC datetime."""
        if not ts:
            return datetime.now(timezone.utc)
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except (ValueError, TypeError):
            return datetime.now(timezone.utc)
