"""Info-Leak Detector agent (Agent 1).

Monitors political prediction markets for anomalous price/volume movements
that may indicate information leakage before public announcements.

Runs on a configurable tick interval (default 60s). On each tick it:
  1. Loads every active political market and its latest snapshot.
  2. Feeds the price into a per-market :class:`~prediction_market.analysis
     .price_analyzer.PriceAnalyzer` and the per-interval volume delta into a
     per-market :class:`~prediction_market.analysis.volume_analyzer
     .VolumeAnalyzer`; both compute z-scores from a rolling-window
     :class:`~prediction_market.analysis.timeseries.RollingStats` of
     returns/volume, not an EWMA baseline.
  2. If either z-score exceeds its threshold, computes a combined anomaly
     score and applies contextual amplifiers/dampeners (scheduled political
     events, pre-existing public news).
  3. Cross-references order-book data for a thin-liquidity annotation.
  4. Emits an AnomalyReport via configured sinks when the final score
     exceeds the combined-score threshold and the market is not in
     cooldown.

State (rolling stats, per-market volume totals, alert cooldowns) is kept
in memory only. A restart resets it: rolling baselines need to warm back up
(3 snapshots, roughly 3 minutes at the default 60s cadence) and cooldowns
are forgotten. This is a deliberate scope choice — DB-backed persistence of
analyzer/cooldown state is not implemented here.
"""

from __future__ import annotations

import logging
import math
import re
from datetime import datetime, timedelta, timezone
from typing import Any

import aiosqlite

from prediction_market.agents.base import BaseAgent
from prediction_market.analysis.price_analyzer import PriceAnalyzer
from prediction_market.analysis.volume_analyzer import VolumeAnalyzer
from prediction_market.config import AppConfig
from prediction_market.data.external.news_checker import NewsChecker
from prediction_market.reporting.anomaly_report import AnomalyReport
from prediction_market.reporting.sink import ReportSink
from prediction_market.store import queries

logger = logging.getLogger(__name__)

_STOPWORDS = frozenset(
    {
        "will",
        "this",
        "that",
        "with",
        "have",
        "been",
        "before",
        "after",
        "than",
        "them",
        "they",
        "there",
        "what",
        "when",
        "does",
        "from",
        "into",
        "over",
        "under",
        "between",
    }
)

_MAX_KEYWORDS = 6


def _extract_keywords(question: str) -> list[str]:
    """Extract up to 6 salient keywords from a market question.

    Splits on non-alphanumeric characters, lowercases, drops tokens under
    4 characters and tokens in :data:`_STOPWORDS`, and caps the result at
    the first ``_MAX_KEYWORDS`` remaining tokens in question order.
    """
    tokens = re.split(r"[^a-zA-Z0-9]+", question.lower())
    keywords: list[str] = []
    for token in tokens:
        if len(token) < 4:
            continue
        if token in _STOPWORDS:
            continue
        keywords.append(token)
        if len(keywords) >= _MAX_KEYWORDS:
            break
    return keywords


def _parse_timestamp(ts: str) -> datetime:
    """Parse a snapshot timestamp (stored as ``%Y-%m-%d %H:%M:%S`` text) to UTC."""
    try:
        return datetime.strptime(ts, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        pass
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError, AttributeError):
        return datetime.now(timezone.utc)


class InfoLeakDetector(BaseAgent):
    """Detects potential information leakage in political prediction markets.

    The detector holds one :class:`PriceAnalyzer` and one
    :class:`VolumeAnalyzer`, each keyed internally by ``market_id``, and
    combines their z-scores with scheduled-event and news-existence context
    into a single anomaly score per market per tick.
    """

    def __init__(
        self,
        config: AppConfig,
        db: aiosqlite.Connection,
        sinks: list[ReportSink] | None = None,
        news_checker: NewsChecker | None = None,
    ) -> None:
        super().__init__(config, db, sinks)
        self._price_analyzer = PriceAnalyzer(config.thresholds)
        self._volume_analyzer = VolumeAnalyzer(config.thresholds)
        self._news_checker = news_checker or NewsChecker(config)

        # In-memory only — see module docstring.
        self._last_processed: dict[str, str] = {}
        self._last_volume_total: dict[str, float] = {}
        self._last_alert: dict[str, datetime] = {}

    # -- BaseAgent properties -------------------------------------------

    @property
    def name(self) -> str:
        return "info_leak"

    @property
    def tick_interval_seconds(self) -> int:
        return self.config.polling.snapshot_interval_seconds

    # -- Core tick --------------------------------------------------------

    async def tick(self) -> None:
        """Run one detection cycle across every tracked market."""
        markets = await queries.get_active_political_markets(self.db)
        if not markets:
            logger.debug("InfoLeakDetector: no tracked markets")
            return

        for market in markets:
            await self._process_market(market)

    async def _process_market(self, market: dict[str, Any]) -> None:
        market_id: str = market["id"]
        question: str = market["question"]
        thresholds = self.config.thresholds

        snap = await queries.get_latest_snapshot(self.db, market_id)
        if snap is None:
            return

        ts_str: str = snap["timestamp"]
        if self._last_processed.get(market_id) == ts_str:
            # Snapshot loop writes every 60s; an unchanged row must not be
            # double-counted into the rolling stats.
            return
        self._last_processed[market_id] = ts_str
        ts = _parse_timestamp(ts_str)

        price = snap.get("price_yes")
        if price is None:
            return

        # --- Volume: feed the per-interval delta, not the rolling total ---
        volume_total = snap.get("volume_total")
        if volume_total is None:
            volume_total = 0.0
        previous_total = self._last_volume_total.get(market_id)
        self._last_volume_total[market_id] = volume_total
        if previous_total is not None:
            delta = max(0.0, volume_total - previous_total)
            self._volume_analyzer.update(market_id, delta, ts)

        # --- Price: always update -------------------------------------
        self._price_analyzer.update(market_id, price, ts)

        price_z = self._price_analyzer.current_z_score(market_id) or 0.0
        vol_z = self._volume_analyzer.current_z_score(market_id) or 0.0

        price_triggered = abs(price_z) >= thresholds.price_zscore
        volume_triggered = abs(vol_z) >= thresholds.volume_zscore
        if not (price_triggered or volume_triggered):
            return

        raw_combined = math.sqrt(price_z**2 + vol_z**2)
        combined = raw_combined
        amplifiers_applied: list[str] = []
        dampeners_applied: list[str] = []

        # --- Event amplifier ---------------------------------------------
        proximity = timedelta(hours=thresholds.event_proximity_hours)
        events = await queries.get_scheduled_events_in_range(
            self.db, ts - proximity, ts + proximity
        )
        calendar_matches: list[dict[str, Any]] = [
            {"source": e["source"], "title": e["title"], "event_date": e["event_date"]}
            for e in events
        ]
        if events:
            combined *= thresholds.event_amplifier
            amplifiers_applied.append(
                f"event_proximity(x{thresholds.event_amplifier}, "
                f"{len(events)} event(s) within +/-{thresholds.event_proximity_hours}h)"
            )

        # --- News dampener -------------------------------------------------
        keywords = _extract_keywords(question)
        news_result = await self._news_checker.check_news_exists(
            keywords, before_time=ts
        )
        if news_result.news_found:
            dampener = 1.0 - thresholds.news_dampener
            combined *= dampener
            dampeners_applied.append(
                f"prior_news(x{dampener:.2f}, "
                f"{len(news_result.articles)} article(s) found)"
            )

        # --- Thin-liquidity annotation (informational only) ----------------
        ob_snap = await queries.get_latest_orderbook_snapshot(self.db, market_id)
        thin_liquidity = (
            ob_snap is not None
            and ob_snap.get("susceptibility_score") is not None
            and ob_snap["susceptibility_score"] >= thresholds.susceptibility_threshold
        )

        if combined < thresholds.combined_score_min:
            return

        last_alert = self._last_alert.get(market_id)
        cooldown = timedelta(minutes=thresholds.alert_cooldown_minutes)
        if last_alert is not None and (ts - last_alert) < cooldown:
            return

        self._last_alert[market_id] = ts

        await self._emit_report(
            market_id=market_id,
            question=question,
            ts=ts,
            price=price,
            price_z=price_z,
            vol_z=vol_z,
            price_triggered=price_triggered,
            volume_triggered=volume_triggered,
            raw_combined=raw_combined,
            combined=combined,
            amplifiers_applied=amplifiers_applied,
            dampeners_applied=dampeners_applied,
            calendar_matches=calendar_matches,
            news_result=news_result,
            thin_liquidity=thin_liquidity,
        )

    # -- Reporting --------------------------------------------------------

    async def _emit_report(
        self,
        *,
        market_id: str,
        question: str,
        ts: datetime,
        price: float,
        price_z: float,
        vol_z: float,
        price_triggered: bool,
        volume_triggered: bool,
        raw_combined: float,
        combined: float,
        amplifiers_applied: list[str],
        dampeners_applied: list[str],
        calendar_matches: list[dict[str, Any]],
        news_result: Any,
        thin_liquidity: bool,
    ) -> None:
        confidence = self._score_to_confidence(combined)
        severity = AnomalyReport.severity_from_score(confidence)

        triggers: list[str] = []
        if price_triggered:
            triggers.append("price")
        if volume_triggered:
            triggers.append("volume")
        trigger_desc = " and ".join(triggers)

        summary = (
            f'Potential info leak detected in "{question}": combined anomaly '
            f"score {combined:.2f} (triggered by {trigger_desc} movement)."
        )

        price_anomaly = self._price_analyzer.check_anomaly(market_id)
        if price_anomaly is not None:
            price_evidence = {
                "z_score": price_anomaly.z_score,
                "current_price": price_anomaly.current_price,
                "baseline_price": price_anomaly.baseline_price,
                "price_return": price_anomaly.price_return,
            }
        else:
            price_evidence = {"z_score": price_z, "current_price": price}

        volume_anomaly = self._volume_analyzer.check_anomaly(market_id)
        if volume_anomaly is not None:
            volume_evidence = {
                "z_score": volume_anomaly.z_score,
                "delta": volume_anomaly.current_volume,
                "mean_volume": volume_anomaly.mean_volume,
            }
        else:
            volume_evidence = {"z_score": vol_z}

        news_check = {
            "news_found": news_result.news_found,
            "article_count": len(news_result.articles),
            "earliest_article_time": (
                news_result.earliest_article_time.isoformat()
                if news_result.earliest_article_time
                else None
            ),
            "query_keywords": news_result.query_keywords,
        }

        details = {
            "score_scale": "raw (not 0-1); confidence is the 0-1 mapped value",
            "raw_combined_score": raw_combined,
            "final_score": combined,
            "price_zscore": price_z,
            "volume_zscore": vol_z,
            "amplifiers_applied": amplifiers_applied,
            "dampeners_applied": dampeners_applied,
            "thin_liquidity": thin_liquidity,
        }

        report = AnomalyReport(
            id=AnomalyReport.new_id(),
            agent="info_leak",
            market_id=market_id,
            market_question=question,
            severity=severity,
            anomaly_score=combined,
            confidence=confidence,
            summary=summary,
            details=details,
            price_evidence=price_evidence,
            volume_evidence=volume_evidence,
            calendar_matches=calendar_matches,
            news_check=news_check,
            created_at=datetime.now(timezone.utc),
        )

        await self.emit(report)

        logger.warning(
            "ANOMALY [info_leak] market=%s score=%.2f severity=%s — %s",
            market_id,
            combined,
            severity,
            summary,
        )

    # -- Helpers ------------------------------------------------------------

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
