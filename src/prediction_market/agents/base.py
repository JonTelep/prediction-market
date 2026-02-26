"""Abstract base agent with built-in async scheduling and error handling."""

from __future__ import annotations

import abc
import asyncio
import logging

import aiosqlite

from prediction_market.config import AppConfig
from prediction_market.reporting.anomaly_report import AnomalyReport
from prediction_market.reporting.sink import ReportSink

logger = logging.getLogger(__name__)


class BaseAgent(abc.ABC):
    """Lifecycle-managed surveillance agent.

    Subclasses implement :meth:`tick` which is invoked on a fixed
    interval.  The base class handles scheduling, graceful shutdown,
    error isolation (a failed tick is logged and the loop continues),
    and report dispatch to configured sinks.
    """

    def __init__(
        self,
        config: AppConfig,
        db: aiosqlite.Connection,
        sinks: list[ReportSink] | None = None,
    ) -> None:
        self.config = config
        self.db = db
        self.sinks: list[ReportSink] = sinks or []
        self._running = False
        self._task: asyncio.Task[None] | None = None

    # -- Properties that subclasses must define -------------------------

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Human-readable name of the agent (e.g. ``'manipulation_guard'``)."""

    @property
    @abc.abstractmethod
    def tick_interval_seconds(self) -> int:
        """Seconds between successive ticks."""

    # -- Abstract tick --------------------------------------------------

    @abc.abstractmethod
    async def tick(self) -> None:
        """Execute one cycle of surveillance logic.

        Any :class:`AnomalyReport` instances produced should be passed
        to :meth:`emit` for dispatch to sinks and persistence.
        """

    # -- Lifecycle ------------------------------------------------------

    async def start(self) -> None:
        """Begin the scheduling loop in the background."""
        if self._running:
            logger.warning("Agent %s is already running", self.name)
            return
        self._running = True
        self._task = asyncio.create_task(self._loop(), name=f"agent-{self.name}")
        logger.info("Agent %s started (interval=%ds)", self.name, self.tick_interval_seconds)

    async def stop(self) -> None:
        """Signal the loop to stop and wait for it to finish."""
        if not self._running:
            return
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("Agent %s stopped", self.name)

    # -- Internal loop --------------------------------------------------

    async def _loop(self) -> None:
        """Run ticks at the configured interval until stopped."""
        while self._running:
            try:
                logger.debug("Agent %s tick starting", self.name)
                await self.tick()
                logger.debug("Agent %s tick complete", self.name)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.on_error(exc)
            try:
                await asyncio.sleep(self.tick_interval_seconds)
            except asyncio.CancelledError:
                break

    # -- Emit & persist -------------------------------------------------

    async def emit(self, report: AnomalyReport) -> None:
        """Dispatch a report to all configured sinks and persist to DB."""
        logger.info(
            "Agent %s emitting report %s (severity=%s, score=%.3f)",
            self.name,
            report.id,
            report.severity,
            report.anomaly_score,
        )
        # Persist to database
        await self._persist_report(report)

        # Fan-out to sinks
        for sink in self.sinks:
            try:
                await sink.write(report)
            except Exception:
                logger.exception(
                    "Sink %s failed for report %s", type(sink).__name__, report.id
                )

    async def _persist_report(self, report: AnomalyReport) -> None:
        """Insert the report into the anomaly_reports table."""
        import json

        try:
            await self.db.execute(
                """
                INSERT INTO anomaly_reports
                    (agent, market_id, severity, anomaly_score, confidence,
                     summary, details, price_evidence, volume_evidence,
                     calendar_matches, news_check, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    report.agent,
                    report.market_id,
                    report.severity,
                    report.anomaly_score,
                    report.confidence,
                    report.summary,
                    json.dumps(report.details, default=str),
                    json.dumps(report.price_evidence, default=str),
                    json.dumps(report.volume_evidence, default=str),
                    json.dumps(report.calendar_matches, default=str),
                    json.dumps(report.news_check, default=str),
                    report.created_at.isoformat(),
                ),
            )
            await self.db.commit()
        except Exception:
            logger.exception("Failed to persist report %s", report.id)

    # -- Error handling -------------------------------------------------

    def on_error(self, exc: Exception) -> None:
        """Handle an exception raised during a tick.

        Default behaviour: log the traceback and continue.  Subclasses
        may override to implement back-off, alerting, etc.
        """
        logger.exception("Agent %s tick failed: %s", self.name, exc)
