"""Main asyncio orchestrator for the Polymarket surveillance system.

Coordinates market discovery, data ingestion, anomaly detection agents,
and report output into a single long-running event loop with graceful
shutdown and error isolation between components.
"""

from __future__ import annotations

import asyncio
import logging
import signal
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import aiosqlite
import httpx

from prediction_market.config import AppConfig
from prediction_market.data.political_filter import PoliticalClassification, PoliticalFilter
from prediction_market.data.polymarket.clob_client import ClobClient
from prediction_market.data.polymarket.data_client import DataClient
from prediction_market.data.polymarket.gamma_client import GammaClient
from prediction_market.data.polymarket.models import GammaMarket
from prediction_market.reporting.sink import (
    CompositeSink,
    FileSink,
    ReportSink,
    StdoutSink,
    WebhookSink,
)
from prediction_market.store.database import init_database

# Lazy imports for modules that may not exist yet ----------------------------
# These are structured so the orchestrator file can be imported even if the
# downstream agent or WebSocket modules are still being developed.
try:
    from prediction_market.agents.info_leak_detector import InfoLeakDetector
except ImportError:  # pragma: no cover
    InfoLeakDetector = None  # type: ignore[assignment,misc]

try:
    from prediction_market.agents.manipulation_guard import ManipulationGuard
except ImportError:  # pragma: no cover
    ManipulationGuard = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tracked-market state
# ---------------------------------------------------------------------------


@dataclass
class TrackedMarket:
    """In-memory representation of a market we are actively monitoring."""

    market: GammaMarket
    classification: PoliticalClassification
    last_snapshot: datetime | None = None
    last_orderbook: datetime | None = None
    extra: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class Orchestrator:
    """Coordinates all surveillance components in a single asyncio event loop.

    Lifecycle::

        orchestrator = Orchestrator(config)
        await orchestrator.start()   # blocks until shutdown signal
        await orchestrator.stop()    # graceful teardown

    Parameters
    ----------
    config:
        Fully resolved :class:`AppConfig` instance.
    agent_filter:
        Optional string restricting which agents are launched.
        ``"info-leak"`` starts only the information-leak detector;
        ``"manipulation"`` starts only the manipulation guard.
        ``None`` (default) starts both.
    """

    def __init__(
        self,
        config: AppConfig,
        *,
        agent_filter: str | None = None,
    ) -> None:
        self.config = config
        self._agent_filter = agent_filter

        # Core resources (initialised in ``start``)
        self._db: aiosqlite.Connection | None = None
        self._http: httpx.AsyncClient | None = None
        self._gamma: GammaClient | None = None
        self._clob: ClobClient | None = None
        self._data: DataClient | None = None
        self._filter: PoliticalFilter | None = None
        self._sink: ReportSink | None = None

        # Tracked state
        self.markets: dict[str, TrackedMarket] = {}

        # Agent handles
        self._agents: list[Any] = []

        # Background tasks
        self._tasks: list[asyncio.Task[None]] = []
        self._shutdown_event = asyncio.Event()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Initialise all subsystems and run until a shutdown signal fires."""
        logger.info("Orchestrator starting up")

        # 1. Database
        self._db = await init_database(self.config)
        logger.info("Database ready")

        # 2. Shared HTTP client
        self._http = httpx.AsyncClient(
            timeout=30.0,
            limits=httpx.Limits(max_connections=50, max_keepalive_connections=20),
        )

        # 3. API clients
        self._gamma = GammaClient(self.config, http_client=self._http)
        self._clob = ClobClient(self.config, http_client=self._http)
        self._data = DataClient(self.config, http_client=self._http)

        # 4. Political filter
        self._filter = PoliticalFilter()

        # 5. Discover political markets
        await self._discover_markets()

        # 6. Report sinks
        self._sink = self._build_sinks()

        # 7. Agents
        self._agents = self._build_agents()
        if self._agents:
            agent_names = [type(a).__name__ for a in self._agents]
            logger.info("Agents initialised: %s", ", ".join(agent_names))
        else:
            logger.warning("No agents were initialised (missing implementations?)")

        # 8. Install signal handlers
        self._install_signal_handlers()

        # 9. Launch periodic background tasks
        self._tasks.append(
            asyncio.create_task(
                self._periodic_market_discovery(),
                name="market-discovery",
            )
        )
        self._tasks.append(
            asyncio.create_task(
                self._periodic_snapshot_loop(),
                name="snapshot-loop",
            )
        )

        # Launch agent loops
        for agent in self._agents:
            task = asyncio.create_task(
                self._run_agent(agent),
                name=f"agent-{type(agent).__name__}",
            )
            self._tasks.append(task)

        logger.info(
            "Orchestrator running  --  tracking %d political markets, %d agents active",
            len(self.markets),
            len(self._agents),
        )

        # Block until shutdown
        await self._shutdown_event.wait()
        await self.stop()

    async def stop(self) -> None:
        """Gracefully shut down all background tasks and release resources."""
        logger.info("Orchestrator shutting down")

        # Cancel tasks
        for task in self._tasks:
            if not task.done():
                task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

        # Close agents
        for agent in self._agents:
            if hasattr(agent, "close"):
                try:
                    await agent.close()
                except Exception:
                    logger.exception("Error closing agent %s", type(agent).__name__)

        # Close sinks
        if self._sink is not None:
            try:
                await self._sink.close()
            except Exception:
                logger.exception("Error closing report sink")

        # Close API clients (they do NOT own the shared http client)
        for client in (self._gamma, self._clob, self._data):
            if client is not None:
                try:
                    await client.close()
                except Exception:
                    logger.exception("Error closing %s", type(client).__name__)

        # Close shared HTTP client
        if self._http is not None:
            await self._http.aclose()

        # Close database
        if self._db is not None:
            await self._db.close()

        logger.info("Orchestrator stopped")

    # ------------------------------------------------------------------
    # One-shot scan (non-continuous)
    # ------------------------------------------------------------------

    async def scan_once(self) -> list[dict[str, Any]]:
        """Run a single pass over all political markets and return results.

        This is the backend for the ``prediction-market scan`` CLI command.
        It does *not* start background loops or install signal handlers.

        Returns
        -------
        list[dict[str, Any]]
            Summary dicts for each tracked political market.
        """
        self._db = await init_database(self.config)
        self._http = httpx.AsyncClient(
            timeout=30.0,
            limits=httpx.Limits(max_connections=50, max_keepalive_connections=20),
        )
        self._gamma = GammaClient(self.config, http_client=self._http)
        self._clob = ClobClient(self.config, http_client=self._http)
        self._data = DataClient(self.config, http_client=self._http)
        self._filter = PoliticalFilter()

        await self._discover_markets()

        results: list[dict[str, Any]] = []
        for market_id, tracked in self.markets.items():
            m = tracked.market
            results.append(
                {
                    "id": m.id,
                    "question": m.question,
                    "volume": m.volume,
                    "liquidity": m.liquidity,
                    "active": m.active,
                    "category": m.category,
                    "political_confidence": tracked.classification.confidence,
                    "political_reasons": tracked.classification.reasons,
                }
            )

        # Tear down
        await self._gamma.close()
        await self._clob.close()
        await self._data.close()
        await self._http.aclose()
        await self._db.close()

        return results

    # ------------------------------------------------------------------
    # Backfill
    # ------------------------------------------------------------------

    async def backfill(self, days: int = 30) -> int:
        """Populate the database with historical price data.

        Parameters
        ----------
        days:
            How many days of history to fetch per market.

        Returns
        -------
        int
            Total number of price history points ingested.
        """
        import time

        from prediction_market.store.snapshots import save_market, save_price_snapshot

        self._db = await init_database(self.config)
        self._http = httpx.AsyncClient(
            timeout=30.0,
            limits=httpx.Limits(max_connections=50, max_keepalive_connections=20),
        )
        self._gamma = GammaClient(self.config, http_client=self._http)
        self._clob = ClobClient(self.config, http_client=self._http)
        self._data = DataClient(self.config, http_client=self._http)
        self._filter = PoliticalFilter()

        await self._discover_markets()

        end_ts = int(time.time())
        start_ts = end_ts - (days * 86400)
        total_points = 0

        for market_id, tracked in self.markets.items():
            m = tracked.market
            logger.info("Backfilling %s (%s)", market_id, m.question[:60])

            # Persist the market record
            await save_market(
                self._db,
                m,
                {
                    "confidence": tracked.classification.confidence,
                    "reasons": tracked.classification.reasons,
                },
            )

            # Fetch price history for each token
            for token_id in m.clob_token_ids:
                try:
                    history = await self._clob.get_price_history(
                        token_id,
                        start_ts=start_ts,
                        end_ts=end_ts,
                        interval="1d",
                        fidelity=days,
                    )
                    for point in history.history:
                        await save_price_snapshot(
                            self._db,
                            market_id,
                            price_yes=point.p if token_id == m.clob_token_ids[0] else None,
                            price_no=point.p if len(m.clob_token_ids) > 1 and token_id == m.clob_token_ids[1] else None,
                        )
                        total_points += 1
                except Exception:
                    logger.exception(
                        "Error backfilling token %s for market %s", token_id, market_id
                    )

            # Fetch and store recent trades
            try:
                from prediction_market.store.snapshots import save_trades_batch

                trades = await self._data.get_all_trades(
                    condition_id=m.condition_id, max_pages=10
                )
                if trades:
                    inserted = await save_trades_batch(self._db, trades, market_id)
                    logger.info("Backfilled %d trades for %s", inserted, market_id)
            except Exception:
                logger.exception("Error backfilling trades for market %s", market_id)

        # Tear down
        await self._gamma.close()
        await self._clob.close()
        await self._data.close()
        await self._http.aclose()
        await self._db.close()

        return total_points

    # ------------------------------------------------------------------
    # Market discovery
    # ------------------------------------------------------------------

    async def _discover_markets(self) -> None:
        """Fetch all active markets from Gamma and filter to political ones."""
        assert self._gamma is not None
        assert self._filter is not None
        assert self._db is not None

        from prediction_market.store.snapshots import save_market

        logger.info("Discovering political markets...")

        try:
            all_markets = await self._gamma.get_all_markets(active=True, closed=False)
        except Exception:
            logger.exception("Failed to fetch markets from Gamma API")
            return

        newly_tracked = 0
        for m in all_markets:
            classification = self._filter.classify(m)
            if not classification.is_political:
                continue
            # Volume filter
            if m.volume < self._filter._min_volume:
                continue

            if m.id not in self.markets:
                newly_tracked += 1

            self.markets[m.id] = TrackedMarket(
                market=m,
                classification=classification,
            )

            # Persist to DB
            try:
                await save_market(
                    self._db,
                    m,
                    {
                        "confidence": classification.confidence,
                        "reasons": classification.reasons,
                    },
                )
            except Exception:
                logger.exception("Failed to persist market %s", m.id)

        logger.info(
            "Market discovery complete: %d political markets tracked (%d new)",
            len(self.markets),
            newly_tracked,
        )

    # ------------------------------------------------------------------
    # Periodic loops
    # ------------------------------------------------------------------

    async def _periodic_market_discovery(self) -> None:
        """Re-run market discovery on a configurable interval."""
        interval = self.config.polling.market_discovery_interval_seconds
        while not self._shutdown_event.is_set():
            try:
                await asyncio.wait_for(
                    self._shutdown_event.wait(),
                    timeout=interval,
                )
                # If we get here, the event was set -- time to exit
                return
            except asyncio.TimeoutError:
                pass
            try:
                await self._discover_markets()
            except Exception:
                logger.exception("Error during periodic market discovery")

    async def _periodic_snapshot_loop(self) -> None:
        """Periodically snapshot prices and order books for tracked markets."""
        from prediction_market.store.snapshots import (
            save_orderbook_snapshot,
            save_price_snapshot,
        )

        snapshot_interval = self.config.polling.snapshot_interval_seconds
        orderbook_interval = self.config.polling.orderbook_interval_seconds

        last_orderbook_run = datetime.now(timezone.utc)

        while not self._shutdown_event.is_set():
            try:
                await asyncio.wait_for(
                    self._shutdown_event.wait(),
                    timeout=snapshot_interval,
                )
                return
            except asyncio.TimeoutError:
                pass

            now = datetime.now(timezone.utc)
            run_orderbook = (now - last_orderbook_run).total_seconds() >= orderbook_interval

            for market_id, tracked in list(self.markets.items()):
                m = tracked.market
                # -- Price snapshot --
                try:
                    if m.clob_token_ids:
                        price_yes = await self._clob.get_midpoint(m.clob_token_ids[0])
                        price_no = None
                        if len(m.clob_token_ids) > 1:
                            price_no = await self._clob.get_midpoint(m.clob_token_ids[1])
                        await save_price_snapshot(
                            self._db,
                            market_id,
                            price_yes=price_yes,
                            price_no=price_no,
                            volume_total=m.volume,
                            liquidity=m.liquidity,
                        )
                        tracked.last_snapshot = now
                except Exception:
                    logger.exception("Snapshot failed for market %s", market_id)

                # -- Order book snapshot --
                if run_orderbook:
                    for token_id in m.clob_token_ids:
                        try:
                            ob = await self._clob.get_order_book(token_id)
                            await save_orderbook_snapshot(
                                self._db, market_id, token_id, ob
                            )
                            tracked.last_orderbook = now
                        except Exception:
                            logger.exception(
                                "Orderbook snapshot failed for market %s token %s",
                                market_id,
                                token_id,
                            )

            if run_orderbook:
                last_orderbook_run = now

    # ------------------------------------------------------------------
    # Agent runner (error isolation)
    # ------------------------------------------------------------------

    async def _run_agent(self, agent: Any) -> None:
        """Run a single agent in an isolated loop.

        If the agent raises, the error is logged and the loop retries
        after a brief backoff.  One agent crashing does not affect others.
        """
        name = type(agent).__name__
        backoff = 5  # seconds
        max_backoff = 300  # 5 minutes

        while not self._shutdown_event.is_set():
            try:
                await agent.run(
                    markets=self.markets,
                    db=self._db,
                    clob=self._clob,
                    data=self._data,
                    sink=self._sink,
                    shutdown=self._shutdown_event,
                    config=self.config,
                )
                # If run() returns cleanly, the agent decided to exit
                logger.info("Agent %s exited cleanly", name)
                return
            except asyncio.CancelledError:
                logger.info("Agent %s cancelled", name)
                return
            except Exception:
                logger.exception(
                    "Agent %s crashed; restarting in %ds", name, backoff
                )
                try:
                    await asyncio.wait_for(
                        self._shutdown_event.wait(),
                        timeout=backoff,
                    )
                    return  # shutdown requested during backoff
                except asyncio.TimeoutError:
                    pass
                backoff = min(backoff * 2, max_backoff)

    # ------------------------------------------------------------------
    # Builders
    # ------------------------------------------------------------------

    def _build_sinks(self) -> ReportSink:
        """Construct the report sink pipeline from config."""
        sinks: list[ReportSink] = []

        # Always write to files
        sinks.append(FileSink(self.config.reporting.output_dir))

        # Stdout for development / interactive use
        sinks.append(StdoutSink())

        # Optional webhook
        if self.config.webhook_url:
            sinks.append(
                WebhookSink(self.config.webhook_url, http_client=self._http)
            )

        if len(sinks) == 1:
            return sinks[0]
        return CompositeSink(sinks)

    def _build_agents(self) -> list[Any]:
        """Instantiate the configured surveillance agents."""
        agents: list[Any] = []

        want_info_leak = self._agent_filter in (None, "info-leak")
        want_manipulation = self._agent_filter in (None, "manipulation")

        if want_info_leak and InfoLeakDetector is not None:
            try:
                agents.append(InfoLeakDetector(self.config))
                logger.info("InfoLeakDetector initialised")
            except Exception:
                logger.exception("Failed to initialise InfoLeakDetector")

        if want_manipulation and ManipulationGuard is not None:
            try:
                agents.append(ManipulationGuard(self.config))
                logger.info("ManipulationGuard initialised")
            except Exception:
                logger.exception("Failed to initialise ManipulationGuard")

        return agents

    # ------------------------------------------------------------------
    # Signal handling
    # ------------------------------------------------------------------

    def _install_signal_handlers(self) -> None:
        """Register SIGINT and SIGTERM to trigger graceful shutdown."""
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, self._handle_signal, sig)
            except NotImplementedError:
                # Windows does not support add_signal_handler
                logger.debug("Cannot install handler for %s on this platform", sig)

    def _handle_signal(self, sig: signal.Signals) -> None:
        """Set the shutdown event when a termination signal is received."""
        logger.info("Received signal %s -- initiating shutdown", sig.name)
        self._shutdown_event.set()
