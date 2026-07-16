"""RTDS WebSocket client for real-time trade data.

Connects to the Polymarket Real-Time Data Service WebSocket feed to receive
live trade events for subscribed markets.  Implemented but not yet wired
into the orchestrator (planned for Phase 3); REST polling via DataClient
is the live trade-data path today.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable
from types import TracebackType
from typing import Any

import websockets
from websockets.asyncio.client import ClientConnection

from prediction_market.config import AppConfig

logger = logging.getLogger(__name__)


class RtdsWebSocket:
    """Async WebSocket client for Polymarket RTDS trade subscriptions.

    Receives real-time trade messages for subscribed market IDs and
    dispatches them to a caller-provided callback.  Supports automatic
    reconnection with exponential backoff.

    Usage::

        async with RtdsWebSocket(config, on_trade=handle_trade) as ws:
            await ws.subscribe(["market-slug-1", "market-slug-2"])
            await some_shutdown_event.wait()
    """

    def __init__(
        self,
        config: AppConfig,
        on_trade: Callable[[dict[str, Any]], Any],
    ) -> None:
        self.config = config
        self._url = config.websocket.rtds_url
        self._reconnect_delay = config.websocket.reconnect_delay_seconds
        self._max_reconnect_attempts = config.websocket.max_reconnect_attempts
        self._on_trade = on_trade

        self._ws: ClientConnection | None = None
        self._listen_task: asyncio.Task[None] | None = None
        self._subscribed_markets: set[str] = set()
        self._closing = False
        self._reconnect_count = 0

    # -- Async context manager --------------------------------------------------

    async def __aenter__(self) -> RtdsWebSocket:
        await self.connect()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.close()

    # -- Connection lifecycle ---------------------------------------------------

    async def connect(self) -> None:
        """Open the WebSocket connection and start the listener loop."""
        if self._ws is not None:
            logger.debug("RtdsWebSocket already connected; skipping connect()")
            return

        self._closing = False
        await self._establish_connection()
        self._listen_task = asyncio.create_task(
            self._listen(), name="rtds-ws-listen"
        )
        logger.info("RtdsWebSocket listener task started")

    async def _establish_connection(self) -> None:
        """Create the raw WebSocket connection."""
        logger.info("Connecting to RTDS WebSocket at %s", self._url)
        self._ws = await websockets.connect(
            self._url,
            ping_interval=20,
            ping_timeout=30,
            close_timeout=10,
        )
        self._reconnect_count = 0
        logger.info("RTDS WebSocket connection established")

    async def close(self) -> None:
        """Gracefully shut down the WebSocket and listener task."""
        self._closing = True
        logger.info("Closing RtdsWebSocket")

        if self._listen_task is not None and not self._listen_task.done():
            self._listen_task.cancel()
            try:
                await self._listen_task
            except asyncio.CancelledError:
                pass
            self._listen_task = None

        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                logger.debug("Exception during WebSocket close; ignoring", exc_info=True)
            self._ws = None

        self._subscribed_markets.clear()
        logger.info("RtdsWebSocket closed")

    # -- Subscriptions ----------------------------------------------------------

    async def subscribe(self, market_ids: list[str]) -> None:
        """Subscribe to trade feeds for the given market IDs.

        Args:
            market_ids: List of market identifiers (slugs or condition IDs)
                to subscribe to for live trade data.
        """
        if not market_ids:
            return

        new_ids = [mid for mid in market_ids if mid not in self._subscribed_markets]
        if not new_ids:
            logger.debug("All requested market IDs already subscribed")
            return

        for market_id in new_ids:
            msg = {
                "type": "subscribe",
                "channel": "trades",
                "market": market_id,
            }
            await self._send(msg)
            self._subscribed_markets.add(market_id)

        logger.info(
            "Subscribed to trade feed for %d new market(s); total subscriptions: %d",
            len(new_ids),
            len(self._subscribed_markets),
        )

    async def unsubscribe(self, market_ids: list[str]) -> None:
        """Unsubscribe from trade feeds for the given market IDs.

        Args:
            market_ids: List of market identifiers to unsubscribe from.
        """
        if not market_ids:
            return

        active_ids = [mid for mid in market_ids if mid in self._subscribed_markets]
        if not active_ids:
            logger.debug("None of the requested market IDs are subscribed")
            return

        for market_id in active_ids:
            msg = {
                "type": "unsubscribe",
                "channel": "trades",
                "market": market_id,
            }
            await self._send(msg)
            self._subscribed_markets.discard(market_id)

        logger.info(
            "Unsubscribed from %d market(s); remaining subscriptions: %d",
            len(active_ids),
            len(self._subscribed_markets),
        )

    # -- Internal helpers -------------------------------------------------------

    async def _send(self, payload: dict[str, Any]) -> None:
        """Serialize and send a JSON message over the WebSocket."""
        if self._ws is None:
            raise RuntimeError("WebSocket is not connected")

        raw = json.dumps(payload)
        await self._ws.send(raw)
        logger.debug("Sent: %s", raw)

    async def _listen(self) -> None:
        """Main receive loop: read messages and dispatch to callback.

        Handles connection drops by attempting automatic reconnection
        with exponential backoff.  On each reconnection the client
        re-subscribes to all previously-subscribed market IDs.
        """
        while not self._closing:
            try:
                await self._receive_loop()
            except asyncio.CancelledError:
                logger.debug("Listen task cancelled")
                return
            except websockets.ConnectionClosed as exc:
                if self._closing:
                    return
                logger.warning(
                    "RTDS WebSocket connection closed (code=%s reason=%s)",
                    exc.rcvd.code if exc.rcvd else "N/A",
                    exc.rcvd.reason if exc.rcvd else "N/A",
                )
                await self._reconnect()
            except Exception:
                if self._closing:
                    return
                logger.exception("Unexpected error in RTDS WebSocket listener")
                await self._reconnect()

    async def _receive_loop(self) -> None:
        """Read messages from the WebSocket until it closes."""
        if self._ws is None:
            return

        async for raw_message in self._ws:
            if self._closing:
                return

            try:
                message = json.loads(raw_message)
            except json.JSONDecodeError:
                logger.warning(
                    "Non-JSON message received: %s",
                    raw_message[:200] if isinstance(raw_message, str) else "<binary>",
                )
                continue

            logger.debug("Received RTDS message: %s", str(message)[:300])
            await self._dispatch(message)

    async def _dispatch(self, message: dict[str, Any]) -> None:
        """Route a parsed message to the on_trade callback.

        Filters out control/ack messages and only dispatches trade data.
        If the message contains a list of trades (batch), each trade is
        dispatched individually.
        """
        msg_type = message.get("type", "")

        # Skip subscription acknowledgements and heartbeat frames
        if msg_type in ("subscribed", "unsubscribed", "heartbeat", "pong"):
            logger.debug("Control message (%s); not dispatching", msg_type)
            return

        # Some RTDS messages wrap trades in a list under a "data" key
        trades = message.get("data") if isinstance(message.get("data"), list) else None

        if trades is not None:
            for trade in trades:
                if isinstance(trade, dict):
                    await self._invoke_callback(trade)
        else:
            # Single trade message or unknown structure -- pass it through
            await self._invoke_callback(message)

    async def _invoke_callback(self, trade_data: dict[str, Any]) -> None:
        """Invoke the on_trade callback, handling both sync and async callables."""
        try:
            result = self._on_trade(trade_data)
            if asyncio.iscoroutine(result):
                await result
        except Exception:
            logger.exception("Error in on_trade callback for RTDS message")

    async def _reconnect(self) -> None:
        """Attempt to reconnect with exponential backoff.

        After reconnection, re-subscribes to all previously-subscribed
        market IDs so the trade feed resumes seamlessly.
        """
        # Clean up the old connection
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None

        while not self._closing and self._reconnect_count < self._max_reconnect_attempts:
            self._reconnect_count += 1
            delay = min(
                self._reconnect_delay * (2 ** (self._reconnect_count - 1)),
                300,  # cap at 5 minutes
            )
            logger.info(
                "RTDS WebSocket reconnect attempt %d/%d in %.1fs",
                self._reconnect_count,
                self._max_reconnect_attempts,
                delay,
            )
            await asyncio.sleep(delay)

            if self._closing:
                return

            try:
                await self._establish_connection()
            except Exception:
                logger.exception(
                    "RTDS WebSocket reconnect attempt %d failed",
                    self._reconnect_count,
                )
                continue

            # Re-subscribe to previously active market IDs
            if self._subscribed_markets:
                saved_markets = list(self._subscribed_markets)
                self._subscribed_markets.clear()
                try:
                    await self.subscribe(saved_markets)
                except Exception:
                    logger.exception("Failed to re-subscribe after reconnect")

            logger.info("RTDS WebSocket reconnected successfully")
            return

        if not self._closing:
            logger.error(
                "RTDS WebSocket exhausted all %d reconnect attempts; giving up",
                self._max_reconnect_attempts,
            )

    @property
    def connected(self) -> bool:
        """Return True if the underlying WebSocket is open."""
        return self._ws is not None and self._ws.state.name == "OPEN"

    @property
    def subscribed_markets(self) -> frozenset[str]:
        """Return the set of currently-subscribed market IDs."""
        return frozenset(self._subscribed_markets)
