"""CLOB WebSocket client for real-time orderbook updates.

Connects to the Polymarket CLOB WebSocket feed to receive live orderbook
deltas (bid/ask additions, removals, and changes) for subscribed asset IDs.
Implemented but not yet wired into the orchestrator (planned for Phase 3);
REST polling via ClobClient is the live data path today.
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


class ClobWebSocket:
    """Async WebSocket client for Polymarket CLOB orderbook subscriptions.

    Receives real-time orderbook delta messages for subscribed asset IDs.
    Supports automatic reconnection with exponential backoff when the
    connection drops.

    Usage::

        async with ClobWebSocket(config, on_update=handle_update) as ws:
            await ws.subscribe(["71321044..."])
            # ws._listen() runs in background via connect()
            await some_shutdown_event.wait()
    """

    def __init__(
        self,
        config: AppConfig,
        on_update: Callable[[dict[str, Any]], Any],
    ) -> None:
        self.config = config
        self._url = config.websocket.clob_url
        self._reconnect_delay = config.websocket.reconnect_delay_seconds
        self._max_reconnect_attempts = config.websocket.max_reconnect_attempts
        self._on_update = on_update

        self._ws: ClientConnection | None = None
        self._listen_task: asyncio.Task[None] | None = None
        self._subscribed_assets: set[str] = set()
        self._closing = False
        self._reconnect_count = 0

    # -- Async context manager --------------------------------------------------

    async def __aenter__(self) -> ClobWebSocket:
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
            logger.debug("ClobWebSocket already connected; skipping connect()")
            return

        self._closing = False
        await self._establish_connection()
        self._listen_task = asyncio.create_task(
            self._listen(), name="clob-ws-listen"
        )
        logger.info("ClobWebSocket listener task started")

    async def _establish_connection(self) -> None:
        """Create the raw WebSocket connection."""
        logger.info("Connecting to CLOB WebSocket at %s", self._url)
        self._ws = await websockets.connect(
            self._url,
            ping_interval=20,
            ping_timeout=30,
            close_timeout=10,
        )
        self._reconnect_count = 0
        logger.info("CLOB WebSocket connection established")

    async def close(self) -> None:
        """Gracefully shut down the WebSocket and listener task."""
        self._closing = True
        logger.info("Closing ClobWebSocket")

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

        self._subscribed_assets.clear()
        logger.info("ClobWebSocket closed")

    # -- Subscriptions ----------------------------------------------------------

    async def subscribe(self, asset_ids: list[str]) -> None:
        """Subscribe to orderbook updates for the given asset IDs.

        Args:
            asset_ids: List of CLOB token IDs (asset IDs) to subscribe to.
        """
        if not asset_ids:
            return

        new_ids = [aid for aid in asset_ids if aid not in self._subscribed_assets]
        if not new_ids:
            logger.debug("All requested asset IDs already subscribed")
            return

        for asset_id in new_ids:
            msg = {
                "type": "subscribe",
                "channel": "market",
                "assets_ids": [asset_id],
            }
            await self._send(msg)
            self._subscribed_assets.add(asset_id)

        logger.info(
            "Subscribed to %d new asset(s); total subscriptions: %d",
            len(new_ids),
            len(self._subscribed_assets),
        )

    async def unsubscribe(self, asset_ids: list[str]) -> None:
        """Unsubscribe from orderbook updates for the given asset IDs.

        Args:
            asset_ids: List of CLOB token IDs to unsubscribe from.
        """
        if not asset_ids:
            return

        active_ids = [aid for aid in asset_ids if aid in self._subscribed_assets]
        if not active_ids:
            logger.debug("None of the requested asset IDs are subscribed")
            return

        for asset_id in active_ids:
            msg = {
                "type": "unsubscribe",
                "channel": "market",
                "assets_ids": [asset_id],
            }
            await self._send(msg)
            self._subscribed_assets.discard(asset_id)

        logger.info(
            "Unsubscribed from %d asset(s); remaining subscriptions: %d",
            len(active_ids),
            len(self._subscribed_assets),
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
        re-subscribes to all previously-subscribed asset IDs.
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
                    "CLOB WebSocket connection closed (code=%s reason=%s)",
                    exc.rcvd.code if exc.rcvd else "N/A",
                    exc.rcvd.reason if exc.rcvd else "N/A",
                )
                await self._reconnect()
            except Exception:
                if self._closing:
                    return
                logger.exception("Unexpected error in CLOB WebSocket listener")
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

            logger.debug("Received CLOB message: %s", str(message)[:300])
            await self._dispatch(message)

    async def _dispatch(self, message: dict[str, Any]) -> None:
        """Route a parsed message to the on_update callback.

        Filters out control/ack messages and only dispatches data updates.
        """
        msg_type = message.get("type", "")

        # Skip subscription acknowledgements and heartbeat frames
        if msg_type in ("subscribed", "unsubscribed", "heartbeat", "pong"):
            logger.debug("Control message (%s); not dispatching", msg_type)
            return

        try:
            result = self._on_update(message)
            if asyncio.iscoroutine(result):
                await result
        except Exception:
            logger.exception("Error in on_update callback for CLOB message")

    async def _reconnect(self) -> None:
        """Attempt to reconnect with exponential backoff.

        After reconnection, re-subscribes to all previously-subscribed
        asset IDs so the feed resumes seamlessly.
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
                "CLOB WebSocket reconnect attempt %d/%d in %.1fs",
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
                    "CLOB WebSocket reconnect attempt %d failed",
                    self._reconnect_count,
                )
                continue

            # Re-subscribe to previously active asset IDs
            if self._subscribed_assets:
                saved_assets = list(self._subscribed_assets)
                self._subscribed_assets.clear()
                try:
                    await self.subscribe(saved_assets)
                except Exception:
                    logger.exception("Failed to re-subscribe after reconnect")

            logger.info("CLOB WebSocket reconnected successfully")
            return

        if not self._closing:
            logger.error(
                "CLOB WebSocket exhausted all %d reconnect attempts; giving up",
                self._max_reconnect_attempts,
            )

    @property
    def connected(self) -> bool:
        """Return True if the underlying WebSocket is open."""
        return self._ws is not None and self._ws.state.name == "OPEN"

    @property
    def subscribed_assets(self) -> frozenset[str]:
        """Return the set of currently-subscribed asset IDs."""
        return frozenset(self._subscribed_assets)
