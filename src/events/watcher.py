"""
QuantEdge — Alpaca WebSocket Watcher.

Persistent WebSocket connection to Alpaca's real-time data stream.
Subscribes to bars, trades, and quotes for a configurable watchlist.
Auto-reconnecting supervisor with exponential backoff.

Official alpaca-py docs:
  https://alpaca.markets/sdks/python/api_reference/data/stock/live.html

IMPORTANT: StockDataStream.run() is blocking — it runs its own event loop.
We run it in a separate thread via asyncio.to_thread() so it doesn't
block the main uvloop event loop.

Subscribe API:
  stream.subscribe_bars(handler, *symbols)     # *args, NOT a list
  stream.subscribe_trades(handler, *symbols)
  stream.subscribe_quotes(handler, *symbols)
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import Optional

from alpaca.data.enums import DataFeed
from alpaca.data.live.stock import StockDataStream
from alpaca.data.models import Bar, Quote, Trade

from src.cache.signal_buffer import SignalBuffer
from src.config import settings
from src.events.classifier import EventClassifier
from src.events.queue import PriorityEventQueue
from src.events.types import TradingEvent

logger = logging.getLogger(__name__)

# Exponential backoff config
_BACKOFF_INITIAL  = 1.0    # seconds
_BACKOFF_MAX      = 60.0   # seconds
_BACKOFF_FACTOR   = 2.0
_BACKOFF_JITTER   = 0.1    # 10% randomization


class AlpacaStreamWatcher:
    """
    Persistent Alpaca WebSocket watcher with auto-reconnect supervisor.

    Architecture:
      - The watcher runs StockDataStream.run() in a thread pool executor
        so it doesn't block the main asyncio event loop.
      - Handlers receive Alpaca model objects, classify them via
        EventClassifier, and enqueue into PriorityEventQueue.
      - The supervisor monitors the stream thread and restarts it
        with exponential backoff on failure.

    Usage:
        watcher = AlpacaStreamWatcher(queue, signal_buffer)
        await watcher.start()          # non-blocking, runs supervisor
        await watcher.stop()           # graceful shutdown
    """

    def __init__(
        self,
        event_queue: PriorityEventQueue,
        signal_buffer: SignalBuffer,
        watchlist: Optional[list[str]] = None,
    ) -> None:
        self._queue       = event_queue
        self._buffer      = signal_buffer
        self._watchlist   = watchlist or settings.watchlist_symbols
        self._classifier  = EventClassifier(signal_buffer)
        self._stream: Optional[StockDataStream] = None
        self._stream_thread: Optional[threading.Thread] = None
        self._supervisor_task: Optional[asyncio.Task] = None
        self._stop_event  = asyncio.Event()
        self._connected   = False
        self._reconnect_count = 0

    # ---- Public API ----

    async def start(self) -> None:
        """Start the watcher supervisor (non-blocking)."""
        logger.info(
            "Starting Alpaca watcher for %d symbols: %s",
            len(self._watchlist),
            self._watchlist[:5],
        )
        self._stop_event.clear()
        self._supervisor_task = asyncio.create_task(
            self._supervisor_loop(),
            name="alpaca_watcher_supervisor",
        )

    async def stop(self) -> None:
        """Gracefully stop the watcher and supervisor."""
        logger.info("Stopping Alpaca watcher...")
        self._stop_event.set()

        # Stop the stream
        if self._stream is not None:
            try:
                self._stream.stop()
            except Exception as exc:
                logger.warning("Error stopping stream: %s", exc)

        # Cancel supervisor task
        if self._supervisor_task is not None:
            self._supervisor_task.cancel()
            try:
                await self._supervisor_task
            except asyncio.CancelledError:
                pass

        logger.info("Alpaca watcher stopped")

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def stats(self) -> dict:
        return {
            "connected": self._connected,
            "reconnect_count": self._reconnect_count,
            "watchlist_size": len(self._watchlist),
            "queue_stats": self._queue.stats,
        }

    # ---- Supervisor ----

    async def _supervisor_loop(self) -> None:
        """
        Supervisor with exponential backoff.
        Restarts the stream connection on any failure.
        """
        backoff = _BACKOFF_INITIAL

        while not self._stop_event.is_set():
            try:
                logger.info(
                    "Connecting to Alpaca WebSocket (attempt %d)...",
                    self._reconnect_count + 1,
                )
                await self._run_stream()
                # If _run_stream() returns normally (stop requested), exit
                if self._stop_event.is_set():
                    break
                # Otherwise it disconnected unexpectedly — reconnect
                logger.warning("Stream disconnected unexpectedly — reconnecting")

            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error(
                    "Stream error (attempt %d): %s — retrying in %.1fs",
                    self._reconnect_count + 1,
                    exc,
                    backoff,
                )

            self._connected = False
            self._reconnect_count += 1

            # Exponential backoff with jitter
            import random
            jitter = random.uniform(-_BACKOFF_JITTER * backoff, _BACKOFF_JITTER * backoff)
            sleep_time = min(backoff + jitter, _BACKOFF_MAX)
            logger.info("Backoff: sleeping %.1fs before reconnect", sleep_time)

            try:
                await asyncio.wait_for(
                    asyncio.shield(self._stop_event.wait()),
                    timeout=sleep_time,
                )
                # stop_event was set during backoff sleep
                break
            except asyncio.TimeoutError:
                # Normal case — timeout expired, try reconnecting
                backoff = min(backoff * _BACKOFF_FACTOR, _BACKOFF_MAX)

        logger.info("Supervisor loop exited")

    async def _run_stream(self) -> None:
        """
        Create a new StockDataStream, subscribe to all symbols,
        and run it in a thread executor so it doesn't block uvloop.
        """
        feed = (
            DataFeed.IEX
            if settings.alpaca.data_feed.upper() == "IEX"
            else DataFeed.SIP
        )

        self._stream = StockDataStream(
            api_key=settings.alpaca.api_key,
            secret_key=settings.alpaca.secret_key,
            feed=feed,
        )

        # Register handlers — *symbols unpacked as positional args (NOT a list)
        symbols = self._watchlist
        self._stream.subscribe_bars(self._handle_bar, *symbols)
        self._stream.subscribe_trades(self._handle_trade, *symbols)
        self._stream.subscribe_quotes(self._handle_quote, *symbols)
        self._stream.subscribe_trading_statuses(self._handle_status, *symbols)

        self._connected = True
        logger.info(
            "Subscribed to bars/trades/quotes for %d symbols via %s feed",
            len(symbols),
            feed.value,
        )

        # StockDataStream.run() is blocking — run in thread executor
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._stream.run)

    # ---- Stream Handlers ----
    # All handlers must be async coroutines per alpaca-py API.
    # They run inside the StockDataStream's internal event loop (thread),
    # so we use asyncio.run_coroutine_threadsafe to hand off to uvloop.

    async def _handle_bar(self, bar: Bar) -> None:
        """Handle incoming 1-minute bar."""
        event = self._classifier.classify_bar(bar)
        if event is not None:
            await self._enqueue(event)

    async def _handle_trade(self, trade: Trade) -> None:
        """Handle incoming trade tick."""
        event = self._classifier.classify_trade(trade)
        if event is not None:
            await self._enqueue(event)

    async def _handle_quote(self, quote: Quote) -> None:
        """Handle incoming NBBO quote update."""
        event = self._classifier.classify_quote(quote)
        if event is not None:
            await self._enqueue(event)

    async def _handle_status(self, status) -> None:
        """Handle trading status updates (halts, resumes)."""
        logger.info(
            "Trading status update: %s — %s",
            getattr(status, "symbol", "?"),
            getattr(status, "halt_reason", status),
        )

    async def _enqueue(self, event: TradingEvent) -> None:
        """Put event into the priority queue."""
        await self._queue.put(event)
        logger.debug(
            "Queued %s urgency=%d for %s",
            event.event_type.value,
            event.urgency,
            event.symbol,
        )
