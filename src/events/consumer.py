"""
QuantEdge — Event Consumer.

Drains the PriorityEventQueue and dispatches events downstream.
In Phase 2, this logs events. In Phase 3+, it routes to the SignalEngine.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Callable, Coroutine, Optional

from src.events.queue import PriorityEventQueue
from src.events.types import TradingEvent

logger = logging.getLogger(__name__)


class EventConsumer:
    """
    Async consumer that drains the PriorityEventQueue.

    Dispatches each TradingEvent to a registered handler coroutine.
    Designed to be replaced/extended in Phase 3 with the full
    SignalEngine pipeline.
    """

    def __init__(
        self,
        queue: PriorityEventQueue,
        handler: Optional[Callable[[TradingEvent], Coroutine]] = None,
    ) -> None:
        self._queue   = queue
        self._handler = handler or self._default_handler
        self._task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()
        self._processed = 0

    async def start(self) -> None:
        """Start the consumer loop as a background task."""
        self._stop_event.clear()
        self._task = asyncio.create_task(
            self._consume_loop(),
            name="event_consumer",
        )
        logger.info("EventConsumer started")

    async def stop(self) -> None:
        """Gracefully stop the consumer."""
        self._stop_event.set()
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("EventConsumer stopped — processed %d events", self._processed)

    def set_handler(self, handler: Callable[[TradingEvent], Coroutine]) -> None:
        """Swap in a new event handler (used in Phase 3+)."""
        self._handler = handler

    async def _consume_loop(self) -> None:
        """Main consumer loop — drains queue indefinitely."""
        while not self._stop_event.is_set():
            try:
                # Wait up to 1 second for an event before checking stop_event
                event = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                try:
                    await self._handler(event)
                    self._processed += 1
                except Exception as exc:
                    logger.error(
                        "Handler error for %s %s: %s",
                        event.event_type,
                        event.symbol,
                        exc,
                        exc_info=True,
                    )
                finally:
                    self._queue.task_done()
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

    @staticmethod
    async def _default_handler(event: TradingEvent) -> None:
        """Default handler — logs the event. Replaced in Phase 3."""
        logger.info(
            "[EVENT] urgency=%d type=%-20s symbol=%-6s regime=%s",
            event.urgency,
            event.event_type.value,
            event.symbol,
            event.context.regime.value if event.context else "unknown",
        )
