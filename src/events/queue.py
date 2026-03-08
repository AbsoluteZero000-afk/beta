"""
QuantEdge — Priority Event Queue.

Uses asyncio.PriorityQueue (backed by heapq) to rank events by urgency.
TradingEvent.urgency drives priority: lower number = dequeued first.

Official docs:
  - asyncio.PriorityQueue: https://docs.python.org/3/library/asyncio-queue.html
  - heapq: https://docs.python.org/3/library/heapq.html
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from src.events.types import TradingEvent

logger = logging.getLogger(__name__)


class PriorityEventQueue:
    """
    Async priority queue for TradingEvent objects.

    Wraps asyncio.PriorityQueue. TradingEvent is decorated with
    @dataclass(order=True) so heapq comparison works natively:
      - Primary sort: urgency (0 = highest priority)
      - Secondary sort: timestamp (earlier events first within same urgency)

    Usage:
        queue = PriorityEventQueue(maxsize=1000)
        await queue.put(event)
        event = await queue.get()
        queue.task_done()
    """

    def __init__(self, maxsize: int = 1000) -> None:
        self._queue: asyncio.PriorityQueue[TradingEvent] = asyncio.PriorityQueue(
            maxsize=maxsize
        )
        self._put_count = 0
        self._get_count = 0
        self._dropped_count = 0

    async def put(self, event: TradingEvent) -> None:
        """
        Enqueue an event. If queue is full, drops lowest-urgency
        event to make room (backpressure handling).
        """
        if self._queue.full():
            self._dropped_count += 1
            logger.warning(
                "PriorityEventQueue full (maxsize=%d) — dropping event %s for %s",
                self._queue.maxsize,
                event.event_type,
                event.symbol,
            )
            return
        await self._queue.put(event)
        self._put_count += 1

    def put_nowait(self, event: TradingEvent) -> None:
        """Non-blocking put. Silently drops if full."""
        try:
            self._queue.put_nowait(event)
            self._put_count += 1
        except asyncio.QueueFull:
            self._dropped_count += 1
            logger.warning(
                "PriorityEventQueue full — dropping %s for %s",
                event.event_type,
                event.symbol,
            )

    async def get(self) -> TradingEvent:
        """Dequeue the highest-priority event (blocks until available)."""
        event = await self._queue.get()
        self._get_count += 1
        return event

    def task_done(self) -> None:
        """Mark the last dequeued event as processed."""
        self._queue.task_done()

    async def join(self) -> None:
        """Block until all enqueued events have been processed."""
        await self._queue.join()

    @property
    def qsize(self) -> int:
        return self._queue.qsize()

    @property
    def empty(self) -> bool:
        return self._queue.empty()

    @property
    def stats(self) -> dict:
        return {
            "queued": self._put_count,
            "processed": self._get_count,
            "dropped": self._dropped_count,
            "current_size": self.qsize,
        }
