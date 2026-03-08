"""
QuantEdge — ExecutionMonitor.

Tracks the lifecycle of submitted orders and reconciles fills
against TimescaleDB. Provides the feedback loop for PositionSizer
to update win/loss statistics after each closed trade.

Order lifecycle:
  SUBMITTED → PENDING → PARTIALLY_FILLED → FILLED → CLOSED

Stored in TimescaleDB table: trades
  (created in Phase 1 schema, populated here)
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from src.db.pool import get_pool
from src.execution.order_router import OrderResult
from src.execution.position_sizer import PositionSizer

logger = logging.getLogger(__name__)


@dataclass
class TrackedOrder:
    client_order_id: str
    alpaca_order_id: Optional[str]
    symbol:          str
    side:            str
    qty:             float
    submitted_at:    datetime
    filled_at:       Optional[datetime] = None
    fill_price:      Optional[float]    = None
    status:          str                = "submitted"
    pnl:             Optional[float]    = None
    pnl_pct:         Optional[float]    = None


class ExecutionMonitor:
    """
    Tracks order submissions, fills, and closed positions.
    Writes trade records to TimescaleDB for audit and analytics.
    Feeds closed-trade PnL back to PositionSizer.

    Usage:
        monitor = ExecutionMonitor(position_sizer)
        monitor.register(order_result)          # after submit
        await monitor.record_fill(order_id, fill_price, fill_time)
        await monitor.record_close(symbol, close_price)
    """

    def __init__(self, position_sizer: PositionSizer) -> None:
        self._sizer   = position_sizer
        self._orders: dict[str, TrackedOrder] = {}  # client_order_id -> order

    def register(self, result: OrderResult) -> None:
        """Register a newly submitted order for tracking."""
        order = TrackedOrder(
            client_order_id=result.client_order_id,
            alpaca_order_id=result.alpaca_order_id,
            symbol=result.symbol,
            side=result.side,
            qty=result.qty,
            submitted_at=result.submitted_at,
        )
        self._orders[result.client_order_id] = order
        logger.debug("Registered order %s for %s", result.client_order_id, result.symbol)

    async def record_fill(
        self,
        client_order_id: str,
        fill_price: float,
        fill_time: Optional[datetime] = None,
    ) -> None:
        """Mark an order as filled and persist to TimescaleDB."""
        order = self._orders.get(client_order_id)
        if order is None:
            logger.warning("record_fill: unknown order %s", client_order_id)
            return

        order.fill_price = fill_price
        order.filled_at  = fill_time or datetime.now(timezone.utc)
        order.status     = "filled"

        await self._persist_trade(order)
        logger.info(
            "Fill recorded: %s %s qty=%.2f @ $%.2f",
            order.side.upper(), order.symbol, order.qty, fill_price,
        )

    async def record_close(
        self,
        symbol: str,
        close_price: float,
        entry_price: Optional[float] = None,
    ) -> None:
        """
        Record a closed position and update PositionSizer stats.
        Finds the most recent filled order for the symbol.
        """
        # Find entry order
        entry = None
        for o in reversed(list(self._orders.values())):
            if o.symbol == symbol and o.status == "filled" and o.side == "buy":
                entry = o
                break

        if entry is None:
            logger.warning("record_close: no open entry found for %s", symbol)
            return

        ep = entry_price or entry.fill_price or close_price
        if ep and ep > 0:
            pnl_pct = (close_price - ep) / ep
            pnl     = pnl_pct * (ep * entry.qty)
            win     = pnl_pct > 0
        else:
            pnl_pct = 0.0
            pnl     = 0.0
            win     = False

        entry.pnl     = round(pnl, 4)
        entry.pnl_pct = round(pnl_pct, 6)
        entry.status  = "closed"

        # Feed back to PositionSizer for Kelly updates
        self._sizer.update_stats(symbol, win=win, pnl_pct=abs(pnl_pct))

        await self._persist_close(entry)
        logger.info(
            "Position closed: %s pnl=$%.2f (%.2f%%) win=%s",
            symbol, pnl, pnl_pct * 100, win,
        )

    async def _persist_trade(self, order: TrackedOrder) -> None:
        """Write fill to TimescaleDB trades table."""
        pool = await get_pool()
        try:
            async with pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO trades
                        (time, symbol, side, qty, fill_price,
                         client_order_id, alpaca_order_id, status)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    ON CONFLICT (client_order_id) DO UPDATE
                        SET fill_price = EXCLUDED.fill_price,
                            status     = EXCLUDED.status
                    """,
                    order.filled_at or datetime.now(timezone.utc),
                    order.symbol,
                    order.side,
                    order.qty,
                    order.fill_price,
                    order.client_order_id,
                    order.alpaca_order_id,
                    order.status,
                )
        except Exception as exc:
            logger.error("DB persist failed for %s: %s", order.client_order_id, exc)

    async def _persist_close(self, order: TrackedOrder) -> None:
        """Update trade record with PnL on close."""
        pool = await get_pool()
        try:
            async with pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO trades
                        (time, symbol, side, qty, fill_price,
                         client_order_id, alpaca_order_id, status)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    """,
                    order.filled_at or datetime.now(timezone.utc),
                    order.symbol, order.side, order.qty,
                    order.fill_price, order.client_order_id,
                    order.alpaca_order_id, order.status,
                )
        except Exception as exc:
            logger.error("DB close update failed for %s: %s", order.client_order_id, exc)

    @property
    def open_orders(self) -> list[TrackedOrder]:
        return [o for o in self._orders.values() if o.status not in ("closed", "rejected")]

    @property
    def stats(self) -> dict:
        closed = [o for o in self._orders.values() if o.status == "closed"]
        wins   = [o for o in closed if o.pnl and o.pnl > 0]
        return {
            "total_submitted": len(self._orders),
            "open":            len(self.open_orders),
            "closed":          len(closed),
            "win_rate":        len(wins) / len(closed) if closed else 0.0,
            "total_pnl":       sum(o.pnl or 0 for o in closed),
        }
