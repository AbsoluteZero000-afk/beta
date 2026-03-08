"""
QuantEdge — PortfolioTracker.

Maintains a live snapshot of account state by periodically
reconciling with the Alpaca TradingClient (synchronous SDK
wrapped in asyncio executor).

Provides:
  - equity, cash, buying_power
  - open positions and their current values
  - intraday drawdown tracking
  - daily PnL per symbol

Updated:
  - On startup
  - Every N seconds (configurable heartbeat)
  - After every order fill
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class PortfolioSnapshot:
    equity:               float = 0.0
    cash:                 float = 0.0
    buying_power:         float = 0.0
    total_exposure:       float = 0.0
    open_position_count:  int   = 0
    positions:            dict[str, float] = field(default_factory=dict)  # symbol -> market_value
    intraday_high_equity: float = 0.0
    intraday_drawdown_pct: float = 0.0
    daily_pnl_by_symbol:  dict[str, float] = field(default_factory=dict)
    last_updated:         Optional[datetime] = None

    def update_drawdown(self) -> None:
        if self.equity > self.intraday_high_equity:
            self.intraday_high_equity = self.equity
        if self.intraday_high_equity > 0:
            self.intraday_drawdown_pct = (
                self.intraday_high_equity - self.equity
            ) / self.intraday_high_equity


class PortfolioTracker:
    """
    Periodic reconciliation of portfolio state with Alpaca.

    TradingClient is synchronous — all calls wrapped in
    asyncio.get_running_loop().run_in_executor() to avoid blocking.
    """

    def __init__(
        self,
        trading_client,
        heartbeat_seconds: int = 30,
    ) -> None:
        self._client    = trading_client
        self._heartbeat = heartbeat_seconds
        self._snapshot  = PortfolioSnapshot()
        self._task: Optional[asyncio.Task] = None

    @property
    def snapshot(self) -> PortfolioSnapshot:
        return self._snapshot

    async def start(self) -> None:
        """Start background reconciliation loop."""
        await self._reconcile()   # Sync immediately on start
        self._task = asyncio.create_task(
            self._heartbeat_loop(), name="portfolio_tracker"
        )
        logger.info(
            "PortfolioTracker started — equity=$%.2f positions=%d",
            self._snapshot.equity,
            self._snapshot.open_position_count,
        )

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def refresh(self) -> PortfolioSnapshot:
        """Force an immediate reconciliation (e.g. after fill)."""
        await self._reconcile()
        return self._snapshot

    async def _heartbeat_loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(self._heartbeat)
                await self._reconcile()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning("Portfolio reconcile error: %s", exc)

    async def _reconcile(self) -> None:
        """Fetch account + positions from Alpaca and update snapshot."""
        loop = asyncio.get_running_loop()
        try:
            account   = await loop.run_in_executor(None, self._client.get_account)
            positions = await loop.run_in_executor(None, self._client.get_all_positions)
        except Exception as exc:
            logger.error("Failed to reconcile portfolio: %s", exc)
            return

        equity       = float(account.equity)
        cash         = float(account.cash)
        buying_power = float(account.buying_power)

        pos_map: dict[str, float] = {}
        total_exposure = 0.0
        for p in positions:
            mv = float(p.market_value)
            pos_map[p.symbol] = mv
            total_exposure += abs(mv)

        snap = self._snapshot
        snap.equity               = equity
        snap.cash                 = cash
        snap.buying_power         = buying_power
        snap.total_exposure       = total_exposure
        snap.open_position_count  = len(pos_map)
        snap.positions            = pos_map
        snap.last_updated         = datetime.now(timezone.utc)
        snap.update_drawdown()

        logger.debug(
            "Portfolio reconciled: equity=$%.2f cash=$%.2f positions=%d drawdown=%.2f%%",
            equity, cash, len(pos_map),
            snap.intraday_drawdown_pct * 100,
        )
