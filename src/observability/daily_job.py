"""
QuantEdge — Daily PnL Aggregation Job.

Runs at market close (16:05 ET) to:
  1. Aggregate daily trades from TimescaleDB
  2. Compute win rate, total PnL, per-symbol breakdown
  3. Send daily summary to Slack
  4. Log structured summary

Scheduled via asyncio.sleep loop in main.py (no external scheduler needed).
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone, time as dtime
from typing import Optional

import zoneinfo
import structlog

from src.db.pool import get_pool
from src.observability.slack_notifier import SlackNotifier

logger = structlog.get_logger(__name__)

EASTERN = zoneinfo.ZoneInfo("America/New_York")
CLOSE_TIME = dtime(16, 5)   # 4:05 PM ET (5 min after close)


class DailyJob:
    """
    Nightly PnL aggregation and reporting job.

    Runs in the background event loop. Wakes up once per day
    at CLOSE_TIME to aggregate and report.
    """

    def __init__(self, notifier: SlackNotifier) -> None:
        self._notifier = notifier
        self._task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        self._task = asyncio.create_task(
            self._scheduler_loop(), name="daily_job"
        )
        logger.info("DailyJob scheduler started")

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _scheduler_loop(self) -> None:
        """Sleep until market close, run, then repeat."""
        while True:
            try:
                await self._sleep_until_close()
                await self.run()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("DailyJob error: %s", exc, exc_info=True)
                await asyncio.sleep(60)   # Back off and retry

    async def run(self) -> None:
        """Execute the daily aggregation and send Slack summary."""
        logger.info("Running daily PnL aggregation job")
        summary = await self._aggregate_today()

        await self._notifier.send_daily_summary(
            pnl=summary["total_pnl"],
            trades=summary["total_trades"],
            win_rate=summary["win_rate"],
            equity=0.0,   # Injected in main.py via portfolio tracker
            top_winners=summary["top_winners"],
            top_losers=summary["top_losers"],
        )

        await logger.ainfo(
            "daily_summary",
            total_pnl=round(summary["total_pnl"], 2),
            total_trades=summary["total_trades"],
            win_rate=round(summary["win_rate"], 4),
            top_symbol=summary["top_winners"][0]["symbol"]
                if summary["top_winners"] else None,
        )

    async def _aggregate_today(self) -> dict:
        """Query TimescaleDB for today's closed trades."""
        pool = await get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT
                    symbol,
                    COUNT(*)                     AS n_trades,
                    SUM(pnl)                     AS total_pnl,
                    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) AS n_wins
                FROM trades
                WHERE
                    status = 'closed'
                    AND time >= NOW() - INTERVAL '1 day'
                GROUP BY symbol
                ORDER BY total_pnl DESC
                """
            )

        if not rows:
            return {
                "total_pnl": 0.0,
                "total_trades": 0,
                "win_rate": 0.0,
                "top_winners": [],
                "top_losers": [],
            }

        total_pnl    = sum(float(r["total_pnl"] or 0) for r in rows)
        total_trades = sum(int(r["n_trades"])          for r in rows)
        total_wins   = sum(int(r["n_wins"])            for r in rows)
        win_rate     = total_wins / total_trades if total_trades else 0.0

        by_symbol = sorted(
            [{"symbol": r["symbol"], "pnl": float(r["total_pnl"] or 0)} for r in rows],
            key=lambda x: x["pnl"],
            reverse=True,
        )

        return {
            "total_pnl":    total_pnl,
            "total_trades": total_trades,
            "win_rate":     win_rate,
            "top_winners":  [x for x in by_symbol if x["pnl"] > 0][:3],
            "top_losers":   [x for x in reversed(by_symbol) if x["pnl"] < 0][:3],
        }

    @staticmethod
    async def _sleep_until_close() -> None:
        """Sleep until the next 16:05 ET."""
        while True:
            now_et = datetime.now(EASTERN)
            target = now_et.replace(
                hour=CLOSE_TIME.hour,
                minute=CLOSE_TIME.minute,
                second=0,
                microsecond=0,
            )
            if now_et.time() >= CLOSE_TIME:
                # Already past close today — wait for tomorrow
                from datetime import timedelta
                target += timedelta(days=1)
            delta = (target - now_et).total_seconds()
            logger.info("DailyJob sleeping %.0f seconds until next close", delta)
            await asyncio.sleep(delta)
            return   # Wake up and run
