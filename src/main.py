"""
QuantEdge — Main Entry Point (Phase 5).

Full pipeline: Infra + Events + Signals + Execution + Observability.
"""

from __future__ import annotations

import asyncio
import logging
import signal

import uvloop

from src.config import settings
from src.observability.logging import configure_logging
from src.observability.slack_notifier import SlackNotifier
from src.observability.daily_job import DailyJob
from src.cache.signal_buffer import SignalBuffer
from src.db.init_db import init_database
from src.db.pool import close_pool
from src.cache.redis_client import close_redis
from src.events.queue import PriorityEventQueue
from src.events.watcher import AlpacaStreamWatcher
from src.events.consumer import EventConsumer
from src.signals.composer import SignalComposer
from src.execution.portfolio_tracker import PortfolioTracker
from src.execution.risk_gate import RiskGate
from src.execution.position_sizer import PositionSizer
from src.execution.order_router import OrderRouter
from src.execution.execution_monitor import ExecutionMonitor
from src.execution.decision_engine import DecisionEngine

from alpaca.trading.client import TradingClient

# ---- Configure structured logging FIRST ----
configure_logging(
    log_level=settings.log_level,
    log_format=getattr(settings, "log_format", "console"),
)

import structlog
logger = structlog.get_logger("quantedge.main")


async def main() -> None:
    await logger.ainfo("quantedge_starting", version="0.5.0", mode="paper")

    # 1. Infrastructure
    await init_database()
    signal_buffer = SignalBuffer(maxlen=settings.signal_buffer_size)
    restored = await signal_buffer.restore_all(settings.watchlist_symbols)
    await logger.ainfo("signal_buffer_restored", counts=restored)

    # 2. Alpaca Trading client (sync SDK)
    trading_client = TradingClient(
        api_key=settings.alpaca.api_key,
        secret_key=settings.alpaca.secret_key,
        paper=True,
    )

    # 3. Observability
    notifier   = SlackNotifier(
        webhook_url=getattr(settings, "slack_webhook_url", None),
    )
    daily_job  = DailyJob(notifier)

    # 4. Execution layer
    tracker    = PortfolioTracker(trading_client)
    risk_gate  = RiskGate(tracker)
    sizer      = PositionSizer()
    router     = OrderRouter(
        trading_client,
        dry_run=getattr(settings, "dry_run", True),
    )
    monitor    = ExecutionMonitor(sizer)
    composer   = SignalComposer()
    engine     = DecisionEngine(composer, risk_gate, sizer, router, monitor, tracker)

    # 5. Event pipeline
    event_queue = PriorityEventQueue(maxsize=2000)
    watcher     = AlpacaStreamWatcher(event_queue, signal_buffer)
    consumer    = EventConsumer(event_queue)
    consumer.set_handler(engine.handle_event)

    # 6. Graceful shutdown
    stop_event = asyncio.Event()

    def _shutdown(sig, frame):
        logger.info("shutdown_signal_received", sig=str(sig))
        stop_event.set()

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # 7. Start everything
    await tracker.start()
    await consumer.start()
    await watcher.start()
    await daily_job.start()

    await notifier.send_startup(
        watchlist=settings.watchlist_symbols,
        mode="paper",
    )

    await logger.ainfo(
        "quantedge_running",
        symbols=len(settings.watchlist_symbols),
        dry_run=getattr(settings, "dry_run", True),
    )

    # 8. Wait for stop signal
    await stop_event.wait()

    # 9. Graceful shutdown
    await watcher.stop()
    await consumer.stop()
    await daily_job.stop()
    await tracker.stop()

    # Run final daily summary on shutdown
    try:
        await daily_job.run()
    except Exception as exc:
        await logger.awarning("final_summary_failed", error=str(exc))

    await signal_buffer.flush_all()
    await notifier.close()
    await close_redis()
    await close_pool()

    await logger.ainfo("quantedge_shutdown_complete")


if __name__ == "__main__":
    uvloop.run(main())
