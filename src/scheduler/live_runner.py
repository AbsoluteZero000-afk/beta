"""
QuantEdge — Live Runner (Phase 9)

Changes from Phase 8:
- Options executor removed — shares only
- VIX regime check at market open (blocks all entries if VIX > 25)
- Hold cap enforced every bar via check_hold_cap_intraday()
- Bar counter incremented each minute for all open positions
"""
from __future__ import annotations

import asyncio
import logging
import os
import signal
from datetime import datetime, time, timezone

import uvloop
from alpaca.trading.client import TradingClient

from src.strategies.signal_router import route_signal, is_vix_regime_ok
from src.scheduler.state_manager import (
    open_position,
    get_position_count,
    get_open_positions,
    increment_bars,
)
from src.scheduler.eod_reconciler import run_eod_close, check_hold_cap_intraday
from src.scheduler.daily_report import send_daily_report

logger = logging.getLogger(__name__)

DRY_RUN   = os.getenv("DRY_RUN", "true").lower() == "true"
BAR_SECS  = 60  # 1-minute bars


def _get_client() -> TradingClient:
    return TradingClient(
        api_key=os.environ["ALPACA_API_KEY"],
        secret_key=os.environ["ALPACA_SECRET_KEY"],
        paper=True,
    )


async def _increment_all_bars() -> None:
    """Phase 9: Tick bar counter for every open position each minute."""
    for symbol in list(get_open_positions().keys()):
        count = increment_bars(symbol)
        logger.debug("bar_incremented", extra={"symbol": symbol, "bars_held": count})


async def _check_hold_caps() -> None:
    """Phase 9: Force close any position that has exceeded MAX_HOLD_BARS."""
    closed = check_hold_cap_intraday()
    if closed:
        logger.warning("hold_cap_force_closed", extra={"symbols": closed})


async def run_market_open() -> None:
    """
    Runs at 9:30 AM ET.
    1. VIX regime check — if VIX > 25, skip all entries today
    2. Scan for gap + volume signals
    3. Route to shares only (Phase 9 — no options)
    """
    logger.info("market_open_routine_start")

    # Phase 9 VIX regime gate
    if not is_vix_regime_ok():
        logger.warning("vix_regime_blocked_all_entries — skipping today")
        return

    # Signal scanning happens via the Alpaca watcher + event consumer pipeline
    # This hook is for any pre-market setup or watchlist refresh
    logger.info("market_open_routine_complete")


async def run_bar_loop() -> None:
    """
    Main intraday loop — runs every BAR_SECS seconds while market is open.
    - Increments bar counters for all open positions
    - Checks hold cap breaches
    - Checks for 3:55 PM EOD trigger
    """
    stop = asyncio.Event()

    def _sig(s, f):
        stop.set()

    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)

    logger.info("bar_loop_started", extra={"bar_secs": BAR_SECS})

    while not stop.is_set():
        now_et = datetime.now(timezone.utc)

        # Increment bar counters for all open positions
        await _increment_all_bars()

        # Check hold cap on every bar (Phase 9)
        await _check_hold_caps()

        # EOD trigger at 3:55 PM ET
        if now_et.hour == 20 and now_et.minute >= 55:  # 20:55 UTC = 3:55 PM ET
            logger.info("eod_trigger_fired")
            run_eod_close()
            await send_daily_report()
            break

        try:
            await asyncio.wait_for(stop.wait(), timeout=BAR_SECS)
        except asyncio.TimeoutError:
            pass

    logger.info("bar_loop_stopped")


async def main() -> None:
    await run_market_open()
    await run_bar_loop()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    uvloop.run(main())
