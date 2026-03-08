from __future__ import annotations
import logging
import time
from datetime import datetime, timezone, date

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest, StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient

from src.scanners.momentum_scanner import MomentumScanner
from src.scanners.breakout_scanner import BreakoutScanner
from src.live.position_manager import PositionManager

logger = logging.getLogger(__name__)

try:
    import slack_sdk
    _SLACK_AVAILABLE = True
except ImportError:
    _SLACK_AVAILABLE = False


class ContinuousMonitor:
    """
    Runs every `interval_sec` seconds during market hours.
    - Checks open positions against stop/target/trail
    - Scans universe for new momentum/breakout setups
    - Posts Slack alerts when configured

    Parameters
    ----------
    slack_token   : Slack Bot OAuth token (optional)
    slack_channel : e.g. '#trading-alerts'
    interval_sec  : seconds between full scan cycles (default 60)
    """

    def __init__(
        self,
        trading_client:   TradingClient,
        data_client:      StockHistoricalDataClient,
        position_manager: PositionManager,
        universe:         list[str],
        slack_token:      str | None = None,
        slack_channel:    str        = "#trading-alerts",
        interval_sec:     int        = 60,
    ):
        self.trading_client = trading_client
        self.data_client    = data_client
        self.pm             = position_manager
        self.universe       = universe
        self.slack_channel  = slack_channel
        self.interval_sec   = interval_sec
        self._alerted:      set[str] = set()

        self._slack = None
        if slack_token and _SLACK_AVAILABLE:
            from slack_sdk import WebClient
            self._slack = WebClient(token=slack_token)
            logger.info("[MON] Slack notifications enabled")

        self._momentum_scanner = MomentumScanner()
        self._breakout_scanner = BreakoutScanner()

    # ------------------------------------------------------------------
    def run_once(self):
        """Single monitor cycle — call in a loop or scheduler."""
        now = datetime.now(timezone.utc)
        logger.info(f"[MON] Cycle at {now.strftime('%H:%M:%S UTC')}")

        # 1. Position management
        self._check_positions()

        # 2. Opportunity scan
        self._scan_opportunities()

    def run_loop(self, stop_at_hour_utc: int = 20):
        """
        Blocking loop. Stops when UTC hour >= stop_at_hour_utc (default 20 = 4 PM ET).
        """
        logger.info(f"[MON] Starting continuous monitor loop (stops at {stop_at_hour_utc}:00 UTC)")
        while True:
            if datetime.now(timezone.utc).hour >= stop_at_hour_utc:
                logger.info("[MON] Market closed. Stopping monitor.")
                break
            try:
                self.run_once()
            except Exception as e:
                logger.error(f"[MON] Cycle error: {e}")
            time.sleep(self.interval_sec)

    # ------------------------------------------------------------------
    def _check_positions(self):
        if not self.pm.positions:
            return
        symbols = list(self.pm.positions.keys())
        try:
            req    = StockLatestQuoteRequest(symbol_or_symbols=symbols)
            quotes = self.data_client.get_stock_latest_quote(req)
        except Exception as e:
            logger.error(f"[MON] Quote fetch error: {e}")
            return

        for sym, quote in quotes.items():
            price  = (quote.ask_price + quote.bid_price) / 2
            action = self.pm.update(sym, price)
            if action:
                logger.info(f"[MON] {sym} exit trigger: {action} at {price:.2f}")
                self.pm.close(sym, action)
                self._alert(f"🔴 *{sym}* closed | reason=`{action}` price=`{price:.2f}`")

    def _scan_opportunities(self):
        try:
            momentum_hits = self._momentum_scanner.scan(self.universe)
            breakout_hits = self._breakout_scanner.scan(self.universe)
        except Exception as e:
            logger.error(f"[MON] Scanner error: {e}")
            return

        all_hits = {r["symbol"]: r for r in momentum_hits}
        for r in breakout_hits:
            sym = r["symbol"]
            if sym in all_hits:
                all_hits[sym]["_high_conviction"] = True
            else:
                all_hits[sym] = r

        for sym, hit in all_hits.items():
            if sym in self._alerted:
                continue
            tag   = "⭐ HIGH CONVICTION" if hit.get("_high_conviction") else "📡 Setup"
            score = hit.get("score", hit.get("momentum_score", 0))
            msg   = (f"{tag} | *{sym}* | score=`{score:.1f}` "
                     f"entry=`{hit.get('suggested_entry', hit.get('price', 'n/a'))}`")
            logger.info(f"[MON] {msg}")
            self._alert(msg)
            self._alerted.add(sym)

    def reset_daily(self):
        self._alerted = set()

    def _alert(self, message: str):
        if self._slack:
            try:
                self._slack.chat_postMessage(
                    channel=self.slack_channel,
                    text=message,
                )
            except Exception as e:
                logger.warning(f"[MON] Slack alert failed: {e}")
