"""
QuantEdge Live Runner
=====================
Orchestrates the full trading day:

  Pre-market  (4:00–9:29 AM ET)  → gap scanner every 5 min
  Open        (9:31 AM ET)       → first-candle confirmation + entry
  Intraday    (9:32 AM–4:00 PM)  → continuous monitor loop
  End of day  (4:00 PM ET)       → flatten all positions, daily reset

Usage:
    python -m src.live.live_runner

Environment variables required (or set in .env):
    ALPACA_API_KEY
    ALPACA_SECRET_KEY
    ALPACA_PAPER=true          # set false for live
    SLACK_TOKEN                # optional
    SLACK_CHANNEL              # optional, default #trading-alerts
"""
from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone, timedelta

from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient

from src.live.position_manager import PositionManager
from src.live.gap_trader import GapAndGoTrader
from src.live.continuous_monitor import ContinuousMonitor

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Top symbols from Phase 6 backtests (Sharpe > 2.4, profit_factor > 1.0) ──
TRADING_UNIVERSE = [
    # High-beta / gap candidates
    "UPST", "AFRM", "HOOD", "SOFI", "COIN", "PYPL", "DASH", "LYFT",
    "SNAP", "UBER", "AMD", "TSLA", "NVDA", "NFLX", "ADBE", "ETSY",
    # Leveraged ETFs (gap plays)
    "SQQQ", "TQQQ", "TZA", "TNA", "SOXL", "LABU",
    # Crypto proxies
    "IBIT", "ETHA",
    # Macro
    "GLD", "USO", "SHV",
    # Large cap anchors
    "AAPL", "MSFT", "GOOGL", "META", "AMZN", "ORCL", "CRM",
]

# ── Market hours in UTC ──
PREMARKET_START_UTC  = 8    # 4:00 AM ET
MARKET_OPEN_UTC      = 13   # 9:00 AM ET (scan fires at 9:30 = 13:30)
MARKET_OPEN_MIN      = 30
FIRST_CANDLE_UTC     = 13
FIRST_CANDLE_MIN     = 31
MARKET_CLOSE_UTC     = 20   # 4:00 PM ET
PREMARKET_SCAN_EVERY = 300  # seconds (5 min)


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def wait_until(hour: int, minute: int = 0):
    target = now_utc().replace(hour=hour, minute=minute, second=0, microsecond=0)
    if now_utc() >= target:
        return
    delta = (target - now_utc()).total_seconds()
    logger.info(f"[RUNNER] Waiting {delta/60:.1f} min until {hour:02d}:{minute:02d} UTC")
    time.sleep(max(0, delta))


def build_clients(paper: bool):
    api_key = os.environ["ALPACA_API_KEY"]
    secret  = os.environ["ALPACA_SECRET_KEY"]
    trading = TradingClient(api_key, secret, paper=paper)
    data    = StockHistoricalDataClient(api_key, secret)
    return trading, data


def main():
    paper   = os.getenv("ALPACA_PAPER", "true").lower() != "false"
    mode    = "PAPER" if paper else "LIVE"
    logger.info(f"[RUNNER] QuantEdge starting — mode={mode}")

    trading_client, data_client = build_clients(paper)
    pm = PositionManager(trading_client, paper=paper)

    gap_trader = GapAndGoTrader(
        trading_client=trading_client,
        data_client=data_client,
        position_manager=pm,
        universe=TRADING_UNIVERSE,
        capital_per_trade=float(os.getenv("CAPITAL_PER_TRADE", "5000")),
        min_gap_pct=float(os.getenv("MIN_GAP_PCT", "2.0")),
        min_vol_ratio=float(os.getenv("MIN_VOL_RATIO", "0.3")),
        risk_reward=float(os.getenv("RISK_REWARD", "2.0")),
        max_positions=int(os.getenv("MAX_POSITIONS", "3")),
    )

    monitor = ContinuousMonitor(
        trading_client=trading_client,
        data_client=data_client,
        position_manager=pm,
        universe=TRADING_UNIVERSE,
        slack_token=os.getenv("SLACK_TOKEN"),
        slack_channel=os.getenv("SLACK_CHANNEL", "#trading-alerts"),
        interval_sec=int(os.getenv("MONITOR_INTERVAL", "60")),
    )

    current = now_utc()

    # ── Phase 1: Pre-market scanning ──────────────────────────────────
    if current.hour < MARKET_OPEN_UTC or (current.hour == MARKET_OPEN_UTC and current.minute < MARKET_OPEN_MIN):
        logger.info("[RUNNER] ── PRE-MARKET PHASE ──")
        while True:
            t = now_utc()
            if t.hour > MARKET_OPEN_UTC or (t.hour == MARKET_OPEN_UTC and t.minute >= MARKET_OPEN_MIN):
                break
            gap_trader.premarket_scan()
            time.sleep(PREMARKET_SCAN_EVERY)
    else:
        # If runner starts after open, do one scan immediately
        gap_trader.premarket_scan()

    # ── Phase 2: Market open — first candle confirmation ──────────────
    logger.info("[RUNNER] ── MARKET OPEN — awaiting first candle ──")
    wait_until(FIRST_CANDLE_UTC, FIRST_CANDLE_MIN)
    gap_trader.confirm_and_enter()

    # ── Phase 3: Continuous intraday monitor ──────────────────────────
    logger.info("[RUNNER] ── INTRADAY MONITOR ──")
    monitor.run_loop(stop_at_hour_utc=MARKET_CLOSE_UTC)

    # ── Phase 4: End of day — flatten all positions ───────────────────
    logger.info("[RUNNER] ── END OF DAY — flattening positions ──")
    for sym in list(pm.positions.keys()):
        pm.close(sym, "end_of_day")

    gap_trader.reset_daily()
    monitor.reset_daily()
    logger.info("[RUNNER] Session complete.")


if __name__ == "__main__":
    main()
