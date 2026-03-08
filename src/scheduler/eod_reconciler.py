"""
QuantEdge — EOD Reconciler (Phase 9)

Changes from Phase 8:
- Options force-close logic removed entirely
- Hold cap check: force close any position >= MAX_HOLD_BARS before EOD
- All positions force closed at 3:55 PM via Alpaca market order
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

from src.scheduler.state_manager import (
    get_open_positions,
    close_position,
    is_hold_cap_breached,
    MAX_HOLD_BARS,
)

logger = logging.getLogger(__name__)

DRY_RUN = os.getenv("DRY_RUN", "true").lower() == "true"


def _get_client() -> TradingClient:
    return TradingClient(
        api_key=os.environ["ALPACA_API_KEY"],
        secret_key=os.environ["ALPACA_SECRET_KEY"],
        paper=True,
    )


def _market_close(client: TradingClient, symbol: str, qty: int, direction: str) -> None:
    """Submit a market order to close a position."""
    side = OrderSide.SELL if direction == "long" else OrderSide.BUY
    if DRY_RUN:
        logger.info("dry_run_close", extra={"symbol": symbol, "qty": qty, "side": side.value})
        return
    req = MarketOrderRequest(symbol=symbol, qty=qty, side=side, time_in_force=TimeInForce.DAY)
    client.submit_order(req)
    logger.info("eod_order_submitted", extra={"symbol": symbol, "qty": qty, "side": side.value})


def run_eod_close() -> None:
    """
    Phase 9 EOD reconciler — runs at 3:55 PM.
    Closes ALL open share positions. No options logic.
    """
    client = _get_client()
    positions = get_open_positions()

    if not positions:
        logger.info("eod_reconciler_no_positions")
        return

    logger.info("eod_reconciler_start", extra={"open_positions": len(positions)})

    for symbol, pos in list(positions.items()):
        bars_held = pos.get("bars_held", 0)
        reason = "hold_cap" if bars_held >= MAX_HOLD_BARS else "eod"

        try:
            _market_close(client, symbol, pos["qty"], pos["direction"])

            # Get current price from Alpaca for PnL calculation
            try:
                asset = client.get_open_position(symbol)
                exit_price = float(asset.current_price)
            except Exception:
                exit_price = pos["entry_price"]  # fallback

            trade = close_position(symbol, exit_price, reason)
            if trade:
                logger.info(
                    "eod_position_closed",
                    extra={
                        "symbol":      symbol,
                        "reason":      reason,
                        "pnl":         trade["pnl"],
                        "bars_held":   bars_held,
                    },
                )
        except Exception as exc:
            logger.error("eod_close_failed", extra={"symbol": symbol, "error": str(exc)})


def check_hold_cap_intraday() -> list[str]:
    """
    Phase 9: Call this each bar to check if any position has breached MAX_HOLD_BARS.
    Returns list of symbols that were force-closed.
    """
    client = _get_client()
    positions = get_open_positions()
    force_closed = []

    for symbol, pos in list(positions.items()):
        if is_hold_cap_breached(symbol):
            logger.warning(
                "hold_cap_breach",
                extra={"symbol": symbol, "bars_held": pos.get("bars_held", 0)},
            )
            try:
                _market_close(client, symbol, pos["qty"], pos["direction"])
                try:
                    asset = client.get_open_position(symbol)
                    exit_price = float(asset.current_price)
                except Exception:
                    exit_price = pos["entry_price"]
                close_position(symbol, exit_price, "hold_cap")
                force_closed.append(symbol)
            except Exception as exc:
                logger.error("hold_cap_close_failed", extra={"symbol": symbol, "error": str(exc)})

    return force_closed


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_eod_close()
