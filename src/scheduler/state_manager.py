"""
QuantEdge — State Manager (Phase 9)

Changes from Phase 8:
- Tracks bar count per position for hold cap enforcement
- MAX_HOLD_BARS = 100: force close any position held > 100 bars
- Removed options state tracking
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

STATE_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "state.json")

# Phase 9: max bars before forced close
MAX_HOLD_BARS = 100


def _load() -> dict:
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            logger.warning("state_file_corrupt_resetting")
    return {"positions": {}, "daily_pnl": 0.0, "trade_log": []}


def _save(state: dict) -> None:
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


def get_state() -> dict:
    return _load()


def open_position(
    symbol: str,
    direction: str,
    entry_price: float,
    stop_price: float,
    target_price: float,
    qty: int,
    gap_pct: float,
    vol_ratio: float,
    conviction: str,
) -> None:
    state = _load()
    state["positions"][symbol] = {
        "direction":    direction,
        "entry_price":  entry_price,
        "stop_price":   stop_price,
        "target_price": target_price,
        "qty":          qty,
        "gap_pct":      gap_pct,
        "vol_ratio":    vol_ratio,
        "conviction":   conviction,
        "entry_time":   datetime.now(timezone.utc).isoformat(),
        "bars_held":    0,          # Phase 9: bar counter
        "route":        "shares",   # Phase 9: always shares
    }
    _save(state)
    logger.info("position_opened", extra={"symbol": symbol, "qty": qty, "entry": entry_price})


def close_position(symbol: str, exit_price: float, exit_reason: str) -> Optional[dict]:
    state = _load()
    pos = state["positions"].pop(symbol, None)
    if pos is None:
        logger.warning("close_position_not_found", extra={"symbol": symbol})
        return None

    pnl = (exit_price - pos["entry_price"]) * pos["qty"]
    if pos["direction"] == "short":
        pnl = -pnl

    trade = {
        "symbol":      symbol,
        "direction":   pos["direction"],
        "entry_price": pos["entry_price"],
        "exit_price":  exit_price,
        "qty":         pos["qty"],
        "pnl":         round(pnl, 2),
        "exit_reason": exit_reason,
        "bars_held":   pos.get("bars_held", 0),
        "gap_pct":     pos.get("gap_pct", 0),
        "vol_ratio":   pos.get("vol_ratio", 0),
        "conviction":  pos.get("conviction", "normal"),
        "entry_time":  pos.get("entry_time"),
        "exit_time":   datetime.now(timezone.utc).isoformat(),
    }

    state["daily_pnl"] = round(state.get("daily_pnl", 0.0) + pnl, 2)
    state["trade_log"].append(trade)
    _save(state)

    logger.info(
        "position_closed",
        extra={"symbol": symbol, "exit_reason": exit_reason, "pnl": round(pnl, 2)},
    )
    return trade


def increment_bars(symbol: str) -> int:
    """
    Phase 9: Increment bar counter for a position.
    Returns the updated bar count.
    """
    state = _load()
    if symbol in state["positions"]:
        state["positions"][symbol]["bars_held"] = (
            state["positions"][symbol].get("bars_held", 0) + 1
        )
        count = state["positions"][symbol]["bars_held"]
        _save(state)
        return count
    return 0


def get_bars_held(symbol: str) -> int:
    """Return how many bars a position has been open."""
    state = _load()
    return state["positions"].get(symbol, {}).get("bars_held", 0)


def is_hold_cap_breached(symbol: str) -> bool:
    """Phase 9: Returns True if position has exceeded MAX_HOLD_BARS."""
    return get_bars_held(symbol) >= MAX_HOLD_BARS


def get_open_positions() -> dict:
    return _load().get("positions", {})


def get_position_count() -> int:
    return len(get_open_positions())


def reset_daily(carry_positions: bool = True) -> None:
    state = _load()
    state["daily_pnl"] = 0.0
    if not carry_positions:
        state["positions"] = {}
    _save(state)
    logger.info("daily_state_reset")
