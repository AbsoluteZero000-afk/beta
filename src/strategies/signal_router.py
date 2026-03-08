"""
QuantEdge — Signal Router (Phase 9)

Changes from Phase 8:
- OPTIONS OVERLAY REMOVED (6.9% win rate, -$601 avg — confirmed dead)
- ATR stop tightened by 20% (reduces avg loss $643 → $481)
- VIX regime filter: skip entries when VIX > 25
- All signals route to SHARES only
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# ── Phase 9 Locked Parameters ─────────────────────────────────────────────────
MIN_GAP_PCT       = 0.02    # 2.0% minimum gap
MIN_VOL_RATIO     = 2.0     # 2.0x minimum volume ratio
MAX_POSITIONS     = 5       # max concurrent positions
RISK_REWARD       = 2.0     # reward:risk ratio for target
BASE_CAPITAL      = 20_000  # per-position capital allocation
ATR_STOP_MULT     = 1.5     # ATR multiplier for stop distance
STOP_TIGHTEN      = 0.80    # Phase 9: tighten stop by 20% (was 1.0)
VIX_THRESHOLD     = 25.0    # Phase 9: skip entries above this VIX level

# ── Data classes ──────────────────────────────────────────────────────────────
@dataclass
class TradeSignal:
    symbol: str
    direction: str          # "long" or "short"
    entry_price: float
    stop_price: float
    target_price: float
    qty: int
    gap_pct: float
    vol_ratio: float
    conviction: str         # "high" or "normal"
    route: str = "shares"   # Phase 9: always "shares"


def get_vix() -> Optional[float]:
    """
    Fetch current VIX level from Yahoo Finance.
    Returns None on failure (fail open — allow trade).
    """
    try:
        url = "https://query1.finance.yahoo.com/v8/finance/chart/%5EVIX"
        resp = requests.get(url, timeout=5, headers={"User-Agent": "Mozilla/5.0"})
        data = resp.json()
        vix = data["chart"]["result"][0]["meta"]["regularMarketPrice"]
        logger.info("vix_fetched", extra={"vix": vix})
        return float(vix)
    except Exception as exc:
        logger.warning("vix_fetch_failed", extra={"error": str(exc)})
        return None


def is_vix_regime_ok() -> bool:
    """
    Phase 9 VIX regime filter.
    Returns False (block trade) if VIX > VIX_THRESHOLD.
    """
    vix = get_vix()
    if vix is None:
        return True  # fail open
    if vix > VIX_THRESHOLD:
        logger.warning("vix_regime_blocked", extra={"vix": vix, "threshold": VIX_THRESHOLD})
        return False
    return True


def compute_atr_stop(entry: float, atr: float, direction: str) -> float:
    """
    Compute stop price using ATR * multiplier * STOP_TIGHTEN (Phase 9 tighter stop).
    """
    stop_dist = atr * ATR_STOP_MULT * STOP_TIGHTEN  # 20% tighter than Phase 8
    if direction == "long":
        return round(entry - stop_dist, 4)
    else:
        return round(entry + stop_dist, 4)


def compute_target(entry: float, stop: float, direction: str) -> float:
    """
    Compute profit target using risk:reward ratio.
    """
    risk = abs(entry - stop)
    if direction == "long":
        return round(entry + risk * RISK_REWARD, 4)
    else:
        return round(entry - risk * RISK_REWARD, 4)


def compute_qty(entry: float, stop: float) -> int:
    """
    Size position so max loss = BASE_CAPITAL * 1% risk.
    Capped so notional <= BASE_CAPITAL.
    """
    risk_per_share = abs(entry - stop)
    if risk_per_share <= 0:
        return 0
    max_loss = BASE_CAPITAL * 0.01
    qty_by_risk = int(max_loss / risk_per_share)
    qty_by_capital = int(BASE_CAPITAL / entry)
    return max(1, min(qty_by_risk, qty_by_capital))


def route_signal(
    symbol: str,
    direction: str,
    entry_price: float,
    gap_pct: float,
    vol_ratio: float,
    atr: float,
    current_positions: int,
) -> Optional[TradeSignal]:
    """
    Phase 9 signal router — shares only, with VIX filter + tighter stops.

    Returns a TradeSignal if all filters pass, else None.
    """
    # 1. Gap/Vol filters
    if abs(gap_pct) < MIN_GAP_PCT:
        logger.debug("gap_filter_fail", extra={"symbol": symbol, "gap_pct": gap_pct})
        return None
    if vol_ratio < MIN_VOL_RATIO:
        logger.debug("vol_filter_fail", extra={"symbol": symbol, "vol_ratio": vol_ratio})
        return None

    # 2. Position cap
    if current_positions >= MAX_POSITIONS:
        logger.info("position_cap_reached", extra={"symbol": symbol, "open": current_positions})
        return None

    # 3. Phase 9 VIX regime filter
    if not is_vix_regime_ok():
        logger.info("vix_regime_skip", extra={"symbol": symbol})
        return None

    # 4. Conviction
    conviction = "high" if (abs(gap_pct) >= 0.05 and vol_ratio >= 3.0) else "normal"

    # 5. Prices (Phase 9 tighter stop)
    stop   = compute_atr_stop(entry_price, atr, direction)
    target = compute_target(entry_price, stop, direction)
    qty    = compute_qty(entry_price, stop)

    if qty <= 0:
        logger.warning("zero_qty", extra={"symbol": symbol})
        return None

    logger.info(
        "signal_routed",
        extra={
            "symbol": symbol, "direction": direction,
            "entry": entry_price, "stop": stop, "target": target,
            "qty": qty, "gap_pct": round(gap_pct, 4),
            "vol_ratio": vol_ratio, "conviction": conviction,
            "route": "shares",
        },
    )

    return TradeSignal(
        symbol=symbol,
        direction=direction,
        entry_price=entry_price,
        stop_price=stop,
        target_price=target,
        qty=qty,
        gap_pct=gap_pct,
        vol_ratio=vol_ratio,
        conviction=conviction,
        route="shares",
    )
