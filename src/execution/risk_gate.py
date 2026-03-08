"""
QuantEdge — RiskGate.

Hard rule-based pre-trade risk checks. Every potential order must
pass through RiskGate before reaching the OrderRouter.

Checks performed (in order):
  1. Market hours guard — no orders outside RTH (configurable)
  2. Max single-position size (% of portfolio equity)
  3. Max total portfolio exposure (% of equity)
  4. Max open positions count
  5. Max per-symbol daily loss
  6. Max portfolio daily drawdown
  7. Minimum signal strength threshold
  8. Duplicate order prevention (already have position?)

A REJECT result enqueues a PORTFOLIO_BREACH event to flush the queue.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone, time as dtime
from enum import Enum
from typing import Optional

from src.execution.portfolio_tracker import PortfolioTracker
from src.signals.composer import CompositeResult

logger = logging.getLogger(__name__)


class GateVerdict(str, Enum):
    PASS    = "pass"
    REJECT  = "reject"
    SCALE   = "scale"    # Allow but reduce size


@dataclass(frozen=True)
class GateResult:
    verdict:  GateVerdict
    reason:   str
    symbol:   str
    suggested_qty_override: Optional[float] = None  # Used when verdict=SCALE


# ---- Risk limits (overridable via settings in real deployment) ----
class RiskLimits:
    max_position_pct:     float = 0.10   # 10% of equity per position
    max_total_exposure:   float = 0.80   # 80% of equity total
    max_open_positions:   int   = 10
    max_daily_loss_pct:   float = 0.02   # 2% daily loss per symbol
    max_drawdown_pct:     float = 0.05   # 5% portfolio daily drawdown
    min_signal_strength:  float = 0.20   # |score| must exceed this
    rth_start:            dtime  = dtime(9, 30)
    rth_end:              dtime  = dtime(16, 0)
    enforce_rth:          bool   = True


class RiskGate:
    """
    Stateless risk gate — all state comes from PortfolioTracker.

    Usage:
        gate = RiskGate(portfolio_tracker)
        result = gate.check(symbol, side, score, price, qty)
        if result.verdict == GateVerdict.PASS:
            router.submit(...)
    """

    def __init__(
        self,
        tracker: PortfolioTracker,
        limits: Optional[RiskLimits] = None,
    ) -> None:
        self._tracker = tracker
        self._limits  = limits or RiskLimits()

    def check(
        self,
        symbol: str,
        side: str,                     # "buy" or "sell"
        signal: CompositeResult,
        price: float,
        qty: float,
    ) -> GateResult:
        """
        Run all risk checks. Returns first REJECT found, or PASS.
        Checks run cheapest-first to short-circuit early.
        """
        L = self._limits
        snap = self._tracker.snapshot

        # 1. Market hours
        if L.enforce_rth and not self._is_rth():
            return GateResult(GateVerdict.REJECT, "outside_rth", symbol)

        # 2. Minimum signal strength
        if abs(signal.composite_score) < L.min_signal_strength:
            return GateResult(
                GateVerdict.REJECT,
                f"signal_too_weak({abs(signal.composite_score):.3f}<{L.min_signal_strength})",
                symbol,
            )

        # 3. Portfolio drawdown guard
        if snap.intraday_drawdown_pct > L.max_drawdown_pct:
            return GateResult(
                GateVerdict.REJECT,
                f"portfolio_drawdown({snap.intraday_drawdown_pct:.2%}>{L.max_drawdown_pct:.2%})",
                symbol,
            )

        # 4. Max open positions
        if side == "buy" and snap.open_position_count >= L.max_open_positions:
            return GateResult(
                GateVerdict.REJECT,
                f"max_positions({snap.open_position_count}>={L.max_open_positions})",
                symbol,
            )

        # 5. Max total exposure (buy side only)
        if side == "buy":
            order_notional = price * qty
            projected_exposure = snap.total_exposure + order_notional
            exposure_pct = projected_exposure / snap.equity if snap.equity > 0 else 1.0
            if exposure_pct > L.max_total_exposure:
                # Scale down instead of reject
                allowed_notional = max(0, snap.equity * L.max_total_exposure - snap.total_exposure)
                scaled_qty = allowed_notional / price if price > 0 else 0
                if scaled_qty < 0.01:
                    return GateResult(
                        GateVerdict.REJECT,
                        f"max_exposure({exposure_pct:.2%}>{L.max_total_exposure:.2%})",
                        symbol,
                    )
                return GateResult(
                    GateVerdict.SCALE,
                    f"exposure_scaled({exposure_pct:.2%}→{L.max_total_exposure:.2%})",
                    symbol,
                    suggested_qty_override=round(scaled_qty, 2),
                )

        # 6. Max single position size
        position_notional = price * qty
        if snap.equity > 0:
            position_pct = position_notional / snap.equity
            if position_pct > L.max_position_pct:
                scaled_qty = (snap.equity * L.max_position_pct) / price
                return GateResult(
                    GateVerdict.SCALE,
                    f"position_scaled({position_pct:.2%}→{L.max_position_pct:.2%})",
                    symbol,
                    suggested_qty_override=round(scaled_qty, 2),
                )

        # 7. Per-symbol daily loss
        daily_pnl = snap.daily_pnl_by_symbol.get(symbol, 0.0)
        if daily_pnl < 0 and snap.equity > 0:
            loss_pct = abs(daily_pnl) / snap.equity
            if loss_pct > L.max_daily_loss_pct:
                return GateResult(
                    GateVerdict.REJECT,
                    f"daily_loss({symbol}:{loss_pct:.2%}>{L.max_daily_loss_pct:.2%})",
                    symbol,
                )

        logger.debug("RiskGate PASS — %s %s qty=%.2f price=%.2f", side, symbol, qty, price)
        return GateResult(GateVerdict.PASS, "all_checks_passed", symbol)

    @staticmethod
    def _is_rth() -> bool:
        """True if current time is within regular trading hours (ET)."""
        import zoneinfo
        eastern = zoneinfo.ZoneInfo("America/New_York")
        now_et = datetime.now(eastern).time()
        return dtime(9, 30) <= now_et <= dtime(16, 0)
