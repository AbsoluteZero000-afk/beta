"""
QuantEdge — PositionSizer.

Fractional Kelly Criterion position sizing.

Full Kelly is theoretically optimal but practically dangerous —
a 50% Kelly fraction (half-Kelly) is the standard institutional
approach, offering ~75% of full Kelly growth with much lower
volatility of outcomes.

Formula:
    f* = (b*p - q) / b
    where:
        b = win/loss ratio (avg_win / avg_loss)
        p = win probability
        q = 1 - p

    fractional_kelly = kelly_fraction * f*
    position_notional = fractional_kelly * portfolio_equity
    qty = position_notional / current_price

References:
    Kelly (1956) — A New Interpretation of Information Rate
    Thorp (1969) — Optimal Gambling Systems for Favorable Games
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np

from src.signals.composer import CompositeResult

logger = logging.getLogger(__name__)

# Safety caps — always enforced regardless of Kelly output
MAX_KELLY_FRACTION  = 0.25   # Never allocate >25% of equity to one trade
MIN_QTY             = 0.01   # Minimum fractional share


@dataclass
class SizeResult:
    symbol:            str
    qty:               float
    notional:          float
    kelly_f:           float   # Raw Kelly fraction
    applied_f:         float   # After fractional multiplier + cap
    signal_score:      float
    sizing_method:     str     # "kelly" or "fixed_pct"


class PositionSizer:
    """
    Fractional Kelly position sizer.

    Converts a composite signal score and historical win-rate stats
    into a share quantity. Falls back to fixed_pct when history
    is insufficient for reliable Kelly estimation.

    Usage:
        sizer = PositionSizer(equity=50000)
        result = sizer.size("AAPL", signal_result, price=175.0)
    """

    def __init__(
        self,
        kelly_fraction:  float = 0.5,    # Half-Kelly (institutional standard)
        fallback_pct:    float = 0.02,   # 2% equity fallback when no history
        min_win_rate:    float = 0.40,   # Floor on win probability estimate
        signal_scale:    bool  = True,   # Scale qty by |signal_score|
    ) -> None:
        self._kelly_fraction = kelly_fraction
        self._fallback_pct   = fallback_pct
        self._min_win_rate   = min_win_rate
        self._signal_scale   = signal_scale
        # Rolling stats per symbol (updated after each closed trade)
        self._stats: dict[str, dict] = {}

    def size(
        self,
        symbol: str,
        signal: CompositeResult,
        price: float,
        equity: float,
    ) -> SizeResult:
        """
        Compute position size for a given signal and price.

        Args:
            symbol: Ticker symbol.
            signal: CompositeResult from SignalComposer.
            price:  Current market price (latest close or bid).
            equity: Portfolio equity from PortfolioTracker.

        Returns:
            SizeResult with qty and sizing diagnostics.
        """
        if price <= 0 or equity <= 0:
            return self._zero_result(symbol, signal.composite_score)

        stats = self._stats.get(symbol)

        if stats and stats["n_trades"] >= 10:
            sizing_method = "kelly"
            f = self._kelly_f(stats["win_rate"], stats["avg_win_loss_ratio"])
        else:
            sizing_method = "fixed_pct"
            f = self._fallback_pct

        # Apply fractional multiplier
        applied_f = f * self._kelly_fraction if sizing_method == "kelly" else f

        # Scale by signal strength (0.2 → 1.0 maps to 20%→100% of allocation)
        if self._signal_scale:
            strength = min(abs(signal.composite_score), 1.0)
            applied_f *= strength

        # Hard cap
        applied_f = min(applied_f, MAX_KELLY_FRACTION)
        applied_f = max(applied_f, 0.0)

        notional = equity * applied_f
        qty = notional / price

        # Round to 2 decimal places (Alpaca supports fractional shares)
        qty = math.floor(qty * 100) / 100.0

        if qty < MIN_QTY:
            return self._zero_result(symbol, signal.composite_score)

        logger.debug(
            "Size[%s] method=%s f=%.4f applied_f=%.4f qty=%.2f notional=$%.2f",
            symbol, sizing_method, f, applied_f, qty, notional,
        )

        return SizeResult(
            symbol=symbol,
            qty=qty,
            notional=round(notional, 2),
            kelly_f=round(f, 6),
            applied_f=round(applied_f, 6),
            signal_score=signal.composite_score,
            sizing_method=sizing_method,
        )

    def update_stats(
        self,
        symbol: str,
        win: bool,
        pnl_pct: float,
    ) -> None:
        """
        Update rolling win/loss stats after a closed trade.
        Called by ExecutionMonitor when a position is closed.
        """
        if symbol not in self._stats:
            self._stats[symbol] = {
                "n_trades": 0,
                "n_wins": 0,
                "total_win_pct": 0.0,
                "total_loss_pct": 0.0,
            }
        s = self._stats[symbol]
        s["n_trades"] += 1
        if win:
            s["n_wins"] += 1
            s["total_win_pct"] += abs(pnl_pct)
        else:
            s["total_loss_pct"] += abs(pnl_pct)

        n = s["n_trades"]
        n_wins = s["n_wins"]
        n_losses = n - n_wins

        s["win_rate"] = n_wins / n
        s["avg_win_loss_ratio"] = (
            (s["total_win_pct"] / n_wins) / (s["total_loss_pct"] / n_losses)
            if n_wins > 0 and n_losses > 0
            else 1.0
        )

    @staticmethod
    def _kelly_f(win_rate: float, win_loss_ratio: float) -> float:
        """
        Kelly fraction: f* = (b*p - q) / b
        Clamped to [0, MAX_KELLY_FRACTION].
        """
        p = max(win_rate, 0.01)
        q = 1.0 - p
        b = max(win_loss_ratio, 0.01)
        f = (b * p - q) / b
        return max(0.0, f)

    @staticmethod
    def _zero_result(symbol: str, score: float) -> SizeResult:
        return SizeResult(
            symbol=symbol, qty=0.0, notional=0.0,
            kelly_f=0.0, applied_f=0.0,
            signal_score=score, sizing_method="zero",
        )
