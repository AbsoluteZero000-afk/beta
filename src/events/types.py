"""
QuantEdge — Event Types & Data Structures.

Defines all typed events that flow through the PriorityEventQueue.
Lower urgency number = higher priority (heapq is min-heap).
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import pandas as pd

from src.models.ohlcv import PortfolioState


# ============================================================
# Event Urgency Levels (lower = higher priority)
# ============================================================
class EventUrgency(enum.IntEnum):
    PORTFOLIO_BREACH = 0   # Immediate — margin/risk limit hit
    BREAKOUT         = 1   # High — price breaking key level
    VOLUME_SPIKE     = 2   # Medium-high — unusual volume
    REGIME_CHANGE    = 3   # Medium — market regime shift
    SCHEDULED_BAR    = 4   # Normal — routine 1-min bar arrival


class EventType(str, enum.Enum):
    BAR             = "bar"
    TRADE           = "trade"
    QUOTE           = "quote"
    TRADING_STATUS  = "trading_status"
    PORTFOLIO_BREACH = "portfolio_breach"
    BREAKOUT        = "breakout"
    VOLUME_SPIKE    = "volume_spike"
    REGIME_CHANGE   = "regime_change"
    SCHEDULED_BAR   = "scheduled_bar"


class MarketRegime(str, enum.Enum):
    TRENDING_UP    = "trending_up"
    TRENDING_DOWN  = "trending_down"
    MEAN_REVERTING = "mean_reverting"
    VOLATILE       = "volatile"
    UNKNOWN        = "unknown"


# ============================================================
# Trading Context — passed with every event downstream
# ============================================================
@dataclass
class TradingContext:
    """
    Full context passed to the signal engine and decision layer.
    Carries everything needed to make a trading decision without
    additional database lookups.
    """
    symbol:          str
    asset_class:     str                        # "us_equity", "crypto"
    timeframe:       str                        # "1Min", "5Min", "1Day"
    ohlcv:           pd.DataFrame               # rolling window from SignalBuffer
    portfolio_state: PortfolioState
    regime:          MarketRegime = MarketRegime.UNKNOWN
    metadata:        dict         = field(default_factory=dict)

    # Convenience properties
    @property
    def latest_close(self) -> Optional[float]:
        if self.ohlcv.empty:
            return None
        return float(self.ohlcv["close"].iloc[-1])

    @property
    def latest_volume(self) -> Optional[float]:
        if self.ohlcv.empty:
            return None
        return float(self.ohlcv["volume"].iloc[-1])

    @property
    def has_sufficient_data(self) -> bool:
        """True if we have at least 20 bars for signal computation."""
        return len(self.ohlcv) >= 20


# ============================================================
# Core Trading Event
# ============================================================
@dataclass(order=True)
class TradingEvent:
    """
    A typed, prioritized trading event that flows through the
    PriorityEventQueue.

    order=True enables heapq comparison by urgency first,
    then timestamp (both are comparable types).
    """
    urgency:    int           # EventUrgency value — heapq sorts by this first
    timestamp:  datetime      # Sort key within same urgency level
    event_type: EventType     = field(compare=False)
    symbol:     str           = field(compare=False)
    context:    Optional[TradingContext] = field(compare=False, default=None)
    raw_data:   Optional[dict] = field(compare=False, default=None)

    @classmethod
    def make(
        cls,
        event_type: EventType,
        urgency: EventUrgency,
        symbol: str,
        timestamp: datetime,
        context: Optional[TradingContext] = None,
        raw_data: Optional[dict] = None,
    ) -> TradingEvent:
        return cls(
            urgency=int(urgency),
            timestamp=timestamp,
            event_type=event_type,
            symbol=symbol,
            context=context,
            raw_data=raw_data,
        )
