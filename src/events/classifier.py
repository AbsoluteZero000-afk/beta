"""
QuantEdge — Event Classifier.

Converts raw Alpaca WebSocket data (Bar, Trade, Quote) into
typed TradingEvent objects with urgency rankings.
Discards noise (zero-volume bars, stale quotes, etc.).

Alpaca model docs:
  https://alpaca.markets/sdks/python/api_reference/data/models.html
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional, Union

from alpaca.data.models import Bar, Quote, Trade

from src.cache.signal_buffer import OHLCVBar, SignalBuffer
from src.events.types import (
    EventType,
    EventUrgency,
    MarketRegime,
    TradingContext,
    TradingEvent,
)
from src.models.ohlcv import PortfolioState

logger = logging.getLogger(__name__)


# ---- Thresholds for event classification ----
VOLUME_SPIKE_MULTIPLIER = 2.0   # Volume > 2x rolling average = spike
BREAKOUT_PCT_THRESHOLD  = 0.005 # 0.5% move from rolling high/low = breakout
MIN_BARS_FOR_SIGNALS    = 20    # Minimum bars before classifying events


class EventClassifier:
    """
    Classifies raw Alpaca stream data into typed TradingEvents.

    Maintains a reference to the SignalBuffer to access rolling
    OHLCV windows for context-aware classification (e.g., volume
    spike detection requires a rolling average).
    """

    def __init__(
        self,
        signal_buffer: SignalBuffer,
        portfolio_state: Optional[PortfolioState] = None,
    ) -> None:
        self._buffer = signal_buffer
        self._portfolio_state = portfolio_state or PortfolioState()
        self._bar_count: dict[str, int] = {}

    def update_portfolio_state(self, state: PortfolioState) -> None:
        """Update the portfolio snapshot used in TradingContext."""
        self._portfolio_state = state

    # ---- Public classification entry points ----

    def classify_bar(self, bar: Bar) -> Optional[TradingEvent]:
        """
        Process a 1-minute OHLCV bar from Alpaca.

        Returns None if the bar is noise (zero volume, weekend, etc.)
        Adds bar to SignalBuffer before classification.
        """
        # Discard noise
        if bar.volume == 0:
            logger.debug("Discarding zero-volume bar for %s", bar.symbol)
            return None

        # Store in rolling buffer
        ohlcv_bar = OHLCVBar(
            time=bar.timestamp,
            symbol=bar.symbol,
            open=float(bar.open),
            high=float(bar.high),
            low=float(bar.low),
            close=float(bar.close),
            volume=int(bar.volume),
            vwap=float(bar.vwap) if bar.vwap else None,
            trade_count=int(bar.trade_count) if bar.trade_count else None,
        )
        self._buffer.append(ohlcv_bar)
        count = self._buffer.size(bar.symbol)

        # Build context
        context = self._build_context(bar.symbol, bar.timestamp)

        # Not enough data yet — treat as scheduled bar
        if count < MIN_BARS_FOR_SIGNALS:
            return TradingEvent.make(
                event_type=EventType.SCHEDULED_BAR,
                urgency=EventUrgency.SCHEDULED_BAR,
                symbol=bar.symbol,
                timestamp=bar.timestamp,
                context=context,
                raw_data={"bar_count": count},
            )

        # Check for breakout
        if self._is_breakout(bar.symbol, bar.close):
            logger.info("BREAKOUT detected: %s @ %.2f", bar.symbol, bar.close)
            return TradingEvent.make(
                event_type=EventType.BREAKOUT,
                urgency=EventUrgency.BREAKOUT,
                symbol=bar.symbol,
                timestamp=bar.timestamp,
                context=context,
            )

        # Check for volume spike
        if self._is_volume_spike(bar.symbol, bar.volume):
            logger.info("VOLUME_SPIKE detected: %s vol=%d", bar.symbol, bar.volume)
            return TradingEvent.make(
                event_type=EventType.VOLUME_SPIKE,
                urgency=EventUrgency.VOLUME_SPIKE,
                symbol=bar.symbol,
                timestamp=bar.timestamp,
                context=context,
            )

        # Default: scheduled bar
        return TradingEvent.make(
            event_type=EventType.SCHEDULED_BAR,
            urgency=EventUrgency.SCHEDULED_BAR,
            symbol=bar.symbol,
            timestamp=bar.timestamp,
            context=context,
        )

    def classify_trade(self, trade: Trade) -> Optional[TradingEvent]:
        """
        Process a real-time trade tick.

        Currently used to detect large block trades (future: dark pool signal).
        Returns None for normal-sized trades to reduce queue pressure.
        """
        # Only surface unusually large trades (top ~1% by size)
        # In production, replace 10000 with a per-symbol dynamic threshold
        if trade.size < 10_000:
            return None

        context = self._build_context(trade.symbol, trade.timestamp)
        return TradingEvent.make(
            event_type=EventType.TRADE,
            urgency=EventUrgency.VOLUME_SPIKE,
            symbol=trade.symbol,
            timestamp=trade.timestamp,
            context=context,
            raw_data={"price": float(trade.price), "size": float(trade.size)},
        )

    def classify_quote(self, quote: Quote) -> Optional[TradingEvent]:
        """
        Process a real-time NBBO quote update.

        Currently discards all quotes to avoid flooding the queue.
        Quotes are very high frequency — only surface extreme spreads.
        """
        # Extreme spread detection: bid/ask spread > 2%
        if quote.bid_price and quote.ask_price and quote.bid_price > 0:
            spread_pct = (quote.ask_price - quote.bid_price) / quote.bid_price
            if spread_pct > 0.02:
                context = self._build_context(quote.symbol, quote.timestamp)
                return TradingEvent.make(
                    event_type=EventType.QUOTE,
                    urgency=EventUrgency.BREAKOUT,
                    symbol=quote.symbol,
                    timestamp=quote.timestamp,
                    context=context,
                    raw_data={
                        "bid": float(quote.bid_price),
                        "ask": float(quote.ask_price),
                        "spread_pct": round(spread_pct * 100, 3),
                    },
                )
        return None

    def make_portfolio_breach_event(
        self, symbol: str, reason: str
    ) -> TradingEvent:
        """
        Manually inject a PORTFOLIO_BREACH event (highest priority).
        Called by the RiskGate in Phase 4 when limits are hit.
        """
        context = self._build_context(symbol, datetime.now(timezone.utc))
        return TradingEvent.make(
            event_type=EventType.PORTFOLIO_BREACH,
            urgency=EventUrgency.PORTFOLIO_BREACH,
            symbol=symbol,
            timestamp=datetime.now(timezone.utc),
            context=context,
            raw_data={"reason": reason},
        )

    # ---- Private helpers ----

    def _build_context(self, symbol: str, timestamp: datetime) -> TradingContext:
        """Build a TradingContext snapshot for a symbol."""
        df = self._buffer.get_dataframe(symbol)
        return TradingContext(
            symbol=symbol,
            asset_class="us_equity",
            timeframe="1Min",
            ohlcv=df,
            portfolio_state=self._portfolio_state,
            regime=self._detect_regime(df),
        )

    def _detect_regime(self, df) -> MarketRegime:
        """
        Simple regime detection from rolling OHLCV data.
        Full implementation in Phase 3 (SignalEngine).
        This is a lightweight version for event classification context.
        """
        if len(df) < 20:
            return MarketRegime.UNKNOWN

        import numpy as np
        closes = df["close"].values.astype(float)
        returns = np.diff(closes) / closes[:-1]
        volatility = float(np.std(returns))
        trend = float(closes[-1] - closes[-20]) / closes[-20]

        if volatility > 0.02:
            return MarketRegime.VOLATILE
        elif trend > 0.005:
            return MarketRegime.TRENDING_UP
        elif trend < -0.005:
            return MarketRegime.TRENDING_DOWN
        else:
            return MarketRegime.MEAN_REVERTING

    def _is_volume_spike(self, symbol: str, current_volume: float) -> bool:
        """True if current bar volume > VOLUME_SPIKE_MULTIPLIER * rolling avg."""
        bars = self._buffer.get_bars(symbol)
        if len(bars) < 10:
            return False
        import numpy as np
        avg_volume = float(np.mean([b.volume for b in bars[-20:-1]]))
        if avg_volume == 0:
            return False
        return current_volume > avg_volume * VOLUME_SPIKE_MULTIPLIER

    def _is_breakout(self, symbol: str, current_close: float) -> bool:
        """True if price breaks above 20-bar high or below 20-bar low."""
        bars = self._buffer.get_bars(symbol)
        if len(bars) < 21:
            return False
        window = bars[-21:-1]  # Exclude current bar
        rolling_high = max(b.high for b in window)
        rolling_low  = min(b.low  for b in window)
        return (
            current_close > rolling_high * (1 + BREAKOUT_PCT_THRESHOLD) or
            current_close < rolling_low  * (1 - BREAKOUT_PCT_THRESHOLD)
        )
