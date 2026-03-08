"""
Tests for Phase 2 — Event Detection & Classification.
Run with: pytest tests/test_events.py -v
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from src.cache.signal_buffer import OHLCVBar, SignalBuffer
from src.events.classifier import EventClassifier
from src.events.queue import PriorityEventQueue
from src.events.types import (
    EventType,
    EventUrgency,
    MarketRegime,
    TradingEvent,
    TradingContext,
)
from src.models.ohlcv import PortfolioState


# ---- Fixtures ----

@pytest.fixture
def signal_buffer():
    return SignalBuffer(maxlen=100)


@pytest.fixture
def classifier(signal_buffer):
    return EventClassifier(signal_buffer)


def make_bar(symbol: str, close: float, volume: int, idx: int = 0) -> OHLCVBar:
    return OHLCVBar(
        time=datetime(2024, 1, 1, 9, 30 + idx, tzinfo=timezone.utc),
        symbol=symbol,
        open=close - 0.5,
        high=close + 1.0,
        low=close - 1.0,
        close=close,
        volume=volume,
    )


def make_alpaca_bar(symbol: str, close: float, volume: int):
    """Mock an alpaca-py Bar object."""
    bar = MagicMock()
    bar.symbol    = symbol
    bar.timestamp = datetime(2024, 1, 2, 10, 0, tzinfo=timezone.utc)
    bar.open      = close - 0.5
    bar.high      = close + 1.0
    bar.low       = close - 1.0
    bar.close     = close
    bar.volume    = volume
    bar.vwap      = close
    bar.trade_count = 100
    return bar


# ---- TradingEvent tests ----

def test_trading_event_priority_ordering():
    """Lower urgency value = higher priority in heapq."""
    t = datetime.now(timezone.utc)
    breach  = TradingEvent.make(EventType.PORTFOLIO_BREACH, EventUrgency.PORTFOLIO_BREACH, "AAPL", t)
    bar     = TradingEvent.make(EventType.SCHEDULED_BAR,    EventUrgency.SCHEDULED_BAR,    "AAPL", t)
    breakout = TradingEvent.make(EventType.BREAKOUT,        EventUrgency.BREAKOUT,          "AAPL", t)

    assert breach < breakout
    assert breakout < bar
    assert breach < bar


def test_trading_event_same_urgency_sorted_by_timestamp():
    """Within same urgency, earlier timestamp = higher priority."""
    t1 = datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc)
    t2 = datetime(2024, 1, 1, 10, 1, tzinfo=timezone.utc)
    e1 = TradingEvent.make(EventType.SCHEDULED_BAR, EventUrgency.SCHEDULED_BAR, "SPY", t1)
    e2 = TradingEvent.make(EventType.SCHEDULED_BAR, EventUrgency.SCHEDULED_BAR, "SPY", t2)
    assert e1 < e2


# ---- PriorityEventQueue tests ----

@pytest.mark.asyncio
async def test_priority_queue_order():
    """Events dequeued in urgency order regardless of insertion order."""
    queue = PriorityEventQueue(maxsize=100)
    t = datetime.now(timezone.utc)

    bar     = TradingEvent.make(EventType.SCHEDULED_BAR,    EventUrgency.SCHEDULED_BAR,    "SPY", t)
    spike   = TradingEvent.make(EventType.VOLUME_SPIKE,     EventUrgency.VOLUME_SPIKE,     "SPY", t)
    breach  = TradingEvent.make(EventType.PORTFOLIO_BREACH, EventUrgency.PORTFOLIO_BREACH, "SPY", t)

    # Insert in wrong order
    await queue.put(bar)
    await queue.put(spike)
    await queue.put(breach)

    # Should come out in priority order
    first  = await queue.get()
    second = await queue.get()
    third  = await queue.get()

    assert first.event_type  == EventType.PORTFOLIO_BREACH
    assert second.event_type == EventType.VOLUME_SPIKE
    assert third.event_type  == EventType.SCHEDULED_BAR


@pytest.mark.asyncio
async def test_priority_queue_drop_when_full():
    """Queue silently drops events when full."""
    queue = PriorityEventQueue(maxsize=2)
    t = datetime.now(timezone.utc)
    for i in range(5):
        await queue.put(
            TradingEvent.make(EventType.SCHEDULED_BAR, EventUrgency.SCHEDULED_BAR, "AAPL", t)
        )
    assert queue.qsize == 2
    assert queue.stats["dropped"] == 3


# ---- EventClassifier tests ----

def test_classifier_discards_zero_volume_bar(classifier):
    """Zero-volume bars are discarded as noise."""
    bar = make_alpaca_bar("AAPL", 150.0, 0)
    result = classifier.classify_bar(bar)
    assert result is None


def test_classifier_scheduled_bar_insufficient_data(classifier):
    """With < 20 bars, always returns SCHEDULED_BAR."""
    bar = make_alpaca_bar("MSFT", 300.0, 5000)
    result = classifier.classify_bar(bar)
    assert result is not None
    assert result.event_type == EventType.SCHEDULED_BAR
    assert result.urgency == int(EventUrgency.SCHEDULED_BAR)


def test_classifier_detects_volume_spike(classifier, signal_buffer):
    """Volume 3x above average should trigger VOLUME_SPIKE."""
    symbol = "TSLA"
    # Load 25 normal bars at avg volume 10000
    for i in range(25):
        signal_buffer.append(make_bar(symbol, 200.0, 10_000, i))

    # Now classify a bar with 3x volume
    bar = make_alpaca_bar(symbol, 200.0, 30_000)
    result = classifier.classify_bar(bar)
    assert result is not None
    assert result.event_type == EventType.VOLUME_SPIKE
    assert result.urgency == int(EventUrgency.VOLUME_SPIKE)


def test_classifier_detects_breakout(classifier, signal_buffer):
    """Price above 20-bar high should trigger BREAKOUT."""
    symbol = "NVDA"
    # Load 25 bars with high=500
    for i in range(25):
        signal_buffer.append(make_bar(symbol, 490.0, 5000, i))

    # Breakout bar: close far above rolling high
    bar = make_alpaca_bar(symbol, 520.0, 5000)  # >0.5% above high=491
    result = classifier.classify_bar(bar)
    assert result is not None
    assert result.event_type == EventType.BREAKOUT
    assert result.urgency == int(EventUrgency.BREAKOUT)


def test_classifier_context_contains_dataframe(classifier, signal_buffer):
    """TradingContext should contain OHLCV DataFrame."""
    symbol = "AAPL"
    for i in range(5):
        signal_buffer.append(make_bar(symbol, 150.0 + i, 5000, i))

    bar = make_alpaca_bar(symbol, 155.0, 5000)
    result = classifier.classify_bar(bar)
    assert result is not None
    assert result.context is not None
    assert not result.context.ohlcv.empty
    assert "close" in result.context.ohlcv.columns


def test_classifier_portfolio_breach_highest_priority(classifier):
    """Portfolio breach events should have urgency=0."""
    event = classifier.make_portfolio_breach_event("AAPL", "max_exposure_exceeded")
    assert event.urgency == int(EventUrgency.PORTFOLIO_BREACH)
    assert event.event_type == EventType.PORTFOLIO_BREACH


def test_regime_detection_volatile(classifier, signal_buffer):
    """High-volatility bars should result in VOLATILE regime."""
    import numpy as np
    symbol = "SPY"
    # Create high-volatility bars
    closes = [400.0 + np.random.normal(0, 10) for _ in range(25)]
    for i, c in enumerate(closes):
        signal_buffer.append(make_bar(symbol, c, 5000, i))

    df = signal_buffer.get_dataframe(symbol)
    regime = classifier._detect_regime(df)
    # With std dev ~10/400 = 2.5% we expect VOLATILE
    assert regime == MarketRegime.VOLATILE
