"""
Tests for Redis and Signal Buffer.
Run with: pytest tests/test_cache.py -v
Requires Docker containers to be running.
"""

from datetime import datetime, timezone

import pytest

from src.cache.redis_client import get_redis, close_redis
from src.cache.signal_buffer import OHLCVBar, SignalBuffer


@pytest.mark.asyncio
async def test_redis_ping():
    """Test Redis connectivity."""
    redis = await get_redis()
    assert await redis.ping() is True
    await close_redis()


@pytest.mark.asyncio
async def test_redis_set_get():
    """Test basic Redis set/get."""
    redis = await get_redis()
    await redis.set("test:key", "hello", ex=60)
    val = await redis.get("test:key")
    assert val == "hello"
    await redis.delete("test:key")
    await close_redis()


def test_signal_buffer_append():
    """Test appending bars to SignalBuffer."""
    buf = SignalBuffer(maxlen=5)
    for i in range(10):
        bar = OHLCVBar(
            time=datetime.now(timezone.utc),
            symbol="AAPL",
            open=150.0 + i,
            high=155.0 + i,
            low=149.0 + i,
            close=153.0 + i,
            volume=1000 * (i + 1),
        )
        buf.append(bar)

    # maxlen=5, so only last 5 should remain
    assert buf.size("AAPL") == 5
    bars = buf.get_bars("AAPL")
    assert bars[0].open == 155.0  # bar index 5


def test_signal_buffer_dataframe():
    """Test DataFrame conversion."""
    buf = SignalBuffer(maxlen=100)
    for i in range(3):
        bar = OHLCVBar(
            time=datetime(2024, 1, 1 + i, tzinfo=timezone.utc),
            symbol="MSFT",
            open=300.0 + i,
            high=305.0 + i,
            low=298.0 + i,
            close=302.0 + i,
            volume=5000,
        )
        buf.append(bar)

    df = buf.get_dataframe("MSFT")
    assert len(df) == 3
    assert "close" in df.columns
    assert "volume" in df.columns


@pytest.mark.asyncio
async def test_signal_buffer_redis_persistence():
    """Test flush/restore to Redis."""
    buf = SignalBuffer(maxlen=100)
    bar = OHLCVBar(
        time=datetime.now(timezone.utc),
        symbol="TSLA",
        open=200.0,
        high=210.0,
        low=195.0,
        close=205.0,
        volume=50000,
    )
    buf.append(bar)

    await buf.flush_to_redis("TSLA")

    # Restore into a fresh buffer
    buf2 = SignalBuffer(maxlen=100)
    restored = await buf2.restore_from_redis("TSLA")
    assert restored == 1
    bars = buf2.get_bars("TSLA")
    assert bars[0].close == 205.0

    await close_redis()
