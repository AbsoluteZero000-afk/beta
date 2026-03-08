"""
QuantEdge — Signal Buffer.

Rolling OHLCV window per symbol using collections.deque for O(1)
append/pop. Backed by Redis for persistence across restarts.

The buffer keeps the most recent N bars in memory (hot path)
and serializes snapshots to Redis for crash recovery.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import orjson
import pandas as pd

from src.cache.redis_client import get_redis
from src.config import settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OHLCVBar:
    """A single OHLCV bar."""

    time: datetime
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    vwap: Optional[float] = None
    trade_count: Optional[int] = None

    def to_dict(self) -> dict:
        return {
            "time": self.time.isoformat(),
            "symbol": self.symbol,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "vwap": self.vwap,
            "trade_count": self.trade_count,
        }

    @classmethod
    def from_dict(cls, data: dict) -> OHLCVBar:
        return cls(
            time=datetime.fromisoformat(data["time"]),
            symbol=data["symbol"],
            open=data["open"],
            high=data["high"],
            low=data["low"],
            close=data["close"],
            volume=data["volume"],
            vwap=data.get("vwap"),
            trade_count=data.get("trade_count"),
        )


class SignalBuffer:
    """
    Per-symbol rolling window of OHLCV bars using collections.deque.

    - O(1) append to right, automatic eviction from left when maxlen exceeded
    - Periodically flushes to Redis for persistence
    - Can reconstruct a pandas DataFrame for signal computation
    """

    def __init__(self, maxlen: int | None = None) -> None:
        self._maxlen = maxlen or settings.signal_buffer_size
        self._buffers: dict[str, deque[OHLCVBar]] = {}

    def _get_or_create(self, symbol: str) -> deque[OHLCVBar]:
        if symbol not in self._buffers:
            self._buffers[symbol] = deque(maxlen=self._maxlen)
        return self._buffers[symbol]

    def append(self, bar: OHLCVBar) -> None:
        """Append a bar to the symbol's rolling window."""
        buf = self._get_or_create(bar.symbol)
        buf.append(bar)

    def get_bars(self, symbol: str) -> list[OHLCVBar]:
        """Get all bars for a symbol (oldest first)."""
        buf = self._buffers.get(symbol, deque())
        return list(buf)

    def get_dataframe(self, symbol: str) -> pd.DataFrame:
        """
        Convert the rolling window to a pandas DataFrame.
        Columns: time, open, high, low, close, volume, vwap, trade_count
        """
        bars = self.get_bars(symbol)
        if not bars:
            return pd.DataFrame(
                columns=["time", "open", "high", "low", "close", "volume", "vwap", "trade_count"]
            )
        records = [b.to_dict() for b in bars]
        df = pd.DataFrame(records)
        df["time"] = pd.to_datetime(df["time"])
        df.set_index("time", inplace=True)
        return df

    @property
    def symbols(self) -> list[str]:
        """List all symbols currently buffered."""
        return list(self._buffers.keys())

    def size(self, symbol: str) -> int:
        """Number of bars currently in the buffer for a symbol."""
        return len(self._buffers.get(symbol, deque()))

    def clear(self, symbol: str | None = None) -> None:
        """Clear buffer for a specific symbol or all symbols."""
        if symbol:
            self._buffers.pop(symbol, None)
        else:
            self._buffers.clear()

    # ----- Redis persistence -----

    async def flush_to_redis(self, symbol: str) -> None:
        """Serialize the symbol buffer to Redis for crash recovery."""
        client = await get_redis()
        bars = self.get_bars(symbol)
        data = orjson.dumps([b.to_dict() for b in bars]).decode()
        await client.set(f"signal_buffer:{symbol}", data, ex=3600)
        logger.debug("Flushed %d bars for %s to Redis", len(bars), symbol)

    async def restore_from_redis(self, symbol: str) -> int:
        """Restore a symbol buffer from Redis. Returns number of bars restored."""
        client = await get_redis()
        raw = await client.get(f"signal_buffer:{symbol}")
        if raw is None:
            return 0

        records = orjson.loads(raw)
        buf = self._get_or_create(symbol)
        buf.clear()
        for record in records:
            buf.append(OHLCVBar.from_dict(record))

        logger.debug("Restored %d bars for %s from Redis", len(buf), symbol)
        return len(buf)

    async def flush_all(self) -> None:
        """Flush all symbol buffers to Redis."""
        for symbol in self.symbols:
            await self.flush_to_redis(symbol)

    async def restore_all(self, symbols: list[str]) -> dict[str, int]:
        """Restore buffers for all given symbols from Redis."""
        results: dict[str, int] = {}
        for symbol in symbols:
            results[symbol] = await self.restore_from_redis(symbol)
        return results
