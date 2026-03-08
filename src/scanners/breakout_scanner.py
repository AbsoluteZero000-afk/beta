"""
Volume breakout scanner.

Detects stocks breaking above key resistance levels on above-average volume.
Best used in conjunction with the gap scanner for high-conviction setups.
"""
from __future__ import annotations

import os
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

logger = logging.getLogger(__name__)


@dataclass
class BreakoutCandidate:
    symbol:           str
    current_price:    float
    resistance_level: float
    breakout_pct:     float     # how far above resistance
    volume_ratio:     float     # today vol / 20d avg vol
    days_at_resistance: int     # consolidation period
    score:            float
    suggested_entry:  float
    suggested_stop:   float
    suggested_target: float


class BreakoutScanner:
    """Scans for volume-confirmed breakouts above resistance."""

    def __init__(
        self,
        min_breakout_pct:  float = 0.5,
        min_volume_ratio:  float = 1.5,
        max_candidates:    int   = 10,
        lookback_days:     int   = 20,
    ) -> None:
        self.min_breakout_pct = min_breakout_pct
        self.min_volume_ratio = min_volume_ratio
        self.max_candidates   = max_candidates
        self.lookback_days    = lookback_days
        self._client = StockHistoricalDataClient(
            os.getenv("ALPACA_API_KEY", ""),
            os.getenv("ALPACA_SECRET_KEY", ""),
        )

    def scan(self, symbols: list[str]) -> list[BreakoutCandidate]:
        candidates = []
        end   = datetime.now(timezone.utc)
        start = end - timedelta(days=self.lookback_days + 30)

        try:
            req  = StockBarsRequest(
                symbol_or_symbols=symbols,
                timeframe=TimeFrame.Day,
                start=start, end=end,
            )
            bars = self._client.get_stock_bars(req).df.reset_index()
        except Exception as e:
            logger.warning(f"BreakoutScanner fetch error: {e}")
            return []

        for sym in symbols:
            try:
                sym_bars = bars[bars["symbol"] == sym].sort_values("timestamp")
                if len(sym_bars) < self.lookback_days + 5:
                    continue

                closes  = sym_bars["close"].values.astype(float)
                volumes = sym_bars["volume"].values.astype(float)

                current    = closes[-1]
                resistance = float(np.max(closes[-self.lookback_days:-1]))
                avg_vol    = float(np.mean(volumes[-20:-1]))
                today_vol  = float(volumes[-1])
                vol_ratio  = today_vol / avg_vol if avg_vol > 0 else 0

                breakout_pct = ((current - resistance) / resistance) * 100
                if breakout_pct < self.min_breakout_pct:
                    continue
                if vol_ratio < self.min_volume_ratio:
                    continue

                # Count days price was near resistance (consolidation)
                near_res = sum(1 for c in closes[-self.lookback_days:-1]
                               if abs(c - resistance) / resistance < 0.02)

                score = self._score(breakout_pct, vol_ratio, near_res)

                stop   = resistance * 0.99     # just below breakout level
                target = current + (current - stop) * 2

                candidates.append(BreakoutCandidate(
                    symbol=sym,
                    current_price=round(current, 2),
                    resistance_level=round(resistance, 2),
                    breakout_pct=round(breakout_pct, 2),
                    volume_ratio=round(vol_ratio, 2),
                    days_at_resistance=near_res,
                    score=round(score, 1),
                    suggested_entry=round(current * 1.001, 2),
                    suggested_stop=round(stop, 2),
                    suggested_target=round(target, 2),
                ))
            except Exception as e:
                logger.warning(f"BreakoutScanner error on {sym}: {e}")

        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates[: self.max_candidates]

    def to_dataframe(self, candidates) -> pd.DataFrame:
        if not candidates:
            return pd.DataFrame()
        return pd.DataFrame([vars(c) for c in candidates])

    def _score(self, breakout_pct, vol_ratio, days_consolidating) -> float:
        score  = min(breakout_pct * 10, 40)
        score += min((vol_ratio - 1) * 20, 40)
        score += min(days_consolidating * 1.5, 20)
        return score
