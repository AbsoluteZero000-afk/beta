"""
5-day momentum scanner.

Finds stocks with the strongest short-term price momentum heading into open.
Useful for riding continuation moves on gap-and-go days.
"""
from __future__ import annotations

import os
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

logger = logging.getLogger(__name__)


@dataclass
class MomentumCandidate:
    symbol:       str
    momentum_5d:  float    # 5-day return %
    momentum_10d: float    # 10-day return %
    momentum_20d: float    # 20-day return %
    rsi_14:       float    # RSI(14)
    above_20ma:   bool
    above_50ma:   bool
    score:        float
    direction:    str      # "long" | "short"


class MomentumScanner:
    """Scans for strongest momentum candidates."""

    def __init__(self, max_candidates: int = 10) -> None:
        self.max_candidates = max_candidates
        self._client = StockHistoricalDataClient(
            os.getenv("ALPACA_API_KEY", ""),
            os.getenv("ALPACA_SECRET_KEY", ""),
        )

    def scan(self, symbols: list[str]) -> list[MomentumCandidate]:
        candidates = []
        end   = datetime.now(timezone.utc)
        start = end - timedelta(days=90)

        try:
            req  = StockBarsRequest(
                symbol_or_symbols=symbols,
                timeframe=TimeFrame.Day,
                start=start, end=end,
            )
            bars = self._client.get_stock_bars(req).df.reset_index()
        except Exception as e:
            logger.warning(f"MomentumScanner fetch error: {e}")
            return []

        for sym in symbols:
            try:
                sym_bars = bars[bars["symbol"] == sym].sort_values("timestamp")
                if len(sym_bars) < 55:
                    continue

                closes = sym_bars["close"].values.astype(float)
                c      = closes[-1]

                mom_5d  = (c / closes[-6]  - 1) * 100 if len(closes) >= 6  else 0
                mom_10d = (c / closes[-11] - 1) * 100 if len(closes) >= 11 else 0
                mom_20d = (c / closes[-21] - 1) * 100 if len(closes) >= 21 else 0

                ma20 = closes[-20:].mean()
                ma50 = closes[-50:].mean()
                rsi  = self._rsi(closes)

                score = self._score(mom_5d, mom_10d, mom_20d, rsi, c > ma20, c > ma50)

                if abs(score) < 20:
                    continue

                candidates.append(MomentumCandidate(
                    symbol=sym,
                    momentum_5d=round(mom_5d, 2),
                    momentum_10d=round(mom_10d, 2),
                    momentum_20d=round(mom_20d, 2),
                    rsi_14=round(rsi, 1),
                    above_20ma=bool(c > ma20),
                    above_50ma=bool(c > ma50),
                    score=round(score, 1),
                    direction="long" if score > 0 else "short",
                ))
            except Exception as e:
                logger.warning(f"MomentumScanner error on {sym}: {e}")

        candidates.sort(key=lambda c: abs(c.score), reverse=True)
        return candidates[: self.max_candidates]

    def to_dataframe(self, candidates) -> pd.DataFrame:
        if not candidates:
            return pd.DataFrame()
        return pd.DataFrame([vars(c) for c in candidates])

    def _rsi(self, closes, period: int = 14) -> float:
        if len(closes) < period + 1:
            return 50.0
        deltas = pd.Series(closes).diff().dropna()
        gain   = deltas.clip(lower=0).rolling(period).mean().iloc[-1]
        loss   = (-deltas.clip(upper=0)).rolling(period).mean().iloc[-1]
        if loss == 0:
            return 100.0
        rs = gain / loss
        return float(100 - 100 / (1 + rs))

    def _score(self, m5, m10, m20, rsi, above_20ma, above_50ma) -> float:
        score  = m5 * 3 + m10 * 2 + m20 * 1
        score += 10 if above_20ma else -10
        score += 10 if above_50ma else -10
        if rsi > 70:  score += 5
        if rsi < 30:  score -= 5
        return score
