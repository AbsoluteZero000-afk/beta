"""
Pre-market gap scanner.

Identifies stocks gapping up or down significantly from the prior close.
Scores candidates by gap size, pre-market volume, and relative strength.

Run at 9:00–9:25am ET for gap-and-go setups.
"""
from __future__ import annotations

import os
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest, StockBarsRequest
from alpaca.data.timeframe import TimeFrame

logger = logging.getLogger(__name__)


@dataclass
class GapCandidate:
    symbol:           str
    prior_close:      float
    premarket_price:  float
    gap_pct:          float          # positive = gap up, negative = gap down
    premarket_volume: float
    avg_volume:       float          # 20-day average daily volume
    volume_ratio:     float          # premarket_vol / avg_daily_vol
    score:            float          # composite score 0–100
    direction:        str            # "long" | "short"
    suggested_entry:  float
    suggested_stop:   float
    suggested_target: float


class GapScanner:
    """
    Scans a watchlist for pre-market gap setups.

    Parameters
    ----------
    min_gap_pct : float
        Minimum absolute gap percentage to qualify (default 2%).
    min_volume_ratio : float
        Minimum premarket vol / avg daily vol ratio (default 0.05 = 5%).
    max_candidates : int
        Maximum number of candidates to return, ranked by score.
    stop_pct : float
        Stop-loss distance as fraction of entry price (default 1%).
    target_ratio : float
        Risk/reward ratio for target calculation (default 2.0 = 2:1).
    """

    def __init__(
        self,
        min_gap_pct:     float = 2.0,
        min_volume_ratio: float = 0.05,
        max_candidates:  int   = 10,
        stop_pct:        float = 0.01,
        target_ratio:    float = 2.0,
    ) -> None:
        self.min_gap_pct      = min_gap_pct
        self.min_volume_ratio = min_volume_ratio
        self.max_candidates   = max_candidates
        self.stop_pct         = stop_pct
        self.target_ratio     = target_ratio
        self._client = StockHistoricalDataClient(
            os.getenv("ALPACA_API_KEY", ""),
            os.getenv("ALPACA_SECRET_KEY", ""),
        )

    def scan(self, symbols: list[str]) -> list[GapCandidate]:
        """
        Scan symbols for gap-and-go candidates.
        Returns list of GapCandidate sorted by score descending.
        """
        candidates = []
        prior_closes = self._get_prior_closes(symbols)
        premarket    = self._get_premarket_quotes(symbols)
        avg_volumes  = self._get_avg_volumes(symbols)

        for sym in symbols:
            try:
                prior_close = prior_closes.get(sym)
                pm_price    = premarket.get(sym)
                avg_vol     = avg_volumes.get(sym, 0)

                if not prior_close or not pm_price or prior_close <= 0:
                    continue

                gap_pct = ((pm_price - prior_close) / prior_close) * 100

                if abs(gap_pct) < self.min_gap_pct:
                    continue

                # Estimate premarket volume from quote (not perfect but directional)
                pm_vol_ratio = 0.1  # placeholder — real vol needs bar data
                pm_bars = self._get_premarket_bars(sym)
                if pm_bars is not None and len(pm_bars) > 0:
                    pm_volume = pm_bars["volume"].sum()
                    pm_vol_ratio = pm_volume / avg_vol if avg_vol > 0 else 0
                else:
                    pm_volume = 0

                if pm_vol_ratio < self.min_volume_ratio:
                    continue

                direction = "long" if gap_pct > 0 else "short"
                score     = self._score(gap_pct, pm_vol_ratio)

                # Entry/stop/target
                if direction == "long":
                    entry  = pm_price * 1.001        # slight buffer above premarket
                    stop   = entry * (1 - self.stop_pct)
                    target = entry + (entry - stop) * self.target_ratio
                else:
                    entry  = pm_price * 0.999
                    stop   = entry * (1 + self.stop_pct)
                    target = entry - (stop - entry) * self.target_ratio

                candidates.append(GapCandidate(
                    symbol=sym,
                    prior_close=round(prior_close, 2),
                    premarket_price=round(pm_price, 2),
                    gap_pct=round(gap_pct, 2),
                    premarket_volume=pm_volume,
                    avg_volume=avg_vol,
                    volume_ratio=round(pm_vol_ratio, 4),
                    score=round(score, 1),
                    direction=direction,
                    suggested_entry=round(entry, 2),
                    suggested_stop=round(stop, 2),
                    suggested_target=round(target, 2),
                ))

            except Exception as e:
                logger.warning(f"GapScanner error on {sym}: {e}")
                continue

        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates[: self.max_candidates]

    def to_dataframe(self, candidates: list[GapCandidate]) -> pd.DataFrame:
        if not candidates:
            return pd.DataFrame()
        return pd.DataFrame([vars(c) for c in candidates])

    # ── Private helpers ──────────────────────────────────────────────────────

    def _get_prior_closes(self, symbols: list[str]) -> dict[str, float]:
        """Get yesterday's closing price for each symbol."""
        end   = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0)
        start = end - timedelta(days=5)
        out   = {}
        try:
            req  = StockBarsRequest(
                symbol_or_symbols=symbols,
                timeframe=TimeFrame.Day,
                start=start, end=end,
            )
            bars = self._client.get_stock_bars(req).df
            if bars.empty:
                return out
            bars = bars.reset_index()
            for sym in symbols:
                sym_bars = bars[bars["symbol"] == sym].sort_values("timestamp")
                if not sym_bars.empty:
                    out[sym] = float(sym_bars.iloc[-1]["close"])
        except Exception as e:
            logger.warning(f"Prior close fetch error: {e}")
        return out

    def _get_premarket_quotes(self, symbols: list[str]) -> dict[str, float]:
        """Get latest bid/ask midpoint as premarket price proxy."""
        out = {}
        try:
            req    = StockLatestQuoteRequest(symbol_or_symbols=symbols)
            quotes = self._client.get_stock_latest_quote(req)
            for sym, q in quotes.items():
                bid = getattr(q, "bid_price", 0) or 0
                ask = getattr(q, "ask_price", 0) or 0
                if bid > 0 and ask > 0:
                    out[sym] = (bid + ask) / 2
        except Exception as e:
            logger.warning(f"Quote fetch error: {e}")
        return out

    def _get_premarket_bars(self, symbol: str) -> Optional[pd.DataFrame]:
        """Get today's pre-market 1-min bars."""
        now   = datetime.now(timezone.utc)
        start = now.replace(hour=8, minute=0, second=0, microsecond=0)
        try:
            req  = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Minute,
                start=start, end=now,
            )
            bars = self._client.get_stock_bars(req).df
            return bars if not bars.empty else None
        except Exception:
            return None

    def _get_avg_volumes(self, symbols: list[str], days: int = 20) -> dict[str, float]:
        """Get 20-day average daily volume per symbol."""
        end   = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0)
        start = end - timedelta(days=days + 5)
        out   = {}
        try:
            req  = StockBarsRequest(
                symbol_or_symbols=symbols,
                timeframe=TimeFrame.Day,
                start=start, end=end,
            )
            bars = self._client.get_stock_bars(req).df.reset_index()
            for sym in symbols:
                sym_bars = bars[bars["symbol"] == sym]
                if not sym_bars.empty:
                    out[sym] = float(sym_bars["volume"].mean())
        except Exception as e:
            logger.warning(f"Avg volume fetch error: {e}")
        return out

    def _score(self, gap_pct: float, volume_ratio: float) -> float:
        """
        Composite score 0–100.
        Weights: 60% gap size, 40% volume confirmation.
        """
        gap_score = min(abs(gap_pct) / 10.0 * 60, 60)       # caps at 60 for 10%+ gap
        vol_score = min(volume_ratio / 0.5 * 40, 40)         # caps at 40 for 50%+ of avg vol
        return gap_score + vol_score
