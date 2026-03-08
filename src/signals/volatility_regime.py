"""
QuantEdge — Volatility Regime Signal.

Measures short-term vs long-term volatility expansion/contraction
and trend strength via linear regression slope.

Expanding volatility = bearish (more uncertainty).
Contracting volatility + uptrend = bullish.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.signals.base import BaseSignal
from src.signals.kernels import compute_volatility_signal, compute_trend_strength


class VolatilityRegimeSignal(BaseSignal):
    """
    Volatility regime signal: vol ratio + trend strength.

    Score > 0 = contracting vol / strong uptrend (bullish)
    Score < 0 = expanding vol / strong downtrend (bearish)
    """

    name = "volatility_regime"
    description = "Vol expansion/contraction + trend strength"

    def __init__(
        self,
        short_period: int = 5,
        long_period:  int = 20,
    ) -> None:
        self._short = short_period
        self._long  = long_period

    @property
    def min_bars(self) -> int:
        return self._long + 2

    def compute(self, data: pd.DataFrame) -> float:
        closes = data["close"].to_numpy(dtype=np.float64)

        vol_score   = compute_volatility_signal(closes, self._short, self._long)
        trend_score = compute_trend_strength(closes, self._long)

        return float((vol_score + trend_score) / 2.0)
