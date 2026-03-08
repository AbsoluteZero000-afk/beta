"""
QuantEdge — Momentum Signal.

Combines RSI and dual moving average momentum via equal weighting.
Score in [-1, 1]: positive = bullish momentum, negative = bearish.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.signals.base import BaseSignal
from src.signals.kernels import compute_rsi, compute_momentum_score


class MomentumSignal(BaseSignal):
    """
    Momentum signal: RSI + dual-MA crossover.

    Uses @numba.jit kernels for all numerical computation.
    pandas DataFrame is converted to numpy arrays before JIT calls.
    """

    name = "momentum"
    description = "RSI + dual moving average momentum"

    def __init__(
        self,
        rsi_period: int = 14,
        fast_ma: int = 5,
        slow_ma: int = 20,
    ) -> None:
        self._rsi_period = rsi_period
        self._fast_ma = fast_ma
        self._slow_ma = slow_ma

    @property
    def min_bars(self) -> int:
        return max(self._rsi_period + 1, self._slow_ma)

    def compute(self, data: pd.DataFrame) -> float:
        # Extract numpy arrays — required before numba JIT calls
        closes = data["close"].to_numpy(dtype=np.float64)

        rsi_score = compute_rsi(closes, self._rsi_period)
        ma_score  = compute_momentum_score(closes, self._fast_ma, self._slow_ma)

        # Equal-weight composite
        return float((rsi_score + ma_score) / 2.0)
