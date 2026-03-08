"""
QuantEdge — Mean Reversion Signal.

Combines Z-score and Bollinger Band position.
Score in [-1, 1]: positive = price below mean (buy), negative = above (sell).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.signals.base import BaseSignal
from src.signals.kernels import compute_zscore, compute_bollinger_position


class MeanReversionSignal(BaseSignal):
    """
    Mean reversion signal: Z-score + Bollinger Band position.

    High score (+1) = price significantly below rolling mean = buy signal.
    Low score (-1) = price significantly above rolling mean = sell signal.
    """

    name = "mean_reversion"
    description = "Z-score + Bollinger Band position"

    def __init__(self, period: int = 20, num_std: float = 2.0) -> None:
        self._period  = period
        self._num_std = num_std

    @property
    def min_bars(self) -> int:
        return self._period

    def compute(self, data: pd.DataFrame) -> float:
        closes = data["close"].to_numpy(dtype=np.float64)

        zscore_score    = compute_zscore(closes, self._period)
        bollinger_score = compute_bollinger_position(closes, self._period, self._num_std)

        return float((zscore_score + bollinger_score) / 2.0)
