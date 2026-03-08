"""
QuantEdge — Volume Anomaly Signal.

Detects directional volume spikes: high volume on an up move is bullish,
high volume on a down move is bearish. Normal volume returns near zero.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.signals.base import BaseSignal
from src.signals.kernels import compute_volume_anomaly


class VolumeAnomalySignal(BaseSignal):
    """
    Volume anomaly signal using price-weighted volume ratio.

    Score > 0 = bullish volume (up move + high volume)
    Score < 0 = bearish volume (down move + high volume)
    Score ~ 0 = normal volume activity
    """

    name = "volume_anomaly"
    description = "Directional volume spike detection"

    def __init__(self, period: int = 20) -> None:
        self._period = period

    @property
    def min_bars(self) -> int:
        return self._period + 2

    def compute(self, data: pd.DataFrame) -> float:
        closes  = data["close"].to_numpy(dtype=np.float64)
        volumes = data["volume"].to_numpy(dtype=np.float64)

        return float(compute_volume_anomaly(closes, volumes, self._period))
