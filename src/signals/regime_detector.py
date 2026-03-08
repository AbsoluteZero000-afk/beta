"""
QuantEdge — Market Regime Detector.

Classifies the current market regime from rolling volatility and
trend strength metrics. Used by SignalComposer to select weight tables.

Regimes:
  TRENDING_UP    — strong positive slope, low vol
  TRENDING_DOWN  — strong negative slope, low vol
  MEAN_REVERTING — weak slope, low vol
  VOLATILE       — high realized volatility regardless of direction
  UNKNOWN        — insufficient data
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.events.types import MarketRegime
from src.signals.kernels import compute_realized_volatility, compute_trend_strength

logger = logging.getLogger(__name__)

# Regime classification thresholds
VOL_THRESHOLD_HIGH   = 0.30   # annualized vol > 30% = volatile
TREND_THRESHOLD_STRONG = 0.35  # |trend score| > 0.35 = trending


class RegimeDetector:
    """
    Classifies market regime from OHLCV data using:
      1. Realized annualized volatility (20-bar)
      2. Linear regression trend strength (20-bar)

    Returns one of: TRENDING_UP, TRENDING_DOWN, MEAN_REVERTING,
                    VOLATILE, UNKNOWN
    """

    def __init__(
        self,
        vol_period: int = 20,
        trend_period: int = 20,
    ) -> None:
        self._vol_period   = vol_period
        self._trend_period = trend_period

    def detect(self, data: pd.DataFrame) -> MarketRegime:
        """
        Detect market regime from OHLCV DataFrame.
        Returns UNKNOWN if insufficient data.
        """
        if data.empty or len(data) < max(self._vol_period, self._trend_period) + 1:
            return MarketRegime.UNKNOWN

        closes = data["close"].to_numpy(dtype=np.float64)

        vol   = compute_realized_volatility(closes, self._vol_period)
        trend = compute_trend_strength(closes, self._trend_period)

        regime = self._classify(vol, trend)
        logger.debug(
            "Regime: %s (vol=%.3f trend=%.3f)",
            regime.value, vol, trend,
        )
        return regime

    def _classify(self, vol: float, trend: float) -> MarketRegime:
        if vol > VOL_THRESHOLD_HIGH:
            return MarketRegime.VOLATILE
        if trend > TREND_THRESHOLD_STRONG:
            return MarketRegime.TRENDING_UP
        if trend < -TREND_THRESHOLD_STRONG:
            return MarketRegime.TRENDING_DOWN
        return MarketRegime.MEAN_REVERTING
