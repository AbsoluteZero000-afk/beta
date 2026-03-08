"""
QuantEdge — Signal Composer.

Applies dynamic regime-based weight tables to combine individual
signal scores into a single composite score in [-1, 1].

Regime weight tables:
  - Different regimes favor different signal types.
  - TRENDING: momentum dominates
  - MEAN_REVERTING: mean reversion dominates
  - VOLATILE: volatility regime + mean reversion dominate
  - UNKNOWN: equal weighting

All weights in each table sum to 1.0.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from src.events.types import MarketRegime
from src.signals.base import BaseSignal
from src.signals.momentum import MomentumSignal
from src.signals.mean_reversion import MeanReversionSignal
from src.signals.volume_anomaly import VolumeAnomalySignal
from src.signals.volatility_regime import VolatilityRegimeSignal
from src.signals.regime_detector import RegimeDetector

logger = logging.getLogger(__name__)


# ============================================================
# Regime-based weight tables
# Keys must match signal.name attributes exactly.
# All weights per row must sum to 1.0.
# ============================================================
REGIME_WEIGHTS: dict[MarketRegime, dict[str, float]] = {
    MarketRegime.TRENDING_UP: {
        "momentum":          0.50,
        "mean_reversion":    0.10,
        "volume_anomaly":    0.25,
        "volatility_regime": 0.15,
    },
    MarketRegime.TRENDING_DOWN: {
        "momentum":          0.50,
        "mean_reversion":    0.10,
        "volume_anomaly":    0.25,
        "volatility_regime": 0.15,
    },
    MarketRegime.MEAN_REVERTING: {
        "momentum":          0.15,
        "mean_reversion":    0.55,
        "volume_anomaly":    0.20,
        "volatility_regime": 0.10,
    },
    MarketRegime.VOLATILE: {
        "momentum":          0.15,
        "mean_reversion":    0.25,
        "volume_anomaly":    0.30,
        "volatility_regime": 0.30,
    },
    MarketRegime.UNKNOWN: {
        "momentum":          0.25,
        "mean_reversion":    0.25,
        "volume_anomaly":    0.25,
        "volatility_regime": 0.25,
    },
}


@dataclass
class SignalResult:
    """Result from a single signal computation."""
    name:   str
    score:  float
    weight: float

    @property
    def weighted_score(self) -> float:
        return self.score * self.weight


@dataclass
class CompositeResult:
    """Full output from SignalComposer for a single symbol."""
    symbol:          str
    regime:          MarketRegime
    composite_score: float                        # [-1, 1]
    signal_results:  list[SignalResult] = field(default_factory=list)

    @property
    def signal_scores_dict(self) -> dict[str, float]:
        """Returns {signal_name: score} for DB storage as JSONB."""
        return {r.name: round(r.score, 6) for r in self.signal_results}

    @property
    def signal_weights_dict(self) -> dict[str, float]:
        return {r.name: round(r.weight, 4) for r in self.signal_results}


class SignalComposer:
    """
    Combines multiple BaseSignal instances into one composite score
    using regime-adaptive weight tables.

    Usage:
        composer = SignalComposer()
        result = composer.compute("AAPL", data_df, regime)
        print(result.composite_score)  # e.g. 0.42
    """

    def __init__(
        self,
        signals: Optional[list[BaseSignal]] = None,
        regime_weights: Optional[dict[MarketRegime, dict[str, float]]] = None,
    ) -> None:
        self._signals = signals or self._default_signals()
        self._weights = regime_weights or REGIME_WEIGHTS
        self._regime_detector = RegimeDetector()
        self._signal_map = {s.name: s for s in self._signals}

    @staticmethod
    def _default_signals() -> list[BaseSignal]:
        """Default signal suite."""
        return [
            MomentumSignal(rsi_period=14, fast_ma=5, slow_ma=20),
            MeanReversionSignal(period=20, num_std=2.0),
            VolumeAnomalySignal(period=20),
            VolatilityRegimeSignal(short_period=5, long_period=20),
        ]

    def compute(
        self,
        symbol: str,
        data: pd.DataFrame,
        regime: Optional[MarketRegime] = None,
    ) -> CompositeResult:
        """
        Compute composite signal score for a symbol.

        Args:
            symbol: Ticker symbol (for logging).
            data:   OHLCV DataFrame from SignalBuffer.
            regime: Optional pre-detected regime. If None, auto-detects.

        Returns:
            CompositeResult with composite score and per-signal breakdown.
        """
        if regime is None:
            regime = self._regime_detector.detect(data)

        weight_table = self._weights.get(regime, self._weights[MarketRegime.UNKNOWN])

        signal_results: list[SignalResult] = []
        composite = 0.0
        weight_sum = 0.0

        for signal in self._signals:
            score  = signal.safe_compute(data)
            weight = weight_table.get(signal.name, 0.0)

            signal_results.append(SignalResult(
                name=signal.name,
                score=score,
                weight=weight,
            ))

            composite  += score * weight
            weight_sum += weight

        # Normalize in case weights don't sum to exactly 1.0
        if weight_sum > 0:
            composite /= weight_sum

        composite = float(np.clip(composite, -1.0, 1.0))

        logger.debug(
            "Composite[%s] regime=%s score=%.4f signals=%s",
            symbol,
            regime.value,
            composite,
            {r.name: round(r.score, 3) for r in signal_results},
        )

        return CompositeResult(
            symbol=symbol,
            regime=regime,
            composite_score=composite,
            signal_results=signal_results,
        )
