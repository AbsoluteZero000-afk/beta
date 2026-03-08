"""
QuantEdge — BaseSignal Abstract Class.

All signals must implement compute(closes, volumes) -> float
returning a normalized score in [-1.0, 1.0]:
  -1.0 = maximum bearish
   0.0 = neutral
  +1.0 = maximum bullish

CRITICAL numba rule: @jit(nopython=True) functions cannot accept
pandas DataFrames. Always pass numpy arrays extracted BEFORE calling
the JIT function. Extract in the Python wrapper, compute in numba.

Official numba docs:
  https://numba.readthedocs.io/en/stable/user/jit.html
  https://numba.readthedocs.io/en/stable/reference/numpysupported.html
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class BaseSignal(ABC):
    """
    Abstract base class for all QuantEdge signals.

    Subclasses implement `compute` which receives a pandas DataFrame
    (for convenience) but must extract numpy arrays before passing
    them to any @numba.jit decorated function.
    """

    name: str = "base"
    description: str = ""

    @abstractmethod
    def compute(self, data: pd.DataFrame) -> float:
        """
        Compute a normalized signal score from OHLCV data.

        Args:
            data: pandas DataFrame with columns:
                  open, high, low, close, volume
                  Index is DatetimeIndex (time).

        Returns:
            float in [-1.0, 1.0].
            Returns 0.0 (neutral) if insufficient data.
        """
        ...

    def safe_compute(self, data: pd.DataFrame) -> float:
        """
        Compute with error handling. Returns 0.0 on any failure.
        Used by SignalComposer to ensure pipeline never crashes.
        """
        try:
            if data.empty or len(data) < self.min_bars:
                return 0.0
            score = self.compute(data)
            # Clip to [-1, 1] as safety net
            return float(np.clip(score, -1.0, 1.0))
        except Exception as exc:
            import logging
            logging.getLogger(f"quantedge.signal.{self.name}").warning(
                "Signal compute error: %s", exc
            )
            return 0.0

    @property
    def min_bars(self) -> int:
        """Minimum bars required before this signal is valid."""
        return 20
