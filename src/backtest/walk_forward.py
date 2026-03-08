"""
Walk-forward validation.

Splits a bars DataFrame into sequential in-sample / out-of-sample
windows and runs a BacktestEngine on each OOS window.

    |--- IS ---|-- OOS --|--- IS ---|-- OOS --|  ...
    window 0              window 1
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.backtest.engine import BacktestEngine, BacktestResult
from src.backtest.metrics import compute_tearsheet


@dataclass
class WalkForwardResult:
    windows:             list[dict]
    combined_equity:     pd.Series
    combined_trades:     pd.DataFrame
    combined_tearsheet:  dict


class WalkForwardValidator:
    """
    Parameters
    ----------
    n_splits : int
        Number of walk-forward windows.
    train_pct : float
        Fraction of each window used for in-sample (0 < train_pct < 1).
    engine_kwargs : dict
        Passed directly to BacktestEngine constructor.
    """

    def __init__(
        self,
        n_splits:      int        = 5,
        train_pct:     float      = 0.7,
        engine_kwargs: dict | None = None,
    ) -> None:
        self.n_splits      = n_splits
        self.train_pct     = train_pct
        self.engine_kwargs = engine_kwargs or {}

    def run(self, bars: pd.DataFrame, symbol: str = "") -> WalkForwardResult:
        window_size = len(bars) // self.n_splits
        windows     = []
        oos_equities: list[pd.Series]    = []
        oos_trades:   list[pd.DataFrame] = []

        for i in range(self.n_splits):
            start = i * window_size
            end   = start + window_size if i < self.n_splits - 1 else len(bars)

            split     = int((end - start) * self.train_pct)
            oos_start = start + split
            oos_bars  = bars.iloc[oos_start:end]

            if len(oos_bars) < 30:
                continue

            engine = BacktestEngine(**self.engine_kwargs)
            result = engine.run(oos_bars, symbol=symbol)

            windows.append({
                "window":    i + 1,
                "oos_start": bars.index[oos_start],
                "oos_end":   bars.index[end - 1],
                "oos_bars":  len(oos_bars),
                **result.tearsheet,
            })

            oos_equities.append(result.equity)
            if len(result.trades) > 0:
                oos_trades.append(result.trades)

        combined_equity    = pd.concat(oos_equities).sort_index() if oos_equities else pd.Series(dtype=float)
        combined_trades    = pd.concat(oos_trades).reset_index(drop=True) if oos_trades else pd.DataFrame()
        combined_tearsheet = compute_tearsheet(
            combined_equity, combined_trades, symbol,
            self.engine_kwargs.get("initial_capital", 100_000.0),
        )

        return WalkForwardResult(
            windows=windows,
            combined_equity=combined_equity,
            combined_trades=combined_trades,
            combined_tearsheet=combined_tearsheet,
        )
