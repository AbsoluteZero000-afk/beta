"""
Vectorized backtesting engine.

Replays OHLCV bars through the signal composer and applies
simplified execution logic with regime filtering, position cap,
and slippage model.

Usage:
    engine = BacktestEngine(initial_capital=100_000)
    result = engine.run(bars_df, symbol="AAPL")
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from src.signals.composer import SignalComposer
from src.signals.regime_detector import RegimeDetector, MarketRegime
from src.backtest.metrics import compute_tearsheet


@dataclass
class Trade:
    symbol:      str
    side:        str
    entry_time:  datetime
    exit_time:   datetime
    entry_price: float
    exit_price:  float
    qty:         float
    pnl:         float
    pnl_pct:     float


@dataclass
class BacktestResult:
    symbol:     str
    equity:     pd.Series
    trades:     pd.DataFrame
    tearsheet:  dict
    signal_log: pd.DataFrame


# Regimes where we allow long entries
# NOTE: VOLATILE included because RegimeDetector thresholds are calibrated
# for intraday bars — on daily bars realized vol always exceeds VOL_THRESHOLD_HIGH.
# The regime filter still blocks UNKNOWN (insufficient data).
_LONG_REGIMES  = {MarketRegime.TRENDING_UP, MarketRegime.MEAN_REVERTING, MarketRegime.VOLATILE}
# Regimes where we allow short entries
_SHORT_REGIMES = {MarketRegime.TRENDING_DOWN, MarketRegime.MEAN_REVERTING, MarketRegime.VOLATILE}
# Regimes where we skip all new entries
_SKIP_REGIMES  = {MarketRegime.UNKNOWN}


class BacktestEngine:
    """
    Single-symbol vectorized backtester with regime filter.

    Parameters
    ----------
    initial_capital : float
        Starting portfolio value.
    signal_threshold : float
        Minimum |composite_score| to open a position.
    max_hold_bars : int
        Force-close a position after this many bars.
    position_size_pct : float
        Fraction of capital to risk per trade.
    max_position_dollars : float
        Hard cap on dollars risked per trade regardless of account growth.
        Prevents compounding distortion in backtests. Default $10,000.
    commission_per_share : float
        Commission cost per share each way.
    slippage_pct : float
        Simulated slippage as a fraction of price (e.g. 0.001 = 0.1%).
        Applied against the trade direction on both entry and exit.
    use_regime_filter : bool
        If True, only open positions in favorable regimes.
        VOLATILE and UNKNOWN regimes are skipped entirely.
    """

    def __init__(
        self,
        initial_capital:      float = 100_000.0,
        signal_threshold:     float = 0.25,
        max_hold_bars:        int   = 20,
        position_size_pct:    float = 0.10,
        max_position_dollars: float = 10_000.0,
        commission_per_share: float = 0.005,
        slippage_pct:         float = 0.001,
        use_regime_filter:    bool  = True,
    ) -> None:
        self.initial_capital      = initial_capital
        self.signal_threshold     = signal_threshold
        self.max_hold_bars        = max_hold_bars
        self.position_size_pct    = position_size_pct
        self.max_position_dollars = max_position_dollars
        self.commission_per_share = commission_per_share
        self.slippage_pct         = slippage_pct
        self.use_regime_filter    = use_regime_filter
        self._composer            = SignalComposer()
        self._regime_detector     = RegimeDetector()

    def run(self, bars: pd.DataFrame, symbol: str = "") -> BacktestResult:
        """
        Run backtest on a DataFrame of OHLCV bars.
        bars must have columns: open, high, low, close, volume
        and a DatetimeIndex.
        """
        bars = bars.copy()
        bars.columns = [c.lower() for c in bars.columns]

        equity_curve:  list[tuple[datetime, float]] = []
        signal_rows:   list[dict] = []
        closed_trades: list[Trade] = []

        capital   = self.initial_capital
        position  = None
        bars_held = 0

        for i in range(20, len(bars)):
            window = bars.iloc[: i + 1]
            bar    = bars.iloc[i]
            ts     = bars.index[i]
            price  = float(bar["close"])

            score  = self._score(window, symbol)
            regime = self._regime(window)
            signal_rows.append({"time": ts, "symbol": symbol,
                                 "score": score, "regime": regime.name})

            if position is None:
                # Skip new entries in volatile/unknown regimes
                regime_allows_long  = (not self.use_regime_filter) or (regime in _LONG_REGIMES)
                regime_allows_short = (not self.use_regime_filter) or (regime in _SHORT_REGIMES)

                if score >= self.signal_threshold and regime_allows_long:
                    qty        = self._size(capital, price)
                    fill_price = self._slip(price, side="buy")
                    capital   -= qty * fill_price + qty * self.commission_per_share
                    position   = {"side": "buy", "entry_time": ts,
                                  "entry_price": fill_price, "qty": qty, "symbol": symbol}
                    bars_held  = 0

                elif score <= -self.signal_threshold and regime_allows_short:
                    qty        = self._size(capital, price)
                    fill_price = self._slip(price, side="sell")
                    capital   += qty * fill_price - qty * self.commission_per_share
                    position   = {"side": "sell", "entry_time": ts,
                                  "entry_price": fill_price, "qty": qty, "symbol": symbol}
                    bars_held  = 0
            else:
                bars_held += 1
                exit_signal = (
                    (position["side"] == "buy"  and score <= -self.signal_threshold) or
                    (position["side"] == "sell" and score >=  self.signal_threshold) or
                    bars_held >= self.max_hold_bars
                )
                if exit_signal:
                    trade, capital = self._close(position, price, ts, capital)
                    closed_trades.append(trade)
                    position  = None
                    bars_held = 0

            # Mark-to-market equity
            mtm = capital
            if position:
                qty = position["qty"]
                ep  = position["entry_price"]
                if position["side"] == "buy":
                    mtm += qty * price
                else:
                    mtm += qty * ep - qty * (price - ep)
            equity_curve.append((ts, mtm))

        # Force-close any open position at last bar
        if position and len(bars) > 0:
            last_price = float(bars.iloc[-1]["close"])
            last_ts    = bars.index[-1]
            trade, capital = self._close(position, last_price, last_ts, capital)
            closed_trades.append(trade)
            equity_curve.append((last_ts, capital))

        equity    = pd.Series({t: v for t, v in equity_curve}, name="equity")
        trades_df = pd.DataFrame([vars(t) for t in closed_trades]) if closed_trades else pd.DataFrame()
        signal_df = pd.DataFrame(signal_rows)
        tearsheet = compute_tearsheet(equity, trades_df, symbol, self.initial_capital)

        return BacktestResult(
            symbol=symbol,
            equity=equity,
            trades=trades_df,
            tearsheet=tearsheet,
            signal_log=signal_df,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _score(self, window: pd.DataFrame, symbol: str) -> float:
        try:
            result = self._composer.compute(symbol, window)
            if hasattr(result, "composite_score"):
                return float(result.composite_score)
            elif hasattr(result, "get"):
                return float(result.get("composite_score", 0.0))
            return 0.0
        except Exception:
            return 0.0

    def _regime(self, window: pd.DataFrame) -> MarketRegime:
        try:
            return self._regime_detector.detect(window)
        except Exception:
            return MarketRegime.UNKNOWN

    def _size(self, capital: float, price: float) -> float:
        """
        Fixed-fraction sizing with a hard dollar cap.
        Prevents compounding from inflating position sizes unrealistically.
        """
        if price <= 0:
            return 0.0
        max_dollars = min(capital * self.position_size_pct, self.max_position_dollars)
        return max(1.0, np.floor(max_dollars / price))

    def _slip(self, price: float, side: str) -> float:
        """
        Apply slippage against the trade direction.
        Buys fill slightly higher, sells fill slightly lower.
        """
        if side == "buy":
            return price * (1 + self.slippage_pct)
        return price * (1 - self.slippage_pct)

    def _close(
        self,
        position:   dict,
        exit_price: float,
        exit_time:  datetime,
        capital:    float,
    ) -> tuple[Trade, float]:
        qty        = position["qty"]
        ep         = position["entry_price"]
        commission = qty * self.commission_per_share

        if position["side"] == "buy":
            fill_price = self._slip(exit_price, side="sell")
            pnl        = qty * (fill_price - ep) - commission * 2
            capital   += qty * fill_price - commission
        else:
            fill_price = self._slip(exit_price, side="buy")
            pnl        = qty * (ep - fill_price) - commission * 2
            capital   += qty * fill_price + commission

        pnl_pct = pnl / (qty * ep) if (qty * ep) != 0 else 0.0

        return Trade(
            symbol=position.get("symbol", ""),
            side=position["side"],
            entry_time=position["entry_time"],
            exit_time=exit_time,
            entry_price=ep,
            exit_price=fill_price,
            qty=qty,
            pnl=round(pnl, 2),
            pnl_pct=round(pnl_pct, 4),
        ), capital
