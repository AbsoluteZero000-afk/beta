"""
Tearsheet metric calculations.
All functions operate on a pandas Series of daily returns.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def _to_daily(equity: pd.Series) -> pd.Series:
    return equity.pct_change().dropna()


def sharpe_ratio(returns: pd.Series, risk_free: float = 0.0) -> float:
    excess = returns - risk_free / TRADING_DAYS
    if excess.std() == 0:
        return 0.0
    return float((excess.mean() / excess.std()) * np.sqrt(TRADING_DAYS))


def sortino_ratio(returns: pd.Series, risk_free: float = 0.0) -> float:
    excess   = returns - risk_free / TRADING_DAYS
    downside = excess[excess < 0]
    if len(downside) == 0 or downside.std() == 0 or np.isnan(downside.std()):
        return 0.0
    return float((excess.mean() / downside.std()) * np.sqrt(TRADING_DAYS))


def max_drawdown(equity: pd.Series) -> float:
    roll_max = equity.cummax()
    drawdown = (equity - roll_max) / roll_max
    return float(drawdown.min())


def calmar_ratio(equity: pd.Series, returns: pd.Series) -> float:
    ann_return = float((1 + returns.mean()) ** TRADING_DAYS - 1)
    mdd = abs(max_drawdown(equity))
    if mdd == 0:
        return 0.0
    return ann_return / mdd


def win_rate(pnl_series: pd.Series) -> float:
    if len(pnl_series) == 0:
        return 0.0
    return float((pnl_series > 0).sum() / len(pnl_series))


def profit_factor(pnl_series: pd.Series) -> float:
    gross_profit = pnl_series[pnl_series > 0].sum()
    gross_loss   = abs(pnl_series[pnl_series < 0].sum())
    if gross_loss == 0:
        return float("inf")
    return float(gross_profit / gross_loss)


def compute_tearsheet(
    equity: pd.Series,
    trades: pd.DataFrame,
    symbol: str = "",
    initial_capital: float = 100_000.0,
) -> dict:
    returns = _to_daily(equity)
    pnl     = trades["pnl"] if "pnl" in trades.columns and len(trades) > 0 else pd.Series(dtype=float)

    total_return = float((equity.iloc[-1] - equity.iloc[0]) / equity.iloc[0]) if len(equity) > 1 else 0.0
    ann_return   = float((1 + returns.mean()) ** TRADING_DAYS - 1) if len(returns) > 0 else 0.0

    return {
        "symbol":           symbol,
        "initial_capital":  initial_capital,
        "final_equity":     float(equity.iloc[-1]) if len(equity) > 0 else initial_capital,
        "total_return_pct": round(total_return * 100, 2),
        "ann_return_pct":   round(ann_return * 100, 2),
        "sharpe":           round(sharpe_ratio(returns), 3),
        "sortino":          round(sortino_ratio(returns), 3),
        "max_drawdown_pct": round(max_drawdown(equity) * 100, 2),
        "calmar":           round(calmar_ratio(equity, returns), 3),
        "total_trades":     len(trades),
        "win_rate_pct":     round(win_rate(pnl) * 100, 2),
        "profit_factor":    round(profit_factor(pnl), 3),
        "avg_trade_pnl":    round(float(pnl.mean()), 2) if len(pnl) > 0 else 0.0,
        "best_trade":       round(float(pnl.max()), 2) if len(pnl) > 0 else 0.0,
        "worst_trade":      round(float(pnl.min()), 2) if len(pnl) > 0 else 0.0,
    }
