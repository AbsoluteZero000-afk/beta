"""
Tests for Phase 6 — Backtesting Engine.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest.metrics import (
    sharpe_ratio, sortino_ratio, max_drawdown,
    calmar_ratio, win_rate, profit_factor, compute_tearsheet,
)
from src.backtest.engine import BacktestEngine, BacktestResult
from src.backtest.walk_forward import WalkForwardValidator, WalkForwardResult


# ---- Fixtures ----

def _make_bars(n: int = 300, trend: float = 0.001, seed: int = 42) -> pd.DataFrame:
    rng    = np.random.default_rng(seed)
    dates  = pd.date_range("2023-01-01", periods=n, freq="D", tz="UTC")
    close  = 100.0 * np.cumprod(1 + rng.normal(trend, 0.015, n))
    open_  = close * (1 + rng.normal(0, 0.003, n))
    high   = np.maximum(close, open_) * (1 + rng.uniform(0, 0.005, n))
    low    = np.minimum(close, open_) * (1 - rng.uniform(0, 0.005, n))
    volume = rng.integers(500_000, 5_000_000, n).astype(float)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


def _make_downtrend_bars(n: int = 300) -> pd.DataFrame:
    return _make_bars(n=n, trend=-0.001, seed=99)


def _make_equity(n: int = 200) -> pd.Series:
    rng   = np.random.default_rng(0)
    dates = pd.date_range("2023-01-01", periods=n, freq="D", tz="UTC")
    vals  = 100_000 * np.cumprod(1 + rng.normal(0.0005, 0.01, n))
    return pd.Series(vals, index=dates, name="equity")


def _make_trades(n: int = 30, win_frac: float = 0.6, seed: int = 7) -> pd.DataFrame:
    rng    = np.random.default_rng(seed)
    wins   = int(n * win_frac)
    losses = n - wins
    pnl    = np.concatenate([rng.uniform(10, 500, wins), rng.uniform(-400, -10, losses)])
    rng.shuffle(pnl)
    return pd.DataFrame({"pnl": pnl})


# ===== metrics.py =====

class TestSharpeRatio:
    def test_positive_returns_positive_sharpe(self):
        assert sharpe_ratio(pd.Series([0.001] * 252)) > 0

    def test_negative_returns_negative_sharpe(self):
        assert sharpe_ratio(pd.Series([-0.001] * 252)) < 0

    def test_zero_std_returns_zero(self):
        assert sharpe_ratio(pd.Series([0.0] * 100)) == 0.0

    def test_scale_by_sqrt_252(self):
        r   = pd.Series([0.001] * 252)
        raw = r.mean() / r.std()
        assert abs(sharpe_ratio(r) - raw * (252 ** 0.5)) < 1e-6


class TestSortinoRatio:
    def test_positive_on_uptrend(self):
        rng = np.random.default_rng(1)
        # 200 positive returns, 52 varied negative — mean > 0, downside std > 0
        r = pd.Series(
            [0.001] * 200 + list(rng.uniform(-0.002, -0.0001, 52))
        )
        assert sortino_ratio(r) > 0

    def test_zero_downside_returns_zero(self):
        assert sortino_ratio(pd.Series([0.001] * 100)) == 0.0


class TestMaxDrawdown:
    def test_monotone_increase_zero_drawdown(self):
        assert max_drawdown(pd.Series([100.0, 101.0, 102.0, 103.0])) == 0.0

    def test_known_drawdown(self):
        eq       = pd.Series([100.0, 120.0, 80.0, 90.0])
        expected = (80 - 120) / 120
        assert abs(max_drawdown(eq) - expected) < 1e-6

    def test_always_nonpositive(self):
        assert max_drawdown(_make_equity()) <= 0.0


class TestWinRate:
    def test_all_wins(self):
        assert win_rate(pd.Series([100.0, 200.0, 50.0])) == 1.0

    def test_all_losses(self):
        assert win_rate(pd.Series([-100.0, -50.0])) == 0.0

    def test_mixed(self):
        assert win_rate(pd.Series([100.0, -50.0, 200.0, -30.0])) == 0.5

    def test_empty_returns_zero(self):
        assert win_rate(pd.Series(dtype=float)) == 0.0


class TestProfitFactor:
    def test_only_wins_returns_inf(self):
        assert profit_factor(pd.Series([100.0, 200.0])) == float("inf")

    def test_balanced(self):
        assert profit_factor(pd.Series([100.0, -100.0])) == pytest.approx(1.0)

    def test_gt_one_when_profitable(self):
        assert profit_factor(_make_trades(win_frac=0.65)["pnl"]) > 1.0


class TestComputeTearsheet:
    def test_all_keys_present(self):
        ts = compute_tearsheet(_make_equity(), _make_trades(), symbol="TEST")
        for k in ["symbol", "initial_capital", "final_equity", "total_return_pct",
                  "ann_return_pct", "sharpe", "sortino", "max_drawdown_pct",
                  "calmar", "total_trades", "win_rate_pct", "profit_factor",
                  "avg_trade_pnl", "best_trade", "worst_trade"]:
            assert k in ts

    def test_symbol_passthrough(self):
        assert compute_tearsheet(_make_equity(), _make_trades(), symbol="AAPL")["symbol"] == "AAPL"

    def test_trade_count(self):
        assert compute_tearsheet(_make_equity(), _make_trades(n=25))["total_trades"] == 25

    def test_empty_trades_no_crash(self):
        ts = compute_tearsheet(_make_equity(), pd.DataFrame())
        assert ts["total_trades"] == 0
        assert ts["win_rate_pct"] == 0.0


# ===== engine.py =====

class TestBacktestEngine:
    def test_returns_backtest_result(self):
        assert isinstance(BacktestEngine().run(_make_bars(), "TEST"), BacktestResult)

    def test_equity_nonempty(self):
        assert len(BacktestEngine().run(_make_bars(300), "TEST").equity) > 0

    def test_equity_starts_near_capital(self):
        r = BacktestEngine(initial_capital=100_000).run(_make_bars(300), "TEST")
        assert abs(r.equity.iloc[0] - 100_000) / 100_000 < 0.15

    def test_trades_columns(self):
        r = BacktestEngine(signal_threshold=0.1).run(_make_bars(300), "TEST")
        if len(r.trades) > 0:
            for col in ["symbol", "side", "entry_price", "exit_price", "qty", "pnl", "pnl_pct"]:
                assert col in r.trades.columns

    def test_signal_log_has_score(self):
        assert "score" in BacktestEngine().run(_make_bars(200), "TEST").signal_log.columns

    def test_tearsheet_symbol(self):
        assert BacktestEngine().run(_make_bars(300), "TSYM").tearsheet["symbol"] == "TSYM"

    def test_high_threshold_fewer_trades(self):
        bars  = _make_bars(300)
        r_low = BacktestEngine(signal_threshold=0.05).run(bars, "X")
        r_hi  = BacktestEngine(signal_threshold=0.80).run(bars, "X")
        assert len(r_low.trades) >= len(r_hi.trades)

    def test_no_crash_on_downtrend(self):
        assert BacktestEngine().run(_make_downtrend_bars(), "DOWN") is not None

    def test_uppercase_columns_normalized(self):
        bars = _make_bars(200)
        bars.columns = [c.upper() for c in bars.columns]
        assert len(BacktestEngine().run(bars, "TEST").equity) > 0

    def test_max_hold_bars_limits_duration(self):
        r = BacktestEngine(signal_threshold=0.05, max_hold_bars=5).run(_make_bars(300), "X")
        if len(r.trades) > 0:
            durations = (r.trades["exit_time"] - r.trades["entry_time"]).dt.days
            assert durations.max() <= 10


# ===== walk_forward.py =====

class TestWalkForwardValidator:
    def test_returns_walk_forward_result(self):
        assert isinstance(WalkForwardValidator(n_splits=3).run(_make_bars(500), "WF"), WalkForwardResult)

    def test_window_count_lte_n_splits(self):
        assert len(WalkForwardValidator(n_splits=4).run(_make_bars(500), "WF").windows) <= 4

    def test_combined_equity_is_series(self):
        assert isinstance(WalkForwardValidator(n_splits=3).run(_make_bars(500), "WF").combined_equity, pd.Series)

    def test_tearsheet_has_keys(self):
        ts = WalkForwardValidator(n_splits=3).run(_make_bars(500), "WF").combined_tearsheet
        assert "sharpe" in ts and "total_trades" in ts

    def test_equity_monotone_index(self):
        eq = WalkForwardValidator(n_splits=4).run(_make_bars(600), "WF").combined_equity
        if len(eq) > 1:
            assert eq.index.is_monotonic_increasing

    def test_engine_kwargs_passthrough(self):
        wf = WalkForwardValidator(
            n_splits=3,
            engine_kwargs={"signal_threshold": 0.80, "initial_capital": 50_000},
        )
        assert wf.run(_make_bars(500), "WF") is not None
