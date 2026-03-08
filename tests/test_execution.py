"""
Tests for Phase 4 — Decision & Execution Layer.
Run with: pytest tests/test_execution.py -v

All tests are pure unit tests — no Alpaca connection required.
Alpaca TradingClient is mocked throughout.
"""

from __future__ import annotations

import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock, AsyncMock, patch

from src.execution.risk_gate import RiskGate, GateVerdict, RiskLimits
from src.execution.position_sizer import PositionSizer
from src.execution.portfolio_tracker import PortfolioSnapshot, PortfolioTracker
from src.execution.order_router import OrderRouter
from src.execution.execution_monitor import ExecutionMonitor, TrackedOrder
from src.signals.composer import CompositeResult, SignalResult
from src.events.types import MarketRegime


# ---- Fixtures ----

def make_signal(score: float, regime: MarketRegime = MarketRegime.TRENDING_UP) -> CompositeResult:
    return CompositeResult(
        symbol="AAPL",
        regime=regime,
        composite_score=score,
        signal_results=[
            SignalResult("momentum", score, 0.5),
            SignalResult("mean_reversion", score * 0.5, 0.5),
        ],
    )


def make_snapshot(
    equity=50000.0,
    exposure=5000.0,
    positions=2,
    drawdown=0.0,
    daily_pnl=None,
) -> PortfolioSnapshot:
    snap = PortfolioSnapshot(
        equity=equity,
        cash=equity - exposure,
        buying_power=equity - exposure,
        total_exposure=exposure,
        open_position_count=positions,
        intraday_drawdown_pct=drawdown,
        daily_pnl_by_symbol=daily_pnl or {},
        intraday_high_equity=equity,
    )
    return snap


def make_tracker(snap: PortfolioSnapshot) -> PortfolioTracker:
    tracker = MagicMock(spec=PortfolioTracker)
    tracker.snapshot = snap
    return tracker


# ---- RiskGate tests ----

class TestRiskGate:
    def test_pass_normal_conditions(self):
        limits = RiskLimits()
        limits.enforce_rth = False
        snap    = make_snapshot()
        tracker = make_tracker(snap)
        gate    = RiskGate(tracker, limits)
        signal  = make_signal(0.60)
        result  = gate.check("AAPL", "buy", signal, price=150.0, qty=10)
        assert result.verdict == GateVerdict.PASS

    def test_reject_weak_signal(self):
        limits = RiskLimits()
        limits.enforce_rth = False
        snap    = make_snapshot()
        tracker = make_tracker(snap)
        gate    = RiskGate(tracker, limits)
        signal  = make_signal(0.05)   # Below min_signal_strength=0.20
        result  = gate.check("AAPL", "buy", signal, price=150.0, qty=10)
        assert result.verdict == GateVerdict.REJECT
        assert "signal_too_weak" in result.reason

    def test_reject_drawdown_exceeded(self):
        limits = RiskLimits()
        limits.enforce_rth = False
        snap    = make_snapshot(drawdown=0.06)   # 6% > 5% limit
        tracker = make_tracker(snap)
        gate    = RiskGate(tracker, limits)
        signal  = make_signal(0.80)
        result  = gate.check("AAPL", "buy", signal, price=150.0, qty=10)
        assert result.verdict == GateVerdict.REJECT
        assert "drawdown" in result.reason

    def test_reject_max_positions(self):
        limits = RiskLimits()
        limits.enforce_rth = False
        limits.max_open_positions = 2
        snap    = make_snapshot(positions=2)
        tracker = make_tracker(snap)
        gate    = RiskGate(tracker, limits)
        signal  = make_signal(0.70)
        result  = gate.check("AAPL", "buy", signal, price=150.0, qty=10)
        assert result.verdict == GateVerdict.REJECT
        assert "max_positions" in result.reason

    def test_scale_on_position_too_large(self):
        limits = RiskLimits()
        limits.enforce_rth = False
        limits.max_position_pct = 0.05  # 5% max
        snap    = make_snapshot(equity=10000.0)
        tracker = make_tracker(snap)
        gate    = RiskGate(tracker, limits)
        signal  = make_signal(0.80)
        # Trying to buy $2000 worth on $10k equity = 20% > 5%
        result  = gate.check("AAPL", "buy", signal, price=100.0, qty=20)
        assert result.verdict == GateVerdict.SCALE
        # Scaled qty should be 10k * 0.05 / 100 = 5 shares
        assert result.suggested_qty_override == pytest.approx(5.0, abs=0.01)

    def test_reject_daily_loss(self):
        limits = RiskLimits()
        limits.enforce_rth = False
        limits.max_daily_loss_pct = 0.01
        snap = make_snapshot(
            equity=10000.0,
            daily_pnl={"AAPL": -200.0},  # 2% loss > 1% limit
        )
        tracker = make_tracker(snap)
        gate = RiskGate(tracker, limits)
        signal = make_signal(0.75)
        # qty=1 → $150 notional = 1.5% of $10k equity, well under 10% position limit
        result = gate.check("AAPL", "buy", signal, price=150.0, qty=1)
        assert result.verdict == GateVerdict.REJECT
        assert "daily_loss" in result.reason


# ---- PositionSizer tests ----

class TestPositionSizer:
    def test_fallback_sizing_no_history(self):
        sizer  = PositionSizer(fallback_pct=0.02, signal_scale=False)
        signal = make_signal(0.80)
        result = sizer.size("AAPL", signal, price=150.0, equity=50000.0)
        # 2% of 50k = $1000 / $150 = 6.66 → 6.66 shares
        assert result.qty == pytest.approx(6.66, abs=0.1)
        assert result.sizing_method == "fixed_pct"

    def test_signal_scale_reduces_qty(self):
        sizer_unscaled = PositionSizer(fallback_pct=0.05, signal_scale=False)
        sizer_scaled   = PositionSizer(fallback_pct=0.05, signal_scale=True)
        signal = make_signal(0.40)   # 40% strength
        r1 = sizer_unscaled.size("AAPL", signal, price=100.0, equity=10000.0)
        r2 = sizer_scaled.size("AAPL", signal, price=100.0, equity=10000.0)
        assert r2.qty < r1.qty

    def test_zero_result_on_zero_price(self):
        sizer  = PositionSizer()
        signal = make_signal(0.80)
        result = sizer.size("AAPL", signal, price=0.0, equity=50000.0)
        assert result.qty == 0.0
        assert result.sizing_method == "zero"

    def test_kelly_sizing_with_history(self):
        sizer = PositionSizer(kelly_fraction=0.5, signal_scale=False)
        # Simulate 15 trades: 10 wins at 3% avg, 5 losses at 1% avg
        for i in range(10):
            sizer.update_stats("TSLA", win=True, pnl_pct=0.03)
        for i in range(5):
            sizer.update_stats("TSLA", win=False, pnl_pct=0.01)
        signal = make_signal(0.80)
        result = sizer.size("TSLA", signal, price=200.0, equity=50000.0)
        assert result.sizing_method == "kelly"
        assert result.qty > 0
        assert result.kelly_f > 0

    def test_kelly_f_computation(self):
        # f* = (b*p - q) / b  where b=3, p=0.67, q=0.33
        # f* = (3*0.67 - 0.33) / 3 = (2.01 - 0.33) / 3 = 0.56
        f = PositionSizer._kelly_f(win_rate=0.67, win_loss_ratio=3.0)
        assert 0.50 < f < 0.65

    def test_max_kelly_cap_enforced(self):
        """Kelly output should never exceed MAX_KELLY_FRACTION=0.25."""
        sizer = PositionSizer(kelly_fraction=1.0, signal_scale=False)  # Full Kelly
        for i in range(20):
            sizer.update_stats("NVDA", win=True, pnl_pct=0.10)  # Unrealistic high win rate
        signal = make_signal(1.0)
        result = sizer.size("NVDA", signal, price=500.0, equity=100000.0)
        max_notional = 100000.0 * 0.25
        assert result.notional <= max_notional + 1.0   # +1 for float rounding


# ---- OrderRouter tests ----

class TestOrderRouter:
    @pytest.mark.asyncio
    async def test_dry_run_returns_success(self):
        client = MagicMock()
        router = OrderRouter(client, dry_run=True)
        from src.execution.position_sizer import SizeResult
        size = SizeResult("AAPL", qty=10.0, notional=1500.0,
                          kelly_f=0.02, applied_f=0.01,
                          signal_score=0.6, sizing_method="fixed_pct")
        result = await router.submit_bracket(size, "buy", price=150.0)
        assert result.success
        assert "dry_" in result.alpaca_order_id

    @pytest.mark.asyncio
    async def test_bracket_tp_sl_prices_buy(self):
        """TP should be above entry, SL below for buy orders."""
        submitted = {}
        def fake_submit(order_data):
            submitted["order"] = order_data
            mock = MagicMock()
            mock.id = "alpaca-123"
            return mock

        client = MagicMock()
        client.submit_order = fake_submit
        router = OrderRouter(client, take_profit_pct=0.02, stop_loss_pct=0.01, dry_run=False)

        from src.execution.position_sizer import SizeResult
        size = SizeResult("SPY", qty=5.0, notional=2500.0,
                          kelly_f=0.05, applied_f=0.025,
                          signal_score=0.7, sizing_method="fixed_pct")
        result = await router.submit_bracket(size, "buy", price=500.0)
        assert result.success
        order = submitted["order"]
        assert float(order.take_profit.limit_price) == pytest.approx(510.0, abs=0.01)
        assert float(order.stop_loss.stop_price)    == pytest.approx(495.0, abs=0.01)

    @pytest.mark.asyncio
    async def test_order_failure_captured(self):
        client = MagicMock()
        client.submit_order.side_effect = Exception("insufficient funds")
        router = OrderRouter(client, dry_run=False)

        from src.execution.position_sizer import SizeResult
        size = SizeResult("AAPL", qty=1.0, notional=150.0,
                          kelly_f=0.01, applied_f=0.005,
                          signal_score=0.5, sizing_method="fixed_pct")
        result = await router.submit_bracket(size, "buy", price=150.0)
        assert not result.success
        assert "insufficient funds" in result.error


# ---- ExecutionMonitor tests ----

class TestExecutionMonitor:
    @pytest.mark.asyncio
    async def test_register_and_fill(self):
        sizer   = PositionSizer()
        monitor = ExecutionMonitor(sizer)

        from src.execution.order_router import OrderResult
        order_result = OrderResult(
            client_order_id="qe_AAPL_abc123",
            symbol="AAPL",
            side="buy",
            qty=10.0,
            order_class="bracket",
            submitted_at=datetime.now(timezone.utc),
            alpaca_order_id="alp-xyz",
        )
        monitor.register(order_result)
        assert len(monitor.open_orders) == 1

        with patch.object(monitor, "_persist_trade", new_callable=AsyncMock):
            await monitor.record_fill("qe_AAPL_abc123", fill_price=150.0)
        assert monitor.open_orders[0].status == "filled"

    @pytest.mark.asyncio
    async def test_record_close_updates_sizer(self):
        sizer   = PositionSizer()
        monitor = ExecutionMonitor(sizer)

        from src.execution.order_router import OrderResult
        order_result = OrderResult(
            client_order_id="qe_TSLA_def456",
            symbol="TSLA",
            side="buy",
            qty=5.0,
            order_class="bracket",
            submitted_at=datetime.now(timezone.utc),
            alpaca_order_id="alp-abc",
        )
        monitor.register(order_result)

        with patch.object(monitor, "_persist_trade", new_callable=AsyncMock):
            await monitor.record_fill("qe_TSLA_def456", fill_price=200.0)

        with patch.object(monitor, "_persist_close", new_callable=AsyncMock):
            await monitor.record_close("TSLA", close_price=210.0, entry_price=200.0)

        stats = monitor.stats
        assert stats["closed"] == 1
        assert stats["win_rate"] == 1.0
        assert "TSLA" in sizer._stats
        assert sizer._stats["TSLA"]["n_wins"] == 1

    def test_monitor_stats_initial(self):
        monitor = ExecutionMonitor(PositionSizer())
        assert monitor.stats["total_submitted"] == 0
        assert monitor.stats["win_rate"] == 0.0
