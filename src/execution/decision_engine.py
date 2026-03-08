"""
QuantEdge — DecisionEngine.

The bridge between Phase 3 (Signal Engine) and Phase 4 (Execution).
Consumes TradingEvent objects from the PriorityEventQueue, runs
the full decision pipeline, and routes qualifying orders.

Pipeline per event:
  TradingEvent
    → SignalComposer.compute()        (Phase 3)
    → RiskGate.check()                (Phase 4)
    → PositionSizer.size()            (Phase 4)
    → OrderRouter.submit_bracket()    (Phase 4)
    → ExecutionMonitor.register()     (Phase 4)
    → PortfolioTracker.refresh()      (Phase 4)
"""

from __future__ import annotations

import logging
from typing import Optional

from src.events.types import TradingEvent, EventType
from src.execution.risk_gate import RiskGate, GateVerdict
from src.execution.position_sizer import PositionSizer
from src.execution.order_router import OrderRouter
from src.execution.execution_monitor import ExecutionMonitor
from src.execution.portfolio_tracker import PortfolioTracker
from src.signals.composer import SignalComposer

logger = logging.getLogger(__name__)


class DecisionEngine:
    """
    Orchestrates the full decision pipeline for each trading event.

    Designed to be registered as the EventConsumer handler:
        consumer.set_handler(engine.handle_event)
    """

    def __init__(
        self,
        composer:  SignalComposer,
        risk_gate: RiskGate,
        sizer:     PositionSizer,
        router:    OrderRouter,
        monitor:   ExecutionMonitor,
        tracker:   PortfolioTracker,
    ) -> None:
        self._composer  = composer
        self._gate      = risk_gate
        self._sizer     = sizer
        self._router    = router
        self._monitor   = monitor
        self._tracker   = tracker

    async def handle_event(self, event: TradingEvent) -> None:
        """
        Main event handler. Registered with EventConsumer.
        Runs the full pipeline for BREAKOUT, VOLUME_SPIKE,
        and SCHEDULED_BAR events that have sufficient context.
        """
        # Portfolio breach: close all and halt
        if event.event_type == EventType.PORTFOLIO_BREACH:
            logger.critical(
                "PORTFOLIO BREACH — %s: %s",
                event.symbol,
                event.raw_data.get("reason") if event.raw_data else "unknown",
            )
            return

        # Skip events without context (insufficient data)
        if event.context is None or not event.context.has_sufficient_data:
            return

        ctx = event.context

        # 1. Compute composite signal
        signal = self._composer.compute(
            symbol=event.symbol,
            data=ctx.ohlcv,
            regime=ctx.regime,
        )

        # 2. Determine trade direction
        side = self._determine_side(signal.composite_score)
        if side is None:
            return   # Neutral — no trade

        # 3. Get current price
        price = ctx.latest_close
        if price is None or price <= 0:
            return

        # 4. Risk gate check
        snap  = self._tracker.snapshot
        qty_estimate = (snap.equity * 0.05) / price   # Rough estimate for gate
        gate_result  = self._gate.check(
            symbol=event.symbol,
            side=side,
            signal=signal,
            price=price,
            qty=qty_estimate,
        )

        if gate_result.verdict == GateVerdict.REJECT:
            logger.debug(
                "Gate REJECT [%s] %s: %s",
                event.symbol, side, gate_result.reason,
            )
            return

        # 5. Size the position
        size = self._sizer.size(
            symbol=event.symbol,
            signal=signal,
            price=price,
            equity=snap.equity,
        )

        # Apply gate scaling if needed
        if gate_result.verdict == GateVerdict.SCALE and gate_result.suggested_qty_override:
            size.qty = min(size.qty, gate_result.suggested_qty_override)

        if size.qty < 0.01:
            logger.debug("Zero qty after sizing — skipping %s", event.symbol)
            return

        # 6. Submit order
        order_result = await self._router.submit_bracket(size, side, price)

        # 7. Register with monitor
        if order_result.success:
            self._monitor.register(order_result)
            # Refresh portfolio after order
            await self._tracker.refresh()

            logger.info(
                "Decision executed: %s %s qty=%.2f @ $%.2f score=%.3f regime=%s",
                side.upper(), event.symbol, size.qty, price,
                signal.composite_score, ctx.regime.value,
            )
        else:
            logger.warning(
                "Order failed: %s %s — %s",
                side.upper(), event.symbol, order_result.error,
            )

    @staticmethod
    def _determine_side(score: float) -> Optional[str]:
        """
        Convert composite score to trade side.
        Only trade when score exceeds neutral zone (handled by RiskGate).
        """
        if score > 0:
            return "buy"
        elif score < 0:
            return "sell"
        return None
