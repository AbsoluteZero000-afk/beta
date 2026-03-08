"""
QuantEdge — OrderRouter.

Translates SizeResult + signal direction into Alpaca bracket orders
and submits them via TradingClient.

Bracket order structure (per Alpaca docs):
  - Entry: market order (filled immediately at best price)
  - Take-profit leg: limit order at +TP_PCT above entry
  - Stop-loss leg: stop order at -SL_PCT below entry

alpaca-py imports used:
  from alpaca.trading.client import TradingClient
  from alpaca.trading.requests import (
      MarketOrderRequest, TakeProfitRequest, StopLossRequest
  )
  from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass

Official docs:
  https://docs.alpaca.markets/docs/working-with-orders
  https://alpaca.markets/sdks/python/trading.html
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderClass, OrderSide, TimeInForce
from alpaca.trading.requests import (
    MarketOrderRequest,
    StopLossRequest,
    TakeProfitRequest,
)

from src.execution.position_sizer import SizeResult

logger = logging.getLogger(__name__)

# Default bracket parameters (configurable per symbol in Phase 6)
DEFAULT_TAKE_PROFIT_PCT = 0.02   # 2% profit target
DEFAULT_STOP_LOSS_PCT   = 0.01   # 1% stop loss  →  2:1 R:R ratio


@dataclass
class OrderResult:
    client_order_id: str
    symbol:          str
    side:            str
    qty:             float
    order_class:     str
    submitted_at:    datetime
    alpaca_order_id: Optional[str] = None
    error:           Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None


class OrderRouter:
    """
    Routes sized orders to Alpaca with bracket SL/TP legs.

    All TradingClient calls are synchronous (SDK limitation) —
    wrapped in run_in_executor to avoid blocking uvloop.

    Usage:
        router = OrderRouter(trading_client)
        result = await router.submit_bracket(size_result, side, price)
    """

    def __init__(
        self,
        client: TradingClient,
        take_profit_pct: float = DEFAULT_TAKE_PROFIT_PCT,
        stop_loss_pct:   float = DEFAULT_STOP_LOSS_PCT,
        dry_run:         bool  = False,
    ) -> None:
        self._client         = client
        self._take_profit_pct = take_profit_pct
        self._stop_loss_pct   = stop_loss_pct
        self._dry_run         = dry_run

    async def submit_bracket(
        self,
        size: SizeResult,
        side: str,         # "buy" or "sell"
        price: float,      # Current market price for SL/TP calculation
    ) -> OrderResult:
        """
        Submit a bracket order (entry + SL + TP in one request).

        Dry-run mode logs the order but does NOT submit to Alpaca.
        Use dry_run=True during backtesting or pre-live validation.
        """
        client_order_id = f"qe_{size.symbol}_{uuid.uuid4().hex[:8]}"
        order_side = OrderSide.BUY if side == "buy" else OrderSide.SELL

        # Calculate SL/TP prices
        if side == "buy":
            tp_price = round(price * (1 + self._take_profit_pct), 2)
            sl_price = round(price * (1 - self._stop_loss_pct),   2)
        else:  # short
            tp_price = round(price * (1 - self._take_profit_pct), 2)
            sl_price = round(price * (1 + self._stop_loss_pct),   2)

        order_request = MarketOrderRequest(
            symbol=size.symbol,
            qty=size.qty,
            side=order_side,
            time_in_force=TimeInForce.DAY,
            order_class=OrderClass.BRACKET,
            take_profit=TakeProfitRequest(limit_price=tp_price),
            stop_loss=StopLossRequest(stop_price=sl_price),
            client_order_id=client_order_id,
        )

        result = OrderResult(
            client_order_id=client_order_id,
            symbol=size.symbol,
            side=side,
            qty=size.qty,
            order_class="bracket",
            submitted_at=datetime.now(timezone.utc),
        )

        if self._dry_run:
            logger.info(
                "[DRY RUN] Bracket order: %s %s qty=%.2f TP=%.2f SL=%.2f id=%s",
                side.upper(), size.symbol, size.qty, tp_price, sl_price, client_order_id,
            )
            result.alpaca_order_id = f"dry_{client_order_id}"
            return result

        # Submit to Alpaca (synchronous SDK → thread executor)
        loop = asyncio.get_running_loop()
        try:
            order = await loop.run_in_executor(
                None,
                lambda: self._client.submit_order(order_data=order_request),
            )
            result.alpaca_order_id = str(order.id)
            logger.info(
                "Order submitted ✓ %s %s qty=%.2f TP=%.2f SL=%.2f alpaca_id=%s",
                side.upper(), size.symbol, size.qty, tp_price, sl_price, order.id,
            )
        except Exception as exc:
            result.error = str(exc)
            logger.error(
                "Order FAILED: %s %s qty=%.2f — %s",
                side.upper(), size.symbol, size.qty, exc,
            )

        return result

    async def close_position(self, symbol: str) -> OrderResult:
        """
        Close an open position by symbol (market order, opposite side).
        Used by ExecutionMonitor for stop-out or end-of-day flatten.
        """
        client_order_id = f"qe_close_{symbol}_{uuid.uuid4().hex[:8]}"
        result = OrderResult(
            client_order_id=client_order_id,
            symbol=symbol,
            side="close",
            qty=0.0,
            order_class="market",
            submitted_at=datetime.now(timezone.utc),
        )

        if self._dry_run:
            logger.info("[DRY RUN] Close position: %s", symbol)
            result.alpaca_order_id = f"dry_{client_order_id}"
            return result

        loop = asyncio.get_running_loop()
        try:
            resp = await loop.run_in_executor(
                None,
                lambda: self._client.close_position(symbol),
            )
            result.alpaca_order_id = str(resp.id)
            logger.info("Position closed ✓ %s", symbol)
        except Exception as exc:
            result.error = str(exc)
            logger.error("Close position FAILED: %s — %s", symbol, exc)

        return result
