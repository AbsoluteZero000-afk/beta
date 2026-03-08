from __future__ import annotations
import logging
from dataclasses import dataclass, field
from datetime import datetime

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

logger = logging.getLogger(__name__)


@dataclass
class ManagedPosition:
    symbol:       str
    side:         str
    qty:          float
    entry_price:  float
    stop_price:   float
    target_price: float
    entry_time:   datetime = field(default_factory=datetime.utcnow)
    trail_pct:    float = 0.015
    peak_price:   float = 0.0

    def __post_init__(self):
        self.peak_price = self.entry_price


class PositionManager:
    def __init__(self, trading_client: TradingClient, paper: bool = True):
        self.client = trading_client
        self.paper  = paper
        self.positions: dict[str, ManagedPosition] = {}

    def add(self, pos: ManagedPosition):
        self.positions[pos.symbol] = pos
        logger.info(f"[PM] Tracking {pos.symbol} | entry={pos.entry_price:.2f} "
                    f"stop={pos.stop_price:.2f} target={pos.target_price:.2f}")

    def update(self, symbol: str, current_price: float) -> str | None:
        pos = self.positions.get(symbol)
        if not pos:
            return None
        if pos.side == "long":
            pos.peak_price = max(pos.peak_price, current_price)
            trail_stop     = pos.peak_price * (1 - pos.trail_pct)
            if current_price <= pos.stop_price:
                return "stop"
            if current_price >= pos.target_price:
                return "target"
            if current_price <= trail_stop and current_price > pos.entry_price:
                return "trail"
        else:
            pos.peak_price = min(pos.peak_price, current_price)
            trail_stop     = pos.peak_price * (1 + pos.trail_pct)
            if current_price >= pos.stop_price:
                return "stop"
            if current_price <= pos.target_price:
                return "target"
            if current_price >= trail_stop and current_price < pos.entry_price:
                return "trail"
        return None

    def close(self, symbol: str, reason: str):
        if symbol not in self.positions:
            return
        pos = self.positions.pop(symbol)
        logger.info(f"[PM] Closing {symbol} — reason: {reason}")
        try:
            side = OrderSide.SELL if pos.side == "long" else OrderSide.BUY
            req  = MarketOrderRequest(
                symbol=symbol,
                qty=pos.qty,
                side=side,
                time_in_force=TimeInForce.DAY,
            )
            self.client.submit_order(req)
            logger.info(f"[PM] Close order submitted for {symbol}")
        except Exception as e:
            logger.error(f"[PM] Failed to close {symbol}: {e}")
