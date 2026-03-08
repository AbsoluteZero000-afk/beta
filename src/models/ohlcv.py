"""QuantEdge — OHLCV Data Models."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class OHLCVBarModel(BaseModel):
    model_config = ConfigDict(frozen=True)

    time: datetime
    symbol: str
    timeframe: str = "1Min"
    open: float
    high: float
    low: float
    close: float
    volume: int = Field(ge=0)
    vwap: Optional[float] = None
    trade_count: Optional[int] = None


class PortfolioState(BaseModel):
    model_config = ConfigDict(frozen=False)

    cash: float = 0.0
    equity: float = 0.0
    buying_power: float = 0.0
    positions: dict[str, float] = Field(default_factory=dict)
    total_exposure: float = 0.0

    @property
    def total_value(self) -> float:
        return self.cash + self.equity
