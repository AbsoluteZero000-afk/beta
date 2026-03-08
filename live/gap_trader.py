from __future__ import annotations
import logging
from datetime import datetime, date
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from src.scanners.gap_scanner import GapScanner
from src.live.position_manager import PositionManager, ManagedPosition

logger = logging.getLogger(__name__)


class GapAndGoTrader:
    def __init__(
        self,
        trading_client:    TradingClient,
        data_client:       StockHistoricalDataClient,
        position_manager:  PositionManager,
        universe:          list[str],
        capital_per_trade: float = 5_000.0,
        min_gap_pct:       float = 2.0,
        min_vol_ratio:     float = 0.3,
        risk_reward:       float = 2.0,
        max_positions:     int   = 3,
    ):
        self.trading_client    = trading_client
        self.data_client       = data_client
        self.pm                = position_manager
        self.universe          = universe
        self.capital_per_trade = capital_per_trade
        self.min_gap_pct       = min_gap_pct
        self.min_vol_ratio     = min_vol_ratio
        self.risk_reward       = risk_reward
        self.max_positions     = max_positions
        self.watchlist:    list[dict] = []
        self.traded_today: set[str]  = set()
        self._scanner = GapScanner(
            min_gap_pct=min_gap_pct,
            min_volume_ratio=min_vol_ratio,
        )

    def premarket_scan(self) -> list[dict]:
        logger.info("[GAP] Running pre-market scan...")
        results = self._scanner.scan(self.universe)
        self.watchlist = [
            r for r in results
            if abs(r["gap_pct"]) < 50
            and r["symbol"] not in self.traded_today
            and r.get("volume_ratio", 0) >= self.min_vol_ratio
        ]
        self.watchlist.sort(key=lambda x: x["score"], reverse=True)
        logger.info(f"[GAP] Watchlist ({len(self.watchlist)}): {[r['symbol'] for r in self.watchlist]}")
        return self.watchlist

    def confirm_and_enter(self):
        if not self.watchlist:
            logger.info("[GAP] Watchlist empty, nothing to enter.")
            return
        if len(self.pm.positions) >= self.max_positions:
            logger.info("[GAP] Max positions reached.")
            return

        for candidate in self.watchlist[:self.max_positions]:
            sym = candidate["symbol"]
            if sym in self.traded_today:
                continue
            if len(self.pm.positions) >= self.max_positions:
                break
            try:
                candle = self._get_first_candle(sym)
            except Exception as e:
                logger.warning(f"[GAP] First candle error {sym}: {e}")
                continue
            if candle is None:
                continue

            gap_dir        = candidate["direction"]
            candle_bullish = candle["close"] > candle["open"]
            candle_bearish = candle["close"] < candle["open"]

            if gap_dir == "long"  and not candle_bullish:
                logger.info(f"[GAP] {sym} — bearish candle, no entry")
                continue
            if gap_dir == "short" and not candle_bearish:
                logger.info(f"[GAP] {sym} — bullish candle, no short entry")
                continue

            self._enter(candidate, candle)

    def _get_first_candle(self, symbol: str) -> dict | None:
        today = date.today()
        req   = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Minute,
            start=datetime(today.year, today.month, today.day, 13, 30),
            limit=1,
        )
        bars = self.data_client.get_stock_bars(req)
        if symbol not in bars or not bars[symbol]:
            return None
        b = bars[symbol][0]
        return {"open": b.open, "high": b.high, "low": b.low,
                "close": b.close, "volume": b.volume}

    def _enter(self, candidate: dict, candle: dict):
        sym    = candidate["symbol"]
        side   = OrderSide.BUY if candidate["direction"] == "long" else OrderSide.SELL
        price  = candle["close"]
        stop   = candidate["suggested_stop"]
        risk   = abs(price - stop)
        if risk == 0:
            logger.warning(f"[GAP] {sym} risk=0, skipping")
            return
        target = (price + self.risk_reward * risk
                  if side == OrderSide.BUY
                  else price - self.risk_reward * risk)
        qty    = max(1, int(self.capital_per_trade / price))

        logger.info(f"[GAP] Entering {sym} | side={side.value} qty={qty} "
                    f"price={price:.2f} stop={stop:.2f} target={target:.2f}")
        try:
            req = MarketOrderRequest(
                symbol=sym,
                qty=qty,
                side=side,
                time_in_force=TimeInForce.DAY,
            )
            self.trading_client.submit_order(req)
            self.pm.add(ManagedPosition(
                symbol=sym,
                side=candidate["direction"],
                qty=qty,
                entry_price=price,
                stop_price=stop,
                target_price=target,
            ))
            self.traded_today.add(sym)
            logger.info(f"[GAP] Order submitted for {sym}")
        except Exception as e:
            logger.error(f"[GAP] Order failed for {sym}: {e}")

    def reset_daily(self):
        self.watchlist    = []
        self.traded_today = set()
