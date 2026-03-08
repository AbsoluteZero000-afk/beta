"""
backtest_options.py — Phase 9 Options Backtest
================================================
Simulates the hybrid strategy on historical data:
  - Normal conviction gaps → shares (exact same as Phase 8)
  - High conviction gaps   → options (simulated via underlying price move + leverage)

Options simulation model:
  - ATM call/put approximated as 50-delta
  - Premium ≈ 0.04 × underlying price (rough ATM weekly estimate)
  - P&L = premium × (price_move / (0.5 × underlying)) × 100 shares
  - Max loss = premium paid

Usage:
    python -m tests.backtest_options
    python -m tests.backtest_options --shares-only   # baseline comparison
"""
from __future__ import annotations

import argparse
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, date
from pathlib import Path

import pandas as pd
import numpy as np
from dotenv import load_dotenv
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)

try:
    from tests.backtest_gap_and_go import SP500, load_all_daily_bars, fetch_minute_bars
except ImportError:
    from backtest_gap_and_go import SP500, load_all_daily_bars, fetch_minute_bars

# ── Locked params ──────────────────────────────────────────────────────────
MIN_GAP_PCT       = 2.0
MIN_VOL_RATIO     = 2.0
MIN_PRICE         = 5.0
MIN_AVG_VOLUME    = 500_000
ATR_PERIOD        = 14
ATR_STOP_MULT     = 1.0
RISK_REWARD       = 2.0
BASE_CAPITAL      = 20_000.0
MAX_POSITIONS     = 5
HIGH_CONV_GAP     = 5.0
HIGH_CONV_VOL     = 3.0
TRAIL_PCT         = 0.015
COMMISSION        = 0.005
INITIAL_CAPITAL   = 100_000.0
START_DATE        = "2024-01-01"
END_DATE          = "2025-12-31"

# ── Options simulation params ──────────────────────────────────────────────
OPTION_PREMIUM_PCT  = 0.04    # ATM weekly premium ≈ 4% of underlying
MAX_OPTION_SPEND    = 1_000.0 # max premium per trade
MAX_OPTION_HOLD     = 3       # days before forced close
OPTION_WIN_HOLD_PCT = 0.10    # hold if up >10% at EOD
DELTA               = 0.50    # ATM delta approximation


@dataclass
class Trade:
    symbol:      str
    date:        date
    direction:   str
    conviction:  str
    route:       str        # "shares" or "options"
    entry_price: float
    exit_price:  float
    qty:         float
    exit_reason: str
    pnl:         float
    pnl_pct:     float
    gap_pct:     float
    vol_ratio:   float


def simulate_option_pnl(
    minute_bars: pd.DataFrame,
    direction: str,
    entry_price: float,
    max_hold_days: int,
    premium_per_contract: float,
    n_contracts: int,
) -> tuple[float, str]:
    """
    Simplified options P&L simulation:
    - Uses underlying price moves scaled by delta
    - Caps loss at premium paid
    - Trails winners using TRAIL_PCT on underlying
    """
    total_premium = premium_per_contract * n_contracts
    peak_underlying = entry_price

    for i, (_, bar) in enumerate(minute_bars.iterrows()):
        price = bar["close"]
        move  = (price - entry_price) if direction == "long" else (entry_price - price)
        pct_move = move / entry_price

        # Option value ≈ delta × price_move × 100 × n_contracts
        option_pnl = DELTA * move * 100 * n_contracts

        # Trail: lock profits if up >50% on premium
        if direction == "long":
            peak_underlying = max(peak_underlying, price)
            trail_stop_underlying = peak_underlying * (1 - TRAIL_PCT)
            if price <= trail_stop_underlying and option_pnl > total_premium * 0.5:
                return round(option_pnl - total_premium, 2), "trail"

        # Target: 2× premium
        if option_pnl >= total_premium * 2:
            return round(option_pnl - total_premium, 2), "target_2x"

        # Stop: lose full premium if underlying drops past ATR stop
        if pct_move < -0.015:
            return round(-total_premium, 2), "stop"

    # EOD
    last_price = float(minute_bars.iloc[-1]["close"]) if len(minute_bars) > 0 else entry_price
    move       = (last_price - entry_price) if direction == "long" else (entry_price - last_price)
    option_pnl = DELTA * move * 100 * n_contracts
    final_pnl  = max(option_pnl - total_premium, -total_premium)
    return round(final_pnl, 2), "eod"


def simulate_share_pnl(minute_bars, direction, entry_price, stop_price, target_price, qty):
    peak = entry_price
    for i, (_, bar) in enumerate(minute_bars.iterrows()):
        price = bar["close"]
        if direction == "long":
            peak = max(peak, price)
            trail_stop = peak * (1 - TRAIL_PCT)
            if bar["low"]  <= stop_price:   return stop_price,   qty*(stop_price-entry_price)   - qty*COMMISSION*2, "stop"
            if bar["high"] >= target_price: return target_price, qty*(target_price-entry_price) - qty*COMMISSION*2, "target"
            if price <= trail_stop and price > entry_price: return trail_stop, qty*(trail_stop-entry_price) - qty*COMMISSION*2, "trail"
        else:
            peak = min(peak, price)
            trail_stop = peak * (1 + TRAIL_PCT)
            if bar["high"] >= stop_price:   return stop_price,   qty*(entry_price-stop_price)   - qty*COMMISSION*2, "stop"
            if bar["low"]  <= target_price: return target_price, qty*(entry_price-target_price) - qty*COMMISSION*2, "target"
            if price >= trail_stop and price < entry_price: return trail_stop, qty*(entry_price-trail_stop) - qty*COMMISSION*2, "trail"
    exit_p = float(minute_bars.iloc[-1]["close"]) if len(minute_bars) > 0 else entry_price
    pnl    = qty*(exit_p-entry_price if direction=="long" else entry_price-exit_p) - qty*COMMISSION*2
    return exit_p, pnl, "eod"


def compute_tearsheet(trades, equity):
    if not trades: return {}
    df      = pd.DataFrame([t.__dict__ for t in trades])
    returns = equity.pct_change().dropna()
    DAYS    = 252
    ann_ret = float((1 + returns.mean()) ** DAYS - 1)
    sharpe  = float((returns.mean() / returns.std()) * np.sqrt(DAYS)) if returns.std() > 0 else 0
    mdd     = float(((equity - equity.cummax()) / equity.cummax()).min())
    wins    = df[df["pnl"] > 0]; losses = df[df["pnl"] < 0]

    opt  = df[df["route"] == "options"]
    shr  = df[df["route"] == "shares"]

    return {
        "total_return_pct":   round((equity.iloc[-1]-INITIAL_CAPITAL)/INITIAL_CAPITAL*100, 2),
        "ann_return_pct":     round(ann_ret*100, 2),
        "sharpe":             round(sharpe, 3),
        "max_drawdown_pct":   round(mdd*100, 2),
        "calmar":             round(ann_ret/abs(mdd) if mdd != 0 else 0, 3),
        "total_trades":       len(df),
        "win_rate_pct":       round(len(wins)/len(df)*100, 1),
        "profit_factor":      round(wins["pnl"].sum()/abs(losses["pnl"].sum()), 3) if len(losses)>0 else float("inf"),
        "avg_trade_pnl":      round(df["pnl"].mean(), 2),
        "options_trades":     len(opt),
        "options_win_rate":   round(len(opt[opt["pnl"]>0])/max(len(opt),1)*100, 1),
        "options_total_pnl":  round(opt["pnl"].sum(), 2),
        "shares_trades":      len(shr),
        "shares_win_rate":    round(len(shr[shr["pnl"]>0])/max(len(shr),1)*100, 1),
        "shares_total_pnl":   round(shr["pnl"].sum(), 2),
        "exit_reasons":       df["exit_reason"].value_counts().to_dict(),
        "by_symbol":          df.groupby("symbol")["pnl"].sum().sort_values(ascending=False).round(2).to_dict(),
    }


def run_options_backtest(client, daily, shares_only=False):
    all_days = sorted(set(d.date() for df in daily.values() for d in df.index))
    logger.info(f"Running {'SHARES ONLY' if shares_only else 'HYBRID'} backtest | "
                f"{len(daily)} symbols | {len(all_days)} days")

    trades: list[Trade] = []
    equity_curve        = []
    capital             = INITIAL_CAPITAL

    for day in all_days:
        candidates = []
        for sym, df in daily.items():
            day_rows = df[df.index.date == day]
            if day_rows.empty: continue
            idx = df.index.get_loc(day_rows.index[0])
            if idx < ATR_PERIOD + 1: continue

            prev_close = float(df.iloc[idx-1]["close"])
            open_price = float(day_rows.iloc[0]["open"])
            gap_pct    = (open_price - prev_close) / prev_close * 100
            if abs(gap_pct) < MIN_GAP_PCT or abs(gap_pct) > 50: continue
            if open_price < MIN_PRICE: continue

            hist      = df.iloc[max(0,idx-20):idx]
            avg_vol   = float(hist["volume"].mean())
            if avg_vol < MIN_AVG_VOLUME: continue
            vol_ratio = float(day_rows.iloc[0]["volume"]) / avg_vol
            if vol_ratio < MIN_VOL_RATIO: continue

            high, low, cp = df["high"], df["low"], df["close"].shift(1)
            tr  = pd.concat([(high-low),(high-cp).abs(),(low-cp).abs()], axis=1).max(axis=1)
            atr = float(tr.iloc[:idx].iloc[-ATR_PERIOD:].mean())

            direction  = "long" if gap_pct > 0 else "short"
            stop       = open_price - ATR_STOP_MULT*atr if direction=="long" else open_price + ATR_STOP_MULT*atr
            risk       = abs(open_price - stop)
            if risk == 0: continue

            conviction = "high" if abs(gap_pct) >= HIGH_CONV_GAP and vol_ratio >= HIGH_CONV_VOL else "normal"

            # Route decision
            if conviction == "high" and not shares_only:
                route         = "options"
                trade_capital = min(MAX_OPTION_SPEND, capital * 0.05)
            else:
                route         = "shares"
                trade_capital = min(BASE_CAPITAL, capital * 0.25)

            candidates.append({
                "symbol": sym, "gap_pct": gap_pct, "vol_ratio": vol_ratio,
                "open": open_price, "stop": stop,
                "target": open_price + RISK_REWARD*risk if direction=="long" else open_price - RISK_REWARD*risk,
                "atr": atr, "direction": direction, "conviction": conviction,
                "route": route, "trade_capital": trade_capital,
                "score": abs(gap_pct) * min(vol_ratio,3) * (1.5 if conviction=="high" else 1.0),
            })

        if not candidates:
            equity_curve.append((day, capital)); continue

        for c in sorted(candidates, key=lambda x: x["score"], reverse=True)[:MAX_POSITIONS]:
            sym = c["symbol"]
            try:
                minute_df = fetch_minute_bars(client, sym, day)
            except Exception:
                continue
            if len(minute_df) < 2: continue

            first = minute_df.iloc[0]
            if c["direction"] == "long"  and first["close"] <= first["open"]: continue
            if c["direction"] == "short" and first["close"] >= first["open"]: continue

            entry_price = float(minute_df.iloc[1]["open"])

            if c["route"] == "options":
                premium_per  = entry_price * OPTION_PREMIUM_PCT * 100  # per contract
                n_contracts  = max(1, int(c["trade_capital"] / premium_per))
                pnl, reason  = simulate_option_pnl(
                    minute_df.iloc[1:], c["direction"], entry_price,
                    MAX_OPTION_HOLD, premium_per, n_contracts,
                )
                exit_price   = entry_price  # not used for options PnL
                qty          = n_contracts
            else:
                qty          = max(1, int(c["trade_capital"] / entry_price))
                risk         = abs(entry_price - c["stop"])
                if risk == 0: continue
                target       = entry_price + RISK_REWARD*risk if c["direction"]=="long" else entry_price - RISK_REWARD*risk
                exit_price, pnl, reason = simulate_share_pnl(
                    minute_df.iloc[1:], c["direction"], entry_price, c["stop"], target, qty)

            trade = Trade(
                symbol=sym, date=day, direction=c["direction"],
                conviction=c["conviction"], route=c["route"],
                entry_price=entry_price, exit_price=exit_price if c["route"]=="shares" else entry_price,
                qty=qty, exit_reason=reason,
                pnl=round(pnl, 2),
                pnl_pct=round(pnl / c["trade_capital"], 4),
                gap_pct=round(c["gap_pct"], 2),
                vol_ratio=round(c["vol_ratio"], 2),
            )
            trades.append(trade)
            capital += pnl

            logger.info(f"  {day} {sym:6s} {c['route']:7s} [{c['conviction']:6s}] "
                        f"gap={c['gap_pct']:+.1f}% pnl=${pnl:+.2f} [{reason}]")

        equity_curve.append((day, capital))

    equity = pd.Series({d: v for d, v in equity_curve}, name="equity")
    ts     = compute_tearsheet(trades, equity)

    label = "SHARES ONLY" if shares_only else "HYBRID (shares + options)"
    print("\n" + "="*68)
    print(f"  PHASE 9 BACKTEST — {label}")
    print("="*68)
    print(f"  Total Return:       {ts.get('total_return_pct',0):+.2f}%")
    print(f"  Ann. Return:        {ts.get('ann_return_pct',0):+.2f}%")
    print(f"  Sharpe:             {ts.get('sharpe',0):.3f}")
    print(f"  Max Drawdown:       {ts.get('max_drawdown_pct',0):.2f}%")
    print(f"  Calmar:             {ts.get('calmar',0):.3f}")
    print(f"  Total Trades:       {ts.get('total_trades',0)}")
    print(f"  Win Rate:           {ts.get('win_rate_pct',0):.1f}%")
    print(f"  Profit Factor:      {ts.get('profit_factor',0):.3f}")
    print()
    print(f"  Shares  trades:     {ts.get('shares_trades',0)}  WR={ts.get('shares_win_rate',0):.1f}%  PnL=${ts.get('shares_total_pnl',0):+,.0f}")
    print(f"  Options trades:     {ts.get('options_trades',0)}  WR={ts.get('options_win_rate',0):.1f}%  PnL=${ts.get('options_total_pnl',0):+,.0f}")
    print()
    print(f"  Final Capital:      ${capital:,.2f}")
    print(f"  Top 10 Symbols:")
    for s, p in list(ts.get("by_symbol", {}).items())[:10]:
        print(f"    {s:8s}: ${p:+,.2f}")
    print("="*68)

    suffix = "shares_only" if shares_only else "hybrid"
    pd.DataFrame([t.__dict__ for t in trades]).to_csv(f"options_trades_{suffix}.csv", index=False)
    equity.to_csv(f"options_equity_{suffix}.csv")
    logger.info(f"Saved: options_trades_{suffix}.csv, options_equity_{suffix}.csv")
    return trades, equity, ts


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--shares-only", action="store_true")
    args = parser.parse_args()

    from dotenv import load_dotenv
    load_dotenv()
    from alpaca.data.historical import StockHistoricalDataClient
    client = StockHistoricalDataClient(os.environ["ALPACA_API_KEY"],
                                       os.environ["ALPACA_SECRET_KEY"])
    daily  = load_all_daily_bars(client)
    if not daily:
        raise RuntimeError("No data fetched.")

    if args.shares_only:
        run_options_backtest(client, daily, shares_only=True)
    else:
        # Run both for side-by-side comparison
        logger.info("Running HYBRID backtest...")
        run_options_backtest(client, daily, shares_only=False)
        logger.info("Running SHARES ONLY baseline...")
        run_options_backtest(client, daily, shares_only=True)
