"""
Gap-and-Go Strategy Backtester — Phase 8 (optimized sweep)
============================================================
Key optimization: daily bars fetched ONCE, reused across all sweep combinations.

Usage:
    python -m tests.backtest_gap_and_go          # full S&P 500 run
    python -m tests.backtest_gap_and_go --sweep  # parameter sweep

Requires:
    ALPACA_API_KEY / ALPACA_SECRET_KEY in .env
    pip install alpaca-py pandas numpy python-dotenv
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

# ── Full universe ──────────────────────────────────────────────────────────
SP500 = [
    "MMM","AOS","ABT","ABBV","ACN","ADBE","AMD","AES","AFL","A","APD","ABNB","AKAM","ALB","ARE","ALGN",
    "ALLE","LNT","ALL","GOOGL","GOOG","MO","AMZN","AMCR","AEE","AAL","AEP","AXP","AIG","AMT","AWK","AMP",
    "AME","AMGN","APH","ADI","ANSS","AON","APA","AAPL","AMAT","APTV","ACGL","ADM","ANET","AJG","AIZ",
    "T","ATO","ADSK","ADP","AZO","AVB","AVY","AXON","BKR","BALL","BAC","BK","BBWI","BAX","BDX","BRO",
    "BSX","BMY","AVGO","BR","BLDR","BLK","BX","BA","BKNG","BWA","BXP","CHRW","CDNS","CZR","CPT","CPB",
    "COF","CAH","KMX","CCL","CARR","CAT","CBOE","CBRE","CDW","CE","COR","CNC","CNX","CDAY","CF","CRL",
    "SCHW","CHTR","CVX","CMG","CB","CHD","CI","CINF","CTAS","CSCO","C","CFG","CLX","CME","CMS","KO",
    "CTSH","CL","CMCSA","CMA","CAG","COP","ED","STZ","CEG","COO","CPRT","GLW","CPAY","CTVA","CSGP",
    "COST","CTRA","CCI","CSX","CMI","CVS","DHI","DHR","DRI","DVA","DAY","DECK","DE","DAL","XRAY","DVN",
    "DXCM","FANG","DLR","DFS","DG","DLTR","D","DPZ","DOV","DOW","DTE","DUK","DD","EMN","ETN","EBAY",
    "ECL","EIX","EW","EA","ELV","LLY","EMR","ENPH","ETR","EOG","EPAM","EQT","EFX","EQIX","EQR","ESS",
    "EL","ETSY","EG","EVRG","ES","EXC","EXPE","EXPD","EXR","XOM","FFIV","FDS","FICO","FAST","FRT",
    "FDX","FIS","FITB","FSLR","FE","FI","FMC","F","FTNT","FTV","FOXA","FOX","BEN","FCX","GRMN","IT",
    "GE","GEHC","GEV","GEN","GNRC","GD","GIS","GPC","GILD","GPN","GL","GDDY","GS","HAL","HIG","HAS",
    "HCA","DOC","HSIC","HSY","HES","HPE","HLT","HOLX","HD","HON","HRL","HST","HWM","HPQ","HUBB","HUM",
    "HBAN","HII","IBM","IEX","IDXX","ITW","INCY","IR","PODD","INTC","ICE","IFF","IP","IPG","INTU",
    "ISRG","IVZ","INVH","IQV","IRM","JBHT","JBL","JKHY","J","JNJ","JCI","JPM","JNPR","K","KVUE","KDP",
    "KEY","KEYS","KMB","KIM","KMI","KLAC","KHC","KR","LHX","LH","LRCX","LW","LVS","LDOS","LEN","LIN",
    "LYV","LKQ","LMT","L","LOW","LULU","LYB","MTB","MRO","MPC","MKTX","MAR","MMC","MLM","MAS","MA",
    "MTCH","MKC","MCD","MCK","MDT","MRK","META","MET","MTD","MGM","MCHP","MU","MSFT","MAA","MRNA",
    "MHK","MOH","TAP","MDLZ","MPWR","MNST","MCO","MS","MOS","MSI","MSCI","NDAQ","NTAP","NFLX","NEM",
    "NWSA","NWS","NEE","NKE","NI","NDSN","NSC","NTRS","NOC","NCLH","NRG","NUE","NVDA","NVR","NXPI",
    "ORLY","OXY","ODFL","OMC","ON","OKE","ORCL","OTIS","PCAR","PKG","PANW","PH","PAYX","PAYC","PYPL",
    "PNR","PEP","PFE","PCG","PM","PSX","PNW","PNC","POOL","PPG","PPL","PFG","PG","PGR","PLD","PRU",
    "PEG","PTC","PSA","PHM","QRVO","PWR","QCOM","DGX","RL","RJF","RTX","O","REG","REGN","RF","RSG",
    "RMD","RVTY","ROK","ROL","ROP","ROST","RCL","SPGI","CRM","SBAC","SLB","STX","SRE","NOW","SHW",
    "SPG","SWKS","SJM","SNA","SOLV","SO","LUV","SWK","SBUX","STT","STLD","STE","SYK","SYF","SNPS",
    "SYY","TMUS","TROW","TTWO","TPR","TRGP","TGT","TEL","TDY","TFX","TER","TSLA","TXN","TXT","TMO",
    "TJX","TSCO","TT","TDG","TRV","TRMB","TFC","TYL","TSN","USB","UBER","UDR","ULTA","UNP","UAL",
    "UPS","URI","UNH","UHS","VLO","VTR","VLTO","VRSN","VRSK","VZ","VRTX","VTRS","VICI","V","VST",
    "VMC","WRB","GWW","WAB","WBA","WMT","DIS","WBD","WM","WAT","WEC","WFC","WELL","WST","WDC","WY",
    "WHR","WMB","WTW","WYNN","XEL","XYL","YUM","ZBRA","ZBH","ZTS",
    # High-momentum extras
    "UPST","AFRM","HOOD","COIN","LYFT","SNAP","DASH","SOFI","PLTR","RBLX","RIVN","SQQQ","TQQQ","TZA","TNA",
]
SP500 = list(dict.fromkeys(SP500))  # dedupe

# ── Strategy parameters ────────────────────────────────────────────────────
MIN_PRICE         = 5.0
MIN_AVG_VOLUME    = 500_000
ATR_PERIOD        = 14
ATR_STOP_MULT     = 1.0
CONVICTION_MULT   = 1.5
HIGH_CONV_GAP     = 5.0
HIGH_CONV_VOL     = 3.0
TRAIL_PCT         = 0.015
COMMISSION        = 0.005
INITIAL_CAPITAL   = 100_000.0
START_DATE        = "2024-01-01"
END_DATE          = "2025-12-31"
BATCH_SIZE        = 50


@dataclass
class Trade:
    symbol:      str
    date:        date
    direction:   str
    entry_price: float
    exit_price:  float
    qty:         float
    exit_reason: str
    pnl:         float
    pnl_pct:     float
    hold_bars:   int
    gap_pct:     float
    vol_ratio:   float
    conviction:  str


# ──────────────────────────────────────────────────────────────────────────
def fetch_daily_bars_batch(client, symbols, start, end):
    result = {}
    try:
        req    = StockBarsRequest(symbol_or_symbols=symbols, timeframe=TimeFrame.Day,
                                  start=start, end=end, adjustment="all")
        df_all = client.get_stock_bars(req).df
        if not df_all.empty:
            for sym in symbols:
                try:
                    df = df_all.xs(sym, level="symbol").copy()
                    df.index = pd.to_datetime(df.index).tz_convert(None)
                    if len(df) > ATR_PERIOD + 2:
                        result[sym] = df
                except KeyError:
                    pass
            return result
    except Exception:
        pass
    for sym in symbols:
        try:
            req = StockBarsRequest(symbol_or_symbols=sym, timeframe=TimeFrame.Day,
                                   start=start, end=end, adjustment="all")
            df  = client.get_stock_bars(req).df
            if df.empty: continue
            if isinstance(df.index, pd.MultiIndex):
                df = df.xs(sym, level="symbol").copy()
            df.index = pd.to_datetime(df.index).tz_convert(None)
            if len(df) > ATR_PERIOD + 2:
                result[sym] = df
        except Exception:
            pass
    return result


def load_all_daily_bars(client) -> dict[str, pd.DataFrame]:
    """Fetch daily bars ONCE for all symbols — reused across sweep combinations."""
    logger.info(f"Fetching daily bars for {len(SP500)} symbols in batches of {BATCH_SIZE}...")
    daily = {}
    batches = [SP500[i:i+BATCH_SIZE] for i in range(0, len(SP500), BATCH_SIZE)]
    for i, batch in enumerate(batches):
        result = fetch_daily_bars_batch(client, batch, START_DATE, END_DATE)
        daily.update(result)
        logger.info(f"  Batch {i+1}/{len(batches)}: {len(result)}/{len(batch)} loaded ({len(daily)} total)")
        time.sleep(0.3)
    logger.info(f"Daily bars loaded for {len(daily)} symbols — reusing for all sweep runs.")
    return daily


def fetch_minute_bars(client, symbol, day):
    req = StockBarsRequest(
        symbol_or_symbols=symbol, timeframe=TimeFrame.Minute,
        start=datetime(day.year, day.month, day.day, 13, 30),
        end=datetime(day.year, day.month, day.day, 20, 0),
        adjustment="all",
    )
    df = client.get_stock_bars(req).df
    if df.empty: return pd.DataFrame()
    if isinstance(df.index, pd.MultiIndex):
        df = df.xs(symbol, level="symbol")
    df.index = pd.to_datetime(df.index).tz_convert(None)
    return df


def compute_atr(df, period=14):
    high, low, cp = df["high"], df["low"], df["close"].shift(1)
    tr = pd.concat([(high-low), (high-cp).abs(), (low-cp).abs()], axis=1).max(axis=1)
    return float(tr.iloc[-period:].mean())


def simulate_trade(minute_bars, direction, entry_price, stop_price, target_price, qty):
    peak = entry_price
    for i, (_, bar) in enumerate(minute_bars.iterrows()):
        price = bar["close"]
        if direction == "long":
            peak = max(peak, price)
            trail_stop = peak * (1 - TRAIL_PCT)
            if bar["low"]  <= stop_price:   return stop_price,   qty*(stop_price-entry_price)   - qty*COMMISSION*2, "stop",   i+1
            if bar["high"] >= target_price: return target_price, qty*(target_price-entry_price) - qty*COMMISSION*2, "target", i+1
            if price <= trail_stop and price > entry_price: return trail_stop, qty*(trail_stop-entry_price) - qty*COMMISSION*2, "trail", i+1
        else:
            peak = min(peak, price)
            trail_stop = peak * (1 + TRAIL_PCT)
            if bar["high"] >= stop_price:   return stop_price,   qty*(entry_price-stop_price)   - qty*COMMISSION*2, "stop",   i+1
            if bar["low"]  <= target_price: return target_price, qty*(entry_price-target_price) - qty*COMMISSION*2, "target", i+1
            if price >= trail_stop and price < entry_price: return trail_stop, qty*(entry_price-trail_stop) - qty*COMMISSION*2, "trail", i+1
    exit_p = float(minute_bars.iloc[-1]["close"]) if len(minute_bars) > 0 else entry_price
    pnl    = qty*(exit_p-entry_price if direction=="long" else entry_price-exit_p) - qty*COMMISSION*2
    return exit_p, pnl, "eod", len(minute_bars)


def compute_tearsheet(trades, equity):
    if not trades: return {}
    df      = pd.DataFrame([t.__dict__ for t in trades])
    returns = equity.pct_change().dropna()
    DAYS    = 252
    ann_ret = float((1 + returns.mean()) ** DAYS - 1) if len(returns) > 0 else 0
    sharpe  = float((returns.mean() / returns.std()) * np.sqrt(DAYS)) if returns.std() > 0 else 0
    mdd     = float(((equity - equity.cummax()) / equity.cummax()).min())
    wins    = df[df["pnl"] > 0]; losses = df[df["pnl"] < 0]
    return {
        "total_trades":     len(df),
        "win_rate_pct":     round(len(wins)/len(df)*100, 1),
        "profit_factor":    round(wins["pnl"].sum()/abs(losses["pnl"].sum()), 3) if len(losses) > 0 else float("inf"),
        "avg_trade_pnl":    round(df["pnl"].mean(), 2),
        "best_trade":       round(df["pnl"].max(), 2),
        "worst_trade":      round(df["pnl"].min(), 2),
        "total_return_pct": round((equity.iloc[-1]-INITIAL_CAPITAL)/INITIAL_CAPITAL*100, 2),
        "ann_return_pct":   round(ann_ret*100, 2),
        "sharpe":           round(sharpe, 3),
        "max_drawdown_pct": round(mdd*100, 2),
        "calmar":           round((ann_ret/abs(mdd)) if mdd != 0 else 0, 3),
        "exit_reasons":     df["exit_reason"].value_counts().to_dict(),
        "by_symbol":        df.groupby("symbol")["pnl"].sum().sort_values(ascending=False).round(2).to_dict(),
        "high_conv_trades": int((df["conviction"]=="high").sum()),
        "high_conv_wr":     round(len(df[(df["conviction"]=="high") & (df["pnl"]>0)]) / max((df["conviction"]=="high").sum(),1)*100, 1),
    }


# ──────────────────────────────────────────────────────────────────────────
def run_backtest(
    client, daily,                          # ← pass pre-loaded data in
    min_gap=2.0, min_vol_ratio=1.5,
    base_capital=20_000.0, max_positions=5,
    rr=2.0, verbose=True,
):
    all_days = sorted(set(d.date() for df in daily.values() for d in df.index))
    if verbose:
        logger.info(f"Running backtest: gap={min_gap} vol={min_vol_ratio} "
                    f"cap=${base_capital:,.0f} pos={max_positions} rr={rr}")
        logger.info(f"  {len(daily)} symbols | {len(all_days)} trading days")

    trades: list[Trade] = []
    equity_curve: list[tuple] = []
    capital = INITIAL_CAPITAL

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
            if abs(gap_pct) < min_gap or abs(gap_pct) > 50: continue
            if open_price < MIN_PRICE: continue
            hist    = df.iloc[max(0, idx-20):idx]
            avg_vol = float(hist["volume"].mean())
            if avg_vol < MIN_AVG_VOLUME: continue
            vol_ratio = float(day_rows.iloc[0]["volume"]) / avg_vol if avg_vol > 0 else 0
            if vol_ratio < min_vol_ratio: continue
            atr       = compute_atr(df.iloc[:idx], ATR_PERIOD)
            direction = "long" if gap_pct > 0 else "short"
            stop      = open_price - ATR_STOP_MULT*atr if direction=="long" else open_price + ATR_STOP_MULT*atr
            risk      = abs(open_price - stop)
            if risk == 0: continue
            conviction = "high" if abs(gap_pct) >= HIGH_CONV_GAP and vol_ratio >= HIGH_CONV_VOL else "normal"
            trade_cap  = min(base_capital * (CONVICTION_MULT if conviction=="high" else 1.0), capital * 0.25)
            candidates.append({
                "symbol": sym, "gap_pct": gap_pct, "vol_ratio": vol_ratio,
                "direction": direction, "open": open_price, "stop": stop,
                "target": open_price + rr*risk if direction=="long" else open_price - rr*risk,
                "atr": atr, "conviction": conviction, "trade_cap": trade_cap,
                "score": abs(gap_pct) * min(vol_ratio, 3) * (1.5 if conviction=="high" else 1.0),
            })

        if not candidates:
            equity_curve.append((day, capital)); continue

        for c in sorted(candidates, key=lambda x: x["score"], reverse=True)[:max_positions]:
            sym = c["symbol"]
            try:
                minute_df = fetch_minute_bars(client, sym, day)
            except Exception as e:
                logger.warning(f"  {sym} {day}: minute fetch failed — {e}"); continue
            if len(minute_df) < 2: continue
            first = minute_df.iloc[0]
            if c["direction"] == "long"  and first["close"] <= first["open"]: continue
            if c["direction"] == "short" and first["close"] >= first["open"]: continue
            entry_price = float(minute_df.iloc[1]["open"])
            qty         = max(1, int(c["trade_cap"] / entry_price))
            risk        = abs(entry_price - c["stop"])
            if risk == 0: continue
            target      = entry_price + rr*risk if c["direction"]=="long" else entry_price - rr*risk
            exit_p, pnl, reason, hold = simulate_trade(
                minute_df.iloc[1:], c["direction"], entry_price, c["stop"], target, qty)
            trade = Trade(
                symbol=sym, date=day, direction=c["direction"],
                entry_price=entry_price, exit_price=exit_p, qty=qty,
                exit_reason=reason, pnl=round(pnl, 2),
                pnl_pct=round(pnl/(qty*entry_price), 4), hold_bars=hold,
                gap_pct=round(c["gap_pct"], 2), vol_ratio=round(c["vol_ratio"], 2),
                conviction=c["conviction"],
            )
            trades.append(trade)
            capital += pnl
            if verbose:
                logger.info(f"  {day} {sym:6s} {c['direction']:5s} [{c['conviction']:6s}] "
                            f"gap={c['gap_pct']:+.1f}% vol={c['vol_ratio']:.1f}x "
                            f"entry={entry_price:.2f} pnl=${pnl:+.2f} [{reason}]")
        equity_curve.append((day, capital))

    equity = pd.Series({d: v for d, v in equity_curve}, name="equity")
    ts     = compute_tearsheet(trades, equity)

    if verbose:
        print("\n" + "="*64)
        print("  GAP-AND-GO BACKTEST RESULTS — Phase 8")
        print("="*64)
        print(f"  Period:            {START_DATE} → {END_DATE}")
        print(f"  Universe:          {len(daily)} symbols")
        print(f"  Capital/Trade:     ${base_capital:,.0f} (high-conv: ${base_capital*CONVICTION_MULT:,.0f})")
        print(f"  Max Positions:     {max_positions}")
        print(f"  Initial Capital:   ${INITIAL_CAPITAL:,.0f}")
        print(f"  Final Capital:     ${capital:,.2f}")
        print(f"  Total Return:      {ts.get('total_return_pct',0):+.2f}%")
        print(f"  Ann. Return:       {ts.get('ann_return_pct',0):+.2f}%")
        print(f"  Sharpe Ratio:      {ts.get('sharpe',0):.3f}")
        print(f"  Max Drawdown:      {ts.get('max_drawdown_pct',0):.2f}%")
        print(f"  Calmar Ratio:      {ts.get('calmar',0):.3f}")
        print(f"  Total Trades:      {ts.get('total_trades',0)}")
        print(f"  Win Rate:          {ts.get('win_rate_pct',0):.1f}%")
        print(f"  Profit Factor:     {ts.get('profit_factor',0):.3f}")
        print(f"  Avg Trade PnL:     ${ts.get('avg_trade_pnl',0):+.2f}")
        print(f"  Best Trade:        ${ts.get('best_trade',0):+.2f}")
        print(f"  Worst Trade:       ${ts.get('worst_trade',0):+.2f}")
        print(f"  High-Conv Trades:  {ts.get('high_conv_trades',0)} (WR: {ts.get('high_conv_wr',0):.1f}%)")
        print()
        print("  Exit Breakdown:")
        for r, cnt in ts.get("exit_reasons", {}).items():
            print(f"    {r:10s}: {cnt}")
        print()
        print("  Top 15 Symbols by PnL:")
        for s, p in list(ts.get("by_symbol", {}).items())[:15]:
            print(f"    {s:8s}: ${p:+,.2f}")
        print("="*64)

    pd.DataFrame([t.__dict__ for t in trades]).to_csv("gap_and_go_trades.csv", index=False)
    equity.to_csv("gap_and_go_equity.csv")
    logger.info("Saved: gap_and_go_trades.csv, gap_and_go_equity.csv")
    return trades, equity, ts


# ── Parameter Sweep ────────────────────────────────────────────────────────
def run_sweep(client, daily):
    """Run all sweep combinations using pre-loaded daily bars."""
    sweep_params = [
        (2.0, 1.5, 20000, 5, 2.0),
        (2.0, 2.0, 20000, 5, 2.0),
        (3.0, 1.5, 20000, 5, 2.0),
        (3.0, 2.0, 20000, 5, 2.0),
        (2.0, 1.5, 25000, 5, 2.0),
        (2.0, 1.5, 20000, 5, 1.5),
        (2.0, 1.5, 20000, 7, 2.0),
        (2.0, 1.5, 20000, 5, 2.5),
    ]
    results = []
    for i, params in enumerate(sweep_params):
        min_gap, min_vol, cap, maxpos, rr = params
        logger.info(f"Sweep {i+1}/{len(sweep_params)}: gap={min_gap} vol={min_vol} cap=${cap} pos={maxpos} rr={rr}")
        try:
            _, _, ts = run_backtest(client, daily, min_gap=min_gap, min_vol_ratio=min_vol,
                                    base_capital=cap, max_positions=maxpos, rr=rr, verbose=False)
            if ts:
                results.append({"min_gap": min_gap, "min_vol_ratio": min_vol,
                                 "base_capital": cap, "max_positions": maxpos, "risk_reward": rr,
                                 **{k: ts.get(k) for k in ["total_return_pct","ann_return_pct","sharpe",
                                     "max_drawdown_pct","calmar","win_rate_pct","profit_factor","total_trades"]}})
                logger.info(f"  → return={ts.get('total_return_pct',0):+.2f}% sharpe={ts.get('sharpe',0):.3f} "
                            f"wr={ts.get('win_rate_pct',0):.1f}% trades={ts.get('total_trades',0)}")
        except Exception as e:
            logger.warning(f"Sweep failed for {params}: {e}")

    if not results:
        logger.error("All sweep runs failed."); return

    sweep_df = pd.DataFrame(results).sort_values("sharpe", ascending=False)
    sweep_df.to_csv("gap_sweep_results.csv", index=False)
    print("\n" + "="*90)
    print("  PARAMETER SWEEP RESULTS (sorted by Sharpe)")
    print("="*90)
    print(sweep_df.to_string(index=False))
    print("\nSaved: gap_sweep_results.csv")


# ──────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", action="store_true", help="Run parameter sweep")
    args = parser.parse_args()

    api_key = os.environ.get("ALPACA_API_KEY")
    secret  = os.environ.get("ALPACA_SECRET_KEY")
    if not api_key or not secret:
        raise EnvironmentError("Set ALPACA_API_KEY and ALPACA_SECRET_KEY in .env")

    client = StockHistoricalDataClient(api_key, secret)

    # ── Fetch daily bars ONCE ──────────────────────────────────────────────
    daily = load_all_daily_bars(client)
    if not daily:
        raise RuntimeError("No data fetched.")

    if args.sweep:
        run_sweep(client, daily)
    else:
        run_backtest(client, daily, verbose=True)
