"""
CLI runner for backtests.

Usage:
    python -m src.backtest.runner --symbol AAPL --days 365
    python -m src.backtest.runner --symbol AAPL,MSFT,NVDA --days 730 --walk-forward
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone

from src.backtest.engine import BacktestEngine
from src.backtest.walk_forward import WalkForwardValidator
from src.data.alpaca_fetcher import fetch_historical_bars


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="QuantEdge Backtester")
    p.add_argument("--symbol",       default="AAPL")
    p.add_argument("--days",         type=int,   default=365)
    p.add_argument("--capital",      type=float, default=100_000.0)
    p.add_argument("--threshold",    type=float, default=0.25)
    p.add_argument("--walk-forward", action="store_true")
    p.add_argument("--n-splits",     type=int,   default=5)
    p.add_argument("--output",       default=None)
    return p.parse_args()


def main() -> None:
    args    = parse_args()
    symbols = [s.strip().upper() for s in args.symbol.split(",")]
    end     = datetime.now(timezone.utc)
    start   = end - timedelta(days=args.days)

    engine_kwargs  = {"initial_capital": args.capital, "signal_threshold": args.threshold}
    all_tearsheets = []

    for sym in symbols:
        print(f"\n{'='*50}\n  Backtesting {sym}  ({args.days} days)\n{'='*50}")
        bars = fetch_historical_bars(sym, start, end)
        if bars is None or len(bars) < 100:
            print(f"  WARNING: Insufficient data for {sym}, skipping.")
            continue

        if args.walk_forward:
            wf = WalkForwardValidator(n_splits=args.n_splits, engine_kwargs=engine_kwargs)
            ts = wf.run(bars, symbol=sym).combined_tearsheet
        else:
            engine = BacktestEngine(**engine_kwargs)
            ts     = engine.run(bars, symbol=sym).tearsheet

        for k, v in ts.items():
            print(f"    {k:25s}: {v}")
        all_tearsheets.append(ts)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(all_tearsheets, f, indent=2)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
