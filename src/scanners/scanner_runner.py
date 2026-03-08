"""
Scanner runner — entry point for daily pre-market scan.

Run at 9:00am ET every trading day:
    python -m src.scanners.scanner_runner

Or schedule via cron:
    0 14 * * 1-5 cd /path/to/quantedge && .venv/bin/python -m src.scanners.scanner_runner
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone

from dotenv import load_dotenv
load_dotenv(override=True)

from src.scanners.gap_scanner       import GapScanner
from src.scanners.momentum_scanner  import MomentumScanner
from src.scanners.breakout_scanner  import BreakoutScanner

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Watchlist ────────────────────────────────────────────────────────────────
# Expand this to S&P 500 / Russell 2000 for full market scan
SCAN_UNIVERSE = [
    # GO tier from backtest sweep
    "MSFT","GOOGL","TSLA","XLC","XLV","XLRE","GLD","SPY",
    # WATCH tier worth scanning
    "QQQ","XLK","USO","VTI","XLF","META","AMZN","IBIT","FBTC",
    # High-momentum names
    "NVDA","NFLX","ADBE","CRM","AMD","ORCL","V","MA","UNH","JPM",
    "COIN","HOOD","SOFI","UBER","ABNB","SHOP","CRWD","PANW","SNOW","DDOG",
]


def run_scans(universe: list[str] = SCAN_UNIVERSE) -> dict:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    print(f"\n{'='*60}")
    print(f"  QuantEdge Pre-Market Scanner — {now}")
    print(f"  Universe: {len(universe)} symbols")
    print(f"{'='*60}\n")

    results = {}

    # ── Gap Scanner ──────────────────────────────────────────────────────────
    print("🔍 Running Gap Scanner...")
    gap_scanner = GapScanner(min_gap_pct=2.0, min_volume_ratio=0.03, max_candidates=10)
    gap_hits    = gap_scanner.scan(universe)
    results["gap"] = [vars(c) for c in gap_hits]

    if gap_hits:
        print(f"\n  📈 Top Gap Candidates ({len(gap_hits)} found):")
        for c in gap_hits:
            arrow = "↑" if c.direction == "long" else "↓"
            print(f"    {arrow} {c.symbol:<8} gap={c.gap_pct:+.1f}%  "
                  f"entry=${c.suggested_entry}  stop=${c.suggested_stop}  "
                  f"target=${c.suggested_target}  score={c.score}")
    else:
        print("  No gap candidates found (market may not be in pre-market hours)")

    # ── Momentum Scanner ─────────────────────────────────────────────────────
    print("\n🔍 Running Momentum Scanner...")
    mom_scanner = MomentumScanner(max_candidates=10)
    mom_hits    = mom_scanner.scan(universe)
    results["momentum"] = [vars(c) for c in mom_hits]

    if mom_hits:
        print(f"\n  🚀 Top Momentum Candidates ({len(mom_hits)} found):")
        for c in mom_hits:
            arrow = "↑" if c.direction == "long" else "↓"
            print(f"    {arrow} {c.symbol:<8} 5d={c.momentum_5d:+.1f}%  "
                  f"10d={c.momentum_10d:+.1f}%  RSI={c.rsi_14:.0f}  score={c.score}")
    else:
        print("  No momentum candidates found")

    # ── Breakout Scanner ─────────────────────────────────────────────────────
    print("\n🔍 Running Breakout Scanner...")
    bo_scanner = BreakoutScanner(min_breakout_pct=0.5, min_volume_ratio=1.3, max_candidates=10)
    bo_hits    = bo_scanner.scan(universe)
    results["breakout"] = [vars(c) for c in bo_hits]

    if bo_hits:
        print(f"\n  💥 Top Breakout Candidates ({len(bo_hits)} found):")
        for c in bo_hits:
            print(f"    ↑ {c.symbol:<8} breakout={c.breakout_pct:+.1f}%  "
                  f"vol_ratio={c.volume_ratio:.1f}x  "
                  f"entry=${c.suggested_entry}  stop=${c.suggested_stop}  "
                  f"target=${c.suggested_target}  score={c.score}")
    else:
        print("  No breakout candidates found")

    # ── High-Conviction: appears in 2+ scanners ──────────────────────────────
    gap_syms = {c.symbol for c in gap_hits}
    mom_syms = {c.symbol for c in mom_hits if c.direction == "long"}
    bo_syms  = {c.symbol for c in bo_hits}

    conviction = (gap_syms & mom_syms) | (gap_syms & bo_syms) | (mom_syms & bo_syms)
    results["high_conviction"] = list(conviction)

    if conviction:
        print(f"\n  ⭐ HIGH CONVICTION (2+ scanners): {', '.join(sorted(conviction))}")

    # ── Save results ─────────────────────────────────────────────────────────
    out_path = f"scan_results_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M')}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved → {out_path}")
    print(f"{'='*60}\n")

    return results


if __name__ == "__main__":
    run_scans()
