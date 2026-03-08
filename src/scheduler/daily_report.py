"""
QuantEdge — Daily Report (Phase 9)

Changes from Phase 8:
- Options P&L section removed
- Added hold_cap exit reason to summary
- Added VIX level to report header
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone

import requests

from src.scheduler.state_manager import get_state
from src.strategies.signal_router import get_vix

logger = logging.getLogger(__name__)

SLACK_WEBHOOK = os.getenv("SLACK_WEBHOOK_URL", "")


def _slack(msg: str) -> None:
    if not SLACK_WEBHOOK:
        logger.info("slack_report_disabled (no webhook configured)")
        print(msg)
        return
    try:
        requests.post(SLACK_WEBHOOK, json={"text": msg}, timeout=5)
    except Exception as exc:
        logger.warning("slack_send_failed", extra={"error": str(exc)})


async def send_daily_report() -> None:
    state   = get_state()
    trades  = state.get("trade_log", [])
    daily   = state.get("daily_pnl", 0.0)
    vix     = get_vix()

    if not trades:
        _slack(f"*QuantEdge Phase 9 — EOD Report*
No trades today.")
        return

    wins    = [t for t in trades if t["pnl"] > 0]
    losses  = [t for t in trades if t["pnl"] <= 0]
    wr      = len(wins) / len(trades) * 100 if trades else 0

    exit_counts = {}
    for t in trades:
        r = t.get("exit_reason", "unknown")
        exit_counts[r] = exit_counts.get(r, 0) + 1

    lines = [
        f"*QuantEdge Phase 9 — EOD Report* | {datetime.now(timezone.utc).strftime('%Y-%m-%d')}",
        f"VIX: {vix:.1f}" if vix else "VIX: N/A",
        f"",
        f"*Trades:* {len(trades)} | *Win Rate:* {wr:.1f}%",
        f"*Daily P&L:* ${daily:+,.2f}",
        f"*Wins:* {len(wins)} avg ${sum(t['pnl'] for t in wins)/max(len(wins),1):+.0f}",
        f"*Losses:* {len(losses)} avg ${sum(t['pnl'] for t in losses)/max(len(losses),1):+.0f}",
        f"",
        f"*Exit Breakdown:*",
    ]
    for reason, count in sorted(exit_counts.items()):
        lines.append(f"  • {reason}: {count}")

    if exit_counts.get("hold_cap", 0) > 0:
        lines.append(f"
⚠️ {exit_counts['hold_cap']} position(s) hit 100-bar hold cap today")

    lines += [
        f"",
        f"*Trade Log:*",
    ]
    for t in sorted(trades, key=lambda x: x["pnl"]):
        lines.append(
            f"  {t['symbol']:<6} {t['direction']:<5} "
            f"${t['pnl']:>+8.2f}  [{t.get('exit_reason','?')}]  "
            f"bars={t.get('bars_held',0)}"
        )

    _slack("
".join(lines))
    logger.info("daily_report_sent", extra={"trades": len(trades), "pnl": daily})


if __name__ == "__main__":
    import asyncio
    logging.basicConfig(level=logging.INFO)
    asyncio.run(send_daily_report())
