"""
QuantEdge — Options Executor (Phase 9)

STATUS: DISABLED
Options overlay removed in Phase 9 — 6.9% win rate, -$601 avg per trade.
This file is kept for reference only. No code here is called.

To re-enable in a future phase:
1. Restore route_signal() options branch in signal_router.py
2. Re-implement contract selection and sizing below
3. Update eod_reconciler.py with options close logic
"""

raise ImportError(
    "options_executor is disabled in Phase 9. "
    "Options overlay removed due to poor performance (6.9% WR, -$601 avg). "
    "All signals route to shares only."
)
