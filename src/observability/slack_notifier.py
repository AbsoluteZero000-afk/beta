"""
QuantEdge — Slack Notifier.

Sends structured Slack alerts via Incoming Webhooks using httpx async client.
No Slack SDK dependency — plain HTTPS POST with JSON payload.

Alert types:
  - order_fill       — bracket order filled
  - portfolio_breach — risk gate triggered
  - daily_summary    — end-of-day PnL report
  - system_error     — unhandled exception

Slack Incoming Webhook docs:
  https://docs.slack.dev/messaging/sending-messages-using-incoming-webhooks

httpx async docs:
  https://www.python-httpx.org/async/
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

import httpx
import structlog

logger = structlog.get_logger(__name__)

# Emoji map for alert types
_ICONS = {
    "fill":    "✅",
    "breach":  "🚨",
    "summary": "📊",
    "error":   "❌",
    "info":    "ℹ️",
    "startup": "🚀",
}

# Color map for Slack attachment side-bar color
_COLORS = {
    "fill":    "#36a64f",   # Green
    "breach":  "#ff0000",   # Red
    "summary": "#439fe0",   # Blue
    "error":   "#ff6600",   # Orange
    "info":    "#dddddd",   # Gray
    "startup": "#7c3aed",   # Purple
}


class AlertLevel(str, Enum):
    FILL    = "fill"
    BREACH  = "breach"
    SUMMARY = "summary"
    ERROR   = "error"
    INFO    = "info"
    STARTUP = "startup"


class SlackNotifier:
    """
    Async Slack notifier using httpx.AsyncClient.

    Uses a persistent async HTTP client with connection pooling.
    All methods are fire-and-forget — failures are logged but never
    propagate to the trading pipeline.

    Usage:
        notifier = SlackNotifier(webhook_url)
        await notifier.send_fill("AAPL", "buy", qty=10, price=150.0, pnl=None)
        await notifier.send_daily_summary(pnl=1250.0, trades=12, win_rate=0.67)
        await notifier.close()   # on shutdown
    """

    def __init__(
        self,
        webhook_url: Optional[str],
        timeout: float = 5.0,
        enabled: bool = True,
    ) -> None:
        self._url     = webhook_url
        self._enabled = enabled and bool(webhook_url)
        self._client  = httpx.AsyncClient(timeout=timeout)

        if not self._enabled:
            logger.info("SlackNotifier disabled (no webhook URL configured)")

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()

    # ---- High-level alert methods ----

    async def send_startup(self, watchlist: list[str], mode: str = "paper") -> None:
        await self._post(
            level=AlertLevel.STARTUP,
            title=f"QuantEdge Started ({mode.upper()})",
            fields={
                "Mode":      mode.upper(),
                "Watchlist": f"{len(watchlist)} symbols",
                "Symbols":   ", ".join(watchlist[:8]) + ("..." if len(watchlist) > 8 else ""),
                "Time":      _now_str(),
            },
        )

    async def send_fill(
        self,
        symbol: str,
        side: str,
        qty: float,
        price: float,
        tp_price: float,
        sl_price: float,
        signal_score: float,
        regime: str,
    ) -> None:
        direction = "📈 LONG" if side == "buy" else "📉 SHORT"
        await self._post(
            level=AlertLevel.FILL,
            title=f"Order Filled — {symbol}",
            fields={
                "Symbol":       symbol,
                "Direction":    direction,
                "Qty":          f"{qty:.2f} shares",
                "Fill Price":   f"${price:.2f}",
                "Take Profit":  f"${tp_price:.2f}",
                "Stop Loss":    f"${sl_price:.2f}",
                "Signal Score": f"{signal_score:+.3f}",
                "Regime":       regime,
                "Time":         _now_str(),
            },
        )

    async def send_portfolio_breach(
        self, symbol: str, reason: str, equity: float, drawdown_pct: float
    ) -> None:
        await self._post(
            level=AlertLevel.BREACH,
            title="🚨 Portfolio Breach — Trading Halted",
            fields={
                "Symbol":    symbol,
                "Reason":    reason,
                "Equity":    f"${equity:,.2f}",
                "Drawdown":  f"{drawdown_pct:.2%}",
                "Time":      _now_str(),
            },
        )

    async def send_daily_summary(
        self,
        pnl: float,
        trades: int,
        win_rate: float,
        equity: float,
        top_winners: list[dict],
        top_losers: list[dict],
    ) -> None:
        pnl_emoji = "📈" if pnl >= 0 else "📉"
        fields = {
            "Daily PnL":   f"{pnl_emoji} ${pnl:+,.2f}",
            "Total Trades": str(trades),
            "Win Rate":    f"{win_rate:.1%}",
            "Equity":      f"${equity:,.2f}",
        }
        if top_winners:
            fields["Top Winners"] = " | ".join(
                f"{w['symbol']} +${w['pnl']:.0f}" for w in top_winners[:3]
            )
        if top_losers:
            fields["Top Losers"] = " | ".join(
                f"{l['symbol']} -${abs(l['pnl']):.0f}" for l in top_losers[:3]
            )
        fields["Date"] = _now_str()

        await self._post(
            level=AlertLevel.SUMMARY,
            title="QuantEdge Daily Summary",
            fields=fields,
        )

    async def send_error(self, component: str, error: str, detail: str = "") -> None:
        await self._post(
            level=AlertLevel.ERROR,
            title=f"System Error — {component}",
            fields={
                "Component": component,
                "Error":     error[:200],
                "Detail":    detail[:300] if detail else "—",
                "Time":      _now_str(),
            },
        )

    # ---- Core send ----

    async def _post(
        self,
        level: AlertLevel,
        title: str,
        fields: dict[str, str],
    ) -> None:
        """Build Slack Block Kit payload and POST to webhook."""
        if not self._enabled:
            logger.debug("Slack alert suppressed (disabled): %s", title)
            return

        payload = self._build_payload(level, title, fields)
        try:
            resp = await self._client.post(self._url, json=payload)
            if resp.status_code != 200:
                logger.warning(
                    "Slack webhook returned %d: %s",
                    resp.status_code, resp.text,
                )
        except Exception as exc:
            # NEVER let Slack failures affect the trading pipeline
            logger.warning("Slack notification failed: %s", exc)

    @staticmethod
    def _build_payload(level: AlertLevel, title: str, fields: dict) -> dict:
        """Build Slack Block Kit message with attachment sidebar color."""
        icon  = _ICONS.get(level.value, "ℹ️")
        color = _COLORS.get(level.value, "#dddddd")

        # Build field blocks (2-column layout)
        field_blocks = []
        items = list(fields.items())
        for i in range(0, len(items), 2):
            row = []
            for k, v in items[i:i + 2]:
                row.append({
                    "type": "mrkdwn",
                    "text": "*" + k + "*\n" + v,
                })
            field_blocks.append({"type": "section", "fields": row})

        return {
            "attachments": [
                {
                    "color": color,
                    "blocks": [
                        {
                            "type": "header",
                            "text": {
                                "type": "plain_text",
                                "text": f"{icon}  {title}",
                                "emoji": True,
                            },
                        },
                    ] + field_blocks,
                }
            ]
        }


def _now_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
