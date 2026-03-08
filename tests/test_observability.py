"""
Tests for Phase 5 — Observability Layer.
Run with: pytest tests/test_observability.py -v
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.observability.slack_notifier import SlackNotifier, AlertLevel, _COLORS, _ICONS


class TestSlackNotifier:
    @pytest.mark.asyncio
    async def test_disabled_when_no_url(self):
        """Notifier with no URL should not attempt HTTP calls."""
        notifier = SlackNotifier(webhook_url=None)
        # Should not raise
        await notifier.send_startup(["AAPL", "MSFT"])
        await notifier.close()

    @pytest.mark.asyncio
    async def test_send_fill_posts_to_webhook(self):
        posted = {}
        async def fake_post(url, json):
            posted["payload"] = json
            resp = MagicMock()
            resp.status_code = 200
            return resp

        notifier = SlackNotifier(webhook_url="https://hooks.slack.com/fake")
        notifier._client.post = fake_post

        await notifier.send_fill(
            symbol="AAPL",
            side="buy",
            qty=10.0,
            price=150.0,
            tp_price=153.0,
            sl_price=148.5,
            signal_score=0.72,
            regime="trending_up",
        )
        assert "payload" in posted
        # Verify structure
        payload = posted["payload"]
        assert "attachments" in payload
        blocks = payload["attachments"][0]["blocks"]
        header_text = blocks[0]["text"]["text"]
        assert "AAPL" in header_text

    @pytest.mark.asyncio
    async def test_send_breach_uses_red_color(self):
        posted = {}
        async def fake_post(url, json):
            posted["payload"] = json
            resp = MagicMock()
            resp.status_code = 200
            return resp

        notifier = SlackNotifier(webhook_url="https://hooks.slack.com/fake")
        notifier._client.post = fake_post

        await notifier.send_portfolio_breach(
            symbol="SPY",
            reason="max_drawdown",
            equity=48000.0,
            drawdown_pct=0.055,
        )
        color = posted["payload"]["attachments"][0]["color"]
        assert color == _COLORS["breach"]  # Red

    @pytest.mark.asyncio
    async def test_failure_does_not_propagate(self):
        """HTTP failure must never raise — just log."""
        notifier = SlackNotifier(webhook_url="https://hooks.slack.com/fake")

        async def failing_post(url, json):
            raise ConnectionError("network down")

        notifier._client.post = failing_post

        # Must not raise
        await notifier.send_error("test_component", "some error")
        await notifier.close()

    @pytest.mark.asyncio
    async def test_daily_summary_fields(self):
        posted = {}
        async def fake_post(url, json):
            posted["payload"] = json
            resp = MagicMock()
            resp.status_code = 200
            return resp

        notifier = SlackNotifier(webhook_url="https://hooks.slack.com/fake")
        notifier._client.post = fake_post

        await notifier.send_daily_summary(
            pnl=1250.50,
            trades=15,
            win_rate=0.667,
            equity=51250.0,
            top_winners=[{"symbol": "NVDA", "pnl": 450.0}],
            top_losers=[{"symbol": "TSLA", "pnl": -120.0}],
        )
        payload_str = str(posted["payload"])
        assert "1,250" in payload_str   # PnL formatted
        assert "15"    in payload_str   # trades
        assert "NVDA"  in payload_str
        assert "TSLA"  in payload_str

    def test_all_alert_levels_have_icon_and_color(self):
        """Every AlertLevel must have a corresponding icon and color."""
        for level in AlertLevel:
            assert level.value in _ICONS,  f"Missing icon for {level}"
            assert level.value in _COLORS, f"Missing color for {level}"

    def test_payload_structure(self):
        """_build_payload must return valid Block Kit structure."""
        payload = SlackNotifier._build_payload(
            level=AlertLevel.FILL,
            title="Test Order",
            fields={"Symbol": "AAPL", "Qty": "10", "Price": "$150"},
        )
        assert "attachments" in payload
        attachment = payload["attachments"][0]
        assert "color"  in attachment
        assert "blocks" in attachment
        # First block must be header
        assert attachment["blocks"][0]["type"] == "header"
