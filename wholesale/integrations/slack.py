"""
slack — internal alerts adapter (dormant)
=========================================

Used for CEO escalations and big wins. Dry-run by default; `_live_post`
is the seam to the Slack MCP send_message tool or an incoming webhook.
"""

from __future__ import annotations

from .base import Adapter, DispatchResult


class SlackAdapter(Adapter):
    channel = "slack"

    def post(self, channel: str, text: str) -> DispatchResult:
        payload = {"channel": channel, "text": text}
        summary = f"Slack #{channel}: {text[:90]}"
        delivered = self._dispatch(self._live_post, payload)
        return self._result("post", channel, summary, payload, delivered)

    def _live_post(self, payload: dict) -> None:  # pragma: no cover
        raise NotImplementedError("Live Slack transport not wired — arm deliberately.")
