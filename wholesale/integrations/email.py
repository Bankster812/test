"""
email — seller-outreach email adapter (dormant)
===============================================

Drafts go to the dashboard outbox; nothing is sent in dry-run. When armed,
`_live_send` is the single seam to a real transport (Gmail MCP draft/send,
or SMTP).
"""

from __future__ import annotations

from .base import Adapter, DispatchResult


class EmailAdapter(Adapter):
    channel = "email"

    def send(self, to: str, subject: str, body: str) -> DispatchResult:
        payload = {"to": to, "subject": subject, "body": body}
        preview = body.replace("\n", " ")[:80]
        summary = f"Email -> {to}: \"{subject}\" — {preview}…"
        delivered = self._dispatch(self._live_send, payload)
        return self._result("send", to, summary, payload, delivered)

    def _live_send(self, payload: dict) -> None:  # pragma: no cover
        raise NotImplementedError("Live email transport not wired — arm deliberately.")
