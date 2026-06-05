"""
dispo — submit a contract to a disposition platform (DispoBridge, etc.)
=======================================================================

Dry-run by default: records the buyer-facing summary that WOULD be sent.
When armed and a transport is wired (`_live_submit`), it submits to the
platform (most use an email intake or a web form; some expose an API).
"""

from __future__ import annotations

from typing import Any

from .base import Adapter, DispatchResult


class DispoAdapter(Adapter):
    channel = "dispo"

    def submit(self, platform: str, deal_id: int, summary: str,
               channel_hint: str = "") -> DispatchResult:
        payload = {"platform": platform, "deal_id": deal_id,
                   "buyer_summary": summary, "channel": channel_hint}
        line = f"Submit #{deal_id} to {platform}"
        if channel_hint:
            line += f" via {channel_hint}"
        delivered = self._dispatch(self._live_submit, payload)
        return self._result("submit", platform, line, payload, delivered)

    def _live_submit(self, payload: dict[str, Any]) -> None:  # pragma: no cover
        # Wire the platform's intake here (their API / email intake / form).
        raise NotImplementedError("Live dispo submission not wired — arm deliberately.")
