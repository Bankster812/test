"""
integrations — outbound adapters (CRM, email, Slack)
====================================================

Every adapter is DORMANT by default: it records the intended action and
returns a dry-run result without touching the network. Nothing is sent
externally unless the company is armed (`config.INTEGRATIONS_ARMED` /
`WS_INTEGRATIONS_ARMED=1`) AND the operator wires the adapter to a live
transport.

This keeps the autonomous company safe-by-default: it will draft the CRM
record, the seller email, and the Slack alert, surface them on the
dashboard, and wait. Arming is a deliberate human (CEO) act.
"""

from __future__ import annotations

from .base import Adapter, DispatchResult
from .crm import CRMAdapter
from .email import EmailAdapter
from .slack import SlackAdapter

__all__ = ["Adapter", "DispatchResult", "CRMAdapter", "EmailAdapter",
           "SlackAdapter", "IntegrationHub"]


class IntegrationHub:
    """Bundles the three adapters and tracks an outbox of dry-run dispatches."""

    def __init__(self, armed: bool = False) -> None:
        self.crm = CRMAdapter(armed=armed)
        self.email = EmailAdapter(armed=armed)
        self.slack = SlackAdapter(armed=armed)
        self.outbox: list[DispatchResult] = []

    def _record(self, r: DispatchResult) -> DispatchResult:
        self.outbox.append(r)
        if len(self.outbox) > 200:
            self.outbox = self.outbox[-200:]
        return r

    def crm_upsert(self, **kw) -> DispatchResult:
        return self._record(self.crm.upsert_deal(**kw))

    def email_seller(self, **kw) -> DispatchResult:
        return self._record(self.email.send(**kw))

    def slack_alert(self, **kw) -> DispatchResult:
        return self._record(self.slack.post(**kw))

    def stats(self) -> dict[str, int]:
        return {
            "queued": len(self.outbox),
            "armed": int(self.crm.armed),
        }
