"""
crm — HubSpot CRM adapter (dormant)
===================================

Mirrors the shape of a HubSpot deal/contact upsert. In dry-run it returns
the payload that *would* be written. When armed, `_live_upsert` is the
single place to wire a real transport (e.g. the operator routes it through
the HubSpot MCP tools, or a private-app token + REST call).
"""

from __future__ import annotations

from typing import Any

from .base import Adapter, DispatchResult


class CRMAdapter(Adapter):
    channel = "crm"

    def upsert_deal(
        self,
        deal_id: int,
        name: str,
        stage: str,
        amount: int,
        contact: str,
        properties: dict[str, Any] | None = None,
    ) -> DispatchResult:
        payload = {
            "objectType": "deal",
            "dealname": f"#{deal_id} {name}",
            "dealstage": stage,
            "amount": amount,
            "associated_contact": contact,
            "properties": properties or {},
        }
        summary = f"CRM upsert deal #{deal_id} -> stage '{stage}', ${amount:,}"
        if self.armed:
            self._live_upsert(payload)
        return self._result("upsert_deal", str(deal_id), summary, payload)

    def _live_upsert(self, payload: dict[str, Any]) -> None:  # pragma: no cover
        # Wire to HubSpot here (MCP manage_crm_objects / REST). Intentionally
        # a no-op until the operator implements live transport.
        raise NotImplementedError("Live CRM transport not wired — arm deliberately.")
