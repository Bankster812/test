"""ScoutAgent — sourcing & lead qualification (owns SOURCED)."""

from __future__ import annotations

from ..core.models import Stage
from .base import BaseAgent


class ScoutAgent(BaseAgent):
    code = "SCOUT"
    name = "Ava (Acquisitions Scout)"
    role = "Sourcing & lead qualification"
    color = "#4cc9f0"
    owns = (Stage.SOURCED,)

    def handle(self, deal) -> None:
        self._set("thinking", f"Qualifying {deal.prop.address}", deal.id)
        p, s = deal.prop, deal.seller

        # Hard screen: is there any conceivable spread? (asking vs market)
        headroom = p.est_market_value - s.asking_price
        if s.asking_price > p.est_market_value * 1.10:
            deal.stage = Stage.DEAD
            deal.log(self.name, "Killed: seller asking far above market, no spread.")
            self.say(f"Passed on {deal.label} — asking ${s.asking_price:,} "
                     f"over market.", level="warn", deal_id=deal.id)
            self.handled += 1
            return

        # Soft work: a one-line qualification note (LLM if available).
        note = self.reason(
            system="You are an acquisitions scout at a US real-estate wholesaling "
                   "firm. In ONE sentence, say why this is a promising motivated-seller "
                   "lead. Be concrete and unsentimental.",
            user=f"{p.beds}bd/{p.baths}ba {p.property_type}, {p.sqft}sqft, built "
                 f"{p.year_built}, {p.city} {p.state}. Distress: {p.distress}. "
                 f"Seller motivation: {s.motivation}. Asking ${s.asking_price:,}, "
                 f"est. market ${p.est_market_value:,}.",
            max_tokens=90,
            fallback=f"{p.distress} signal with ${headroom:,} of apparent headroom "
                     f"vs market — worth underwriting.",
        )
        deal.log(self.name, f"Qualified: {note}")
        deal.stage = Stage.UNDERWRITING
        self.company.integrations.crm_upsert(
            deal_id=deal.id, name=deal.prop.address, stage="sourced",
            amount=s.asking_price, contact=s.name,
        )
        self.say(f"Sourced {deal.label} ({p.distress}) → underwriting.",
                 deal_id=deal.id)
        self.handled += 1
