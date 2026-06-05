"""DispoAgent — disposition / buyer matching (owns DISPOSITION → ASSIGNED).

Markets the *contract* (not the property) to the cash-buyer book and
matches the best-fit, highest-appetite buyer. If nothing fits, it shaves
our assignment fee to widen the pool before giving up.
"""

from __future__ import annotations

from ..core.models import Stage
from .base import BaseAgent

_MAX_ROUNDS = 3


class DispoAgent(BaseAgent):
    code = "DISPO"
    name = "Theo (Disposition Manager)"
    role = "Cash-buyer matching & assignment"
    team = "Disposition"
    desc = ("Markets the contract to the cash-buyer book and matching dispo "
            "platforms, and assigns to the best-fit buyer.")
    color = "#43aa8b"
    owns = (Stage.DISPOSITION,)

    def _fit(self, deal, buyer) -> bool:
        return (
            deal.prop.metro in buyer.metros
            and deal.prop.property_type in buyer.property_types
            and (deal.contract_price + buyer.min_spread) <= buyer.max_price
            and deal.assignment_fee >= buyer.min_spread
        )

    def handle(self, deal) -> None:
        self._set("acting", f"Marketing {deal.prop.address} to buyers", deal.id)
        first_time = deal.dispo_rounds == 0
        deal.dispo_rounds += 1
        rng = self.company.rng

        # On first marketing, submit the buyer summary to disposition platforms
        # (dry-run unless armed) so the deal is listed where buyers are.
        if first_time:
            from ..disposition.submission import buyer_summary
            summary = buyer_summary(deal, self.company)
            for plat, hint in (("OfferMarket", "free marketplace"),
                               ("DispoBridge", "pay-on-close intake")):
                self.company.integrations.dispo_submit(
                    platform=plat, deal_id=deal.id, summary=summary, channel_hint=hint)

        candidates = [b for b in self.company.buyers if self._fit(deal, b)]
        # Weight by appetite; verified-cash buyers preferred.
        rng.shuffle(candidates)
        candidates.sort(key=lambda b: (b.cash_verified, b.appetite), reverse=True)

        for b in candidates:
            if rng.random() <= b.appetite:
                deal.buyer = b
                deal.stage = Stage.ASSIGNED
                deal.log(self.name, f"Assigned to {b.name} — fee ${deal.assignment_fee:,}.")
                self.company.integrations.crm_upsert(
                    deal_id=deal.id, name=deal.prop.address, stage="assigned",
                    amount=deal.contract_price, contact=b.name,
                )
                self.say(f"Matched {deal.label} → {b.name} "
                         f"(fee ${deal.assignment_fee:,}).", level="win", deal_id=deal.id)
                self.handled += 1
                return

        # No taker this round.
        if deal.dispo_rounds >= _MAX_ROUNDS:
            deal.stage = Stage.DEAD
            deal.log(self.name, "No buyer after 3 rounds — released contract.")
            self.say(f"Couldn't place {deal.label} — released.", level="warn",
                     deal_id=deal.id)
        else:
            # Shave $2k off our fee to widen the pool next round.
            shave = min(2_000, deal.assignment_fee)
            deal.assignment_fee -= shave
            deal.log(self.name, f"Round {deal.dispo_rounds}: no taker, fee → "
                                f"${deal.assignment_fee:,}.")
            self.say(f"Re-marketing {deal.label} (round {deal.dispo_rounds}).",
                     deal_id=deal.id)
        self.handled += 1
