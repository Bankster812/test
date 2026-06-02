"""CloserAgent — seller outreach & price negotiation (owns OUTREACH, NEGOTIATION).

Drafts the seller's first-touch message (LLM), then runs a bounded
negotiation. The seller's behaviour is modelled from their motivation and
flexibility; the agent will only sign at or below the underwritten MAO.
"""

from __future__ import annotations

from .. import config
from ..core.models import Stage
from .base import BaseAgent

_MAX_ROUNDS = 3


class CloserAgent(BaseAgent):
    code = "CLOSER"
    name = "Sofia (Acquisition Manager)"
    role = "Seller outreach & negotiation"
    color = "#f3722c"
    owns = (Stage.OUTREACH, Stage.NEGOTIATION)

    def handle(self, deal) -> None:
        if deal.stage == Stage.OUTREACH:
            self._outreach(deal)
        else:
            self._negotiate(deal)

    # -- first touch --------------------------------------------------------
    def _outreach(self, deal) -> None:
        self._set("acting", f"Reaching out to {deal.seller.name}", deal.id)
        s, p = deal.seller, deal.prop
        body = self.reason(
            system="You are an acquisition manager at a real-estate wholesaling firm. "
                   "Write a SHORT, warm, no-pressure first-touch message to a motivated "
                   "seller offering a fast, as-is cash close. 3 sentences max. No price yet.",
            user=f"Seller {s.name}, motivation: {s.motivation}. Property: "
                 f"{p.address}, {p.city} {p.state}, {p.beds}bd/{p.baths}ba.",
            max_tokens=160,
            fallback=(f"Hi {s.name.split()[0]}, I buy homes in {p.city} for cash and can "
                      f"close on your timeline, as-is — no repairs or fees. Would you be "
                      f"open to a quick no-obligation offer on {p.address}?"),
        )
        self.company.integrations.email_seller(
            to=f"{s.name.split()[0].lower()}@example.com",
            subject=f"Cash offer for {p.address}?",
            body=body,
        )
        deal.log(self.name, f"First-touch via {s.reachable_via}: {body[:80]}…")
        deal.stage = Stage.NEGOTIATION
        self.say(f"Contacted {s.name} re {deal.label} → negotiating.", deal_id=deal.id)
        self.handled += 1

    # -- price discovery ----------------------------------------------------
    def _negotiate(self, deal) -> None:
        self._set("thinking", f"Negotiating {deal.prop.address}", deal.id)
        s, uw = deal.seller, deal.uw
        deal.nego_rounds += 1
        rng = self.company.rng

        # The cash buyer's all-in ceiling (their own 70% rule). Our spread lives
        # between what we pay the seller and this number.
        buyer_ceiling = uw.mao + config.TARGET_ASSIGNMENT_FEE

        # Seller softens a touch each round; their hidden floor is the real wall.
        floor = int(round(s.walk_floor * (1 - 0.02 * deal.nego_rounds), -3))
        our_offer = min(uw.target_offer + deal.nego_rounds * 3_000, uw.mao)

        if our_offer >= floor:
            # Settle between their floor and our ceiling offer (we keep the rest).
            price = int(round((our_offer + floor) / 2, -3))
            price = min(price, uw.mao)
            deal.contract_price = price
            deal.assignment_fee = max(0, buyer_ceiling - price)
            deal.stage = Stage.UNDER_CONTRACT
            deal.needs_ceo = price >= config.CEO_APPROVAL_THRESHOLD
            deal.log(self.name, f"Agreed ${price:,} (floor ${floor:,}). "
                                f"Projected fee ${deal.assignment_fee:,}.")
            self.company.integrations.crm_upsert(
                deal_id=deal.id, name=deal.prop.address, stage="under_contract",
                amount=price, contact=s.name,
            )
            lvl = "escalate" if deal.needs_ceo else "win"
            tail = " — needs CEO sign-off." if deal.needs_ceo else "."
            self.say(f"Locked {deal.label} under contract at ${price:,}{tail}",
                     level=lvl, deal_id=deal.id)
            self.handled += 1
            return

        # No deal yet. Keep trying, or walk after max rounds.
        if deal.nego_rounds >= _MAX_ROUNDS:
            deal.stage = Stage.DEAD
            deal.log(self.name, f"Walked: seller floor ${floor:,} above our MAO "
                                f"${uw.mao:,}.")
            self.say(f"Couldn't bridge {deal.label} (floor ${floor:,} > MAO "
                     f"${uw.mao:,}).", level="warn", deal_id=deal.id)
        else:
            deal.log(self.name, f"Round {deal.nego_rounds}: offered ${our_offer:,}, "
                                f"seller floor ~${floor:,}. Holding.")
            self.say(f"Negotiating {deal.label}: round {deal.nego_rounds}.",
                     deal_id=deal.id)
        self.handled += 1
