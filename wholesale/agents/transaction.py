"""CoordinatorAgent — transaction coordination (owns ASSIGNED, CLOSING).

Opens title/escrow on an assigned deal, then drives the close. Most close;
a minority fall through (title/financing) and bounce back to disposition.
On funding, the assignment fee is booked as company revenue.
"""

from __future__ import annotations

from ..core.models import Stage
from .base import BaseAgent


class CoordinatorAgent(BaseAgent):
    code = "COORD"
    name = "Priya (Transaction Coordinator)"
    role = "Title, escrow & closing"
    team = "Transaction"
    desc = ("Opens title/escrow on assigned deals and drives closing; books the "
            "assignment fee on funding.")
    color = "#577590"
    owns = (Stage.ASSIGNED, Stage.CLOSING)

    def handle(self, deal) -> None:
        if deal.stage == Stage.ASSIGNED:
            self._open_escrow(deal)
        else:
            self._close(deal)

    def _open_escrow(self, deal) -> None:
        self._set("acting", f"Opening escrow #{deal.id}", deal.id)
        deal.stage = Stage.CLOSING
        deal.log(self.name, f"Opened title/escrow with {deal.buyer.name}; "
                            f"assignment agreement sent.")
        self.say(f"Escrow opened on {deal.label} with {deal.buyer.name}.",
                 deal_id=deal.id)
        self.handled += 1

    def _close(self, deal) -> None:
        self._set("acting", f"Closing #{deal.id}", deal.id)
        rng = self.company.rng

        if rng.random() < 0.85:               # clean close
            deal.stage = Stage.CLOSED_WON
            self.company.book_revenue(deal)
            deal.log(self.name, f"FUNDED. Assignment fee ${deal.assignment_fee:,} "
                                f"collected.")
            self.company.integrations.slack_alert(
                channel="wins",
                text=f"💰 Closed {deal.label}: +${deal.assignment_fee:,} "
                     f"(buyer {deal.buyer.name}).",
            )
            self.say(f"CLOSED {deal.label}: +${deal.assignment_fee:,}.",
                     level="win", deal_id=deal.id)
        elif deal.dispo_rounds < 4:           # fell through — re-assign once
            deal.log(self.name, "Buyer fell through (title/financing). Back to dispo.")
            deal.buyer = None
            deal.stage = Stage.DISPOSITION
            self.say(f"{deal.label} fell through — re-marketing.", level="warn",
                     deal_id=deal.id)
        else:
            deal.stage = Stage.DEAD
            deal.log(self.name, "Repeated fall-through — released.")
            self.say(f"{deal.label} dead after fall-through.", level="warn",
                     deal_id=deal.id)
        self.handled += 1
