"""ComplianceAgent — legal/risk gate (owns UNDER_CONTRACT).

Reviews each signed contract before it can be marketed: assignment
mechanics, state disclosure rules, and risk flags. Deals above the CEO
approval threshold (or carrying serious flags) are held in the CEO queue
until the human operator decides.
"""

from __future__ import annotations

from ..core.models import Stage
from .base import BaseAgent

# Illustrative state-specific wholesaling notes (NOT legal advice — the seam
# where a real compliance ruleset/counsel review would plug in).
_STATE_NOTES = {
    "FL": "FL: assignor must have equitable interest; disclose assignment in contract.",
    "TX": "TX: written disclosure of equitable-interest position required to buyer.",
    "GA": "GA: ensure no unlicensed brokerage; assign contract, don't market the property.",
    "OH": "OH: avoid advertising the property itself; market the contract/equitable interest.",
    "NC": "NC: contract assignment permitted; verify not acting as unlicensed broker.",
}


class ComplianceAgent(BaseAgent):
    code = "COMPLY"
    name = "Dana (Compliance & Risk)"
    role = "Legal review & CEO escalation gate"
    color = "#9b5de5"
    owns = (Stage.UNDER_CONTRACT,)

    def handle(self, deal) -> None:
        if not deal.compliance_done:
            self._review(deal)
            return

        # Review done — resolve the CEO gate.
        if deal.needs_ceo and deal.ceo_decision is None:
            self._set("idle", f"Awaiting CEO sign-off on #{deal.id}", deal.id)
            return
        if deal.ceo_decision == "rejected":
            deal.stage = Stage.DEAD
            deal.log(self.name, "CEO rejected — deal killed.")
            self.say(f"CEO rejected {deal.label}. Closed out.", level="warn",
                     deal_id=deal.id)
            self.handled += 1
            return

        deal.stage = Stage.DISPOSITION
        deal.log(self.name, "Cleared compliance → disposition.")
        self.say(f"Cleared {deal.label} for disposition.", deal_id=deal.id)
        self.handled += 1

    def _review(self, deal) -> None:
        self._set("acting", f"Reviewing contract #{deal.id}", deal.id)
        st = deal.prop.state
        note = _STATE_NOTES.get(st, f"{st}: confirm assignment legality & disclosure.")
        deal.flags = [note]

        # Risk flags.
        if deal.uw.rehab_ratio > 0.4:
            deal.flags.append("Heavy rehab — buyer scope risk.")
        if deal.assignment_fee < 6_000:
            deal.flags.append("Thin assignment fee — margin risk.")
        if deal.contract_price >= self.company.cfg.CEO_APPROVAL_THRESHOLD:
            deal.needs_ceo = True
            deal.flags.append(f"Contract ≥ ${self.company.cfg.CEO_APPROVAL_THRESHOLD:,.0f} "
                              f"— CEO approval required.")

        # Pull a legal-research memo from the LegalAnalyst sub-agent.
        legal = self.company.agents.get("LEGAL")
        if legal is not None:
            deal.legal_memo = legal.analyze(
                st, deal_type="wholesale assignment",
                pre_foreclosure=("foreclosure" in deal.prop.distress.lower()),
            )

        deal.compliance_done = True
        deal.log(self.name, "Compliance review: " + " ".join(deal.flags))

        if deal.needs_ceo:
            self.company.integrations.slack_alert(
                channel="deals",
                text=f"🟣 CEO approval needed: {deal.label} at ${deal.contract_price:,} "
                     f"(fee ${deal.assignment_fee:,}).",
            )
            self.say(f"Escalated {deal.label} to CEO ({' | '.join(deal.flags)}).",
                     level="escalate", deal_id=deal.id)
        else:
            self.say(f"Reviewed {deal.label}: {deal.flags[0]}", deal_id=deal.id)
        self.handled += 1
