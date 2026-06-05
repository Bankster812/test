"""
QAAuditAgent — final audit before anything reaches the human (playbook)
=======================================================================

Fails any deal object containing arithmetic errors, impossible economics, or
missing-but-required artifacts. Deterministic (no LLM) so the check itself is
trustworthy. Runs each tick over recent deals and stamps `deal.qa`.
"""

from __future__ import annotations

from .. import config, policy
from ..core.models import Stage
from .base import BaseAgent

_AUDIT_PER_TICK = 6


class QAAuditAgent(BaseAgent):
    code = "QA"
    name = "Quinn (QA & Audit)"
    role = "Final correctness & compliance audit"
    team = "Governance"
    desc = ("Audits every deal for arithmetic errors, impossible spreads, policy "
            "violations, and missing disclosures before it reaches the CEO.")
    color = "#e0e0e0"
    owns = ()

    def run(self) -> None:
        recent = [d for d in self.company.deals if not d.stage.is_terminal][-_AUDIT_PER_TICK:]
        if not recent:
            return
        self._set("acting", f"Auditing {len(recent)} deal(s)", None)
        fails = 0
        for d in recent:
            issues = self.audit(d)
            d.qa = {"pass_fail": "fail" if issues else "pass", "issues": issues}
            if issues:
                fails += 1
        self.handled += 1
        if fails:
            self.say(f"Audit flagged {fails} deal(s) with issues.", level="warn")

    def audit(self, deal) -> list[str]:
        issues: list[str] = []
        uw = deal.uw

        # Policy: never have a live deal in a blocked state.
        if not deal.stage.is_terminal and not policy.is_operable(deal.prop.state):
            issues.append(f"Deal active in non-operable state {deal.prop.state}.")

        # Arithmetic: assignment fee must equal buyer ceiling minus contract price.
        if deal.contract_price and deal.assignment_fee:
            ceiling = uw.mao + config.TARGET_ASSIGNMENT_FEE
            expected = ceiling - deal.contract_price
            if abs(expected - deal.assignment_fee) > 3_000:
                issues.append(
                    f"Fee math off: expected ~${expected:,}, got ${deal.assignment_fee:,}.")
            if deal.assignment_fee < 0:
                issues.append("Negative assignment fee.")
            if deal.contract_price > uw.arv:
                issues.append("Contract price exceeds ARV.")

        # Required artifacts by stage.
        if deal.stage in (Stage.DISPOSITION, Stage.ASSIGNED, Stage.CLOSING):
            if not deal.compliance_done:
                issues.append("Past compliance stage without a compliance review.")
            if not deal.legal_memo:
                issues.append("Missing legal memo at/after disposition.")

        # Thin-margin sanity.
        if deal.stage == Stage.ASSIGNED and deal.assignment_fee < config.MIN_ASSIGNMENT_FEE:
            issues.append(
                f"Assigned below minimum fee (${deal.assignment_fee:,} < "
                f"${config.MIN_ASSIGNMENT_FEE:,}).")
        return issues
