"""
LegalAnalyst — AI legal-research sub-agent (informational, NOT legal advice)
===========================================================================

Produces a structured legal-risk TRIAGE for a state + transaction type:
licensing, anti-brokerage/equitable-interest, disclosures, pre-foreclosure
(equity-purchaser/foreclosure-consultant) rules, and TCPA — plus a list of
questions to take to a licensed attorney.

Hard rule baked into every output: this is informational triage to prepare
for counsel; it is NOT legal advice and does not create an attorney-client
relationship. The company still cannot contact real owners in a state until
`compliance.gate` is satisfied (which requires a real attorney engaged).
"""

from __future__ import annotations

from typing import Any

from .base import BaseAgent
from ..compliance.gate import STATE_RULES

DISCLAIMER = ("AI legal-research triage — NOT legal advice, no attorney-client "
              "relationship. Verify with a licensed attorney in the state.")

_SYSTEM = (
    "You are a paralegal-style legal RESEARCH assistant for a US real-estate "
    "wholesaling firm. Produce concise, accurate, source-aware triage on the "
    "legality of assigning purchase contracts in a given state. Never claim to "
    "give legal advice. Flag uncertainty plainly. Emphasise licensing rules, "
    "anti-brokerage/equitable-interest lines, required disclosures, "
    "pre-foreclosure 'equity purchaser'/'foreclosure consultant' statutes "
    "(disclosures, rescission rights, penalties), and TCPA/DNC for outreach. "
    "End with concrete questions to ask a licensed attorney."
)


class LegalAnalyst(BaseAgent):
    code = "LEGAL"
    name = "Counsel-AI (Legal Research)"
    role = "Legal-risk triage — informational, not legal advice"
    team = "Compliance"
    desc = ("Produces per-state legal triage (licensing, disclosures, "
            "pre-foreclosure rules, TCPA) and attorney questions. Not legal advice.")
    color = "#bdb2ff"
    owns = ()  # not a pipeline-stage owner; invoked on demand

    def analyze(self, state: str, deal_type: str = "wholesale assignment",
                pre_foreclosure: bool = False) -> dict[str, Any]:
        self._set("thinking", f"Researching {state} {deal_type}", None)
        st = state.upper()
        rule = STATE_RULES.get(st)

        user = (
            f"State: {st}. Transaction: {deal_type}. "
            f"Pre-foreclosure / distressed seller: {pre_foreclosure}. "
            "Give: (1) license requirement, (2) anti-brokerage/equitable-interest, "
            "(3) disclosures, (4) pre-foreclosure equity-purchaser rules, "
            "(5) TCPA/DNC, (6) red flags, (7) 5 questions for a licensed attorney."
        )
        fallback = self._fallback(st, rule, pre_foreclosure)
        body = self.reason(_SYSTEM, user, max_tokens=600, temperature=0.3,
                           fallback=fallback)

        memo = {
            "state": st,
            "deal_type": deal_type,
            "pre_foreclosure": pre_foreclosure,
            "analysis": body,
            "source": "claude" if self.llm.is_available() else "heuristic",
            "disclaimer": DISCLAIMER,
        }
        self.handled += 1
        self.say(f"Legal triage prepared for {st} ({deal_type}). {DISCLAIMER}",
                 level="info")
        return memo

    def _fallback(self, st: str, rule, pre_foreclosure: bool) -> str:
        lines = [f"{st} wholesaling triage (heuristic — no LLM key set):"]
        if rule:
            lines += [
                f"- Licensing/brokerage risk: {rule.license_risk}.",
                f"- Wholesale note: {rule.wholesale_note}",
                f"- Foreclosure note: {rule.foreclosure_note}",
            ]
        else:
            lines.append("- No state rule on file; confirm licensing and "
                         "assignment legality with counsel.")
        lines.append("- Disclosures: disclose intent to assign for profit; "
                     "market the CONTRACT, not the property.")
        if pre_foreclosure:
            lines.append("- PRE-FORECLOSURE: equity-purchaser/foreclosure-"
                         "consultant statutes likely apply — mandatory written "
                         "contract, specific disclosures, and a rescission "
                         "window; penalties can be severe. Counsel REQUIRED.")
        lines.append("- Outreach: calls/texts must honour TCPA + Do-Not-Call; "
                     "prefer direct mail for first touch.")
        lines.append("Questions for your attorney: (1) Do I need a license to "
                     "wholesale here? (2) How do I keep equitable interest and "
                     "avoid unlicensed brokerage? (3) What seller/buyer "
                     "disclosures are mandatory? (4) What pre-foreclosure "
                     "disclosures/rescission apply? (5) Is my assignment "
                     "contract compliant as drafted?")
        return "\n".join(lines)
