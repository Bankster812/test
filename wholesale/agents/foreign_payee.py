"""
ForeignPayeeAgent — non-resident payout & tax readiness (playbook Part 9)
========================================================================

Critical for a German, non-U.S.-resident operator: before any closing, the
operator must be able to actually RECEIVE funds and document the transaction.
This agent produces a readiness checklist for CPA/attorney review — it does
NOT give tax or legal advice.

Covers: entity path, EIN/ITIN, W-8BEN-E, FIRPTA/withholding posture, title-
company payout to a foreign payee, and banking/KYC.
"""

from __future__ import annotations

from typing import Any

from .base import BaseAgent

DISCLAIMER = ("Readiness checklist for CPA/attorney review — NOT tax or legal "
              "advice. Confirm withholding/treaty posture with a cross-border CPA.")

_SYSTEM = (
    "You are a foreign-payee readiness assistant for a German non-U.S.-resident "
    "doing U.S. real-estate wholesaling (principal/assignor only). You do NOT "
    "give tax or legal advice. Produce a concise checklist for CPA and attorney "
    "review covering: entity path (US LLC vs foreign vs JV), EIN/ITIN need, tax "
    "forms (e.g. W-8BEN-E), FIRPTA/withholding considerations, title-company "
    "payout to a foreign payee, and banking/KYC. Flag what must be resolved "
    "before the first closing."
)


class ForeignPayeeAgent(BaseAgent):
    code = "FOREIGN"
    name = "Mei (Capital & Foreign-Payee)"
    role = "Non-resident payout & tax readiness"
    team = "Capital"
    desc = ("Builds the non-resident payout/tax readiness checklist (EIN/ITIN, "
            "W-8BEN-E, FIRPTA, title-company payout). Not tax advice.")
    color = "#ffd6a5"
    owns = ()

    def readiness(self) -> dict[str, Any]:
        self._set("acting", "Assessing foreign-payee readiness", None)
        fallback = (
            "Foreign-payee readiness (heuristic — confirm with cross-border CPA):\n"
            "- Entity: a US LLC (single-member, foreign-owned) is common; confirm "
            "vs. foreign entity / JV with counsel + CPA.\n"
            "- EIN: required for the US LLC (Form SS-4). ITIN: may be required for "
            "the individual owner depending on filings (Form W-7).\n"
            "- Tax forms: foreign entity typically provides W-8BEN-E to establish "
            "foreign status/beneficial ownership; treatment depends on entity/payment.\n"
            "- FIRPTA: withholding applies to dispositions of US real property by "
            "foreign persons — confirm whether an assignment fee is in scope.\n"
            "- Title-company payout: confirm IN WRITING the title company will "
            "disburse an assignment fee to a foreign-owned entity, and what KYC "
            "docs they need.\n"
            "- Banking/KYC: US business account or compliant payment rail ready "
            "before first close.\n"
            "Must resolve before first closing: EIN, title-company payout "
            "confirmation, CPA withholding memo, banking ready."
        )
        body = self.reason(_SYSTEM,
                           "Give the readiness checklist and pre-closing blockers "
                           "for a German non-resident operator, first market Texas.",
                           max_tokens=600, temperature=0.3, fallback=fallback)
        self.handled += 1
        self.say("Foreign-payee readiness checklist prepared. " + DISCLAIMER)
        return {
            "analysis": body,
            "source": "claude" if self.llm.is_available() else "heuristic",
            "must_resolve_before_close": [
                "EIN obtained", "Title-company payout confirmed in writing",
                "CPA withholding/FIRPTA memo", "Banking/KYC ready",
                "W-8BEN-E or correct tax form prepared",
            ],
            "disclaimer": DISCLAIMER,
        }
