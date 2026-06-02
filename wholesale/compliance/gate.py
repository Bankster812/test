"""
gate — pre-flight compliance checks for REAL operation
======================================================

The simulated company runs freely. The moment real homeowners would be
contacted or real contracts executed, this gate stands in the way. It does
NOT give legal advice and is NOT a substitute for counsel — it is a hard
checklist that blocks live outreach until the operator has attested, per
state, that the legal prerequisites are met.

Why this exists: pre-foreclosure outreach and contract assignment are
heavily regulated and vary by state. Getting this wrong carries civil and
sometimes criminal liability. Default posture: BLOCKED.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class StateRule:
    """A compact, NON-EXHAUSTIVE summary of why a state needs counsel sign-off.

    These are pointers for your attorney, not legal conclusions. Verify the
    current statute — these laws change.
    """
    state: str
    wholesale_note: str
    foreclosure_note: str
    license_risk: str          # "high" | "medium" | "low"


# Illustrative pointers for the company's launch markets. Confirm with counsel.
STATE_RULES: dict[str, StateRule] = {
    "TX": StateRule("TX",
        "Equitable interest must be disclosed in writing to the buyer; advertising "
        "the property itself (vs. the contract) can constitute unlicensed brokerage.",
        "Pre-foreclosure has statutory notice/timing rules; tread carefully.",
        "medium"),
    "FL": StateRule("FL",
        "Assignor must hold genuine equitable interest; market the contract, not "
        "the property, to avoid unlicensed brokerage.",
        "Foreclosure-rescue/equity-purchaser conduct is scrutinized.",
        "medium"),
    "GA": StateRule("GA",
        "Recent legislation has tightened wholesaling; confirm current licensing "
        "requirements before any outreach.",
        "Notice-of-default timelines apply; counsel review required.",
        "high"),
    "OH": StateRule("OH",
        "Advertise the equitable interest/contract, not the property; avoid acting "
        "as an unlicensed broker.",
        "Judicial foreclosure state; confirm permissible contact windows.",
        "medium"),
    "NC": StateRule("NC",
        "Assignment permitted but anti-brokerage line is strict; verify you are a "
        "principal, not a broker.",
        "Confirm pre-foreclosure contact rules with counsel.",
        "medium"),
}

# The attestations the operator must explicitly confirm, per state, before the
# company may contact real owners or execute real contracts there.
REQUIRED_ATTESTATIONS = (
    "entity_formed",            # operating LLC exists and is in good standing
    "licensed_or_exempt",       # licensed where required, or confirmed exempt by counsel
    "attorney_engaged",         # a real-estate attorney reviews contracts in this state
    "contracts_reviewed",       # the PSA/assignment templates were attorney-approved
    "foreclosure_law_reviewed", # equity-purchaser/foreclosure-consultant rules cleared
    "tcpa_dnc_process",         # calling/texting honors TCPA + Do-Not-Call scrubbing
    "data_source_compliant",    # lead/contact data obtained via ToS/FCRA-compliant means
)


class ComplianceGate:
    """Blocks real outreach until per-state attestations are satisfied."""

    def __init__(self) -> None:
        # state -> set of satisfied attestation keys
        self._attested: dict[str, set[str]] = {}

    def attest(self, state: str, **flags: bool) -> None:
        """Operator records confirmations, e.g. gate.attest('TX', entity_formed=True...)."""
        bucket = self._attested.setdefault(state.upper(), set())
        for key, ok in flags.items():
            if key not in REQUIRED_ATTESTATIONS:
                raise ValueError(f"Unknown attestation: {key}")
            if ok:
                bucket.add(key)
            else:
                bucket.discard(key)

    def missing(self, state: str) -> list[str]:
        have = self._attested.get(state.upper(), set())
        return [a for a in REQUIRED_ATTESTATIONS if a not in have]

    def can_contact_real_owners(self, state: str) -> bool:
        return not self.missing(state)

    def explain(self, state: str) -> str:
        st = state.upper()
        rule = STATE_RULES.get(st)
        miss = self.missing(st)
        head = f"[{st}] " + ("CLEARED for live outreach." if not miss
                             else f"BLOCKED — missing: {', '.join(miss)}.")
        if rule:
            head += (f"\n  wholesale: {rule.wholesale_note}"
                     f"\n  foreclosure: {rule.foreclosure_note}"
                     f"\n  license risk: {rule.license_risk}")
        return head
