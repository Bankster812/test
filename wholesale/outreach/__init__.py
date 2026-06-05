"""Compliant B2B outreach drafts (realtors, co-wholesale partners, cash buyers).

These target BUSINESS contacts (licensed agents, active investors), not
distressed homeowners — so they sit outside the homeowner-protection regime.
Calls/texts must still honour TCPA/DNC; email business contacts per CAN-SPAM
(clear identity + opt-out). Nothing here sends; it drafts for your review.
"""

from __future__ import annotations

from .templates import draft, TEMPLATES

__all__ = ["draft", "TEMPLATES"]
