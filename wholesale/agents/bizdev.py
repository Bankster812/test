"""
BizDevAgent ("Riley") — B2B outreach to realtors, partners & cash buyers
========================================================================

Works a contact queue (business contacts only — agents, wholesalers, cash
buyers), drafts a personalised message per contact (LLM if available, else
the templates in `wholesale.outreach`), and queues it to the integration
outbox in DRY-RUN. Nothing is sent until integrations are armed.

This is the legitimate, fast-cash channel: it never contacts distressed
homeowners (that path stays behind `compliance.gate`).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict

from .base import BaseAgent
from ..outreach import draft

_PER_TICK = 2  # how many contacts Riley works per tick


@dataclass
class Contact:
    name: str
    kind: str                 # "realtor" | "cowholesale" | "cashbuyer"
    market: str = "Dallas, TX"
    area: str = "75215"
    email: str = ""
    url: str = ""             # public website / contact-form (preferred channel)
    status: str = "queued"    # "queued" | "drafted" | "sent"
    drafted: str = ""
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["drafted"] = (self.drafted[:160] + "…") if len(self.drafted) > 160 else self.drafted
        return d


_SYSTEM = (
    "You are a real-estate investor's business-development rep. Write a SHORT, "
    "specific, friendly B2B outreach message (4-6 sentences) to the named "
    "contact. Goal depends on their type: realtor=ask for off-market/expired "
    "deals & cash buyers; cowholesale=propose a 50/50 JV; cashbuyer=ask for "
    "their buy-box to add them to deal alerts. No hype, no spam, be concrete."
)


class BizDevAgent(BaseAgent):
    code = "BIZDEV"
    name = "Riley (Business Development)"
    role = "Realtor / partner / cash-buyer outreach (B2B)"
    team = "Disposition"
    desc = ("Works a queue of real B2B contacts (agents, wholesalers, cash buyers) "
            "and drafts tailored outreach. Never contacts homeowners.")
    color = "#80ed99"
    owns = ()  # driven by company.tick via run(), not a pipeline stage

    def run(self) -> None:
        """Process up to _PER_TICK queued contacts this tick."""
        pending = [c for c in self.company.contacts if c.status == "queued"]
        if not pending:
            return
        for contact in pending[:_PER_TICK]:
            self._work(contact)

    def _work(self, contact: Contact) -> None:
        self._set("acting", f"Drafting outreach → {contact.name}", None)
        # Deterministic template as the base / fallback.
        base = draft(contact.kind, name=contact.name, market=contact.market,
                     area=contact.area,
                     signature=f"{self.company.cfg.CEO_NAME}\n"
                               f"{self.company.cfg.COMPANY_NAME}")
        user = (f"Contact: {contact.name}. Type: {contact.kind}. "
                f"Market: {contact.market} ({contact.area}). "
                f"Company: {self.company.cfg.COMPANY_NAME}. "
                f"Here is a baseline template to improve:\n{base}")
        msg = self.reason(_SYSTEM, user, max_tokens=320, temperature=0.7,
                          fallback=base)
        contact.drafted = msg
        contact.status = "drafted"

        to = contact.email or f"{contact.name.split()[0].lower()}@example.com"
        self.company.integrations.email_seller(
            to=to, subject=f"{contact.kind} outreach — {contact.market}", body=msg,
        )
        self.handled += 1
        self.say(f"Drafted {contact.kind} outreach to {contact.name} "
                 f"(queued, dry-run).", level="info")
