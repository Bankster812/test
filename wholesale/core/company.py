"""
company — the autonomous wholesaling firm
=========================================

`Company` is the org. It sources leads, runs the agent roster over the
pipeline once per tick, books revenue on closings, and exposes a
thread-safe snapshot for the dashboard. The human operator is the CEO and
acts only through the approval queue (`approve`/`reject`).

Tick discipline: stages are processed in *reverse* pipeline order so a
deal advances at most one stage per tick — making the value chain visible
on the board instead of teleporting end-to-end.
"""

from __future__ import annotations

import random
import threading
import time
from typing import Any

from .. import config
from .. import policy
from ..data.buyers import default_book
from ..data.market import MarketFeed
from ..disposition import PLATFORMS as _DISPO_PLATFORMS
from ..integrations import IntegrationHub
from .eventbus import EventBus
from .cockpit import build_action_queue
from .models import Deal, Stage, PIPELINE_ORDER


class Company:
    def __init__(self, cfg=config, seed: int | None = None) -> None:
        self.cfg = cfg
        self.rng = random.Random(seed)
        self.bus = EventBus()
        self.integrations = IntegrationHub(armed=cfg.INTEGRATIONS_ARMED)
        self.market = MarketFeed(cfg.HQ_MARKETS, seed=seed)
        self.buyers = default_book(cfg.HQ_MARKETS, seed=seed)

        self.deals: list[Deal] = []
        self.contacts: list = []          # B2B outreach queue (Riley/BizDev)
        self.oversight: dict = {}         # Chief-of-Staff accountability report
        self.tick_count = 0
        self.started = time.time()

        # Financials
        self.revenue = 0
        self.closed_count = 0
        self.dead_count = 0

        # Roster is built lazily to avoid an import cycle (agents import core).
        from ..agents import build_roster
        self.agents = build_roster(self)
        self._stage_owner = {}
        for agent in self.agents.values():
            for stage in agent.owns:
                self._stage_owner[stage] = agent

        self._seed_contacts()

        self._lock = threading.Lock()
        self._running = False
        self.paused = False
        self._thread: threading.Thread | None = None

        self.bus.publish(self.cfg.COMPANY_NAME,
                         f"Company online. CEO: {self.cfg.CEO_NAME}. "
                         f"Markets: {', '.join(self.cfg.HQ_MARKETS)}.", level="info")

    # ----------------------------------------------------------------- ticks
    def tick(self) -> None:
        with self._lock:
            self.tick_count += 1
            tick_start = time.time()

            # 1. Source fresh leads on cadence.
            if self.tick_count % self.cfg.NEW_LEADS_TICKS == 1:
                for prop, seller in self.market.next_leads(self.cfg.LEADS_PER_BATCH):
                    self.deals.append(Deal(prop=prop, seller=seller))

            # 2. Advance pipeline (reverse order → one stage/deal/tick).
            for stage in reversed(PIPELINE_ORDER):
                if stage.is_terminal:
                    continue
                owner = self._stage_owner.get(stage)
                if owner is None:
                    continue
                for deal in self.deals:
                    if deal.stage == stage:
                        if self._blocked(deal):
                            continue
                        try:
                            owner.handle(deal)
                        except Exception as e:  # noqa: BLE001
                            self.bus.publish(owner.name,
                                             f"Error on {deal.label}: {e}", level="warn",
                                             deal_id=deal.id)

            # 3. Business development: work the B2B outreach queue.
            self.agents["BIZDEV"].run()

            # 4. Governance: QA/Audit recent deals, then Chief-of-Staff oversight.
            self.agents["QA"].run()
            self.oversight = self.agents["CHIEF"].oversee()

            # 5. Idle any agent that didn't act this tick.
            for agent in self.agents.values():
                if agent.last_active < tick_start:
                    agent.idle()

    def _seed_contacts(self) -> None:
        """Seed the BizDev queue with REAL public DFW B2B targets.

        These are publicly-advertised investor groups and cash-buyer
        companies (business contacts, not consumer PII). Riley drafts
        outreach; you submit via each org's public contact form.
        """
        from ..outreach.dfw_targets import load_contacts
        self.contacts = load_contacts()

    def add_contact(self, name: str, kind: str, market: str = "Dallas, TX",
                    area: str = "75215", email: str = "") -> None:
        from ..agents.bizdev import Contact
        with self._lock:
            self.contacts.append(Contact(name, kind, market, area, email))

    def legal_review(self, state: str, deal_type: str = "wholesale assignment",
                     pre_foreclosure: bool = False) -> dict:
        """On-demand legal triage from the LegalAnalyst sub-agent."""
        return self.agents["LEGAL"].analyze(state, deal_type, pre_foreclosure)

    def foreign_payee_readiness(self) -> dict:
        """Non-resident payout/tax readiness checklist (Mei)."""
        return self.agents["FOREIGN"].readiness()

    def arm(self, on: bool = True) -> bool:
        """Flip integrations live/dry-run (live still needs a wired transport)."""
        self.integrations.arm(on)
        self.bus.publish(self.cfg.CEO_NAME,
                         f"Integrations {'ARMED (live)' if on else 'set to dry-run'} by CEO.",
                         level="escalate" if on else "info")
        return on

    def dispo_submission(self, deal_id: int) -> dict | None:
        """Buyer-facing summary + target platforms for a deal."""
        from ..disposition.submission import submission_packet
        d = self.get_deal(deal_id)
        return submission_packet(d, self) if d else None

    def do_action(self, action_id: str, decision: str | None = None) -> bool:
        """Execute one Action-Queue item (the human-gated last mile).

        Draft-only: 'send' marks the outreach done and logs it (nothing leaves
        the machine unless integrations are armed). 'approve' routes to the CEO
        decision. Other kinds are acknowledged.
        """
        with self._lock:
            kind, _, ref = action_id.partition(":")
            if kind == "send":
                try:
                    contact = self.contacts[int(ref)]
                except (ValueError, IndexError):
                    return False
                if contact.status != "drafted":
                    return False
                contact.status = "sent"
                self.bus.publish("Riley (Business Development)",
                                 f"Marked outreach to {contact.name} as sent "
                                 f"({'live' if self.integrations.crm.armed else 'dry-run'}).",
                                 level="info")
                return True
            if kind == "ceo":
                return self._decide_locked(int(ref), decision or "approved")
            if kind in ("sign", "list"):
                self.bus.publish(self.cfg.CEO_NAME,
                                 f"Acknowledged {kind} action {ref}.", level="info")
                return True
        return False

    def action_queue(self) -> list[dict]:
        return build_action_queue(self)

    def _blocked(self, deal: Deal) -> bool:
        # Deals awaiting an undecided CEO call are parked (compliance still
        # "handles" them to keep the awaiting-status visible, so not blocked
        # here — this hook exists for future holds).
        return False

    # ------------------------------------------------------------- financials
    def book_revenue(self, deal: Deal) -> None:
        self.revenue += deal.assignment_fee
        self.closed_count += 1

    # --------------------------------------------------------------- CEO acts
    def decide(self, deal_id: int, decision: str) -> bool:
        """CEO approves/rejects an escalated deal. Returns True if applied."""
        with self._lock:
            return self._decide_locked(deal_id, decision)

    def _decide_locked(self, deal_id: int, decision: str) -> bool:
        """Caller must hold self._lock."""
        if decision not in ("approved", "rejected"):
            return False
        for d in self.deals:
            if d.id == deal_id and d.needs_ceo and d.ceo_decision is None:
                d.ceo_decision = decision
                d.log(self.cfg.CEO_NAME, f"CEO {decision} the deal.")
                self.bus.publish(self.cfg.CEO_NAME,
                                 f"{decision.title()} {d.label}.",
                                 level="escalate", deal_id=d.id)
                return True
        return False

    # ------------------------------------------------------------- run thread
    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="ws-company")
        self._thread.start()

    def _loop(self) -> None:
        while self._running:
            if not self.paused:
                self.tick()
            time.sleep(self.cfg.TICK_SECONDS)

    def pause(self) -> bool:
        self.paused = True
        self.bus.publish(self.cfg.COMPANY_NAME, "Autopilot paused by CEO.", level="warn")
        return True

    def resume(self) -> bool:
        self.paused = False
        self.bus.publish(self.cfg.COMPANY_NAME, "Autopilot resumed by CEO.", level="info")
        return True

    def tick_now(self) -> bool:
        """Run a single tick on demand (used by the UI 'step' button)."""
        self.tick()
        return True

    def get_deal(self, deal_id: int):
        for d in self.deals:
            if d.id == deal_id:
                return d
        return None

    def contract_packet(self, deal_id: int) -> dict | None:
        """Generate the attorney-review document packet for a deal."""
        from ..contracts import ContractGenerator
        d = self.get_deal(deal_id)
        if d is None:
            return None
        return ContractGenerator(buyer_entity=self.cfg.COMPANY_NAME).preforeclosure_packet(d)

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    # ---------------------------------------------------------------- reports
    def _counts(self) -> dict[str, int]:
        counts = {s.value: 0 for s in PIPELINE_ORDER}
        for d in self.deals:
            counts[d.stage.value] += 1
        return counts

    def pipeline_value(self) -> int:
        return sum(d.assignment_fee for d in self.deals
                   if not d.stage.is_terminal and d.assignment_fee)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            counts = self._counts()
            dead = counts.get(Stage.DEAD.value, 0)
            active = [d for d in self.deals if not d.stage.is_terminal]
            ceo_queue = [d.to_dict() for d in self.deals
                         if d.needs_ceo and d.ceo_decision is None
                         and d.stage == Stage.UNDER_CONTRACT]
            return {
                "company": self.cfg.COMPANY_NAME,
                "ceo": self.cfg.CEO_NAME,
                "uptime_s": round(time.time() - self.started, 1),
                "tick": self.tick_count,
                "paused": self.paused,
                "armed": self.integrations.crm.armed,
                "llm": {
                    "available": self.agents["SCOUT"].llm.is_available(),
                    "backend": self.agents["SCOUT"].llm.backend,
                    "model": self.cfg.LLM_MODEL,
                    "describe": self.agents["SCOUT"].llm.describe(),
                    "calls": self.agents["SCOUT"].llm.calls,
                    "failures": self.agents["SCOUT"].llm.failures,
                },
                "financials": {
                    "revenue": self.revenue,
                    "closed": self.closed_count,
                    "dead": dead,
                    "active": len(active),
                    "total_leads": len(self.deals),
                    "pipeline_fee_value": self.pipeline_value(),
                    "avg_fee": (self.revenue // self.closed_count) if self.closed_count else 0,
                },
                "counts": counts,
                "pipeline_order": [s.value for s in PIPELINE_ORDER],
                "agents": [a.snapshot() for a in self.agents.values()],
                "ceo_queue": ceo_queue,
                "action_queue": build_action_queue(self),
                "deals": [d.to_dict() for d in self.deals[-60:]],
                "activity": self.bus.snapshot(80),
                "outbox": [r.to_dict() for r in self.integrations.outbox[-30:]],
                "contacts": [c.to_dict() for c in self.contacts[-30:]],
                "dispo_platforms": [p.to_dict() for p in _DISPO_PLATFORMS],
                "policy": policy.summary(),
                "oversight": self.oversight,
            }
