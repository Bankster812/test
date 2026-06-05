"""
models — the company's domain objects
=====================================

Plain dataclasses, JSON-serialisable, no per-object behaviour beyond
small conveniences. A `Deal` is the unit of work that flows through the
value chain; everything else hangs off it.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any


# ----------------------------------------------------------------------------
# Pipeline stages — the wholesale value chain
# ----------------------------------------------------------------------------
class Stage(str, Enum):
    SOURCED       = "sourced"        # lead identified by the Scout
    UNDERWRITING  = "underwriting"   # Analyst running the numbers
    OUTREACH      = "outreach"       # Closer contacting the seller
    NEGOTIATION   = "negotiation"    # price discovery with seller
    UNDER_CONTRACT = "under_contract" # assignable purchase agreement signed
    DISPOSITION   = "disposition"    # Dispo marketing to cash buyers
    ASSIGNED      = "assigned"       # matched to a buyer, fee locked
    CLOSING       = "closing"        # Coordinator running title/escrow
    CLOSED_WON    = "closed_won"     # funded — assignment fee collected
    DEAD          = "dead"           # killed (no spread / seller walked / risk)

    @property
    def is_terminal(self) -> bool:
        return self in (Stage.CLOSED_WON, Stage.DEAD)


# Ordered for the kanban board on the dashboard.
PIPELINE_ORDER = [
    Stage.SOURCED, Stage.UNDERWRITING, Stage.OUTREACH, Stage.NEGOTIATION,
    Stage.UNDER_CONTRACT, Stage.DISPOSITION, Stage.ASSIGNED, Stage.CLOSING,
    Stage.CLOSED_WON, Stage.DEAD,
]

_deal_counter = itertools.count(1001)
_buyer_counter = itertools.count(1)


@dataclass
class Property:
    address: str
    city: str
    state: str
    zip: str
    metro: str                       # matches config.HQ_MARKETS, e.g. "TX-Dallas"
    beds: int
    baths: float
    sqft: int
    year_built: int
    property_type: str               # "SFR", "duplex", "townhome"
    distress: str                    # why it's a lead: "pre-foreclosure", "tired-landlord"...
    est_market_value: int            # as-is comp guess before underwriting

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Seller:
    name: str
    motivation: str                  # "relocation", "inherited", "divorce", "distress"
    asking_price: int
    reachable_via: str               # "phone", "email", "text"
    flexibility: float               # 0..1 — how far they'll move off asking
    walk_floor: int = 0              # hidden: lowest price they'll actually accept


@dataclass
class Buyer:
    name: str
    metros: list[str]
    max_price: int
    property_types: list[str]
    min_spread: int                  # smallest assignment fee they'll tolerate baked in
    cash_verified: bool
    appetite: float                  # 0..1 — likelihood to take a fitting deal
    id: int = field(default_factory=lambda: next(_buyer_counter))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Underwriting:
    arv: int = 0                     # after-repair value
    repair_estimate: int = 0
    mao: int = 0                     # maximum allowable offer (base case)
    mao_aggressive: int = 0          # thinner buffer
    mao_conservative: int = 0        # fatter buffer
    target_offer: int = 0
    target_assignment_fee: int = 0
    route: str = "assignment"        # "assignment" | "double_close" | "do_not_pursue"
    rehab_ratio: float = 0.0
    go: bool = False
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Deal:
    prop: Property
    seller: Seller
    stage: Stage = Stage.SOURCED
    uw: Underwriting = field(default_factory=Underwriting)
    contract_price: int = 0          # what we agreed to pay the seller
    assignment_fee: int = 0          # our spread
    buyer: Buyer | None = None
    history: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)   # compliance / risk notes
    legal_memo: dict = field(default_factory=dict)   # LegalAnalyst triage output
    qa: dict = field(default_factory=dict)           # QA/Audit result
    needs_ceo: bool = False
    ceo_decision: str | None = None  # "approved" / "rejected" / None
    compliance_done: bool = False
    nego_rounds: int = 0
    dispo_rounds: int = 0
    id: int = field(default_factory=lambda: next(_deal_counter))
    created: float = field(default_factory=time.time)
    updated: float = field(default_factory=time.time)

    def log(self, who: str, msg: str) -> None:
        self.history.append(f"[{who}] {msg}")
        self.updated = time.time()

    @property
    def label(self) -> str:
        return f"#{self.id} {self.prop.address}, {self.prop.city} {self.prop.state}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "stage": self.stage.value,
            "label": self.label,
            "property": self.prop.to_dict(),
            "seller": asdict(self.seller),
            "uw": self.uw.to_dict(),
            "contract_price": self.contract_price,
            "assignment_fee": self.assignment_fee,
            "buyer": self.buyer.to_dict() if self.buyer else None,
            "flags": self.flags,
            "legal_memo": self.legal_memo,
            "qa": self.qa,
            "needs_ceo": self.needs_ceo,
            "ceo_decision": self.ceo_decision,
            "history": self.history[-12:],
            "created": self.created,
            "updated": self.updated,
        }
