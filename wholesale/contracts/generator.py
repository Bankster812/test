"""
generator — fill draft contracts from a Deal
============================================

Produces the two documents a wholesale deal needs:
  * Purchase & Sale Agreement (assignable) — we buy from the seller.
  * Assignment of Contract — we sell our position to an end buyer.

Output is a DRAFT for attorney review (see templates.BANNER). Nothing here
files, signs, or sends anything.
"""

from __future__ import annotations

import datetime as _dt
import os

from .. import config
from . import templates


class ContractGenerator:
    def __init__(self, buyer_entity: str | None = None) -> None:
        # The wholesaling LLC (the user is setting this up).
        self.buyer_entity = buyer_entity or config.COMPANY_NAME

    def _ctx(self, deal) -> dict:
        today = _dt.date.today().isoformat()
        p = deal.prop
        return {
            "generated": today,
            "state": p.state,
            "banner": templates.BANNER.format(generated=today, state=p.state),
            "buyer_entity": self.buyer_entity,
            "seller_name": deal.seller.name,
            "address": p.address, "city": p.city, "zip": p.zip,
            "contract_price": deal.contract_price or deal.uw.target_offer,
            "assignment_fee": deal.assignment_fee,
            "earnest": 1_000,
            "closing_days": 30,
            "inspection_days": 7,
            "assignee": deal.buyer.name if deal.buyer else "[END BUYER — TBD]",
        }

    def purchase_agreement(self, deal) -> str:
        return templates.PURCHASE_AGREEMENT.format(**self._ctx(deal))

    def assignment_agreement(self, deal) -> str:
        return templates.ASSIGNMENT_AGREEMENT.format(**self._ctx(deal))

    def write_pair(self, deal, out_dir: str = "contracts_out") -> tuple[str, str]:
        """Write both drafts to disk; returns the two file paths."""
        os.makedirs(out_dir, exist_ok=True)
        psa = os.path.join(out_dir, f"deal_{deal.id}_purchase_agreement.txt")
        asg = os.path.join(out_dir, f"deal_{deal.id}_assignment.txt")
        with open(psa, "w") as f:
            f.write(self.purchase_agreement(deal))
        with open(asg, "w") as f:
            f.write(self.assignment_agreement(deal))
        return psa, asg
