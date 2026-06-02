"""AnalystAgent — underwriting / deal math (owns UNDERWRITING).

The numbers here are deterministic (never LLM-generated): ARV from comps,
repair estimate from age/size/distress, then the 70% rule to a Maximum
Allowable Offer. The LLM only writes the human rationale.
"""

from __future__ import annotations

from .. import config
from ..core.models import Stage, Underwriting
from .base import BaseAgent


# Per-distress baseline repair $/sqft (condition proxy).
_REPAIR_PPSF = {
    "pre-foreclosure": 28, "tax-delinquent": 24, "tired-landlord": 32,
    "inherited / probate": 38, "code violations": 45, "vacant": 30,
    "divorce": 20, "job relocation": 18,
}


class AnalystAgent(BaseAgent):
    code = "ANALYST"
    name = "Marcus (Underwriting Analyst)"
    role = "ARV, repairs & Maximum Allowable Offer"
    color = "#f9c74f"
    owns = (Stage.UNDERWRITING,)

    def handle(self, deal) -> None:
        self._set("acting", f"Underwriting {deal.prop.address}", deal.id)
        p = deal.prop

        # ARV: as-is market value uplifted for a renovated comp (8–18%).
        renovated_uplift = 1.0 + (0.18 - min(0.10, (2025 - p.year_built) / 1000))
        arv = int(round(p.est_market_value * max(1.08, renovated_uplift), -3))

        # Repairs: condition proxy x sqft, with an age surcharge.
        ppsf = _REPAIR_PPSF.get(p.distress, 28)
        age_surcharge = 1.0 + max(0.0, (1990 - p.year_built)) * 0.004
        repairs = int(round(p.sqft * ppsf * age_surcharge, -2))
        rehab_ratio = repairs / arv

        # 70% rule → MAO, reserving our target assignment fee.
        mao = int(round(arv * config.ARV_RULE - repairs - config.TARGET_ASSIGNMENT_FEE, -3))
        # We aim a touch under MAO to leave negotiating room.
        target_offer = int(round(mao * 0.96, -3))

        go = (
            mao > 0
            and rehab_ratio <= config.MAX_REHAB_RATIO
            and (arv - repairs - target_offer) >= config.MIN_ASSIGNMENT_FEE
        )

        rationale = self.reason(
            system="You are a real-estate wholesaling underwriter. Given the figures, "
                   "write ONE crisp sentence justifying the go/no-go decision. "
                   "Reference the spread or the killer (over-rehab, thin margin).",
            user=f"ARV ${arv:,}, repairs ${repairs:,} ({rehab_ratio:.0%} of ARV), "
                 f"MAO ${mao:,}, target offer ${target_offer:,}, decision="
                 f"{'GO' if go else 'NO-GO'}.",
            max_tokens=90,
            fallback=(f"Projected spread ${arv - repairs - target_offer:,} clears "
                      f"minimum — proceed." if go else
                      f"Rehab {rehab_ratio:.0%} of ARV / margin too thin — pass."),
        )

        deal.uw = Underwriting(arv=arv, repair_estimate=repairs, mao=mao,
                               target_offer=target_offer, rehab_ratio=round(rehab_ratio, 3),
                               go=go, rationale=rationale)
        deal.log(self.name, f"ARV ${arv:,} | repairs ${repairs:,} | MAO ${mao:,} | "
                            f"{'GO' if go else 'NO-GO'} — {rationale}")

        if go:
            deal.stage = Stage.OUTREACH
            self.say(f"Underwrote {deal.label}: MAO ${mao:,}, target ${target_offer:,} "
                     f"→ outreach.", level="info", deal_id=deal.id)
        else:
            deal.stage = Stage.DEAD
            self.say(f"No-go on {deal.label}: {rationale}", level="warn", deal_id=deal.id)
        self.handled += 1
