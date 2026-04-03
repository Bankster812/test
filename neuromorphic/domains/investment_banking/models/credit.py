"""Credit / Leverage Analysis"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


# Approximate S&P leverage thresholds (simplified)
_RATING_THRESHOLDS = [
    ("AAA",  0.5), ("AA",  1.0), ("A",  2.0), ("BBB", 3.5),
    ("BB",   4.5), ("B",   5.5), ("CCC", 7.0), ("CC", 9.0), ("D", 999.0),
]


@dataclass
class CreditInputs:
    ebitda:          float
    revenue:         float
    total_debt:      float
    cash:            float
    interest_expense: float
    capex:           float
    tax_rate:        float
    net_income:      float
    # Optional for more detail
    senior_debt:     float = 0.0
    sub_debt:        float = 0.0
    fixed_charges:   float = 0.0     # rent, lease, etc.
    debt_maturity_schedule: dict[int, float] = None   # year → repayment amount


@dataclass
class CreditResult:
    net_debt:                float
    leverage_total:          float     # Total Debt / EBITDA
    leverage_net:            float     # Net Debt / EBITDA
    interest_coverage:       float     # EBITDA / Interest
    fixed_charge_coverage:   float     # EBITDA / (Interest + FixedCharges)
    dscr:                    float     # (EBITDA - CapEx - Tax) / Total Debt Service
    debt_capacity_5x:        float     # Max debt at 5x leverage
    debt_capacity_6x:        float     # Max debt at 6x leverage
    implied_rating:          str
    covenants_ok:            bool
    covenant_headroom_pct:   float     # headroom vs typical 6x covenant
    risk_level:              str       # "low", "moderate", "high", "distressed"

    def __str__(self) -> str:
        cov = "✓ PASS" if self.covenants_ok else "✗ BREACH"
        return (
            f"  Total Leverage:       {self.leverage_total:.1f}x Debt/EBITDA\n"
            f"  Net Leverage:         {self.leverage_net:.1f}x Net Debt/EBITDA\n"
            f"  Interest Coverage:    {self.interest_coverage:.1f}x\n"
            f"  DSCR:                 {self.dscr:.2f}x\n"
            f"  Implied Rating:       {self.implied_rating}\n"
            f"  Debt Capacity (5x):   ${self.debt_capacity_5x/1e6:.0f}M\n"
            f"  Debt Capacity (6x):   ${self.debt_capacity_6x/1e6:.0f}M\n"
            f"  Covenant Check:       {cov}  (headroom: {self.covenant_headroom_pct:.0f}%)\n"
            f"  Risk Level:           {self.risk_level.upper()}"
        )


class CreditModel:
    def compute(self, inp: CreditInputs) -> CreditResult:
        net_debt = inp.total_debt - inp.cash
        lev_tot  = inp.total_debt / max(inp.ebitda, 1)
        lev_net  = net_debt        / max(inp.ebitda, 1)

        # Interest coverage
        int_cov  = inp.ebitda / max(inp.interest_expense, 1)

        # Fixed charge coverage
        total_fc = inp.interest_expense + inp.fixed_charges
        fcc      = inp.ebitda / max(total_fc, 1)

        # Debt service coverage (uses post-tax, post-capex cash flow)
        ebt         = inp.ebitda - inp.interest_expense
        taxes       = max(ebt, 0) * inp.tax_rate
        post_tax_cf = inp.ebitda - inp.capex - taxes
        debt_service = inp.interest_expense + (inp.total_debt * 0.05)  # assume 5% amortization
        dscr        = post_tax_cf / max(debt_service, 1)

        # Implied rating
        rating = "D"
        for r, threshold in _RATING_THRESHOLDS:
            if lev_tot <= threshold:
                rating = r
                break

        # Debt capacity
        dc5 = 5.0 * inp.ebitda
        dc6 = 6.0 * inp.ebitda

        # Covenant check (typical: 6x leverage covenant)
        covenant_max = 6.0
        covenants_ok = lev_tot <= covenant_max
        headroom_pct = (covenant_max - lev_tot) / covenant_max * 100.0

        # Risk level
        if lev_tot < 3.0 and int_cov > 5.0:
            risk = "low"
        elif lev_tot < 5.0 and int_cov > 2.5:
            risk = "moderate"
        elif lev_tot < 7.0 and int_cov > 1.5:
            risk = "high"
        else:
            risk = "distressed"

        return CreditResult(
            net_debt               = net_debt,
            leverage_total         = lev_tot,
            leverage_net           = lev_net,
            interest_coverage      = int_cov,
            fixed_charge_coverage  = fcc,
            dscr                   = dscr,
            debt_capacity_5x       = dc5,
            debt_capacity_6x       = dc6,
            implied_rating         = rating,
            covenants_ok           = covenants_ok,
            covenant_headroom_pct  = headroom_pct,
            risk_level             = risk,
        )

    @staticmethod
    def from_brain_params(params, user_inputs: dict) -> CreditInputs:
        def pick(k, bk, d): return user_inputs.get(k, params.value(bk, d))
        ebitda = user_inputs.get("ebitda", 50e6)
        return CreditInputs(
            ebitda           = ebitda,
            revenue          = user_inputs.get("revenue", ebitda / 0.20),
            total_debt       = pick("total_debt", "leverage_ratio", 4.0) * ebitda,
            cash             = user_inputs.get("cash", 10e6),
            interest_expense = user_inputs.get("interest_expense",
                               pick("total_debt", "leverage_ratio", 4.0) * ebitda * pick("interest_rate", "interest_rate", 0.06)),
            capex            = pick("capex_pct", "capex_pct", 0.05) * user_inputs.get("revenue", ebitda / 0.20),
            tax_rate         = pick("tax_rate", "tax_rate", 0.25),
            net_income       = user_inputs.get("net_income", ebitda * 0.5),
        )
