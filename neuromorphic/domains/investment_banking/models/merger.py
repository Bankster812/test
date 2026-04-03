"""Merger / Accretion-Dilution Model"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class MergerInputs:
    # Acquirer
    acq_net_income:      float
    acq_shares:          float
    acq_share_price:     float
    acq_pe:              float
    # Target
    tgt_net_income:      float
    tgt_share_price:     float
    tgt_shares:          float
    # Deal terms
    premium_pct:         float           # e.g. 0.30 = 30% premium
    cash_pct:            float           # % of consideration paid in cash
    stock_pct:           float           # % paid in stock (cash_pct + stock_pct = 1)
    # Synergies
    annual_synergies:    float           # Pre-tax synergies
    one_time_costs:      float           # Integration costs
    synergy_ramp_years:  int  = 3        # Years to fully realise synergies
    tax_rate:            float = 0.25
    # Financing
    new_debt:            float = 0.0     # New debt raised to fund cash consideration
    debt_rate:           float = 0.06


@dataclass
class MergerResult:
    offer_price:         float
    offer_value:         float           # total deal value
    cash_consideration:  float
    stock_consideration: float
    shares_issued:       float
    combined_shares:     float
    acq_standalone_eps:  float
    pro_forma_eps_yr1:   float
    pro_forma_eps_with_synergies: float
    accretion_dilution_pct: float        # vs standalone EPS
    breakeven_synergies: float           # synergies needed for zero accretion/dilution
    goodwill:            float
    exchange_ratio:      float | None

    def __str__(self) -> str:
        ad = self.accretion_dilution_pct
        return (
            f"  Offer price:          ${self.offer_price:.2f}  "
            f"({self.offer_price/max(self.offer_price/(1+0.001),1e-6)*100:.0f}% premium)\n"
            f"  Total deal value:     ${self.offer_value/1e6:.1f}M\n"
            f"  Acquirer EPS (base):  ${self.acq_standalone_eps:.3f}\n"
            f"  Pro-forma EPS (Yr1):  ${self.pro_forma_eps_yr1:.3f}\n"
            f"  With full synergies:  ${self.pro_forma_eps_with_synergies:.3f}\n"
            f"  Accretion/(Dilution): {ad*100:+.2f}% "
            f"({'ACCRETIVE' if ad > 0 else 'DILUTIVE'})\n"
            f"  Breakeven synergies:  ${self.breakeven_synergies/1e6:.1f}M/yr\n"
            f"  Exchange ratio:       {self.exchange_ratio:.4f}x" if self.exchange_ratio else ""
        )


class MergerModel:
    def compute(self, inp: MergerInputs) -> MergerResult:
        # Offer mechanics
        offer_price       = inp.tgt_share_price * (1.0 + inp.premium_pct)
        offer_value       = offer_price * inp.tgt_shares
        cash_consid       = offer_value * inp.cash_pct
        stock_consid      = offer_value * inp.stock_pct

        # Shares issued to target shareholders
        shares_issued     = stock_consid / max(inp.acq_share_price, 0.01)
        combined_shares   = inp.acq_shares + shares_issued
        exchange_ratio    = shares_issued / max(inp.tgt_shares, 1) if inp.stock_pct > 0 else None

        # Goodwill = offer value - book value of target (approximated as target NI * acq P/E * 0.5)
        tgt_book_approx = inp.tgt_net_income * inp.acq_pe * 0.3
        goodwill          = max(offer_value - tgt_book_approx, 0)

        # Acquirer standalone EPS
        acq_eps           = inp.acq_net_income / max(inp.acq_shares, 1)

        # Combined net income (no synergies, year 1)
        interest_on_debt  = inp.new_debt * inp.debt_rate * (1.0 - inp.tax_rate)
        combined_ni_yr1   = inp.acq_net_income + inp.tgt_net_income - interest_on_debt
        pf_eps_yr1        = combined_ni_yr1 / max(combined_shares, 1)

        # With full synergies
        net_synergies     = inp.annual_synergies * (1.0 - inp.tax_rate)
        combined_ni_syn   = combined_ni_yr1 + net_synergies
        pf_eps_syn        = combined_ni_syn / max(combined_shares, 1)

        ad_pct            = (pf_eps_yr1 - acq_eps) / max(abs(acq_eps), 1e-9)

        # Breakeven synergies: how much pre-tax synergies needed to break even
        eps_gap           = (acq_eps - pf_eps_yr1) * combined_shares
        breakeven_syn     = eps_gap / max(1.0 - inp.tax_rate, 0.01)

        return MergerResult(
            offer_price                   = offer_price,
            offer_value                   = offer_value,
            cash_consideration            = cash_consid,
            stock_consideration           = stock_consid,
            shares_issued                 = shares_issued,
            combined_shares               = combined_shares,
            acq_standalone_eps            = acq_eps,
            pro_forma_eps_yr1             = pf_eps_yr1,
            pro_forma_eps_with_synergies  = pf_eps_syn,
            accretion_dilution_pct        = ad_pct,
            breakeven_synergies           = max(breakeven_syn, 0),
            goodwill                      = goodwill,
            exchange_ratio                = exchange_ratio,
        )

    @staticmethod
    def from_brain_params(params, user_inputs: dict) -> MergerInputs:
        def pick(k, bk, d): return user_inputs.get(k, params.value(bk, d))
        return MergerInputs(
            acq_net_income   = user_inputs.get("acq_net_income",  100e6),
            acq_shares       = user_inputs.get("acq_shares",      50e6),
            acq_share_price  = user_inputs.get("acq_share_price", 40.0),
            acq_pe           = pick("acq_pe",      "pe_ratio",    20.0),
            tgt_net_income   = user_inputs.get("tgt_net_income",  30e6),
            tgt_share_price  = user_inputs.get("tgt_share_price", 25.0),
            tgt_shares       = user_inputs.get("tgt_shares",      20e6),
            premium_pct      = pick("premium_pct", "premium_pct", 0.30),
            cash_pct         = user_inputs.get("cash_pct",        0.50),
            stock_pct        = user_inputs.get("stock_pct",       0.50),
            annual_synergies = pick("synergies",   "synergy_pct", 0.05) * user_inputs.get("tgt_net_income", 30e6),
            one_time_costs   = user_inputs.get("one_time_costs",  10e6),
            tax_rate         = pick("tax_rate",    "tax_rate",    0.25),
            new_debt         = user_inputs.get("new_debt",        0.0),
            debt_rate        = pick("debt_rate",   "interest_rate",0.06),
        )
