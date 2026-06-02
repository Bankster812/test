"""
Worked IB models — end-to-end walkthrough
=========================================
A single fictional mid-market target ("Rheinwerk GmbH", a European specialty
industrials business) run through every model so you can see the deterministic
math the platform produces.

    python -m neuromorphic.domains.investment_banking.examples.worked_models
    python -m neuromorphic.domains.investment_banking.examples.worked_models --excel out/

All figures are illustrative. Nothing here touches the network or the spiking
network — these are the plain financial models reached via ``IBBrain.build_model``.
"""
from __future__ import annotations

import argparse
import os

from neuromorphic.domains.investment_banking.models.dcf import DCFModel, DCFInputs
from neuromorphic.domains.investment_banking.models.lbo import LBOModel, LBOInputs
from neuromorphic.domains.investment_banking.models.merger import MergerModel, MergerInputs
from neuromorphic.domains.investment_banking.models.comps import (
    CompsModel, CompsInputs, CompanyData,
)
from neuromorphic.domains.investment_banking.models.credit import CreditModel, CreditInputs

M = 1e6  # millions → absolute

# --- The target -------------------------------------------------------------
REVENUE = 250.0 * M
EBITDA_MARGIN = 0.24
EBITDA = REVENUE * EBITDA_MARGIN     # €60.0M
NET_DEBT = 80.0 * M


def _rule(title: str) -> None:
    print("\n" + "=" * 68)
    print(title)
    print("=" * 68)


def run_dcf():
    _rule("1) DCF — intrinsic value (perpetuity vs exit-multiple cross-check)")
    inp = DCFInputs(
        ebitda=EBITDA, revenue=REVENUE, revenue_growth=0.06, ebitda_margin=EBITDA_MARGIN,
        wacc=0.095, terminal_growth=0.02, tax_rate=0.27, capex_pct=0.05,
        nwc_pct=0.03, depreciation_pct=0.045, projection_years=5,
        net_debt=NET_DEBT, mid_year_convention=True,
    )
    r = DCFModel().compute(inp)
    print(r)
    print(f"\n  Cross-check: perpetuity TV implies a {r.implied_exit_multiple:.1f}x exit "
          f"EV/EBITDA.\n  If that looks rich/cheap vs comps, revisit the terminal growth.")
    return r


def run_comps() -> None:
    _rule("2) Trading comps — relative value")
    peers = [
        CompanyData("Peer Alpha", ev=720 * M, revenue=300 * M, ebitda=78 * M, ebit=60 * M,
                    net_income=42 * M, shares=30 * M, price=21.0, net_debt=90 * M),
        CompanyData("Peer Beta",  ev=540 * M, revenue=230 * M, ebitda=55 * M, ebit=43 * M,
                    net_income=30 * M, shares=25 * M, price=18.0, net_debt=90 * M),
        CompanyData("Peer Gamma", ev=900 * M, revenue=360 * M, ebitda=96 * M, ebit=75 * M,
                    net_income=52 * M, shares=40 * M, price=20.0, net_debt=100 * M),
    ]
    target = CompanyData("Rheinwerk", ev=0, revenue=REVENUE, ebitda=EBITDA,
                         ebit=EBITDA * 0.8, net_income=EBITDA * 0.45, net_debt=NET_DEBT)
    r = CompsModel().compute(CompsInputs(target=target, comparables=peers))
    print(r)
    lo, hi = r.implied_eq_range
    print(f"\n  Implied equity value: ${lo/M:.0f}M - ${hi/M:.0f}M")


def run_lbo():
    _rule("3) LBO — can a sponsor hit ~20% IRR?")
    inp = LBOInputs(
        ebitda=EBITDA, revenue=REVENUE, entry_ev_ebitda=11.0, revenue_growth=0.06,
        ebitda_margin_exit=0.26, leverage_ratio=5.5, senior_debt_rate=0.075,
        hold_period=5, exit_multiple=11.0, tax_rate=0.27, capex_pct=0.05,
        nwc_pct=0.03, depreciation_pct=0.045, debt_paydown_rate=1.0,
    )
    r = LBOModel().compute(inp)
    print(r)
    print(f"\n  Deleveraging: ${r.total_debt/M:.0f}M debt at entry -> "
          f"${r.debt_at_exit/M:.0f}M at exit (cash build ${r.cash_at_exit/M:.0f}M).")
    return r


def run_merger() -> None:
    _rule("4) Merger — accretion / dilution to a listed acquirer")
    inp = MergerInputs(
        acq_net_income=180 * M, acq_shares=120 * M, acq_share_price=32.0, acq_pe=21.0,
        tgt_net_income=27 * M, tgt_share_price=18.0, tgt_shares=20 * M,
        premium_pct=0.30, cash_pct=0.40, stock_pct=0.60,
        annual_synergies=12 * M, one_time_costs=15 * M, tax_rate=0.27,
    )
    r = MergerModel().compute(inp)
    print(r)


def run_credit() -> None:
    _rule("5) Credit — is the capital structure sustainable?")
    inp = CreditInputs(
        ebitda=EBITDA, revenue=REVENUE, total_debt=5.5 * EBITDA, cash=20 * M,
        interest_expense=5.5 * EBITDA * 0.075, capex=REVENUE * 0.05, tax_rate=0.27,
        net_income=EBITDA * 0.30,
    )
    r = CreditModel().compute(inp)
    print(r)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Worked IB model examples")
    ap.add_argument("--excel", metavar="DIR", default=None,
                    help="also write formatted .xlsx outputs to DIR")
    args = ap.parse_args(argv)

    print(__doc__.split("\n\n")[1])  # the scenario blurb
    dcf = run_dcf()
    run_comps()
    lbo = run_lbo()
    run_merger()
    run_credit()

    if args.excel:
        from neuromorphic.domains.investment_banking.excel.writer import ExcelWriter
        os.makedirs(args.excel, exist_ok=True)
        w = ExcelWriter()
        w.write_dcf(dcf, os.path.join(args.excel, "dcf.xlsx"))
        w.write_lbo(lbo, os.path.join(args.excel, "lbo.xlsx"))
        print(f"\nWrote dcf.xlsx, lbo.xlsx to {args.excel}/")

    print("\nDone. (Illustrative figures — not investment advice.)")


if __name__ == "__main__":
    main()
