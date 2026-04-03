#!/usr/bin/env python3
"""
Neuromorphic IB Platform — End-to-End Demo
==========================================
Demonstrates the full capability stack:
  1. Brain initialisation (SCALE=0.01 for demo speed)
  2. Knowledge base lookup (instant, no neural settling)
  3. DCF model build
  4. LBO model build
  5. Natural language query (neural settling)
  6. Continuous learning daemon start
  7. Status report

Run:
    cd /home/user/test
    python -m neuromorphic.domains.investment_banking.demo

Or:
    python neuromorphic/domains/investment_banking/demo.py
"""

from __future__ import annotations

import sys
import os

# Ensure project root is on the path when run directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

import numpy as np


def banner(text: str, char: str = "=") -> None:
    line = char * 70
    print(f"\n{line}")
    print(f"  {text}")
    print(line)


def run_demo():
    banner("NEUROMORPHIC IB PLATFORM — DEMO")
    print("  Scale: 0.01 (10K neurons, ~120K synapses)")
    print("  Full-scale: 1M neurons, 1.2B synapses\n")

    # ------------------------------------------------------------------
    # 0. Configure scale for demo speed
    # ------------------------------------------------------------------
    import neuromorphic.config as base_cfg
    base_cfg.SCALE = 0.01  # demo scale — change to 1.0 for full scale

    from neuromorphic.domains.investment_banking import ib_config
    ib_config.SCALE = 0.01

    # ------------------------------------------------------------------
    # 1. Initialise IBBrain
    # ------------------------------------------------------------------
    banner("1. INITIALISING BRAIN", "-")
    from neuromorphic.domains.investment_banking.ib_brain import IBBrain
    brain = IBBrain(config=ib_config, verbose=True)

    # ------------------------------------------------------------------
    # 2. Knowledge base lookup (no settling required)
    # ------------------------------------------------------------------
    banner("2. KNOWLEDGE BASE", "-")
    terms = ["WACC", "LBO", "accretion/dilution", "covenant", "terminal value"]
    for term in terms:
        entry = brain.knowledge_base.lookup(term)
        if entry:
            print(f"\n[{term.upper()}]")
            print(f"  {entry.definition}")
            if entry.formula:
                print(f"  Formula: {entry.formula}")
        else:
            print(f"  [not found: {term}]")

    # ------------------------------------------------------------------
    # 3. DCF Model
    # ------------------------------------------------------------------
    banner("3. DCF MODEL", "-")
    dcf_inputs = {
        "revenue_m":       500.0,     # $500M revenue
        "ebitda_margin":   0.25,      # 25% EBITDA margin
        "revenue_growth":  0.10,      # 10% CAGR
        "wacc":            0.10,      # 10% discount rate
        "terminal_growth": 0.025,     # 2.5% terminal growth
        "tax_rate":        0.25,      # 25% tax rate
        "capex_pct_rev":   0.05,      # 5% of revenue
        "nwc_pct_rev":     0.08,      # 8% of revenue
        "da_pct_rev":      0.04,      # 4% D&A
        "projection_years": 5,
    }

    print("\nInputs:")
    for k, v in dcf_inputs.items():
        print(f"  {k}: {v}")

    dcf_result = brain.build_model("dcf", dcf_inputs, verbose=False)

    if "error" not in dcf_result:
        print(f"\nDCF Results:")
        print(f"  Enterprise Value (base):  ${dcf_result.get('enterprise_value_m', 0):.1f}M")
        print(f"  Equity Value:             ${dcf_result.get('equity_value_m', 0):.1f}M")
        print(f"  EV/EBITDA implied:        {dcf_result.get('implied_ev_ebitda', 0):.1f}x")
        tv_pct = dcf_result.get("terminal_value_pct", 0)
        print(f"  Terminal value % of EV:   {tv_pct:.1f}%")
        if "sensitivity" in dcf_result:
            print(f"\n  Sensitivity (EV $M, WACC rows × TGR cols):")
            sens = dcf_result["sensitivity"]
            if isinstance(sens, list):
                for row in sens[:3]:
                    print(f"    {row}")
    else:
        print(f"  [DCF error: {dcf_result['error']}]")

    # ------------------------------------------------------------------
    # 4. LBO Model
    # ------------------------------------------------------------------
    banner("4. LBO MODEL", "-")
    lbo_inputs = {
        "ebitda_m":        125.0,     # $125M EBITDA
        "entry_multiple":  10.0,      # 10x entry
        "exit_multiple":   12.0,      # 12x exit
        "leverage":        6.0,       # 6x debt/EBITDA
        "interest_rate":   0.07,      # 7% debt cost
        "revenue_growth":  0.08,
        "ebitda_margin":   0.25,
        "hold_years":      5,
        "tax_rate":        0.25,
        "capex_pct_rev":   0.04,
        "nwc_pct_rev":     0.05,
        "da_pct_rev":      0.03,
        "revenue_m":       500.0,
    }

    print("\nInputs:")
    for k, v in lbo_inputs.items():
        print(f"  {k}: {v}")

    lbo_result = brain.build_model("lbo", lbo_inputs, verbose=False)

    if "error" not in lbo_result:
        print(f"\nLBO Results:")
        print(f"  Entry EV:    ${lbo_result.get('entry_ev_m', 0):.0f}M")
        print(f"  Exit EV:     ${lbo_result.get('exit_ev_m', 0):.0f}M")
        print(f"  IRR:         {lbo_result.get('irr', 0)*100:.1f}%")
        print(f"  MOIC:        {lbo_result.get('moic', 0):.2f}x")
        print(f"  Equity in:   ${lbo_result.get('entry_equity_m', 0):.0f}M")
        print(f"  Equity out:  ${lbo_result.get('exit_equity_m', 0):.0f}M")
    else:
        print(f"  [LBO error: {lbo_result['error']}]")

    # ------------------------------------------------------------------
    # 5. Credit Analysis
    # ------------------------------------------------------------------
    banner("5. CREDIT MODEL", "-")
    credit_inputs = {
        "ebitda_m":        125.0,
        "total_debt_m":    750.0,     # 6x leverage
        "cash_m":          50.0,
        "interest_m":      52.5,      # 7% on $750M
        "capex_m":         25.0,
        "revenue_m":       500.0,
    }
    credit_result = brain.build_model("credit", credit_inputs, verbose=False)
    if "error" not in credit_result:
        print(f"\nCredit Results:")
        print(f"  Total Leverage:    {credit_result.get('total_leverage', 0):.1f}x")
        print(f"  Net Leverage:      {credit_result.get('net_leverage', 0):.1f}x")
        print(f"  Interest Coverage: {credit_result.get('interest_coverage', 0):.1f}x")
        print(f"  Implied Rating:    {credit_result.get('implied_rating', 'N/A')}")
        print(f"  Risk Level:        {credit_result.get('risk_level', 'N/A')}")
    else:
        print(f"  [Credit error: {credit_result['error']}]")

    # ------------------------------------------------------------------
    # 6. Risk Engine
    # ------------------------------------------------------------------
    banner("6. RISK ENGINE", "-")
    risky_deal = {
        "leverage":   9.5,
        "wacc":       0.08,
        "irr":        0.18,
        "ebitda_m":   125.0,
        "premium_pct": 55.0,
    }
    risk_report = brain.risk_engine.analyse(risky_deal)
    print(f"\nRisk report: {risk_report.summary}")
    for flag in risk_report.flags:
        print(f"  {flag}")

    # ------------------------------------------------------------------
    # 7. Natural language query (neural settling — takes a few seconds)
    # ------------------------------------------------------------------
    banner("7. NATURAL LANGUAGE QUERY  (neural settling...)", "-")
    question = "What WACC should I use for a tech LBO with 60% equity at 12% cost of equity and 5% cost of debt at 25% tax rate?"
    print(f"\nQuery: \"{question}\"")
    print("  (Running 200 simulation steps...)\n")

    response = brain.query(question, verbose=False)

    print(f"Answer:\n{response.answer_text}")
    print(f"\nConfidence: {response.confidence*100:.0f}%")
    if response.risk_flags:
        print(f"Risk flags: {len(response.risk_flags)}")
        for f in response.risk_flags[:3]:
            print(f"  {f}")

    # ------------------------------------------------------------------
    # 8. Continuous learning daemon
    # ------------------------------------------------------------------
    banner("8. CONTINUOUS LEARNING DAEMON", "-")
    print("\nStarting 24/7 IB learning daemon...")
    brain.start_continuous_learning(interval_minutes=60)
    import time
    time.sleep(1)  # let daemon thread initialise
    s = brain.status()
    d = s["learning_daemon"]
    print(f"Daemon running: {d.get('running', False)}")
    print(f"Topics: {d.get('topics', [])[:3]}")

    # Stop daemon for clean exit
    brain.stop_continuous_learning()

    # ------------------------------------------------------------------
    # 9. Status
    # ------------------------------------------------------------------
    banner("9. BRAIN STATUS", "-")
    brain.print_status()

    banner("DEMO COMPLETE")
    print("  The neuromorphic IB platform is operational.")
    print("  To use at full scale, set SCALE=1.0 in neuromorphic/config.py")
    print("  Typical full-scale initialisation: ~30s, ~8GB RAM, <5W CPU\n")


if __name__ == "__main__":
    run_demo()
