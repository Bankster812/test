"""
submission — package a deal for a disposition platform / buyer
==============================================================

Builds the compliant buyer-facing summary (markets the CONTRACT, not the
property) and lists where to submit it. Per-deal details are auto-filled;
only a couple of operator specifics ([phone], terms) remain to insert.
"""

from __future__ import annotations

from .platforms import PLATFORMS


def buyer_summary(deal, company=None) -> str:
    p, uw = deal.prop, deal.uw
    structure = (uw.route or "assignment").replace("_", " ")
    fee = deal.assignment_fee or uw.target_assignment_fee
    return (
        f"Contractual interest available — {p.metro} ({p.city}, {p.state}).\n"
        f"Property profile: {p.beds}bd/{p.baths}ba {p.property_type}, {p.sqft} sqft, "
        f"built {p.year_built}. Distress: {p.distress}.\n"
        f"Numbers: ARV ~${uw.arv:,}, est. repairs ~${uw.repair_estimate:,}.\n"
        f"Expected structure: {structure}.\n"
        f"Asking assignment/spread: ${fee:,}.\n"
        f"Title/escrow path: [title company / closing attorney].\n"
        f"Additional due diligence available on request. Contact: [phone] / [email].\n"
    )


def target_platforms(deal) -> list[dict]:
    """Which dispo channels fit this deal (all national here) with submit hints."""
    out = []
    for pf in PLATFORMS:
        out.append({"name": pf.name, "url": pf.url, "payout": pf.payout,
                    "submit_hint": pf.best_for})
    return out


def submission_packet(deal, company=None) -> dict:
    return {
        "deal_id": deal.id,
        "buyer_summary": buyer_summary(deal, company),
        "platforms": target_platforms(deal),
    }
