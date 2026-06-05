"""
platforms — where to quickly sell/assign a contract, and how you get paid
=========================================================================

Two ways you earn on a wholesale deal:

  * ASSIGNMENT SPREAD — you have it under contract at X, the end buyer pays
    X + fee; the fee (the difference) is yours. You control the spread.
  * FINDER / REFERRAL FEE — you bring a buyer or a lead to someone else's
    deal and take an agreed cut (co-wholesale split). Faster, smaller, no
    origination needed.

`payout` below flags which model each platform/channel supports. Details
(pricing, buyer counts) are as of 2026 and change — verify on the site.
NOT an endorsement; do your own due diligence and read each platform's terms.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass
class DispoPlatform:
    name: str
    url: str
    payout: str            # "assignment" | "finder" | "both"
    cost: str              # what it costs you
    reach: str             # buyer network / coverage
    best_for: str
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


PLATFORMS: list[DispoPlatform] = [
    DispoPlatform(
        name="OfferMarket",
        url="https://www.offermarket.us/",
        payout="assignment",
        cost="Free marketplace (optional paid 'boost' for visibility)",
        reach="Verified cash buyers; AS-IS offer comparison",
        best_for="Best free starting point — list a deal and collect offers",
        notes="Commission-free; good for your first placements.",
    ),
    DispoPlatform(
        name="DispoBridge",
        url="https://dispobridge.com/",
        payout="assignment",
        cost="$0 upfront — takes a split of the assignment fee on close",
        reach="500+ verified cash buyers across 800+ cities, all 50 states",
        best_for="Pay-on-success dispo when you don't want monthly cost",
        notes="Founded 2025; split set per-deal at submission.",
    ),
    DispoPlatform(
        name="InvestorLift",
        url="https://investorlift.com/",
        payout="assignment",
        cost="~$500/mo entry up to $4k+/mo enterprise",
        reach="~5.5M cash buyers; AI buyer-matching, SMS/email blasts",
        best_for="High volume (20+ deals/mo); max spread via bidding wars",
        notes="Overkill until you have steady deal flow.",
    ),
    DispoPlatform(
        name="DealFlow AI",
        url="https://dealrun.ai/",
        payout="assignment",
        cost="Lower-cost InvestorLift alternative",
        reach="Pulls cash buyers from county deed records, auto-qualifies",
        best_for="AI dispo + state-specific assignment contracts w/ e-sign",
        notes="Verify current product/URL before relying on it.",
    ),
    DispoPlatform(
        name="New Western",
        url="https://www.newwestern.com/",
        payout="both",
        cost="No platform fee — they broker to their investor base",
        reach="Large vetted investor buyer base, many metros",
        best_for="Moving investment property to active buyers fast",
        notes="Understand their agent/brokerage model before relying on it.",
    ),
    DispoPlatform(
        name="Sundae",
        url="https://sundae.com/",
        payout="both",
        cost="Marketplace; investor-buyer side",
        reach="Investor buyers for distressed/as-is homes",
        best_for="Distressed sellers → investor buyer pool",
        notes="More seller-side, but a buyer pool exists.",
    ),
    DispoPlatform(
        name="Local cash-buyer list (your own)",
        url="(public county deed records → recent cash purchases)",
        payout="assignment",
        cost="Free — your time pulling public records",
        reach="As big as you build it; highest spread (no middleman)",
        best_for="The durable asset — you keep the full fee",
        notes="Maps to wholesale/data/buyers.py; build it from day one.",
    ),
    DispoPlatform(
        name="REI groups / BiggerPockets marketplace",
        url="https://www.biggerpockets.com/forums",
        payout="both",
        cost="Free",
        reach="Local investors, JV / co-wholesale partners",
        best_for="Co-wholesale splits and finding buyers this week",
        notes="Fastest finder-fee route for a brand-new operator.",
    ),
]


def by_payout(model: str) -> list[DispoPlatform]:
    """Filter to platforms supporting a payout model ('assignment'|'finder'|'both')."""
    return [p for p in PLATFORMS if p.payout == model or p.payout == "both"]


def for_market(_market: str) -> list[DispoPlatform]:
    """All listed channels operate nationally; returns the full list.

    Hook kept so a future version can rank by metro coverage.
    """
    return list(PLATFORMS)
