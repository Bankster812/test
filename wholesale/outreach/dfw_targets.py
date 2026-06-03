"""
dfw_targets — real, PUBLIC B2B targets for the Dallas–Fort Worth launch
======================================================================

Publicly-advertised businesses (investor associations and cash-buyer /
"we buy houses" companies). These are commercial entities that exist to
network and transact — exactly the right first outreach for the buyer /
co-wholesale side. This is public business info, not consumer PII.

Preferred channel is each org's own website/contact form (not scraped
email). Riley drafts the message; you submit it through their form. Mind
CAN-SPAM if you email (accurate header, your identity, opt-out).

Verify details before sending — businesses move and rebrand.
"""

from __future__ import annotations

# (name, kind, url, note)
INVESTOR_GROUPS = [
    ("REIA Dallas (Texas Wealth Network)", "cowholesale", "https://reiadallas.com/",
     "Largest TX REIA chapter — deals, buyers, partners; join + pitch deals"),
    ("REIA DFW", "cowholesale", "https://reiadfw.com/",
     "20,000+ member community; networking + deal pitching"),
    ("DFW Real Estate Investor Club", "cowholesale", "https://www.dfwreiclub.com/",
     "Local investor club — buyers and JV partners"),
    ("Dallas-Ft. Worth REIA (Meetup)", "cowholesale",
     "https://www.meetup.com/dallas-fort-worth-real-estate-investment-association/",
     "Meets 4th Tuesday monthly 6-9pm; in-person networking"),
]

CASH_BUYERS = [
    ("Cash House Buyers DFW", "cashbuyer", "https://www.cashhousebuyersdfw.com/",
     "Local cash buyer — add to buyer list / co-wholesale"),
    ("Legacy Home Buyers", "cashbuyer", "https://www.legacybuyers.com/",
     "DFW cash buyer since 2016"),
    ("Big State Home Buyers", "cashbuyer", "https://www.bigstatehomebuyers.com/",
     "10,000+ TX closings"),
    ("Ninebird Properties", "cashbuyer", "https://www.ninebp.com/",
     "DFW cash home-buying company"),
    ("Emerald House Buyers", "cashbuyer", "https://www.emeraldhousebuyers.com/",
     "Fort Worth investment group / cash buyer"),
    ("Super Cash For Houses", "cashbuyer", "https://supercashforhouses.com/",
     "Long-established Dallas cash buyer"),
    ("Southern Hills Home Buyers", "cashbuyer", "https://www.southernhillshomebuyers.com/",
     "Dallas-area cash buyer"),
    ("Pioneer Home Buyers", "cashbuyer", "https://www.pioneerhb.com/",
     "Fort Worth cash buyer"),
]

ALL_TARGETS = INVESTOR_GROUPS + CASH_BUYERS


def load_contacts(market: str = "Dallas, TX", area: str = "DFW / 75215"):
    """Return Contact objects for every DFW target, ready for the BizDev queue."""
    from ..agents.bizdev import Contact
    out = []
    for name, kind, url, _note in ALL_TARGETS:
        out.append(Contact(name=name, kind=kind, market=market, area=area, url=url))
    return out
