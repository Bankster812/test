"""
templates — B2B outreach drafts for the fast-cash channels
==========================================================

Three audiences, all business contacts:
  * realtor       — agents who feed you off-market/expired deals & cash buyers
  * cowholesale   — active wholesalers to JV with (split a fee this week)
  * cashbuyer     — investors for your buyer list / to assign contracts to

`draft(kind, **fields)` returns a filled message. Keep it honest and
specific; generic blasts get ignored (and risk CAN-SPAM issues).
"""

from __future__ import annotations

TEMPLATES = {
    "realtor": (
        "Subject: Cash investor buying in {market} — clean closings for your pipeline\n\n"
        "Hi {name},\n\n"
        "I'm a local cash investor actively buying SFRs in {area}. I close fast, "
        "as-is, no financing contingency, and I bring my own funds — the kind of "
        "clean, on-time closing that's easy commission for you.\n\n"
        "Two ways we can work together:\n"
        "  1) Send me off-market, expired, or price-dropped listings that need a "
        "cash buyer.\n"
        "  2) If you represent cash buyers, I sometimes have contracts to move.\n\n"
        "Open to a quick 10-minute call this week?\n\n"
        "{signature}"
    ),
    "cowholesale": (
        "Subject: JV / co-wholesale in {market}?\n\n"
        "Hey {name},\n\n"
        "I saw you're active wholesaling in {area}. I'm building buyer and deal "
        "flow here and would love to JV.\n\n"
        "  - If you have a property under contract that needs a buyer, I may have "
        "one — happy to split the assignment fee.\n"
        "  - If you've got hungry cash buyers and need deals, I can feed you some.\n\n"
        "Standard 50/50 on anything we close together, papered with a simple JV "
        "agreement. Worth a quick call?\n\n"
        "{signature}"
    ),
    "cashbuyer": (
        "Subject: Off-market {market} deals — want on the buyers list?\n\n"
        "Hi {name},\n\n"
        "I source off-market SFRs in {area} at investor pricing. I noticed you've "
        "been buying in the area, so I'd like to add you to my deal alerts.\n\n"
        "Quick so I only send deals that fit your box:\n"
        "  - Target metros / ZIPs:\n"
        "  - Max all-in price:\n"
        "  - SFR / multi / both, and min beds/baths:\n"
        "  - Buy-and-hold or flip:\n"
        "  - Proof of funds available? Typical close timeline:\n\n"
        "Send those back and I'll only ping you on matches.\n\n"
        "{signature}"
    ),
}


def draft(kind: str, name: str = "there", market: str = "Dallas, TX",
          area: str | None = None, signature: str = "[Your name]\n[LLC] · [phone]") -> str:
    """Fill a B2B outreach template. `area` defaults to `market`."""
    if kind not in TEMPLATES:
        raise ValueError(f"Unknown outreach kind: {kind!r}. "
                         f"Choose from {sorted(TEMPLATES)}.")
    return TEMPLATES[kind].format(name=name, market=market,
                                  area=area or market, signature=signature)
