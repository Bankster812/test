"""
disclosures — state-specific pre-foreclosure disclosures & compliant mail
=========================================================================

Operationalises the "win-win, in writing" requirements for buying from a
homeowner in or near foreclosure. These are DRAFTS for attorney review, not
legal advice — but they put the mandatory transparency (you may profit, price
may be below market, cancellation rights) into the documents, which is what
the equity-purchaser / assignment-disclosure statutes require.

Sources summarised in docs/legal_triage_5state.md. Verify current text with
counsel before use; statutes and effective dates change.

Channels: prefer DIRECT MAIL for first touch (exempt from TCPA, unlike
calls/texts). The mail templates embed each state's required solicitation
legend where one exists (e.g., Georgia SB 90 / HB 240).
"""

from __future__ import annotations

import datetime as _dt

_NLA = ("[DRAFT — NOT LEGAL ADVICE — attorney review required. Confirm the "
        "current statute in {state} before use.]")

# Mandatory / best-practice written disclosure language by state. Placeholders
# are filled from the deal. These summarise statutory intent; counsel must
# confirm exact wording, type size, and formatting (several require bold/caps).
EQUITY_DISCLOSURE = {
    "TX": ("TEXAS EQUITABLE-INTEREST DISCLOSURE (Prop. Code §5.0205): The "
           "undersigned buyer is acquiring an EQUITABLE INTEREST only and does "
           "NOT hold legal title. Buyer intends to ASSIGN this contract and may "
           "profit on the assignment. This is disclosed in writing to both the "
           "property owner and any potential assignee."),
    "FL": ("FLORIDA EQUITY-PURCHASER DISCLOSURE (Fla. Stat. §501.1377): Seller "
           "is or may be in default/foreclosure. Buyer may resell or assign this "
           "contract for profit at a price that may exceed the price paid to "
           "Seller. Seller has the RIGHT TO CANCEL this agreement within the "
           "statutory cancellation period without penalty. [Counsel: render in "
           "≥12-pt UPPERCASE; deliver signed copy within 3 hours.]"),
    "OH": ("OHIO WHOLESALER DISCLOSURE (O.R.C. §5301.95): The buyer is a "
           "wholesaler, is NOT the end buyer, intends to assign/market this "
           "contractual interest, MAY PROFIT, and the price may be BELOW market "
           "value. Seller may consult an attorney. If this disclosure is not "
           "provided, Seller may cancel before close of escrow without penalty. "
           "[Counsel: standalone document, bold ≥12-pt, signed & dated.]"),
    "GA": ("GEORGIA DISCLOSURE: Buyer intends to assign this contract and may "
           "profit; the price offered may be below fair market value. [Counsel: "
           "confirm whether a transactional disclosure beyond the SB 90 / HB 240 "
           "marketing legends is required.]"),
    "NC": ("NORTH CAROLINA DISCLOSURE: Buyer is acquiring and may ASSIGN this "
           "contract for profit and is acting as a principal, not a broker. This "
           "is NOT a foreclosure-rescue sale-leaseback; Seller is selling outright. "
           "[Counsel: ensure compliance with N.C.G.S. §§75-120–122; a "
           "sale-leaseback for gain requires paying ≥50% of FMV.]"),
}

# Georgia requires legends on unsolicited written offers (SB 90 / HB 240).
GA_SOLICITATION_LEGEND = (
    "THIS IS A SOLICITATION. THE SENDER IS CONTACTING YOU TO INQUIRE AS TO YOUR "
    "INTEREST IN SELLING YOUR HOME OR OTHER REAL ESTATE. YOU ARE UNDER NO "
    "OBLIGATION TO RESPOND."
)
GA_PRICE_LEGEND = (
    "THIS OFFER MAY OR MAY NOT BE THE FAIR MARKET VALUE OF THE PROPERTY."
)


def equity_disclosure(state: str, buyer_entity: str) -> str:
    st = state.upper()
    body = EQUITY_DISCLOSURE.get(
        st, f"{st}: Confirm required equity-purchaser/assignment disclosures "
            f"with counsel before contracting with a homeowner in default.")
    today = _dt.date.today().isoformat()
    return (f"{_NLA.format(state=st)}\n\n"
            f"PRE-FORECLOSURE / ASSIGNMENT DISCLOSURE — {today}\n"
            f"Buyer/Assignor: {buyer_entity}\n\n{body}\n\n"
            f"Seller signature: __________________________  Date: __________\n")


def direct_mail_letter(state: str, owner_name: str, address: str, city: str,
                       buyer_entity: str, names_price: bool = False) -> str:
    """A TCPA-safe first-touch letter. Embeds GA legends when state == GA."""
    st = state.upper()
    first = owner_name.split()[0] if owner_name else "Neighbor"
    legend = ""
    if st == "GA":
        legend = GA_SOLICITATION_LEGEND + ("\n\n" + GA_PRICE_LEGEND if names_price else "")
        legend += "\n\n"
    return (
        f"{legend}"
        f"Dear {first},\n\n"
        f"My name is [Your name] with {buyer_entity}. I buy homes for cash in "
        f"{city} and I'm reaching out about {address}.\n\n"
        f"If you're facing a tough situation, a fast as-is cash sale can let you "
        f"avoid the time, cost, and credit damage of a foreclosure — you pick the "
        f"closing date, pay no repairs or agent fees, and walk away clean. I may "
        f"resell or assign the contract, so my offer will be below full retail, "
        f"but it's fast and certain.\n\n"
        f"If that's helpful, call or text [phone], or mail the enclosed card. "
        f"You're under no obligation.\n\n"
        f"Warm regards,\n[Your name] — {buyer_entity}\n[phone] · [address]\n\n"
        f"{_NLA.format(state=st)}\n"
    )
