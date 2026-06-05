"""
county_records — public pre-foreclosure record access (legitimate sourcing)
==========================================================================

Pulls from PUBLIC county legal notices — the records governments publish
specifically so the public (including investors) can see upcoming
foreclosure sales. This is not people-search PII and not scraping a
ToS-protected listing site; it is the open-records channel the pros use.

Currently models Dallas County, TX (covers ZIP 75215 and the wider
DFW launch market). The county clerk posts Notice of Trustee's Sale PDFs,
organised by month and city, plus a public search portal for newer filings.

IMPORTANT — sourcing is allowed; CONTACT is still gated. Accessing public
records does not by itself authorise calling/texting these owners (TCPA,
equity-purchaser / foreclosure-consultant laws). Route any outreach through
`wholesale.compliance.gate` and prefer compliant channels (direct mail,
attorney-reviewed contracts). See docs/compliant_sourcing.md.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

# Dallas County public foreclosure notices.
DALLAS_BASE = "https://www.dallascounty.org/department/countyclerk/media/foreclosure"
DALLAS_INDEX = "https://www.dallascounty.org/government/county-clerk/recording/foreclosures.php"
DALLAS_PUBLIC_SEARCH = "https://dallas.tx.publicsearch.us/"

# Cities within Dallas County most relevant to the 75215 launch market.
DALLAS_CITIES = [
    "Dallas", "DeSoto", "Garland", "Mesquite", "Irving", "Grand_Prairie",
    "Duncanville", "Lancaster", "Cedar_Hill", "Richardson", "Carrollton",
]

# County clerk / public-record portals for the other launch markets, so the
# sourcing seam is documented per market (verify URLs as counties change them).
LAUNCH_MARKET_SOURCES = {
    "TX-Dallas": DALLAS_INDEX,
    "FL-Tampa": "https://www.hillsclerk.com/Records-and-Reports/Official-Records",
    "GA-Atlanta": "https://www.gsccca.org/search/lien/namesearch.asp",  # GA lien/lis pendens
    "OH-Columbus": "https://countyclerk.franklincountyohio.gov/",
    "NC-Charlotte": "https://meckrod.manatron.com/",  # Mecklenburg Register of Deeds
}


@dataclass
class NoticeDoc:
    """A pointer to one public foreclosure-notice PDF (legal notice)."""
    city: str
    month: str
    url: str

    def to_dict(self) -> dict:
        return asdict(self)


def dallas_notice_urls(month: str, cities: list[str] | None = None,
                       max_per_city: int = 6) -> list[NoticeDoc]:
    """Build the candidate public-PDF URLs for a given month.

    The county names files `City_N.pdf` (N starting at 1). We enumerate the
    first `max_per_city` per city; callers fetch and keep the ones that exist.
    Nothing here downloads — it just builds the public URLs to review.
    """
    docs: list[NoticeDoc] = []
    for city in (cities or DALLAS_CITIES):
        for n in range(1, max_per_city + 1):
            docs.append(NoticeDoc(
                city=city.replace("_", " "),
                month=month,
                url=f"{DALLAS_BASE}/{month}/{city}_{n}.pdf",
            ))
    return docs


def public_search_url(market: str = "TX-Dallas") -> str:
    """Return the human-usable public records portal for a launch market."""
    if market == "TX-Dallas":
        return DALLAS_PUBLIC_SEARCH
    return LAUNCH_MARKET_SOURCES.get(market, "")


def market_source(market: str) -> str:
    """Where to pull public distressed-owner records for a given market."""
    return LAUNCH_MARKET_SOURCES.get(market, "")
