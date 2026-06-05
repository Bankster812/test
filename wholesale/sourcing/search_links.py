"""
search_links — motivated-seller search-link generator
=====================================================

Builds a CSV of Google search URLs that surface PUBLIC listings matching
distressed/motivated-seller keywords in a given location, using the
`site:` search operator. This generates *links a human clicks* — it does
not scrape, log in, or retrieve any listing data itself.

    python -m wholesale.sourcing.search_links 75215
    python -m wholesale.sourcing.search_links "Tampa, FL" --site realtor.com

Output columns: keyword, site, query, google_url
"""

from __future__ import annotations

import argparse
import csv
import urllib.parse

# Common keywords that flag motivated sellers / value-add opportunities on
# public listing sites.
KEYWORDS = [
    "fixer upper",
    "as-is",
    "investor special",
    "handyman special",
    "needs work",
    "needs TLC",
    "motivated seller",
    "probate",
    "estate sale",
    "must sell",
    "bring all offers",
    "cash only",
    "tear down",
    "value add",
    "sold as-is where-is",
]


def build_rows(location: str, site: str = "zillow.com",
               keywords: list[str] | None = None) -> list[dict]:
    """Return one row per keyword with a ready-to-click Google search URL."""
    rows = []
    for kw in (keywords or KEYWORDS):
        query = f'site:{site} "{kw}" "{location}"'
        url = "https://www.google.com/search?q=" + urllib.parse.quote(query)
        rows.append({"keyword": kw, "site": site, "query": query, "google_url": url})
    return rows


def write_csv(location: str, path: str | None = None, site: str = "zillow.com",
              keywords: list[str] | None = None) -> str:
    """Write the CSV and return its path."""
    rows = build_rows(location, site=site, keywords=keywords)
    safe = location.replace(",", "").replace(" ", "_")
    out = path or f"seller_search_links_{safe}_{site.split('.')[0]}.csv"
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["keyword", "site", "query", "google_url"])
        writer.writeheader()
        writer.writerows(rows)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("location", help='ZIP code or "City, ST"')
    ap.add_argument("--site", default="zillow.com",
                    help="listing site to target (default zillow.com)")
    ap.add_argument("--out", default=None, help="output CSV path")
    args = ap.parse_args(argv)
    path = write_csv(args.location, path=args.out, site=args.site)
    rows = build_rows(args.location, site=args.site)
    print(f"Wrote {len(rows)} search links for '{args.location}' -> {path}\n")
    for r in rows:
        print(f"  {r['keyword']:<22} {r['google_url']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
