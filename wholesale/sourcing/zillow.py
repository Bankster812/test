"""
zillow — distressed-listing scraper (Camoufox-driven)
====================================================

Pulls listings from Zillow search results filtered for motivated-seller
signals: pre-foreclosure, foreclosed, auction, FSBO. Uses Camoufox
(patched Firefox) — see `camoufox_browser.py` — to bypass PerimeterX /
Cloudflare detection. Falls back gracefully if Camoufox is not installed.

URL FILTERS (verified against Zillow's own UI):
  https://www.zillow.com/{state}/pre-foreclosure_lt/   pre-foreclosure
  https://www.zillow.com/{state}/foreclosures/         REO (bank-owned)
  https://www.zillow.com/{state}/auctions/             auction
  https://www.zillow.com/{state}/fsbo/                 for-sale-by-owner

EXTRACTION: parses `<script id="__NEXT_DATA__">` JSON (most stable) and
falls back to DOM selectors `[data-testid='property-card']` / `article[id^='zpid']`.

ToS WARNING: Zillow's Terms of Use prohibit automated access. Camoufox
defeats the technical detection but not the legal claim. Operator's
responsibility to decide whether use complies with applicable law.
"""

from __future__ import annotations

import json
import logging
import re

from ..core.models import Property, Seller
from .camoufox_browser import CamoufoxBrowser

logger = logging.getLogger("wholesale.sourcing.zillow")


class ZillowScraper:
    name = "zillow"

    FILTERS = {
        "pre-foreclosure": "pre-foreclosure_lt",
        "foreclosed": "foreclosures",
        "auction": "auctions",
        "fsbo": "fsbo",
    }

    def __init__(self, markets: list[str]) -> None:
        self.markets = markets

    def available(self) -> bool:
        return CamoufoxBrowser.is_available()

    def fetch(self, limit: int = 25) -> list[tuple[Property, Seller]]:
        """Scrape up to `limit` distressed listings across configured markets."""
        if not self.available():
            return []
        results: list[tuple[Property, Seller]] = []
        try:
            with CamoufoxBrowser() as b:
                for market in self.markets:
                    state = market.split("-")[0].lower()
                    for distress, slug in self.FILTERS.items():
                        if len(results) >= limit:
                            return results
                        url = f"https://www.zillow.com/{state}/{slug}/"
                        try:
                            html = b.fetch(url,
                                wait_selector="[data-testid='property-card']")
                        except Exception as e:
                            logger.warning("Zillow %s/%s fetch failed: %s",
                                           state, slug, e)
                            continue
                        for prop, seller in self._parse(html, market, distress):
                            results.append((prop, seller))
                            if len(results) >= limit:
                                return results
        except RuntimeError as e:
            logger.warning("Zillow scrape skipped: %s", e)
        return results

    def _parse(self, html: str, metro: str, distress: str
               ) -> list[tuple[Property, Seller]]:
        """Extract listings from Zillow's __NEXT_DATA__ JSON blob."""
        out: list[tuple[Property, Seller]] = []
        m = re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>',
                      html, re.S)
        if not m:
            return out
        try:
            data = json.loads(m.group(1))
        except json.JSONDecodeError:
            return out
        for item in _find_list_results(data)[:40]:
            try:
                addr_obj = item.get("address") or {}
                hdp = (item.get("hdpData") or {}).get("homeInfo") or {}
                price = int(item.get("unformattedPrice") or hdp.get("price") or 0)
                address = (item.get("addressStreet")
                           or addr_obj.get("streetAddress")
                           or hdp.get("streetAddress") or "")
                if not address or price <= 0:
                    continue
                prop = Property(
                    address=address,
                    city=item.get("addressCity") or addr_obj.get("city")
                         or hdp.get("city", ""),
                    state=(item.get("addressState") or addr_obj.get("state")
                           or hdp.get("state") or "TX").upper(),
                    zip=str(item.get("addressZipcode") or addr_obj.get("zipcode")
                            or hdp.get("zipcode", "00000")),
                    metro=metro,
                    beds=int(item.get("beds") or hdp.get("bedrooms") or 0),
                    baths=float(item.get("baths") or hdp.get("bathrooms") or 0),
                    sqft=int(item.get("area") or hdp.get("livingArea") or 1200),
                    year_built=int(hdp.get("yearBuilt") or 1980),
                    property_type=str(hdp.get("homeType") or "SFR"),
                    distress=distress,
                    est_market_value=int(item.get("zestimate")
                                         or hdp.get("zestimate") or price),
                )
                seller = Seller(
                    name="Zillow listing (owner not disclosed)",
                    motivation=_motivation_for(distress),
                    asking_price=price,
                    reachable_via="listing-agent",
                    flexibility=0.30,
                    walk_floor=int(prop.est_market_value * 0.62),
                )
                out.append((prop, seller))
            except (KeyError, ValueError, TypeError) as e:
                logger.debug("Zillow listing skipped: %s", e)
                continue
        return out


def _find_list_results(obj) -> list[dict]:
    """Locate the listing array in Zillow's __NEXT_DATA__ (varies by page)."""
    if isinstance(obj, dict):
        for key in ("listResults", "searchResults", "results"):
            if key in obj and isinstance(obj[key], list):
                return obj[key]
        for v in obj.values():
            r = _find_list_results(v)
            if r:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = _find_list_results(v)
            if r:
                return r
    return []


def _motivation_for(distress: str) -> str:
    return {
        "pre-foreclosure": "distress",
        "foreclosed": "bank-owned",
        "auction": "distress",
        "fsbo": "by-owner",
    }.get(distress, "distress")
