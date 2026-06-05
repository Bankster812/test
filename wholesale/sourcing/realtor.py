"""
realtor — Realtor.com scraper (browser-driven)
=============================================

Pulls listings from Realtor.com filtered for distressed signals
(foreclosure, short sale, fixer-upper, "as-is"). Requires Playwright.

ToS WARNING: same caveat as the Zillow scraper. Realtor.com prohibits
automated access and uses bot detection.
"""

from __future__ import annotations

import json
import logging
import re

from ..core.models import Property, Seller
from .browser import StealthBrowser

logger = logging.getLogger("wholesale.sourcing.realtor")


class RealtorScraper:
    name = "realtor"

    # URL patterns Realtor.com accepts for distressed filters.
    PATHS = {
        "foreclosure": "/realestateandhomes-search/{state}/type-foreclosure",
        "fixer": "/realestateandhomes-search/{state}/show-fixer-upper",
    }

    def __init__(self, markets: list[str]) -> None:
        self.markets = markets

    def available(self) -> bool:
        return StealthBrowser.is_available()

    def fetch(self, limit: int = 25) -> list[tuple[Property, Seller]]:
        if not self.available():
            return []
        results: list[tuple[Property, Seller]] = []
        try:
            with StealthBrowser() as b:
                for market in self.markets:
                    if len(results) >= limit:
                        break
                    state = market.split("-")[0].lower()
                    for distress, path in self.PATHS.items():
                        if len(results) >= limit:
                            break
                        try:
                            url = "https://www.realtor.com" + path.format(state=state)
                            html = b.fetch(url, wait_selector="[data-testid='property-card']")
                            for prop, seller in self._parse(html, market, distress):
                                results.append((prop, seller))
                                if len(results) >= limit:
                                    return results
                        except Exception as e:
                            logger.warning("Realtor %s/%s failed: %s", state, distress, e)
        except RuntimeError as e:
            logger.warning("Realtor scrape skipped: %s", e)
        return results

    def _parse(self, html: str, metro: str, distress: str
               ) -> list[tuple[Property, Seller]]:
        out: list[tuple[Property, Seller]] = []
        m = re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', html, re.S)
        if not m:
            return out
        try:
            data = json.loads(m.group(1))
        except json.JSONDecodeError:
            return out
        for listing in _walk_listings(data)[:40]:
            try:
                loc = listing.get("location") or {}
                addr = loc.get("address") or {}
                price = int(listing.get("list_price") or 0)
                if not addr.get("line") or price <= 0:
                    continue
                desc = listing.get("description") or {}
                prop = Property(
                    address=addr.get("line", ""),
                    city=addr.get("city", ""),
                    state=addr.get("state_code", "TX"),
                    zip=str(addr.get("postal_code", "00000")),
                    metro=metro,
                    beds=int(desc.get("beds") or 3),
                    baths=float(desc.get("baths") or 2),
                    sqft=int(desc.get("sqft") or 1200),
                    year_built=int(desc.get("year_built") or 1980),
                    property_type=str(desc.get("type") or "SFR"),
                    distress=distress,
                    est_market_value=int(price * 1.05),
                )
                seller = Seller(
                    name="Realtor.com listing (owner not disclosed)",
                    motivation="distress" if distress == "foreclosure" else "as-is sale",
                    asking_price=price,
                    reachable_via="listing-agent",
                    flexibility=0.25,
                    walk_floor=int(price * 0.65),
                )
                out.append((prop, seller))
            except (ValueError, KeyError, TypeError) as e:
                logger.debug("Realtor listing skipped: %s", e)
                continue
        return out


def _walk_listings(obj) -> list[dict]:
    """Find any list of listing dicts in Realtor's __NEXT_DATA__."""
    if isinstance(obj, dict):
        if "properties" in obj and isinstance(obj["properties"], list):
            return obj["properties"]
        if "results" in obj and isinstance(obj["results"], list):
            return obj["results"]
        for v in obj.values():
            r = _walk_listings(v)
            if r:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = _walk_listings(v)
            if r:
                return r
    return []
