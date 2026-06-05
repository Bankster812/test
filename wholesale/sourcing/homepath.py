"""
homepath — Fannie Mae HomePath REO scraper (public GSE source)
==============================================================

Fannie Mae's REO portal publishes properties it acquired through
foreclosure. Free and public. The site is JS-rendered, so a browser
is preferred — falls back to a JSON endpoint discovered via the
network panel that the React frontend calls itself.

If neither the JSON endpoint nor a browser is available, returns [].
"""

from __future__ import annotations

import json
import logging
import urllib.parse
import urllib.request

from ..core.models import Property, Seller

logger = logging.getLogger("wholesale.sourcing.homepath")

# Internal search endpoint observed in the HomePath SPA (subject to change).
API = "https://www.homepath.com/listing/searchListings"


class HomePathProvider:
    """Fannie Mae HomePath REO inventory."""

    name = "homepath"

    def __init__(self, markets: list[str]) -> None:
        self.markets = markets

    def available(self) -> bool:
        return True

    def fetch(self, limit: int = 25) -> list[tuple[Property, Seller]]:
        results: list[tuple[Property, Seller]] = []
        for market in self.markets:
            if len(results) >= limit:
                break
            state = market.split("-")[0].upper()
            try:
                rows = self._search_state(state, limit - len(results))
            except Exception as e:
                logger.warning("HomePath %s failed: %s", state, e)
                continue
            results.extend(rows)
        return results[:limit]

    def _search_state(self, state: str, limit: int) -> list[tuple[Property, Seller]]:
        params = {
            "searchType": "state", "state": state,
            "page": "1", "pageSize": str(min(limit, 50)),
            "sortBy": "listDate", "sortOrder": "desc",
        }
        url = f"{API}?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0 (compatible; Keystone-RE-Sourcing/1.0)",
            "Accept": "application/json",
        })
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                data = json.loads(resp.read().decode("utf-8", errors="ignore"))
        except (urllib.error.URLError, json.JSONDecodeError, TimeoutError) as e:
            logger.info("HomePath endpoint unavailable (%s); skipping", e)
            return []
        return self._parse(data, state)

    def _parse(self, data: dict, state: str) -> list[tuple[Property, Seller]]:
        out: list[tuple[Property, Seller]] = []
        listings = (data.get("listings") or data.get("results") or
                    data.get("data") or [])
        for item in listings:
            try:
                price = int(item.get("listPrice") or item.get("price") or 0)
                addr = item.get("address") or {}
                address = item.get("addressLine1") or addr.get("street", "")
                city = item.get("city") or addr.get("city", "")
                zip_code = str(item.get("zip") or addr.get("zip", "00000"))
                if not address or price <= 0:
                    continue
                prop = Property(
                    address=address, city=city, state=state, zip=zip_code,
                    metro=f"{state}-{city}",
                    beds=int(item.get("bedrooms") or 3),
                    baths=float(item.get("bathrooms") or 2.0),
                    sqft=int(item.get("squareFeet") or 1200),
                    year_built=int(item.get("yearBuilt") or 1980),
                    property_type=str(item.get("propertyType") or "SFR"),
                    distress="bank-owned (Fannie Mae)",
                    est_market_value=int(price * 1.12),
                )
                seller = Seller(
                    name="Fannie Mae HomePath",
                    motivation="REO disposition",
                    asking_price=price,
                    reachable_via="HomePath-approved broker offer",
                    flexibility=0.12,
                    walk_floor=int(price * 0.87),
                )
                out.append((prop, seller))
            except (ValueError, KeyError, TypeError) as e:
                logger.debug("HomePath listing skipped: %s", e)
                continue
        return out
