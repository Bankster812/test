"""
providers — pluggable lead-data sources (the live-data seam)
============================================================

One interface, swappable implementations:

  * SyntheticProvider — the built-in MarketFeed (default; runs offline).
  * AttomProvider     — ATTOM Data property API (needs ATTOM_API_KEY).
  * CountyPublicProvider — public county foreclosure records (manual/assisted).

Select via `WS_LEAD_PROVIDER` (default "synthetic"). Real providers return
no leads until credentialed — by design, so nothing pretends to have live
data it doesn't. `get_provider()` is what the company calls.
"""

from __future__ import annotations

import os

from ..core.models import Property, Seller
from ..data.market import MarketFeed


class LeadProvider:
    name = "base"

    def available(self) -> bool:
        return False

    def fetch_leads(self, market: str, limit: int) -> list[tuple[Property, Seller]]:
        raise NotImplementedError


class SyntheticProvider(LeadProvider):
    """Default: realistic synthetic leads, fully offline."""
    name = "synthetic"

    def __init__(self, markets: list[str], seed: int | None = None) -> None:
        self._feed = MarketFeed(markets, seed=seed)

    def available(self) -> bool:
        return True

    def fetch_leads(self, market: str, limit: int):
        return self._feed.next_leads(limit)


class AttomProvider(LeadProvider):
    """ATTOM Data API seam. Returns nothing until ATTOM_API_KEY + transport.

    To go live: implement `_query` (ATTOM property/preforeclosure endpoints),
    map the response into Property/Seller. Keep seller contact acquisition to
    a licensed skip-trace provider and behind the compliance gate.
    """
    name = "attom"

    def __init__(self) -> None:
        self.api_key = os.environ.get("ATTOM_API_KEY", "")

    def available(self) -> bool:
        return bool(self.api_key)

    def fetch_leads(self, market: str, limit: int):
        if not self.available():
            return []
        return self._query(market, limit)  # pragma: no cover

    def _query(self, market: str, limit: int):  # pragma: no cover
        raise NotImplementedError("Wire ATTOM endpoints here.")


class CountyPublicProvider(LeadProvider):
    """Public county foreclosure records (see sourcing.county_records).

    These are PUBLIC legal notices; sourcing is allowed, contact is gated.
    Parsing scanned PDFs is assisted/manual, so this returns nothing
    automatically — it documents the source rather than fabricating leads.
    """
    name = "county_public"

    def available(self) -> bool:
        return False

    def fetch_leads(self, market: str, limit: int):
        return []


def get_provider(markets: list[str], seed: int | None = None) -> LeadProvider:
    """Factory keyed off WS_LEAD_PROVIDER (default 'synthetic')."""
    choice = os.environ.get("WS_LEAD_PROVIDER", "synthetic").lower()
    if choice == "attom":
        p = AttomProvider()
        if p.available():
            return p
        # Fall back to synthetic so the company keeps running.
        return SyntheticProvider(markets, seed=seed)
    if choice == "county_public":
        return CountyPublicProvider()
    return SyntheticProvider(markets, seed=seed)
