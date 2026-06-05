"""
providers — pluggable lead-data sources (the live-data seam)
============================================================

One interface, swappable implementations:

  * SyntheticProvider — the built-in MarketFeed (demo mode only).
  * CsvProvider       — reads a user-supplied CSV of real leads (live mode).
  * AttomProvider     — ATTOM Data property API (needs ATTOM_API_KEY).
  * CountyPublicProvider — public county foreclosure records (manual/assisted).

In "live" mode (WS_LEAD_SOURCE=live, the default) the company starts with an
empty pipeline and only CsvProvider / AttomProvider can add leads.
In "demo" mode (WS_LEAD_SOURCE=demo) SyntheticProvider auto-generates leads
every tick for testing and demonstrations.

CSV format (one row per lead, header required):
  address, city, state, zip, metro, beds, baths, sqft, year_built,
  property_type, distress, est_market_value, seller_name, motivation,
  asking_price, reachable_via[, flexibility][, walk_floor]
"""

from __future__ import annotations

import csv
import io
import os
import random

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


class CsvProvider(LeadProvider):
    """Read real leads from a user-supplied CSV file.

    Set WS_LEADS_CSV to the file path. The CSV is read once at init; rows are
    consumed in order. When the file is exhausted the provider returns [].

    Required columns: address, city, state, zip, metro, beds, baths, sqft,
    year_built, property_type, distress, est_market_value, seller_name,
    motivation, asking_price, reachable_via

    Optional: flexibility (0.0–1.0), walk_floor (int USD)
    """
    name = "csv"

    def __init__(self, csv_path: str) -> None:
        self._path = csv_path
        self._rows: list[dict] = []
        self._cursor = 0
        if os.path.isfile(csv_path):
            with open(csv_path, newline="", encoding="utf-8-sig") as fh:
                self._rows = list(csv.DictReader(fh))

    def available(self) -> bool:
        return bool(self._rows) and self._cursor < len(self._rows)

    def fetch_leads(self, market: str, limit: int) -> list[tuple[Property, Seller]]:
        results: list[tuple[Property, Seller]] = []
        while self._cursor < len(self._rows) and len(results) < limit:
            row = self._rows[self._cursor]
            self._cursor += 1
            try:
                prop = Property(
                    address=row["address"].strip(),
                    city=row["city"].strip(),
                    state=row["state"].strip().upper(),
                    zip=row.get("zip", row.get("zip_code", "00000")).strip(),
                    metro=row["metro"].strip(),
                    beds=int(float(row.get("beds", 3))),
                    baths=float(row.get("baths", 2)),
                    sqft=int(float(row.get("sqft", 1500))),
                    year_built=int(float(row.get("year_built", 1980))),
                    property_type=row.get("property_type", "SFR").strip(),
                    distress=row.get("distress", "pre-foreclosure").strip(),
                    est_market_value=int(float(row["est_market_value"])),
                )
                market_val = prop.est_market_value
                asking = int(float(row.get("asking_price", market_val)))
                flexibility = float(row.get("flexibility", 0.35))
                walk_floor = int(float(row.get("walk_floor", market_val * 0.60)))
                seller = Seller(
                    name=row.get("seller_name", "Unknown Owner").strip(),
                    motivation=row.get("motivation", "distress").strip(),
                    asking_price=asking,
                    reachable_via=row.get("reachable_via", "mail").strip(),
                    flexibility=flexibility,
                    walk_floor=walk_floor,
                )
                results.append((prop, seller))
            except (KeyError, ValueError):
                continue  # skip malformed rows silently
        return results

    @classmethod
    def from_text(cls, text: str) -> "CsvProvider":
        """Create from raw CSV text (used by the dashboard /api/leads/import)."""
        inst = cls.__new__(cls)
        inst._path = "<upload>"
        inst._cursor = 0
        inst._rows = list(csv.DictReader(io.StringIO(text)))
        return inst


def get_provider(markets: list[str], seed: int | None = None) -> LeadProvider:
    """Factory: choose provider based on WS_LEAD_PROVIDER and WS_LEADS_CSV."""
    csv_path = os.environ.get("WS_LEADS_CSV", "")
    if csv_path:
        p = CsvProvider(csv_path)
        if p.available():
            return p

    choice = os.environ.get("WS_LEAD_PROVIDER", "synthetic").lower()
    if choice == "attom":
        p = AttomProvider()
        if p.available():
            return p
    if choice == "county_public":
        return CountyPublicProvider()
    return SyntheticProvider(markets, seed=seed)
