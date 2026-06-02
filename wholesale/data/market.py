"""
market — synthetic US property + motivated-seller lead feed
==========================================================

Generates plausible distressed-property leads across the company's
markets. Numbers are intentionally coherent (price tracks sqft and metro)
so that downstream underwriting produces sensible spreads.

To go live, replace `MarketFeed.next_leads` with an adapter that pulls
from a real list source (county pre-foreclosure, tax-delinquent, absentee
owner) and a skip-trace provider for seller contact info.
"""

from __future__ import annotations

import random

from ..core.models import Property, Seller

# Rough as-is $/sqft by metro — keeps generated prices believable.
_METRO_PPSF = {
    "TX-Dallas": 185, "FL-Tampa": 240, "GA-Atlanta": 205,
    "OH-Columbus": 165, "NC-Charlotte": 215,
}
_CITY_BY_METRO = {
    "TX-Dallas": ("Dallas", "TX", "752"),
    "FL-Tampa": ("Tampa", "FL", "336"),
    "GA-Atlanta": ("Atlanta", "GA", "303"),
    "OH-Columbus": ("Columbus", "OH", "432"),
    "NC-Charlotte": ("Charlotte", "NC", "282"),
}
_STREETS = [
    "Maple", "Oak", "Cedar", "Birch", "Elm", "Sycamore", "Magnolia",
    "Pinecrest", "Lakeview", "Briarwood", "Hillcrest", "Ridgeway",
]
_STREET_TYPES = ["St", "Ave", "Dr", "Ln", "Ct", "Way", "Blvd"]
_DISTRESS = [
    "pre-foreclosure", "tax-delinquent", "tired-landlord", "inherited / probate",
    "code violations", "vacant", "divorce", "job relocation",
]
_MOTIVATION = {
    "pre-foreclosure": "distress", "tax-delinquent": "distress",
    "tired-landlord": "burnout", "inherited / probate": "inherited",
    "code violations": "distress", "vacant": "carrying-cost",
    "divorce": "divorce", "job relocation": "relocation",
}
_TYPES = ["SFR", "SFR", "SFR", "duplex", "townhome"]  # SFR weighted
_FIRST = ["James", "Mary", "Robert", "Linda", "Michael", "Patricia", "David",
          "Barbara", "William", "Elizabeth", "Carlos", "Maria", "Aisha", "Wei"]
_LAST = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
         "Davis", "Rodriguez", "Martinez", "Nguyen", "Patel", "Okafor", "Chen"]


class MarketFeed:
    """Deterministic-ish synthetic feed (seedable for reproducible demos)."""

    def __init__(self, markets: list[str], seed: int | None = None) -> None:
        self.markets = markets
        self.rng = random.Random(seed)

    def _make_property(self) -> Property:
        metro = self.rng.choice(self.markets)
        city, state, zip3 = _CITY_BY_METRO[metro]
        sqft = self.rng.randint(950, 2600)
        beds = max(2, min(5, sqft // 600))
        baths = float(self.rng.choice([1, 1.5, 2, 2.5, 3]))
        year = self.rng.randint(1955, 2008)
        ppsf = _METRO_PPSF[metro] * self.rng.uniform(0.82, 1.05)  # as-is discount
        distress = self.rng.choice(_DISTRESS)
        return Property(
            address=f"{self.rng.randint(100, 9899)} {self.rng.choice(_STREETS)} "
                    f"{self.rng.choice(_STREET_TYPES)}",
            city=city, state=state, zip=f"{zip3}{self.rng.randint(10, 99)}",
            metro=metro, beds=beds, baths=baths, sqft=sqft, year_built=year,
            property_type=self.rng.choice(_TYPES), distress=distress,
            est_market_value=int(round(sqft * ppsf, -3)),
        )

    def _make_seller(self, prop: Property) -> Seller:
        distress = prop.distress
        motivation = _MOTIVATION[distress]
        # Hidden walk-away as a fraction of as-is market value. Deeply motivated
        # sellers (distress/carrying-cost) accept the steepest discounts; only
        # the bottom of each range clears a wholesale MAO — most leads won't.
        lo, hi = {
            "distress":      (0.55, 0.74),
            "carrying-cost": (0.58, 0.76),
            "inherited":     (0.60, 0.80),
            "divorce":       (0.62, 0.82),
            "burnout":       (0.63, 0.83),
            "relocation":    (0.66, 0.86),
        }[motivation]
        floor_frac = self.rng.uniform(lo, hi)
        walk_floor = int(round(prop.est_market_value * floor_frac, -3))
        asking = int(round(prop.est_market_value * self.rng.uniform(0.97, 1.12), -3))
        return Seller(
            name=f"{self.rng.choice(_FIRST)} {self.rng.choice(_LAST)}",
            motivation=motivation,
            asking_price=asking,
            reachable_via=self.rng.choice(["phone", "phone", "text", "email"]),
            flexibility=round((1.0 - floor_frac), 2),
            walk_floor=walk_floor,
        )

    def next_leads(self, n: int) -> list[tuple[Property, Seller]]:
        out = []
        for _ in range(n):
            p = self._make_property()
            out.append((p, self._make_seller(p)))
        return out
