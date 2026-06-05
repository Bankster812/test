"""
buyers — the cash-buyer book the Disposition agent sells into
=============================================================

A seeded roster of investor buy-boxes. Disposition matches an
under-contract deal against these. To go live, replace `default_book`
with a pull from your CRM (buyer list segment) or a proptech buyer API.
"""

from __future__ import annotations

import random

from ..core.models import Buyer

_BUYER_NAMES = [
    "Lone Star REI", "Gulf Coast Holdings", "Peachtree Capital", "Buckeye Rentals",
    "Queen City Homes", "Sunbelt BRRRR Fund", "Magnolia Equity", "Redbrick Partners",
    "Anchor Cash Buyers", "Evergreen Property Group", "Foundry Real Estate",
    "Cardinal Investments",
]


def default_book(markets: list[str], seed: int | None = 7) -> list[Buyer]:
    rng = random.Random(seed)
    buyers: list[Buyer] = []
    for name in _BUYER_NAMES:
        # Each buyer covers 1–3 of the company's metros.
        cover = rng.sample(markets, k=rng.randint(1, min(3, len(markets))))
        buyers.append(Buyer(
            name=name,
            metros=cover,
            max_price=int(round(rng.uniform(180_000, 420_000), -3)),
            property_types=rng.sample(["SFR", "duplex", "townhome"],
                                      k=rng.randint(1, 3)),
            min_spread=rng.choice([8_000, 10_000, 12_000, 15_000]),
            cash_verified=rng.random() > 0.2,
            appetite=round(rng.uniform(0.45, 0.95), 2),
        ))
    return buyers
