"""
county_score — rank counties for operability (playbook formula)
===============================================================

A county is worth working only if records are accessible AND there's enough
cash-buyer demand to exit. This implements the playbook's weighted score so
sourcing can be pointed at the best counties first. Inputs are 0..1 quality
signals (you supply them from real data; sensible defaults included).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

WEIGHTS = {
    "foreclosure_visibility": 0.20,
    "tax_visibility": 0.15,
    "probate_visibility": 0.10,
    "cash_buyer_density": 0.20,
    "institutional_presence": 0.10,
    "price_band_fit": 0.10,
    "title_friendliness": 0.10,
    "data_cleanliness": 0.05,
}


@dataclass
class CountyScore:
    state: str
    county: str
    score_total: float
    components: dict

    def to_dict(self) -> dict:
        return asdict(self)


def score_county(state: str, county: str, **signals: float) -> CountyScore:
    """signals are 0..1; missing ones default to 0.5 (neutral)."""
    comp = {}
    total = 0.0
    for key, w in WEIGHTS.items():
        v = float(signals.get(key, 0.5))
        v = max(0.0, min(1.0, v))
        comp[key] = v
        total += v * w
    return CountyScore(state=state.upper(), county=county,
                       score_total=round(total * 100, 1), components=comp)
