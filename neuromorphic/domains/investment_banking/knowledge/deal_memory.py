"""
DealMemory — Precedent Transaction Store with Neural Pattern Completion
=======================================================================
Stores every deal the brain has seen or been told about.
Backed by a NumPy embedding matrix so the brain can do pattern-completion
retrieval: "what's most similar to this deal context?"

No external DB required — everything lives in memory and can be serialised
to a .npz file for persistence.

Architecture
------------
  Each deal → fixed-length embedding vector (128-d float32)
  Retrieval   → cosine similarity scan (O(N × 128), fast for N < 100 K)
  Storage     → NumPy structured array + overflow list
  Persistence → np.savez / np.load

Embedding dimensions (128 total)
---------------------------------
  0-15   : sector one-hot (16 sectors)
  16-31  : buyer-type one-hot (strategic, PE, SPAC, …)
  32-47  : year normalised (2000-2030 window)
  48-63  : deal size log-normalised
  64-79  : EV/EBITDA multiple normalised
  80-95  : premium normalised
  96-111 : outcome flags (completed, broken, renegotiated, …)
  112-127: concept bag-of-words (16 high-freq IB concepts)
"""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np

logger = logging.getLogger("ib_brain.deal_memory")

# ── Vocabulary & sector maps ────────────────────────────────────────────────

SECTORS = [
    "technology", "healthcare", "energy", "financials", "industrials",
    "consumer", "telecom", "real_estate", "materials", "utilities",
    "media", "retail", "aerospace", "pharma", "biotech", "other",
]
BUYER_TYPES = [
    "strategic", "pe_buyout", "spac", "sovereign_wealth", "family_office",
    "corp_carveout", "management_buyout", "hostile", "merger_of_equals", "other",
    "unknown", "vc", "hedge_fund", "pension", "endowment", "other2",
]
IB_CONCEPTS = [
    "synergies", "premium", "accretion", "dilution", "leverage", "covenant",
    "earnout", "collar", "break_fee", "reverse_break", "go_shop", "no_shop",
    "fiduciary", "appraisal", "recapitalisation", "spin_off",
]


# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class Deal:
    # Identity
    deal_name:       str
    target:          str
    acquirer:        str
    year:            int

    # Economics
    deal_size_m:     float          # $ millions
    ev_ebitda:       float = 0.0
    ev_revenue:      float = 0.0
    premium_pct:     float = 0.0    # % premium to unaffected price
    irr:             float = 0.0    # for PE deals
    moic:            float = 0.0

    # Classification
    sector:          str = "other"
    buyer_type:      str = "strategic"
    completed:       bool = True
    broken:          bool = False
    renegotiated:    bool = False

    # Free-form tags and notes
    concepts:        list[str] = field(default_factory=list)
    notes:           str = ""
    source:          str = ""

    # Internal
    ingested_at:     float = field(default_factory=time.time)
    embedding:       Optional[np.ndarray] = field(default=None, repr=False)


# ── Core class ───────────────────────────────────────────────────────────────

class DealMemory:
    """
    Persistent, searchable store for precedent M&A transactions.

    Parameters
    ----------
    capacity : int   max deals kept in the fast embedding index
    """

    EMB_DIM = 128

    def __init__(self, capacity: int = 50_000):
        self.capacity   = capacity
        self._deals:    list[Deal]       = []
        self._emb_mat:  np.ndarray | None = None   # (N, 128) float32, built lazily
        self._dirty     = False

        # Seed with a handful of landmark deals so the brain is never blank
        self._seed_landmark_deals()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, deal: Deal) -> None:
        """Ingest one deal into memory."""
        deal.embedding = self._embed(deal)
        self._deals.append(deal)
        self._dirty = True
        if len(self._deals) % 100 == 0:
            logger.info(f"DealMemory: {len(self._deals)} deals stored")

    def add_many(self, deals: list[Deal]) -> None:
        for d in deals:
            self.add(d)

    def search(
        self,
        query_deal: Deal | None = None,
        sector: str | None = None,
        buyer_type: str | None = None,
        min_year: int | None = None,
        max_year: int | None = None,
        min_size_m: float | None = None,
        top_k: int = 10,
    ) -> list[tuple[float, Deal]]:
        """
        Hybrid search: filter by metadata, then rank remaining by embedding similarity.

        Returns list of (similarity_score, Deal) sorted descending.
        """
        candidates = self._filter(sector, buyer_type, min_year, max_year, min_size_m)
        if not candidates:
            return []

        if query_deal is not None:
            q_emb = self._embed(query_deal).reshape(1, -1)
            c_emb = np.stack([d.embedding for d in candidates])  # (M, 128)
            sims  = self._cosine_sim(q_emb, c_emb)[0]           # (M,)
            idx   = np.argsort(sims)[::-1][:top_k]
            return [(float(sims[i]), candidates[i]) for i in idx]
        else:
            # No query vector — return most recent
            recent = sorted(candidates, key=lambda d: d.ingested_at, reverse=True)
            return [(1.0, d) for d in recent[:top_k]]

    def get_sector_stats(self, sector: str) -> dict:
        """Compute aggregate multiple stats for a sector."""
        deals = [d for d in self._deals if d.sector.lower() == sector.lower()
                 and d.ev_ebitda > 0]
        if not deals:
            return {}
        evs = np.array([d.ev_ebitda for d in deals])
        prems = np.array([d.premium_pct for d in deals if d.premium_pct > 0])
        return {
            "count":           len(deals),
            "ev_ebitda_mean":  float(np.mean(evs)),
            "ev_ebitda_median": float(np.median(evs)),
            "ev_ebitda_p25":   float(np.percentile(evs, 25)),
            "ev_ebitda_p75":   float(np.percentile(evs, 75)),
            "premium_mean":    float(np.mean(prems)) if len(prems) else 0.0,
            "premium_median":  float(np.median(prems)) if len(prems) else 0.0,
        }

    def __len__(self) -> int:
        return len(self._deals)

    def save(self, path: str) -> None:
        """Serialise to .npz file."""
        if not self._deals:
            return
        emb_mat = np.stack([d.embedding for d in self._deals])
        meta = [
            {k: v for k, v in asdict(d).items() if k != "embedding"}
            for d in self._deals
        ]
        import json
        np.savez_compressed(path, embeddings=emb_mat, meta=json.dumps(meta))
        logger.info(f"DealMemory saved {len(self._deals)} deals to {path}")

    def load(self, path: str) -> None:
        """Restore from .npz file."""
        import json
        data = np.load(path, allow_pickle=False)
        meta = json.loads(str(data["meta"]))
        embs = data["embeddings"]
        self._deals = []
        for i, m in enumerate(meta):
            d = Deal(**{k: v for k, v in m.items()})
            d.embedding = embs[i]
            self._deals.append(d)
        self._dirty = False
        logger.info(f"DealMemory loaded {len(self._deals)} deals from {path}")

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def _embed(self, deal: Deal) -> np.ndarray:
        v = np.zeros(self.EMB_DIM, dtype=np.float32)

        # 0-15: sector one-hot
        s_idx = SECTORS.index(deal.sector.lower()) if deal.sector.lower() in SECTORS else 15
        v[s_idx] = 1.0

        # 16-31: buyer-type one-hot
        b_idx = BUYER_TYPES.index(deal.buyer_type.lower()) if deal.buyer_type.lower() in BUYER_TYPES else 9
        v[16 + b_idx] = 1.0

        # 32-47: year (normalised to 2000-2030)
        v[32] = np.clip((deal.year - 2000) / 30.0, 0, 1)

        # 48-63: deal size log-normalised ($10M → 0.0, $100B → 1.0)
        if deal.deal_size_m > 0:
            v[48] = float(np.clip(np.log10(deal.deal_size_m / 10.0) / 4.0, 0, 1))

        # 64-79: EV/EBITDA (0x → 0.0, 30x → 1.0)
        v[64] = float(np.clip(deal.ev_ebitda / 30.0, 0, 1))

        # 80-95: premium (0% → 0.0, 100% → 1.0)
        v[80] = float(np.clip(deal.premium_pct / 100.0, 0, 1))

        # 96-111: outcome flags
        v[96]  = 1.0 if deal.completed else 0.0
        v[97]  = 1.0 if deal.broken else 0.0
        v[98]  = 1.0 if deal.renegotiated else 0.0

        # 112-127: concept BoW
        for concept in deal.concepts:
            c = concept.lower()
            if c in IB_CONCEPTS:
                v[112 + IB_CONCEPTS.index(c)] = 1.0

        return v

    def _filter(
        self,
        sector, buyer_type, min_year, max_year, min_size_m
    ) -> list[Deal]:
        out = self._deals
        if sector:
            out = [d for d in out if d.sector.lower() == sector.lower()]
        if buyer_type:
            out = [d for d in out if d.buyer_type.lower() == buyer_type.lower()]
        if min_year:
            out = [d for d in out if d.year >= min_year]
        if max_year:
            out = [d for d in out if d.year <= max_year]
        if min_size_m:
            out = [d for d in out if d.deal_size_m >= min_size_m]
        return out

    @staticmethod
    def _cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """a: (1, D), b: (N, D) → (1, N)"""
        a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
        b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
        return a_norm @ b_norm.T

    # ------------------------------------------------------------------
    # Seed data — landmark deals every IB analyst should know
    # ------------------------------------------------------------------

    def _seed_landmark_deals(self):
        seeds = [
            Deal("AOL-Time Warner", "Time Warner", "AOL", 2000, 182_000, ev_ebitda=26.0, premium_pct=71, sector="media", buyer_type="strategic", completed=True, concepts=["synergies", "premium"], notes="Largest merger in history at time; value destroyed"),
            Deal("Vodafone-Mannesmann", "Mannesmann", "Vodafone", 2000, 183_000, ev_ebitda=32.0, premium_pct=58, sector="telecom", buyer_type="strategic", completed=True, concepts=["premium", "synergies"]),
            Deal("RJR Nabisco LBO", "RJR Nabisco", "KKR", 1989, 31_400, ev_ebitda=11.2, irr=0.24, moic=3.1, sector="consumer", buyer_type="pe_buyout", completed=True, concepts=["leverage", "earnout"]),
            Deal("LinkedIn-Microsoft", "LinkedIn", "Microsoft", 2016, 26_200, ev_ebitda=18.0, premium_pct=49, sector="technology", buyer_type="strategic", completed=True, concepts=["premium", "synergies"]),
            Deal("WhatsApp-Facebook", "WhatsApp", "Meta", 2014, 19_000, premium_pct=35, sector="technology", buyer_type="strategic", completed=True),
            Deal("Dell-EMC", "EMC", "Dell", 2016, 67_000, ev_ebitda=12.0, premium_pct=28, sector="technology", buyer_type="strategic", completed=True, concepts=["leverage", "synergies"]),
            Deal("Kraft-Heinz", "Kraft Foods", "Heinz/3G", 2015, 46_000, ev_ebitda=14.5, premium_pct=30, sector="consumer", buyer_type="strategic", completed=True, concepts=["synergies", "recapitalisation"]),
            Deal("AB InBev-SABMiller", "SABMiller", "AB InBev", 2016, 107_000, ev_ebitda=16.8, premium_pct=44, sector="consumer", buyer_type="strategic", completed=True, concepts=["synergies", "premium"]),
            Deal("Exxon-Mobil", "Mobil", "Exxon", 1999, 81_000, ev_ebitda=8.5, premium_pct=26, sector="energy", buyer_type="strategic", completed=True, concepts=["synergies"]),
            Deal("Pfizer-Warner-Lambert", "Warner-Lambert", "Pfizer", 2000, 87_000, ev_ebitda=24.0, premium_pct=60, sector="pharma", buyer_type="strategic", completed=True, concepts=["premium", "go_shop"]),
            Deal("Allergan-Actavis", "Allergan", "Actavis", 2015, 70_500, ev_ebitda=22.0, premium_pct=30, sector="pharma", buyer_type="strategic", completed=True, concepts=["synergies"]),
            Deal("Sprint-T-Mobile", "Sprint", "T-Mobile", 2020, 26_500, ev_ebitda=6.5, premium_pct=18, sector="telecom", buyer_type="strategic", completed=True, concepts=["synergies", "collar"]),
            Deal("Broadcom-VMware", "VMware", "Broadcom", 2023, 69_000, ev_ebitda=14.0, premium_pct=44, sector="technology", buyer_type="strategic", completed=True, concepts=["synergies", "leverage"]),
            Deal("Twitter-Musk", "Twitter", "E. Musk", 2022, 44_000, ev_ebitda=32.0, premium_pct=38, sector="media", buyer_type="strategic", completed=True, concepts=["leverage", "fiduciary", "renegotiated"], renegotiated=True),
            Deal("Activision-Microsoft", "Activision Blizzard", "Microsoft", 2023, 68_700, ev_ebitda=19.0, premium_pct=45, sector="media", buyer_type="strategic", completed=True, concepts=["synergies", "premium"]),
        ]
        for d in seeds:
            d.embedding = self._embed(d)
            self._deals.append(d)
        logger.info(f"DealMemory seeded with {len(seeds)} landmark deals")
