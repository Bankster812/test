"""Precedent Transaction Analysis"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


@dataclass
class TransactionData:
    name:        str
    year:        int
    sector:      str
    ev:          float
    revenue:     float
    ebitda:      float
    premium_pct: float = 0.0
    strategic:   bool  = True     # strategic vs financial buyer


@dataclass
class PrecedentsInputs:
    target_ebitda:   float
    target_revenue:  float
    target_sector:   str
    transactions:    list[TransactionData]
    year_filter:     int  = 2015    # only use deals after this year
    strategic_only:  bool = False


@dataclass
class PrecedentsResult:
    filtered_transactions: list[TransactionData]
    multiples:             dict[str, dict[str, float]]   # metric → stats
    implied_ev_range:      tuple[float, float]
    mean_premium:          float
    football_field:        dict[str, tuple[float, float]]

    def __str__(self) -> str:
        n    = len(self.filtered_transactions)
        lo, hi = self.implied_ev_range
        m    = self.multiples.get("ev_ebitda", {})
        return (
            f"  Precedents used:   {n} transactions\n"
            f"  EV/EBITDA median:  {m.get('median', 0):.1f}x  "
            f"[{m.get('p25',0):.1f}x – {m.get('p75',0):.1f}x]\n"
            f"  Mean premium:      {self.mean_premium*100:.0f}%\n"
            f"  Implied EV range:  ${lo/1e6:.0f}M – ${hi/1e6:.0f}M"
        )


class PrecedentsModel:
    def compute(self, inp: PrecedentsInputs) -> PrecedentsResult:
        # Filter
        txns = [
            t for t in inp.transactions
            if t.year >= inp.year_filter
            and (not inp.strategic_only or t.strategic)
        ]
        if not txns:
            txns = inp.transactions  # fallback: use all

        # Calculate multiples
        ev_ebitda_vals = [t.ev / max(t.ebitda, 1) for t in txns if t.ebitda > 0]
        ev_rev_vals    = [t.ev / max(t.revenue, 1) for t in txns if t.revenue > 0]
        premiums       = [t.premium_pct for t in txns if t.premium_pct > 0]

        def stats(vals):
            if not vals:
                return {"mean": 0, "median": 0, "p25": 0, "p75": 0, "std": 0}
            a = np.array(vals)
            return {
                "mean":   float(a.mean()),
                "median": float(np.median(a)),
                "p25":    float(np.percentile(a, 25)),
                "p75":    float(np.percentile(a, 75)),
                "std":    float(a.std()),
            }

        s_ev_ebitda = stats(ev_ebitda_vals)
        s_ev_rev    = stats(ev_rev_vals)

        # Implied EV (median ± std)
        ev_lo = (s_ev_ebitda["median"] - s_ev_ebitda["std"]) * inp.target_ebitda
        ev_hi = (s_ev_ebitda["median"] + s_ev_ebitda["std"]) * inp.target_ebitda

        football = {}
        if inp.target_ebitda > 0:
            football["EV/EBITDA (prec.)"] = (s_ev_ebitda["p25"] * inp.target_ebitda,
                                              s_ev_ebitda["p75"] * inp.target_ebitda)
        if inp.target_revenue > 0:
            football["EV/Revenue (prec.)"] = (s_ev_rev["p25"] * inp.target_revenue,
                                               s_ev_rev["p75"] * inp.target_revenue)

        return PrecedentsResult(
            filtered_transactions = txns,
            multiples             = {"ev_ebitda": s_ev_ebitda, "ev_revenue": s_ev_rev},
            implied_ev_range      = (max(ev_lo, 0), max(ev_hi, 0)),
            mean_premium          = float(np.mean(premiums)) if premiums else 0.0,
            football_field        = football,
        )

    @staticmethod
    def from_brain_params(params, user_inputs: dict) -> PrecedentsInputs:
        txn_dicts = user_inputs.get("transactions", [])
        return PrecedentsInputs(
            target_ebitda  = user_inputs.get("target_ebitda",  50e6),
            target_revenue = user_inputs.get("target_revenue", 250e6),
            target_sector  = user_inputs.get("target_sector",  "general"),
            transactions   = [TransactionData(**t) for t in txn_dicts],
            year_filter    = user_inputs.get("year_filter",    2015),
        )
