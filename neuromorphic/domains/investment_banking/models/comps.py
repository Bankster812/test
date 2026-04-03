"""Comparable Company Analysis — Trading Comps"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


@dataclass
class CompanyData:
    name:        str
    ev:          float          # Enterprise value
    revenue:     float
    ebitda:      float
    ebit:        float
    net_income:  float
    shares:      float = 0.0
    price:       float = 0.0
    net_debt:    float = 0.0
    sector:      str   = ""


@dataclass
class CompsInputs:
    target:      CompanyData
    comparables: list[CompanyData]


@dataclass
class CompsResult:
    multiples_table:    dict[str, dict[str, float]]  # company → {ev_ebitda, ev_rev, pe}
    stats:              dict[str, dict[str, float]]  # metric → {mean, median, p25, p75}
    implied_ev_range:   tuple[float, float]          # (low, high) from median ±1 std
    implied_eq_range:   tuple[float, float]
    football_field:     dict[str, tuple[float, float]]  # methodology → (lo, hi)

    def __str__(self) -> str:
        lines = []
        for metric, s in self.stats.items():
            lines.append(
                f"  {metric:12s}: median={s['median']:.1f}x  "
                f"mean={s['mean']:.1f}x  range=[{s['p25']:.1f}x – {s['p75']:.1f}x]"
            )
        lo, hi = self.implied_ev_range
        lines.append(f"\n  Implied EV:  ${lo/1e6:.0f}M – ${hi/1e6:.0f}M")
        return "\n".join(lines)


class CompsModel:
    _METRICS = ["ev_ebitda", "ev_revenue", "pe_ratio"]

    def compute(self, inp: CompsInputs) -> CompsResult:
        rows: dict[str, dict[str, float]] = {}
        for c in inp.comparables:
            rows[c.name] = {
                "ev_ebitda":  c.ev / max(c.ebitda, 1),
                "ev_revenue": c.ev / max(c.revenue, 1),
                "pe_ratio":   (c.price * c.shares) / max(c.net_income, 1) if c.net_income > 0 else np.nan,
            }

        stats: dict[str, dict[str, float]] = {}
        for m in self._METRICS:
            vals = [v[m] for v in rows.values() if not np.isnan(v[m])]
            if vals:
                arr = np.array(vals)
                stats[m] = {
                    "mean":   float(arr.mean()),
                    "median": float(np.median(arr)),
                    "p25":    float(np.percentile(arr, 25)),
                    "p75":    float(np.percentile(arr, 75)),
                    "std":    float(arr.std()),
                }

        # Implied EV range (use EV/EBITDA median ± 1 std)
        tgt = inp.target
        ev_low = ev_high = 0.0
        if "ev_ebitda" in stats and tgt.ebitda > 0:
            s    = stats["ev_ebitda"]
            ev_low  = (s["median"] - s["std"]) * tgt.ebitda
            ev_high = (s["median"] + s["std"]) * tgt.ebitda
        eq_low  = max(ev_low  - tgt.net_debt, 0)
        eq_high = max(ev_high - tgt.net_debt, 0)

        football_field = {}
        for m, label in [("ev_ebitda","EV/EBITDA"),("ev_revenue","EV/Revenue")]:
            if m in stats and (tgt.ebitda if m == "ev_ebitda" else tgt.revenue) > 0:
                metric_val = tgt.ebitda if m == "ev_ebitda" else tgt.revenue
                football_field[label] = (
                    stats[m]["p25"] * metric_val,
                    stats[m]["p75"] * metric_val,
                )

        return CompsResult(
            multiples_table  = rows,
            stats            = stats,
            implied_ev_range = (ev_low, ev_high),
            implied_eq_range = (eq_low, eq_high),
            football_field   = football_field,
        )

    @staticmethod
    def from_brain_params(params, user_inputs: dict) -> CompsInputs:
        target = CompanyData(**user_inputs.get("target", {
            "name": "Target", "ev": 0, "revenue": 100e6,
            "ebitda": 20e6, "ebit": 15e6, "net_income": 10e6,
        }))
        comps  = [CompanyData(**c) for c in user_inputs.get("comparables", [])]
        return CompsInputs(target=target, comparables=comps)
