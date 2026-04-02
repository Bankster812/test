"""
DCF Model — Discounted Cash Flow
=================================
Deterministic unlevered DCF with terminal value, sensitivity table,
and full year-by-year projection. Brain provides the parameters;
this class does the math.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass
class DCFInputs:
    ebitda:              float           # Current-year EBITDA
    revenue:             float           # Current-year revenue
    revenue_growth:      float | list[float]  # Single rate or per-year list
    ebitda_margin:       float           # EBITDA / Revenue
    wacc:                float           # Discount rate
    terminal_growth:     float           # Terminal growth rate (Gordon Growth)
    tax_rate:            float
    capex_pct:           float           # CapEx / Revenue
    nwc_pct:             float           # Change in NWC / Change in Revenue
    depreciation_pct:    float           # D&A / Revenue
    projection_years:    int  = 5
    net_debt:            float = 0.0     # For bridge from EV to equity value
    shares_outstanding:  float = 0.0    # For per-share valuation (optional)


@dataclass
class DCFResult:
    projected_revenue:   list[float]
    projected_ebitda:    list[float]
    projected_fcf:       list[float]
    terminal_value:      float
    pv_fcfs:             list[float]
    pv_terminal:         float
    enterprise_value:    float
    equity_value:        float | None
    price_per_share:     float | None
    sensitivity_table:   np.ndarray    # shape (n_wacc, n_tg)
    sensitivity_waccs:   list[float]
    sensitivity_tgs:     list[float]
    assumptions:         dict

    def __str__(self) -> str:
        lines = [
            f"  Revenue (Yr1):     ${self.projected_revenue[0]/1e6:.1f}M",
            f"  EBITDA  (Yr1):     ${self.projected_ebitda[0]/1e6:.1f}M",
            f"  FCF     (Yr1):     ${self.projected_fcf[0]/1e6:.1f}M",
            f"  Terminal Value:    ${self.terminal_value/1e6:.1f}M",
            f"  PV of FCFs:        ${sum(self.pv_fcfs)/1e6:.1f}M",
            f"  PV of TV:          ${self.pv_terminal/1e6:.1f}M",
            f"  Enterprise Value:  ${self.enterprise_value/1e6:.1f}M",
        ]
        if self.equity_value:
            lines.append(f"  Equity Value:      ${self.equity_value/1e6:.1f}M")
        if self.price_per_share:
            lines.append(f"  Price per Share:   ${self.price_per_share:.2f}")
        return "\n".join(lines)


class DCFModel:
    """Deterministic DCF computation."""

    def compute(self, inp: DCFInputs) -> DCFResult:
        years = inp.projection_years
        # Handle scalar or list growth rates
        if isinstance(inp.revenue_growth, (int, float)):
            growth_rates = [inp.revenue_growth] * years
        else:
            gr = list(inp.revenue_growth)
            growth_rates = (gr + [gr[-1]] * years)[:years]

        # Project revenues
        revenues = []
        rev = inp.revenue
        for g in growth_rates:
            rev = rev * (1.0 + g)
            revenues.append(rev)

        # Project EBITDA
        ebitdas = [r * inp.ebitda_margin for r in revenues]

        # Project UFCF = EBIT*(1-t) + D&A - CapEx - dNWC
        fcfs = []
        prev_rev = inp.revenue
        for i, (rev, ebt) in enumerate(zip(revenues, ebitdas)):
            da        = rev * inp.depreciation_pct
            ebit      = ebt - da
            nopat     = ebit * (1.0 - inp.tax_rate)
            capex     = rev * inp.capex_pct
            d_nwc     = (rev - prev_rev) * inp.nwc_pct
            fcf       = nopat + da - capex - d_nwc
            fcfs.append(fcf)
            prev_rev  = rev

        # Terminal value (Gordon Growth)
        denom = inp.wacc - inp.terminal_growth
        if abs(denom) < 1e-6:
            denom = 1e-6
        tv = fcfs[-1] * (1.0 + inp.terminal_growth) / denom

        # Discount to PV
        pv_fcfs = [
            f / (1.0 + inp.wacc) ** (t + 1)
            for t, f in enumerate(fcfs)
        ]
        pv_tv = tv / (1.0 + inp.wacc) ** years

        ev = sum(pv_fcfs) + pv_tv
        eq = ev - inp.net_debt if inp.net_debt != 0 else None
        pps = eq / inp.shares_outstanding if (eq and inp.shares_outstanding > 0) else None

        # Sensitivity table: WACC ±200bp, terminal growth ±100bp
        wacc_range = [inp.wacc + d * 0.005 for d in range(-2, 3)]
        tg_range   = [inp.terminal_growth + d * 0.005 for d in range(-2, 3)]
        sens = np.zeros((len(wacc_range), len(tg_range)), dtype=np.float32)
        for wi, w in enumerate(wacc_range):
            for ti, tg in enumerate(tg_range):
                d2 = w - tg
                if abs(d2) < 1e-6:
                    d2 = 1e-6
                tv2 = fcfs[-1] * (1 + tg) / d2
                pv2 = sum(f / (1 + w) ** (t + 1) for t, f in enumerate(fcfs))
                sens[wi, ti] = pv2 + tv2 / (1 + w) ** years

        return DCFResult(
            projected_revenue  = revenues,
            projected_ebitda   = ebitdas,
            projected_fcf      = fcfs,
            terminal_value     = tv,
            pv_fcfs            = pv_fcfs,
            pv_terminal        = pv_tv,
            enterprise_value   = ev,
            equity_value       = eq,
            price_per_share    = pps,
            sensitivity_table  = sens,
            sensitivity_waccs  = wacc_range,
            sensitivity_tgs    = tg_range,
            assumptions        = {
                "wacc": inp.wacc, "terminal_growth": inp.terminal_growth,
                "ebitda_margin": inp.ebitda_margin, "tax_rate": inp.tax_rate,
                "projection_years": years,
            },
        )

    @staticmethod
    def from_brain_params(params, user_inputs: dict) -> DCFInputs:
        """Merge brain params with user-supplied inputs. User inputs take priority."""
        def pick(key, brain_key, default):
            return user_inputs.get(key, params.value(brain_key, default))
        return DCFInputs(
            ebitda           = user_inputs.get("ebitda", 50e6),
            revenue          = user_inputs.get("revenue", user_inputs.get("ebitda", 50e6) / max(params.value("ebitda_margin", 0.20), 0.01)),
            revenue_growth   = pick("revenue_growth", "revenue_growth", 0.08),
            ebitda_margin    = pick("ebitda_margin",  "ebitda_margin",  0.20),
            wacc             = pick("wacc",           "wacc",           0.10),
            terminal_growth  = pick("terminal_growth","terminal_growth",0.025),
            tax_rate         = pick("tax_rate",       "tax_rate",       0.25),
            capex_pct        = pick("capex_pct",      "capex_pct",      0.05),
            nwc_pct          = pick("nwc_pct",        "nwc_pct",        0.03),
            depreciation_pct = pick("depreciation_pct","depreciation_pct",0.04),
            projection_years = user_inputs.get("projection_years", 5),
            net_debt         = user_inputs.get("net_debt", 0.0),
            shares_outstanding = user_inputs.get("shares_outstanding", 0.0),
        )
