"""
DCF Model — Discounted Cash Flow
=================================
Deterministic unlevered DCF with terminal value, sensitivity table,
and full year-by-year projection. Brain provides the parameters;
this class does the math.

Conventions
-----------
* Unlevered free cash flow:  UFCF = EBIT*(1-t) + D&A - CapEx - dNWC.
* Terminal value, two methods (cross-checked against each other):
    - "perpetuity"    : Gordon Growth, TV = FCF_N*(1+g) / (WACC - g).
    - "exit_multiple" : TV = exit_multiple * terminal-year EBITDA.
* Discounting:
    - Full-year:  FCF_t discounted at (1+WACC)^t,    perpetuity TV at ^N.
    - Mid-year :  FCF_t discounted at (1+WACC)^(t-0.5),
                  perpetuity TV at ^(N-0.5) (consistent with the mid-year
                  cash-flow stream). An exit-multiple TV represents a sale
                  on the terminal date and is always discounted at ^N.
"""

from __future__ import annotations

from dataclasses import dataclass
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
    shares_outstanding:  float = 0.0     # For per-share valuation (optional)
    # --- terminal value / discounting options (all backward-compatible) ---
    terminal_method:        str   = "perpetuity"  # "perpetuity" | "exit_multiple"
    terminal_exit_multiple: float = 0.0           # EV/EBITDA used if exit_multiple
    mid_year_convention:    bool  = False


@dataclass
class DCFResult:
    projected_revenue:   list[float]
    projected_ebitda:    list[float]
    projected_fcf:       list[float]
    terminal_value:      float           # TV from the selected method
    pv_fcfs:             list[float]
    pv_terminal:         float
    enterprise_value:    float
    equity_value:        float           # EV - net_debt (== EV when debt-free)
    price_per_share:     float | None
    sensitivity_table:   np.ndarray      # shape (n_wacc, n_tg)
    sensitivity_waccs:   list[float]
    sensitivity_tgs:     list[float]
    assumptions:         dict
    # --- terminal value cross-check ---
    tv_perpetuity:            float       # Gordon-growth TV (undiscounted)
    tv_exit_multiple:         float       # exit-multiple TV (undiscounted)
    implied_exit_multiple:    float       # perpetuity TV / terminal EBITDA
    implied_perpetuity_growth: float      # g implied by the exit-multiple TV
    mid_year:                 bool = False

    def __str__(self) -> str:
        lines = [
            f"  Revenue (Yr1):     ${self.projected_revenue[0]/1e6:.1f}M",
            f"  EBITDA  (Yr1):     ${self.projected_ebitda[0]/1e6:.1f}M",
            f"  FCF     (Yr1):     ${self.projected_fcf[0]/1e6:.1f}M",
            f"  Terminal Value:    ${self.terminal_value/1e6:.1f}M",
            f"  PV of FCFs:        ${sum(self.pv_fcfs)/1e6:.1f}M",
            f"  PV of TV:          ${self.pv_terminal/1e6:.1f}M  "
            f"({self.pv_terminal/max(self.enterprise_value,1e-9)*100:.0f}% of EV)",
            f"  Enterprise Value:  ${self.enterprise_value/1e6:.1f}M",
            f"  Implied exit EV/EBITDA (perp.): {self.implied_exit_multiple:.1f}x",
            f"  Equity Value:      ${self.equity_value/1e6:.1f}M",
        ]
        if self.price_per_share:
            lines.append(f"  Price per Share:   ${self.price_per_share:.2f}")
        return "\n".join(lines)


class DCFModel:
    """Deterministic DCF computation."""

    def compute(self, inp: DCFInputs) -> DCFResult:
        self._validate(inp)
        years = inp.projection_years

        growth_rates = self._growth_schedule(inp.revenue_growth, years)

        # Project revenue and EBITDA
        revenues = []
        rev = inp.revenue
        for g in growth_rates:
            rev = rev * (1.0 + g)
            revenues.append(rev)
        ebitdas = [r * inp.ebitda_margin for r in revenues]

        # Project unlevered FCF = EBIT*(1-t) + D&A - CapEx - dNWC
        fcfs = []
        prev_rev = inp.revenue
        for rev, ebt in zip(revenues, ebitdas):
            da        = rev * inp.depreciation_pct
            ebit      = ebt - da
            nopat     = ebit * (1.0 - inp.tax_rate)
            capex     = rev * inp.capex_pct
            d_nwc     = (rev - prev_rev) * inp.nwc_pct
            fcf       = nopat + da - capex - d_nwc
            fcfs.append(fcf)
            prev_rev  = rev

        terminal_ebitda = ebitdas[-1]

        # --- Terminal value: compute both methods, select one ---
        tv_perp = self._tv_perpetuity(fcfs[-1], inp.wacc, inp.terminal_growth)
        tv_exit = inp.terminal_exit_multiple * terminal_ebitda
        if inp.terminal_method == "exit_multiple":
            tv = tv_exit
        else:
            tv = tv_perp

        # Cross-check metrics
        implied_exit_mult = tv_perp / max(terminal_ebitda, 1e-9)
        implied_perp_g = self._implied_growth(tv_exit, fcfs[-1], inp.wacc) if tv_exit > 0 else 0.0

        # --- Discount to present value ---
        pv_fcfs = [f * self._discount_factor(inp.wacc, t + 1, inp.mid_year_convention)
                   for t, f in enumerate(fcfs)]
        # Perpetuity TV follows the mid-year stream; exit-multiple TV is a
        # terminal-date sale and is discounted at the full final year.
        if inp.terminal_method == "exit_multiple":
            tv_period = float(years)
        else:
            tv_period = years - 0.5 if inp.mid_year_convention else float(years)
        pv_tv = tv / (1.0 + inp.wacc) ** tv_period

        ev = sum(pv_fcfs) + pv_tv
        eq = ev - inp.net_debt
        pps = eq / inp.shares_outstanding if inp.shares_outstanding > 0 else None

        sens, wacc_range, tg_range = self._sensitivity(inp, fcfs, terminal_ebitda)

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
                "projection_years": years, "terminal_method": inp.terminal_method,
                "mid_year_convention": inp.mid_year_convention,
            },
            tv_perpetuity             = tv_perp,
            tv_exit_multiple          = tv_exit,
            implied_exit_multiple     = implied_exit_mult,
            implied_perpetuity_growth = implied_perp_g,
            mid_year                  = inp.mid_year_convention,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate(inp: DCFInputs) -> None:
        if inp.projection_years < 1:
            raise ValueError("projection_years must be >= 1")
        if inp.wacc <= 0:
            raise ValueError(f"wacc must be positive, got {inp.wacc}")
        if not (0.0 <= inp.tax_rate < 1.0):
            raise ValueError(f"tax_rate must be in [0, 1), got {inp.tax_rate}")
        if inp.revenue <= 0:
            raise ValueError("revenue must be positive")
        if inp.terminal_method not in ("perpetuity", "exit_multiple"):
            raise ValueError(f"unknown terminal_method: {inp.terminal_method!r}")
        if inp.terminal_method == "perpetuity" and inp.terminal_growth >= inp.wacc:
            raise ValueError(
                f"terminal_growth ({inp.terminal_growth:.3f}) must be < wacc "
                f"({inp.wacc:.3f}) for a Gordon-growth terminal value"
            )
        if inp.terminal_method == "exit_multiple" and inp.terminal_exit_multiple <= 0:
            raise ValueError("terminal_exit_multiple must be > 0 when "
                             "terminal_method='exit_multiple'")

    @staticmethod
    def _growth_schedule(growth, years: int) -> list[float]:
        if isinstance(growth, (int, float)):
            return [float(growth)] * years
        gr = list(growth)
        if not gr:
            return [0.0] * years
        return (gr + [gr[-1]] * years)[:years]

    @staticmethod
    def _tv_perpetuity(fcf_n: float, wacc: float, g: float) -> float:
        denom = wacc - g
        if denom <= 0:                      # guarded upstream, belt-and-braces
            denom = 1e-6
        return fcf_n * (1.0 + g) / denom

    @staticmethod
    def _implied_growth(tv: float, fcf_n: float, wacc: float) -> float:
        # Invert Gordon Growth: tv = fcf_n*(1+g)/(wacc-g)  ->  solve for g.
        if tv <= 0 or fcf_n == 0:
            return 0.0
        return (tv * wacc - fcf_n) / (tv + fcf_n)

    @staticmethod
    def _discount_factor(wacc: float, period: int, mid_year: bool) -> float:
        exp = period - 0.5 if mid_year else period
        return 1.0 / (1.0 + wacc) ** exp

    def _sensitivity(self, inp: DCFInputs, fcfs, terminal_ebitda):
        """WACC ±200bp × terminal-growth ±100bp grid of enterprise values."""
        years = inp.projection_years
        wacc_range = [inp.wacc + d * 0.005 for d in range(-2, 3)]
        tg_range   = [inp.terminal_growth + d * 0.005 for d in range(-2, 3)]
        sens = np.zeros((len(wacc_range), len(tg_range)), dtype=np.float32)
        for wi, w in enumerate(wacc_range):
            if w <= 0:
                sens[wi, :] = np.nan
                continue
            pv_explicit = sum(f * self._discount_factor(w, t + 1, inp.mid_year_convention)
                              for t, f in enumerate(fcfs))
            for ti, tg in enumerate(tg_range):
                if inp.terminal_method == "exit_multiple":
                    tv2 = inp.terminal_exit_multiple * terminal_ebitda
                    period = float(years)
                else:
                    if tg >= w:                      # invalid perpetuity cell
                        sens[wi, ti] = np.nan
                        continue
                    tv2 = self._tv_perpetuity(fcfs[-1], w, tg)
                    period = years - 0.5 if inp.mid_year_convention else float(years)
                sens[wi, ti] = pv_explicit + tv2 / (1.0 + w) ** period
        return sens, wacc_range, tg_range

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
            terminal_method        = user_inputs.get("terminal_method", "perpetuity"),
            terminal_exit_multiple = user_inputs.get("terminal_exit_multiple", 0.0),
            mid_year_convention    = user_inputs.get("mid_year_convention", False),
        )
