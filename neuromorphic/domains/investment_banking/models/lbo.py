"""
LBO Model — Leveraged Buyout
============================
Single, internally-consistent projection: interest is taken from the actual
debt balance each year (not a separate approximation), levered free cash flow
sweeps debt at ``debt_paydown_rate`` and any un-swept cash accumulates on the
balance sheet, and the equity IRR is solved from the full cash-flow vector.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class LBOInputs:
    ebitda:                float
    revenue:               float
    entry_ev_ebitda:       float
    revenue_growth:        float
    ebitda_margin_exit:    float           # Exit EBITDA margin
    leverage_ratio:        float           # Total debt / Entry EBITDA
    senior_debt_rate:      float
    hold_period:           int
    exit_multiple:         float
    tax_rate:              float
    capex_pct:             float
    nwc_pct:               float
    depreciation_pct:      float
    debt_paydown_rate:     float = 1.00   # % of levered FCF swept to debt (rest builds cash)


@dataclass
class LBOResult:
    entry_enterprise_value: float
    equity_contribution:    float
    total_debt:             float
    projected_ebitda:       list[float]
    projected_fcf:          list[float]    # levered FCF available for debt paydown
    debt_schedule:          list[float]    # Debt outstanding at year-end
    interest_expense:       list[float]
    cash_schedule:          list[float]    # accumulated balance-sheet cash
    exit_enterprise_value:  float
    exit_equity_value:      float
    debt_at_exit:           float
    cash_at_exit:           float
    net_debt_at_exit:       float
    irr:                    float
    moic:                   float
    sensitivity_irr:        np.ndarray    # shape (n_entry, n_exit)
    sensitivity_entries:    list[float]
    sensitivity_exits:      list[float]

    def __str__(self) -> str:
        entry = max(self.entry_enterprise_value, 1e-9)
        return (
            f"  Entry EV:    ${self.entry_enterprise_value/1e6:.1f}M  "
            f"({self.total_debt/entry*100:.0f}% debt)\n"
            f"  Equity In:   ${self.equity_contribution/1e6:.1f}M\n"
            f"  Exit EV:     ${self.exit_enterprise_value/1e6:.1f}M\n"
            f"  Net Debt @Exit: ${self.net_debt_at_exit/1e6:.1f}M\n"
            f"  Equity Out:  ${self.exit_equity_value/1e6:.1f}M\n"
            f"  IRR:         {self.irr*100:.1f}%\n"
            f"  MOIC:        {self.moic:.2f}x"
        )


class LBOModel:
    def compute(self, inp: LBOInputs) -> LBOResult:
        self._validate(inp)

        entry_ev   = inp.ebitda * inp.entry_ev_ebitda
        total_debt = min(inp.ebitda * inp.leverage_ratio, entry_ev)   # can't lever past EV
        equity_in  = entry_ev - total_debt

        # Linear EBITDA path from entry level to exit-margin × terminal revenue.
        rev_exit   = inp.revenue * (1.0 + inp.revenue_growth) ** inp.hold_period
        ebitda_exit_target = inp.ebitda_margin_exit * rev_exit

        ebitdas, fcfs, interests, debts, cashes = [], [], [], [], []
        debt = total_debt
        cash = 0.0
        rev = inp.revenue
        prev_rev = inp.revenue
        for yr in range(1, inp.hold_period + 1):
            rev = rev * (1.0 + inp.revenue_growth)
            # interpolate EBITDA from entry to exit target
            frac = yr / inp.hold_period
            ebt = inp.ebitda + (ebitda_exit_target - inp.ebitda) * frac
            ebt = max(ebt, 0.0)
            ebitdas.append(ebt)

            da       = rev * inp.depreciation_pct
            ebit     = ebt - da
            interest = debt * inp.senior_debt_rate          # on opening balance
            ebt_pretax = ebit - interest
            taxes    = max(ebt_pretax, 0.0) * inp.tax_rate
            net_income = ebt_pretax - taxes
            capex    = rev * inp.capex_pct
            d_nwc    = (rev - prev_rev) * inp.nwc_pct
            levered_fcf = net_income + da - capex - d_nwc   # cash available pre-financing
            fcfs.append(levered_fcf)

            sweep   = min(max(levered_fcf, 0.0) * inp.debt_paydown_rate, debt)
            debt    = debt - sweep
            cash    = cash + (levered_fcf - sweep)          # remainder builds (or draws) cash
            interests.append(interest)
            debts.append(debt)
            cashes.append(cash)
            prev_rev = rev

        exit_ebitda = ebitdas[-1]
        exit_ev     = exit_ebitda * inp.exit_multiple
        net_debt_exit = debts[-1] - cashes[-1]
        exit_eq     = exit_ev - net_debt_exit

        moic = max(exit_eq, 0.0) / max(equity_in, 1e-9)
        cashflows = [-equity_in] + [0.0] * (inp.hold_period - 1) + [exit_eq]
        irr = self._irr(cashflows)

        sens_irr, entry_range, exit_range = self._sensitivity(inp, exit_ebitda, net_debt_exit)

        return LBOResult(
            entry_enterprise_value = entry_ev,
            equity_contribution    = equity_in,
            total_debt             = total_debt,
            projected_ebitda       = ebitdas,
            projected_fcf          = fcfs,
            debt_schedule          = debts,
            interest_expense       = interests,
            cash_schedule          = cashes,
            exit_enterprise_value  = exit_ev,
            exit_equity_value      = exit_eq,
            debt_at_exit           = debts[-1],
            cash_at_exit           = cashes[-1],
            net_debt_at_exit       = net_debt_exit,
            irr                    = irr,
            moic                   = moic,
            sensitivity_irr        = sens_irr,
            sensitivity_entries    = entry_range,
            sensitivity_exits      = exit_range,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate(inp: LBOInputs) -> None:
        if inp.hold_period < 1:
            raise ValueError("hold_period must be >= 1")
        if inp.ebitda <= 0:
            raise ValueError("ebitda must be positive")
        if inp.entry_ev_ebitda <= 0 or inp.exit_multiple <= 0:
            raise ValueError("entry/exit multiples must be positive")
        if inp.leverage_ratio < 0:
            raise ValueError("leverage_ratio must be >= 0")
        if not (0.0 <= inp.tax_rate < 1.0):
            raise ValueError("tax_rate must be in [0, 1)")
        if not (0.0 <= inp.debt_paydown_rate <= 1.0):
            raise ValueError("debt_paydown_rate must be in [0, 1]")

    def _sensitivity(self, inp: LBOInputs, exit_ebitda: float, net_debt_exit: float):
        """Entry × exit multiple IRR grid (debt path held at base case)."""
        entry_range = [inp.entry_ev_ebitda + d for d in range(-2, 3)]
        exit_range  = [inp.exit_multiple    + d for d in range(-2, 3)]
        sens = np.zeros((len(entry_range), len(exit_range)), dtype=np.float32)
        for ei, em in enumerate(entry_range):
            if em <= 0:
                sens[ei, :] = np.nan
                continue
            ev_in = inp.ebitda * em
            d_in  = min(inp.ebitda * inp.leverage_ratio, ev_in)
            eq_in = max(ev_in - d_in, 1e-9)
            for xi, xm in enumerate(exit_range):
                if xm <= 0:
                    sens[ei, xi] = np.nan
                    continue
                eq_out = exit_ebitda * xm - net_debt_exit
                cfs = [-eq_in] + [0.0] * (inp.hold_period - 1) + [eq_out]
                sens[ei, xi] = self._irr(cfs)
        return sens, entry_range, exit_range

    @staticmethod
    def _irr(cashflows: list[float], lo: float = -0.999, hi: float = 10.0) -> float:
        """IRR via bisection on NPV. Returns clipped guess if no sign change."""
        def npv(r: float) -> float:
            return sum(cf / (1.0 + r) ** t for t, cf in enumerate(cashflows))

        f_lo, f_hi = npv(lo), npv(hi)
        if np.isnan(f_lo) or np.isnan(f_hi) or f_lo * f_hi > 0:
            # No bracketed root (e.g. total loss or all-positive flows).
            pos = sum(cf for cf in cashflows if cf > 0)
            neg = -sum(cf for cf in cashflows if cf < 0)
            if neg <= 0 or pos <= 0:
                return 0.0
            n = max(len(cashflows) - 1, 1)
            return float(np.clip((pos / neg) ** (1.0 / n) - 1.0, lo, hi))
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            f_mid = npv(mid)
            if abs(f_mid) < 1e-9:
                break
            if f_lo * f_mid < 0:
                hi = mid
            else:
                lo, f_lo = mid, f_mid
        return float(np.clip(0.5 * (lo + hi), -0.999, 10.0))

    @staticmethod
    def from_brain_params(params, user_inputs: dict) -> LBOInputs:
        def pick(key, brain_key, default):
            return user_inputs.get(key, params.value(brain_key, default))
        ebitda = user_inputs.get("ebitda", 50e6)
        return LBOInputs(
            ebitda             = ebitda,
            revenue            = user_inputs.get("revenue", ebitda / 0.20),
            entry_ev_ebitda    = pick("entry_multiple",   "ev_ebitda",      10.0),
            revenue_growth     = pick("revenue_growth",   "revenue_growth",  0.08),
            ebitda_margin_exit = pick("ebitda_margin",    "ebitda_margin",   0.22),
            leverage_ratio     = pick("leverage_ratio",   "leverage_ratio",  4.5),
            senior_debt_rate   = pick("interest_rate",    "interest_rate",   0.06),
            hold_period        = int(pick("hold_period",  "hold_period",     5)),
            exit_multiple      = pick("exit_multiple",    "exit_multiple",   11.0),
            tax_rate           = pick("tax_rate",         "tax_rate",        0.25),
            capex_pct          = pick("capex_pct",        "capex_pct",       0.05),
            nwc_pct            = pick("nwc_pct",          "nwc_pct",         0.03),
            depreciation_pct   = pick("depreciation_pct","depreciation_pct", 0.04),
            debt_paydown_rate  = pick("debt_paydown_rate","debt_paydown_rate",1.00),
        )
