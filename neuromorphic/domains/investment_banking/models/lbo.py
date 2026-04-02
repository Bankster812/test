"""LBO Model — Leveraged Buyout"""
from __future__ import annotations
from dataclasses import dataclass, field
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
    debt_paydown_rate:     float = 0.50   # % FCF allocated to debt repayment


@dataclass
class LBOResult:
    entry_enterprise_value: float
    equity_contribution:    float
    total_debt:             float
    projected_ebitda:       list[float]
    projected_fcf:          list[float]
    debt_schedule:          list[float]    # Debt outstanding per year
    interest_expense:       list[float]
    exit_enterprise_value:  float
    exit_equity_value:      float
    irr:                    float
    moic:                   float
    debt_at_exit:           float
    sensitivity_irr:        np.ndarray    # shape (n_entry, n_exit)
    sensitivity_entries:    list[float]
    sensitivity_exits:      list[float]

    def __str__(self) -> str:
        return (
            f"  Entry EV:    ${self.entry_enterprise_value/1e6:.1f}M  "
            f"({self.total_debt/self.entry_enterprise_value*100:.0f}% debt)\n"
            f"  Equity In:   ${self.equity_contribution/1e6:.1f}M\n"
            f"  Exit EV:     ${self.exit_enterprise_value/1e6:.1f}M\n"
            f"  Equity Out:  ${self.exit_equity_value/1e6:.1f}M\n"
            f"  IRR:         {self.irr*100:.1f}%\n"
            f"  MOIC:        {self.moic:.2f}x"
        )


class LBOModel:
    def compute(self, inp: LBOInputs) -> LBOResult:
        entry_ev  = inp.ebitda * inp.entry_ev_ebitda
        total_debt = inp.ebitda * inp.leverage_ratio
        equity_in  = entry_ev - total_debt

        # Project financials
        ebitdas = []
        fcfs    = []
        rev     = inp.revenue
        prev_rev = inp.revenue
        ebt      = inp.ebitda
        for yr in range(inp.hold_period):
            rev  = rev * (1.0 + inp.revenue_growth)
            margin = inp.ebitda + (inp.ebitda_margin_exit * rev - inp.ebitda) * (yr + 1) / inp.hold_period
            ebt  = max(margin, 0)
            ebitdas.append(ebt)
            da   = rev * inp.depreciation_pct
            ebit = ebt - da
            int_exp_approx = total_debt * inp.senior_debt_rate * (0.9 ** yr)  # rough declining
            nopat = (ebit - int_exp_approx) * (1.0 - inp.tax_rate)
            capex = rev * inp.capex_pct
            d_nwc = (rev - prev_rev) * inp.nwc_pct
            fcf   = nopat + da - capex - d_nwc
            fcfs.append(max(fcf, 0))
            prev_rev = rev

        # Debt schedule (paydown from FCF)
        debt    = total_debt
        debts   = []
        int_exp = []
        for yr, fcf in enumerate(fcfs):
            ie = debt * inp.senior_debt_rate
            int_exp.append(ie)
            paydown = min(fcf * inp.debt_paydown_rate, debt)
            debt   = max(0, debt - paydown)
            debts.append(debt)

        # Exit
        exit_ebitda = ebitdas[-1]
        exit_ev     = exit_ebitda * inp.exit_multiple
        exit_eq     = exit_ev - debts[-1]

        # IRR / MOIC
        moic   = max(exit_eq, 0) / max(equity_in, 1)
        # Newton-Raphson IRR
        irr    = self._solve_irr(equity_in, exit_eq, inp.hold_period)

        # Sensitivity: entry x exit multiples
        entry_range = [inp.entry_ev_ebitda + d for d in range(-2, 3)]
        exit_range  = [inp.exit_multiple    + d for d in range(-2, 3)]
        sens_irr = np.zeros((len(entry_range), len(exit_range)), dtype=np.float32)
        for ei, em in enumerate(entry_range):
            for xi, xm in enumerate(exit_range):
                ev_in  = inp.ebitda * em
                d_in   = min(inp.ebitda * inp.leverage_ratio, ev_in)
                eq_in  = max(ev_in - d_in, 1.0)
                ev_out = exit_ebitda * xm
                eq_out = max(ev_out - debts[-1], 0)
                sens_irr[ei, xi] = self._solve_irr(eq_in, eq_out, inp.hold_period)

        return LBOResult(
            entry_enterprise_value = entry_ev,
            equity_contribution    = equity_in,
            total_debt             = total_debt,
            projected_ebitda       = ebitdas,
            projected_fcf          = fcfs,
            debt_schedule          = debts,
            interest_expense       = int_exp,
            exit_enterprise_value  = exit_ev,
            exit_equity_value      = exit_eq,
            irr                    = irr,
            moic                   = moic,
            debt_at_exit           = debts[-1],
            sensitivity_irr        = sens_irr,
            sensitivity_entries    = entry_range,
            sensitivity_exits      = exit_range,
        )

    @staticmethod
    def _solve_irr(eq_in: float, eq_out: float, years: int) -> float:
        """Simple MOIC-based IRR approximation via Newton's method."""
        if eq_in <= 0 or eq_out <= 0:
            return 0.0
        # Initial guess
        r = (eq_out / eq_in) ** (1.0 / max(years, 1)) - 1.0
        for _ in range(50):
            f  = -eq_in + eq_out / (1 + r) ** years
            df = -years * eq_out / (1 + r) ** (years + 1)
            if abs(df) < 1e-12:
                break
            r = r - f / df
        return float(np.clip(r, -0.99, 10.0))

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
            debt_paydown_rate  = pick("debt_paydown_rate","debt_paydown_rate",0.50),
        )
