# Worked IB models — a learning reference

These examples drive the **deterministic** financial models directly (via the
same classes `IBBrain.build_model()` calls). No neural network, no network
access — they run instantly and reproducibly, which makes them a good way to
learn the mechanics and to sanity-check your own numbers.

```bash
# Print a full walkthrough for one fictional mid-market target
python -m neuromorphic.domains.investment_banking.examples.worked_models

# Also write formatted .xlsx outputs
python -m neuromorphic.domains.investment_banking.examples.worked_models --excel out/
```

All figures are illustrative. This is a practice sandbox — see the compliance
note in the repo before pointing any of this at real deal data.

## What each model does

| Model | Question it answers | Key outputs |
|-------|--------------------|-------------|
| **DCF** (`models/dcf.py`) | What is the business worth intrinsically? | Enterprise/equity value, PV of FCF + terminal value, sensitivity grid |
| **Comps** (`models/comps.py`) | What does the market pay for similar businesses *today*? | EV/EBITDA, EV/Revenue, P/E stats; implied EV/equity range |
| **Precedents** (`models/precedents.py`) | What did acquirers pay in past deals? | Transaction multiples, mean premium, implied EV range |
| **LBO** (`models/lbo.py`) | What return can a sponsor make with leverage? | IRR, MOIC, debt paydown schedule, entry/exit bridge |
| **Merger** (`models/merger.py`) | Does buying this help or hurt the acquirer's EPS? | Accretion/dilution (with & without synergies), breakeven synergies |
| **Credit** (`models/credit.py`) | Is the capital structure sustainable? | Leverage, coverage, DSCR, implied rating, covenant headroom |

## Conventions worth knowing

- **Units.** The model classes work in absolute currency (e.g. `60_000_000`).
  The `IBBrain` query layer uses `_m` suffixes for millions and multiplies by
  `1e6`. The examples define `M = 1e6` and do the same.
- **DCF terminal value.** Two methods, always cross-checked against each other:
  - *Perpetuity (Gordon growth):* `TV = FCF_N·(1+g) / (WACC − g)`. Requires
    `g < WACC` (the model raises `ValueError` otherwise).
  - *Exit multiple:* `TV = exit_multiple · terminal-year EBITDA`.
  The result reports the **implied exit multiple** from the perpetuity TV and
  the **implied growth** from the exit-multiple TV — if those look unreasonable
  versus comps, your terminal assumptions need a second look.
- **Mid-year convention.** Set `mid_year_convention=True` to discount each
  year's cash flow half a period earlier (the banker default); it raises value
  versus full-year discounting.
- **LBO cash flow is internally consistent.** Interest is taken from the actual
  debt balance each year, levered FCF sweeps debt at `debt_paydown_rate`, and
  any un-swept cash accrues on the balance sheet. The equity IRR is solved from
  the full cash-flow vector, so it ties to MOIC: with a single entry and exit
  cash flow, `IRR = MOIC^(1/years) − 1`.

## Verifying

The models are covered by `tests/` at the repo root:

```bash
pytest tests/ -q
```
