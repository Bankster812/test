"""
IB Configuration — Investment Banking Brain Parameters
=======================================================
Duck-types as neuromorphic.config so Brain.__init__ accepts it directly.
All base config attributes are inherited; IB-specific ones are added.
"""

from __future__ import annotations
import numpy as np
import neuromorphic.config as _base

# ---- Inherit all base config attributes ----
SCALE          = _base.SCALE
DT             = _base.DT
V_REST         = _base.V_REST
V_RESET        = _base.V_RESET
V_THRESH       = _base.V_THRESH
V_THRESH_MIN   = _base.V_THRESH_MIN
V_THRESH_MAX   = _base.V_THRESH_MAX
TAU_MEM        = _base.TAU_MEM
TAU_SYN        = _base.TAU_SYN
T_REFRAC       = _base.T_REFRAC
R_MEM          = _base.R_MEM
A_PLUS         = 0.012          # slightly stronger LTP for concept acquisition
A_MINUS        = 0.0126
TAU_PLUS       = _base.TAU_PLUS
TAU_MINUS      = _base.TAU_MINUS
W_MIN          = _base.W_MIN
W_MAX          = _base.W_MAX
W_INIT_SCALE   = _base.W_INIT_SCALE
DA_DECAY       = _base.DA_DECAY
DA_BASELINE    = _base.DA_BASELINE
ACH_INIT       = _base.ACH_INIT
SHT_INIT       = _base.SHT_INIT
NM_ALPHA       = _base.NM_ALPHA
TARGET_RATE    = 8.0            # slightly higher for denser associations
ETA_HOMEOSTASIS= _base.ETA_HOMEOSTASIS
HOMEOSTASIS_INTERVAL = _base.HOMEOSTASIS_INTERVAL
RATE_EMA_ALPHA = _base.RATE_EMA_ALPHA
MAX_DELAY_MS   = _base.MAX_DELAY_MS
REGION_ORDER   = _base.REGION_ORDER
CONNECTIVITY   = _base.CONNECTIVITY
INHIBITORY_REGIONS = _base.INHIBITORY_REGIONS

# Safety bounds (permissive for financial output; real validation is FinancialConstraints)
JOINT_ANGLE_MIN   = -10.0
JOINT_ANGLE_MAX   =  10.0
MAX_JOINT_VELOCITY = 10.0
MAX_JOINT_FORCE    = 999.0
COLLISION_ZONES    = []

# ---- IB-specific ----
N_DOF: int = 32   # 32 financial parameter slots (replaces 6 robot DOF)

N_FINANCIAL_PARAMS: int = 32
QUERY_STEPS:       int  = 200   # brain steps per query (~200ms settling time)
INGESTION_STEPS:   int  = 300   # brain steps per document chunk
REWARD_ON_COHERENT: float = 0.7 # reward when output passes sanity checks

# Financial parameter slot assignments (index → name)
PARAM_SLOTS: dict[str, int] = {
    "discount_rate":      0,
    "wacc":               1,
    "cost_of_equity":     2,
    "cost_of_debt":       3,
    "revenue_growth":     4,
    "ebitda_margin":      5,
    "terminal_growth":    6,
    "exit_multiple":      7,
    "leverage_ratio":     8,
    "interest_rate":      9,
    "tax_rate":           10,
    "capex_pct":          11,
    "nwc_pct":            12,
    "depreciation_pct":   13,
    "fcf_margin":         14,
    "ev_ebitda":          15,
    "pe_ratio":           16,
    "ev_revenue":         17,
    "irr":                18,
    "moic":               19,
    "hold_period":        20,
    "debt_equity_ratio":  21,
    "equity_contribution":22,
    "debt_paydown_rate":  23,
    "accretion_dilution": 24,
    "synergy_pct":        25,
    "premium_pct":        26,
    "confidence":         27,
    "deal_risk":          28,
    "sector_adjustment":  29,
    "size_adjustment":    30,
    "liquidity_premium":  31,
}
PARAM_NAMES: list[str] = [None] * N_FINANCIAL_PARAMS
for _name, _idx in PARAM_SLOTS.items():
    PARAM_NAMES[_idx] = _name

# Physical ranges for each parameter (used to scale tanh output to real values)
PARAM_RANGES: dict[str, tuple[float, float]] = {
    "discount_rate":      (0.04, 0.25),
    "wacc":               (0.04, 0.20),
    "cost_of_equity":     (0.06, 0.30),
    "cost_of_debt":       (0.02, 0.15),
    "revenue_growth":     (-0.20, 0.50),
    "ebitda_margin":      (0.00, 0.60),
    "terminal_growth":    (0.00, 0.05),
    "exit_multiple":      (4.0,  25.0),
    "leverage_ratio":     (0.5,  8.0),
    "interest_rate":      (0.03, 0.15),
    "tax_rate":           (0.10, 0.40),
    "capex_pct":          (0.01, 0.20),
    "nwc_pct":            (0.00, 0.15),
    "depreciation_pct":   (0.01, 0.10),
    "fcf_margin":         (0.00, 0.40),
    "ev_ebitda":          (4.0,  30.0),
    "pe_ratio":           (5.0,  60.0),
    "ev_revenue":         (0.5,  20.0),
    "irr":                (0.05, 0.60),
    "moic":               (1.0,  5.0),
    "hold_period":        (3.0,  7.0),
    "debt_equity_ratio":  (0.5,  5.0),
    "equity_contribution":(0.10, 0.50),
    "debt_paydown_rate":  (0.05, 0.30),
    "accretion_dilution": (-0.10, 0.15),
    "synergy_pct":        (0.00, 0.20),
    "premium_pct":        (0.10, 0.60),
    "confidence":         (0.00, 1.00),
    "deal_risk":          (0.00, 1.00),
    "sector_adjustment":  (-0.05, 0.05),
    "size_adjustment":    (-0.03, 0.03),
    "liquidity_premium":  (0.00, 0.04),
}

# Sector benchmarks (typical EV/EBITDA ranges by sector)
SECTOR_BENCHMARKS: dict[str, dict[str, tuple[float, float]]] = {
    "technology":    {"ev_ebitda": (12, 25), "ev_revenue": (3, 12), "growth": (0.15, 0.40)},
    "saas":          {"ev_ebitda": (15, 40), "ev_revenue": (5, 20), "growth": (0.20, 0.50)},
    "healthcare":    {"ev_ebitda": (10, 20), "ev_revenue": (2, 6),  "growth": (0.05, 0.15)},
    "industrials":   {"ev_ebitda": (6, 12),  "ev_revenue": (0.8, 2),"growth": (0.02, 0.08)},
    "consumer":      {"ev_ebitda": (7, 15),  "ev_revenue": (1, 3),  "growth": (0.03, 0.12)},
    "financial":     {"ev_ebitda": (8, 15),  "ev_revenue": (2, 5),  "growth": (0.05, 0.15)},
    "energy":        {"ev_ebitda": (5, 10),  "ev_revenue": (0.5, 2),"growth": (-0.05, 0.10)},
    "real_estate":   {"ev_ebitda": (12, 22), "ev_revenue": (4, 10), "growth": (0.02, 0.08)},
    "media":         {"ev_ebitda": (8, 16),  "ev_revenue": (1, 4),  "growth": (0.03, 0.12)},
    "mid_market":    {"ev_ebitda": (7, 14),  "ev_revenue": (1, 4),  "growth": (0.05, 0.20)},
}

# IB concept vocabulary (2000 terms → neuron block indices)
# Grouped by category for semantic organization
IB_VOCABULARY: dict[str, int] = {}
_vocab_terms = [
    # Valuation
    "dcf","wacc","discount_rate","terminal_value","free_cash_flow","enterprise_value",
    "equity_value","net_debt","ev_ebitda","ev_revenue","pe_ratio","price_earnings",
    "gordon_growth","perpetuity","sensitivity","scenario","football_field",
    # LBO
    "lbo","leveraged_buyout","private_equity","sponsor","entry_multiple","exit_multiple",
    "irr","moic","debt_schedule","amortization","bullet","revolver","term_loan",
    "senior_debt","subordinated","mezzanine","preferred_equity","pik","covenant",
    "leverage_ratio","debt_ebitda","interest_coverage","dscr","hold_period",
    # M&A
    "merger","acquisition","deal","transaction","synergy","accretion","dilution",
    "eps","earnings_per_share","premium","takeover","hostile","friendly","loi",
    "letter_of_intent","due_diligence","data_room","exclusivity","binding","non_binding",
    "stock_deal","cash_deal","merger_consideration","exchange_ratio","collar",
    # Financial Statements
    "revenue","ebitda","ebit","net_income","gross_profit","gross_margin","operating_income",
    "operating_margin","net_margin","cash_flow","capex","working_capital","nwc",
    "depreciation","amortization","fcf","unlevered","levered","balance_sheet",
    "income_statement","cash_flow_statement","assets","liabilities","equity",
    "debt","cash","goodwill","intangibles","pp_and_e","accounts_receivable","inventory",
    # Comps & Precedents
    "comparable","comps","precedent","transaction","multiple","median","mean",
    "25th_percentile","75th_percentile","range","control_premium","trading_multiple",
    # Credit & Debt
    "credit","rating","investment_grade","high_yield","spread","libor","sofr",
    "fixed_rate","floating_rate","maturity","refinancing","covenant_lite","pari_passu",
    # Sectors
    "technology","healthcare","industrials","consumer","energy","financial","media",
    "real_estate","saas","fintech","biotech","pharma","medtech","enterprise_software",
    # Size descriptors
    "mid_market","large_cap","small_cap","mega_deal","cross_border","strategic","financial_buyer",
    # Deal types
    "carve_out","spin_off","ipo","secondary","recapitalization","dividend_recap","bolt_on",
    # General IB
    "pitch_book","ib","investment_banking","advisory","buy_side","sell_side","mandate",
]
for _i, _term in enumerate(_vocab_terms):
    IB_VOCABULARY[_term] = _i


# Proxy for base config functions
def get_region_sizes() -> dict[str, int]:
    return _base.get_region_sizes()

def get_total_neurons() -> int:
    return _base.get_total_neurons()

def compute_id_ranges(region_sizes):
    return _base.compute_id_ranges(region_sizes)

def estimate_synapse_count(region_sizes=None):
    return _base.estimate_synapse_count(region_sizes)
