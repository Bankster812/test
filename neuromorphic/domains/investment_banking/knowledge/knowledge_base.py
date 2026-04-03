"""
KnowledgeBase — Structured IB Concept Graph
============================================
Pre-seeded with ~60 IB concepts.  Supports exact lookup and fuzzy search.
No external dependencies — pure Python dict + simple string matching.

Usage
-----
kb = KnowledgeBase()
entry = kb.lookup("wacc")          # → KBEntry or None
results = kb.search("cost of debt") # → [(score, KBEntry), ...]
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field

logger = logging.getLogger("ib_brain.knowledge_base")


# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class KBEntry:
    term:          str
    aliases:       list[str]
    definition:    str
    formula:       str          = ""
    related_terms: list[str]    = field(default_factory=list)
    examples:      list[str]    = field(default_factory=list)
    category:      str          = "general"   # valuation | structuring | credit | lbo | accounting

    def __str__(self):
        lines = [f"[{self.category.upper()}] {self.term.upper()}"]
        lines.append(self.definition)
        if self.formula:
            lines.append(f"  Formula: {self.formula}")
        if self.related_terms:
            lines.append(f"  See also: {', '.join(self.related_terms)}")
        if self.examples:
            lines.append(f"  Example: {self.examples[0]}")
        return "\n".join(lines)


# ── KnowledgeBase ─────────────────────────────────────────────────────────────

class KnowledgeBase:
    """
    In-memory IB knowledge graph with ~60 pre-loaded concepts.
    Thread-safe for reads (no mutation after init).
    """

    def __init__(self):
        self._entries: dict[str, KBEntry] = {}   # canonical_term → KBEntry
        self._alias_map: dict[str, str]   = {}   # alias → canonical_term
        self._seed()
        logger.info(f"KnowledgeBase loaded {len(self._entries)} concepts")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def lookup(self, term: str) -> KBEntry | None:
        """Exact lookup (case-insensitive, alias-aware)."""
        key = self._normalise(term)
        canon = self._alias_map.get(key, key)
        return self._entries.get(canon)

    def search(self, query: str, top_k: int = 5) -> list[tuple[float, KBEntry]]:
        """
        Fuzzy search: returns (score, KBEntry) sorted by relevance.
        Score in [0, 1].
        """
        q_tokens = set(self._tokenise(query))
        scored = []
        for canon, entry in self._entries.items():
            # tokens from term + definition + aliases
            e_tokens = set(self._tokenise(entry.term + " " + entry.definition + " " + " ".join(entry.aliases)))
            if not e_tokens:
                continue
            overlap = len(q_tokens & e_tokens)
            score = overlap / max(len(q_tokens), 1)
            if score > 0:
                scored.append((score, entry))
        scored.sort(key=lambda x: -x[0])
        return scored[:top_k]

    def add(self, entry: KBEntry) -> None:
        """Add a new entry (or overwrite existing)."""
        canon = self._normalise(entry.term)
        self._entries[canon] = entry
        self._alias_map[canon] = canon
        for alias in entry.aliases:
            self._alias_map[self._normalise(alias)] = canon

    def categories(self) -> list[str]:
        return sorted(set(e.category for e in self._entries.values()))

    def by_category(self, cat: str) -> list[KBEntry]:
        return [e for e in self._entries.values() if e.category == cat]

    def __len__(self):
        return len(self._entries)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise(s: str) -> str:
        return re.sub(r"[^a-z0-9]", "_", s.lower().strip())

    @staticmethod
    def _tokenise(s: str) -> list[str]:
        return re.findall(r"[a-z]{2,}", s.lower())

    # ------------------------------------------------------------------
    # Seed — the 60 concepts every IB analyst must know
    # ------------------------------------------------------------------

    def _seed(self):
        entries = [

            # ── WACC & Cost of Capital ───────────────────────────────
            KBEntry("WACC", ["weighted average cost of capital", "discount rate"],
                "Blended rate at which a company is expected to pay all its security holders to finance its assets.",
                "WACC = (E/V)×Ke + (D/V)×Kd×(1−t)",
                ["cost of equity", "cost of debt", "beta", "CAPM", "tax shield"],
                ["Tech company with 70% equity at 12% Ke and 30% debt at 5% Kd, 25% tax → WACC ≈ 9.5%"],
                "valuation"),

            KBEntry("Cost of Equity", ["Ke", "required return on equity"],
                "Return demanded by equity investors, estimated via CAPM or build-up.",
                "Ke = Rf + β × ERP",
                ["CAPM", "beta", "equity risk premium", "WACC"],
                ["Risk-free 4%, ERP 5.5%, β 1.2 → Ke = 10.6%"],
                "valuation"),

            KBEntry("CAPM", ["capital asset pricing model"],
                "Model that prices an asset based on its systematic risk (beta) and the market risk premium.",
                "E(r) = Rf + β(E(Rm) − Rf)",
                ["beta", "equity risk premium", "cost of equity"],
                [], "valuation"),

            KBEntry("Beta", ["levered beta", "equity beta"],
                "Measure of a stock's sensitivity to market movements. Levered beta includes financial risk.",
                "β_levered = β_unlevered × (1 + (1−t) × D/E)",
                ["CAPM", "Hamada equation", "unlevered beta"],
                ["Market beta=1.0 → moves with index; beta=1.5 → 50% more volatile"],
                "valuation"),

            KBEntry("Equity Risk Premium", ["ERP", "market risk premium"],
                "Excess return investors expect from holding equities over risk-free assets. Typically 4.5–6%.",
                "ERP = E(Rm) − Rf",
                ["CAPM", "cost of equity", "Damodaran"],
                [], "valuation"),

            KBEntry("Cost of Debt", ["Kd", "pre-tax cost of debt"],
                "Effective interest rate a company pays on its debt before the tax shield.",
                "Kd_after_tax = Kd_pre_tax × (1 − tax_rate)",
                ["WACC", "tax shield", "credit spread"],
                [], "valuation"),

            # ── DCF ──────────────────────────────────────────────────
            KBEntry("DCF", ["discounted cash flow", "discounted cashflow"],
                "Intrinsic valuation method: project free cash flows, discount at WACC, add terminal value.",
                "EV = Σ [UFCF_t / (1+WACC)^t] + TV / (1+WACC)^N",
                ["WACC", "terminal value", "UFCF", "Gordon growth model"],
                ["5-year projection with 10% WACC and 2.5% terminal growth"],
                "valuation"),

            KBEntry("Terminal Value", ["TV", "residual value", "continuing value"],
                "Value of cash flows beyond the explicit projection period. Gordon Growth or Exit Multiple method.",
                "TV_GGM = UFCF_N+1 / (WACC − g);  TV_Exit = EBITDA_N × Exit_Multiple",
                ["DCF", "Gordon growth model", "exit multiple", "WACC"],
                [], "valuation"),

            KBEntry("UFCF", ["unlevered free cash flow", "free cash flow to firm", "FCFF"],
                "Cash flow available to all capital providers, before financing costs.",
                "UFCF = EBIT×(1−t) + D&A − CapEx − ΔNWC",
                ["DCF", "EBITDA", "CapEx", "NWC"],
                [], "valuation"),

            KBEntry("Gordon Growth Model", ["GGM", "perpetuity growth model"],
                "Terminal value formula assuming cash flows grow at a constant rate forever.",
                "TV = CF_N+1 / (r − g)  where g < r",
                ["terminal value", "DCF", "WACC"],
                [], "valuation"),

            # ── Multiples / Comps ────────────────────────────────────
            KBEntry("EV/EBITDA", ["enterprise value to EBITDA", "EBITDA multiple"],
                "Most common deal multiple in M&A. Capital-structure neutral. Typical range 6-15x in M&A.",
                "EV/EBITDA = Enterprise Value / EBITDA",
                ["EBITDA", "enterprise value", "comps", "precedent transactions"],
                ["SaaS deal at 20x EBITDA is premium; industrials at 7x is typical"],
                "valuation"),

            KBEntry("EV/Revenue", ["revenue multiple", "EV/Sales"],
                "Used for pre-profit companies or high-growth sectors (SaaS, biotech).",
                "EV/Revenue = Enterprise Value / Revenue",
                ["enterprise value", "comps"],
                [], "valuation"),

            KBEntry("P/E Ratio", ["price to earnings", "PE", "price earnings ratio"],
                "Equity value per share divided by EPS. Sensitive to capital structure.",
                "P/E = Share Price / EPS",
                ["EPS", "equity value", "comps"],
                [], "valuation"),

            KBEntry("Comps", ["comparable company analysis", "trading comps", "public comps"],
                "Value a company by comparing it to similar publicly traded peers on key multiples.",
                "Implied EV = Median Peer EV/EBITDA × Target EBITDA",
                ["EV/EBITDA", "precedent transactions", "football field"],
                [], "valuation"),

            KBEntry("Precedent Transactions", ["deal comps", "transaction comps", "M&A comps"],
                "Value a company using multiples paid in comparable historical M&A deals. Includes control premium.",
                "Implied EV = Median Transaction EV/EBITDA × Target EBITDA",
                ["control premium", "comps", "acquisition premium"],
                [], "valuation"),

            KBEntry("Football Field", ["valuation football field", "valuation summary"],
                "Visual summary of valuation ranges from multiple methodologies (DCF, comps, precedents, LBO).",
                "",
                ["DCF", "comps", "LBO", "precedent transactions"],
                ["Output: horizontal bar chart with low/high range per method"],
                "valuation"),

            # ── Enterprise Value / Equity Bridge ────────────────────
            KBEntry("Enterprise Value", ["EV", "firm value"],
                "Total value of a business, capital-structure neutral. EV = Equity + Net Debt + Minorities + Preferred.",
                "EV = Market Cap + Total Debt − Cash + Minorities + Preferred",
                ["equity value", "bridge", "net debt", "WACC"],
                [], "valuation"),

            KBEntry("Equity Value", ["market cap", "market capitalisation"],
                "Value attributable to common equity holders only.",
                "Equity Value = EV − Net Debt − Minorities − Preferred",
                ["enterprise value", "bridge", "net debt"],
                [], "valuation"),

            KBEntry("Net Debt", ["net financial debt"],
                "Total debt less cash and cash equivalents.",
                "Net Debt = Total Debt − Cash & Equivalents",
                ["enterprise value", "leverage", "debt schedule"],
                [], "accounting"),

            # ── LBO ──────────────────────────────────────────────────
            KBEntry("LBO", ["leveraged buyout"],
                "Acquisition financed primarily with debt, using the target's assets/cash flows as collateral.",
                "Entry EV = Entry Multiple × EBITDA; Equity = EV − Debt",
                ["IRR", "MOIC", "debt schedule", "sponsor", "exit multiple"],
                ["KKR buys company at 10x EBITDA, 6x debt/EBITDA; sells at 12x 5yr later"],
                "lbo"),

            KBEntry("IRR", ["internal rate of return"],
                "Annualised return on an investment; the discount rate that makes NPV=0.",
                "0 = −Initial_Equity + Σ [CF_t / (1+IRR)^t] + Exit_Equity / (1+IRR)^N",
                ["MOIC", "LBO", "NPV", "hurdle rate"],
                ["PE fund buys for $100M equity, sells 5yr later for $300M → IRR ≈ 25%"],
                "lbo"),

            KBEntry("MOIC", ["multiple on invested capital", "money-on-money", "MOM"],
                "Total cash returned divided by initial equity invested.  Complements IRR.",
                "MOIC = Total Cash Returned / Equity Invested",
                ["IRR", "LBO"],
                ["Invest $100M, return $300M → MOIC = 3.0x"],
                "lbo"),

            KBEntry("Debt Schedule", ["amortisation schedule", "debt waterfall"],
                "Year-by-year table of debt tranches, interest payments, and mandatory/FCF-based amortisation.",
                "",
                ["LBO", "leverage", "FCF", "covenant"],
                [], "lbo"),

            KBEntry("Sponsor", ["financial sponsor", "PE sponsor", "private equity"],
                "Private equity firm that provides the equity in an LBO and manages the portfolio company.",
                "",
                ["LBO", "IRR", "MOIC", "hurdle rate"],
                [], "lbo"),

            KBEntry("Hurdle Rate", ["preferred return", "PE hurdle"],
                "Minimum IRR the PE fund must achieve before sharing profits with carried interest.",
                "",
                ["IRR", "carried interest", "LBO"],
                ["Typical PE hurdle: 8%; target IRR 20%+"],
                "lbo"),

            KBEntry("Carried Interest", ["carry", "performance fee"],
                "PE fund manager's share of profits above the hurdle rate, typically 20%.",
                "Carry = 20% × (Returns − Preferred Return)",
                ["hurdle rate", "LBO", "GP/LP"],
                [], "lbo"),

            # ── M&A Structuring ──────────────────────────────────────
            KBEntry("Accretion/Dilution", ["EPS accretion", "EPS dilution", "a/d analysis"],
                "Whether a deal increases (accretive) or decreases (dilutive) acquirer EPS post-close.",
                "Pro-forma EPS = (Acquirer NI + Target NI + Synergies − Financing Cost) / Pro-forma Shares",
                ["EPS", "synergies", "exchange ratio", "merger"],
                ["Deal is accretive if seller P/E < acquirer P/E (all-stock)"],
                "structuring"),

            KBEntry("Synergies", ["cost synergies", "revenue synergies"],
                "Value created by combining two businesses: cost cuts or revenue uplift not achievable standalone.",
                "Synergy Value = PV of Synergies / (1+WACC)^t",
                ["accretion/dilution", "merger", "integration"],
                ["Typical cost synergy = 2-4% of combined cost base"],
                "structuring"),

            KBEntry("Exchange Ratio", ["stock ratio", "share exchange ratio"],
                "Number of acquirer shares issued per target share in an all-stock deal.",
                "Exchange Ratio = Offer Price per Target Share / Acquirer Share Price",
                ["accretion/dilution", "fixed exchange ratio", "collar"],
                [], "structuring"),

            KBEntry("Acquisition Premium", ["control premium", "deal premium", "takeover premium"],
                "% above the unaffected target share price paid by the acquirer.",
                "Premium = (Offer Price − Unaffected Price) / Unaffected Price",
                ["precedent transactions", "EV", "M&A"],
                ["Typical M&A premium: 25–45%; hostile bids often 40-60%"],
                "structuring"),

            KBEntry("Break Fee", ["termination fee", "reverse break fee"],
                "Cash penalty paid if either party walks away from a deal after signing.",
                "",
                ["merger agreement", "reverse break fee", "go-shop"],
                ["Typical break fee: 3-4% of deal value"],
                "structuring"),

            KBEntry("Go-Shop", ["go-shop period", "active solicitation"],
                "Period after signing during which the target can solicit competing bids.",
                "",
                ["no-shop", "fiduciary out", "merger agreement"],
                [], "structuring"),

            KBEntry("Collar", ["price collar", "floating exchange ratio"],
                "Mechanism that adjusts exchange ratio or cash if acquirer share price moves beyond a range.",
                "",
                ["exchange ratio", "fixed collar", "walk-away rights"],
                [], "structuring"),

            KBEntry("Earnout", ["contingent consideration", "earnout structure"],
                "Post-close payments to target shareholders if performance milestones are met.",
                "",
                ["merger", "SPA", "purchase price"],
                ["Seller receives extra $50M if EBITDA exceeds $100M in Year 1 post-close"],
                "structuring"),

            KBEntry("Fairness Opinion", ["investment bank fairness opinion", "board fairness opinion"],
                "Written opinion from an investment bank that the deal consideration is fair from a financial point of view.",
                "",
                ["board of directors", "fiduciary duty", "valuation"],
                [], "structuring"),

            # ── Credit / Leverage ────────────────────────────────────
            KBEntry("DSCR", ["debt service coverage ratio"],
                "Ratio of operating cash flow to total debt service (principal + interest). Min 1.0x to service debt.",
                "DSCR = EBITDA / (Interest + Principal Repayment)",
                ["leverage", "covenant", "interest coverage"],
                [], "credit"),

            KBEntry("Interest Coverage", ["EBIT/interest", "TIE ratio"],
                "How many times EBIT or EBITDA covers interest expense. Key credit covenant metric.",
                "Interest Coverage = EBIT / Interest Expense",
                ["DSCR", "covenant", "leverage"],
                ["Coverage < 1.5x → HIGH risk; > 3x → comfortable"],
                "credit"),

            KBEntry("Covenant", ["debt covenant", "financial covenant", "maintenance covenant"],
                "Legal restrictions in debt agreements that borrowers must satisfy (leverage, coverage, CapEx limits).",
                "",
                ["leverage", "interest coverage", "DSCR", "waiver"],
                ["Maximum leverage covenant: 5.0x net debt/EBITDA"],
                "credit"),

            KBEntry("Leverage", ["financial leverage", "debt/EBITDA"],
                "Amount of debt relative to earnings. Key metric for credit risk and LBO feasibility.",
                "Leverage = Total Debt / EBITDA  (or Net Debt / EBITDA)",
                ["net debt", "DSCR", "covenant", "credit rating"],
                ["Investment grade typically < 3x; leveraged buyout typically 5-7x"],
                "credit"),

            KBEntry("Credit Rating", ["S&P rating", "Moody's rating", "debt rating"],
                "Rating agency assessment of default risk. Investment grade ≥ BBB−; HY/junk < BB+.",
                "",
                ["leverage", "interest coverage", "spread"],
                ["S&P: AAA, AA, A, BBB, BB, B, CCC, CC, C, D"],
                "credit"),

            KBEntry("High Yield", ["junk bond", "HY bond", "sub-investment grade"],
                "Debt rated below investment grade (BB+ or lower). Higher coupon to compensate for higher default risk.",
                "",
                ["credit rating", "LBO", "spread"],
                [], "credit"),

            # ── Accounting / Financials ──────────────────────────────
            KBEntry("EBITDA", ["earnings before interest tax depreciation amortisation"],
                "Proxy for operating cash flow. Most common deal metric. Excludes non-cash & financing items.",
                "EBITDA = Revenue − COGS − SG&A + D&A",
                ["EBIT", "UFCF", "EV/EBITDA", "leverage"],
                [], "accounting"),

            KBEntry("EBIT", ["earnings before interest and tax", "operating income", "NOPAT base"],
                "Operating profit before financing and taxes.",
                "EBIT = Revenue − COGS − SG&A − D&A",
                ["EBITDA", "NOPAT", "UFCF"],
                [], "accounting"),

            KBEntry("NOPAT", ["net operating profit after tax"],
                "After-tax operating profit used in UFCF and ROIC calculations.",
                "NOPAT = EBIT × (1 − Tax Rate)",
                ["UFCF", "ROIC", "DCF"],
                [], "accounting"),

            KBEntry("CapEx", ["capital expenditure", "capex"],
                "Cash spent on fixed assets. Reduces FCF. Maintenance CapEx vs Growth CapEx distinction important.",
                "CapEx = ΔPP&E + D&A",
                ["UFCF", "NWC", "FCF"],
                [], "accounting"),

            KBEntry("NWC", ["net working capital", "working capital"],
                "Short-term operating liquidity: current assets minus current liabilities (excl. cash & debt).",
                "NWC = (AR + Inventory + Prepaid) − (AP + Accruals)",
                ["UFCF", "cash conversion", "DSO", "DPO"],
                [], "accounting"),

            KBEntry("D&A", ["depreciation and amortisation", "depreciation", "amortisation"],
                "Non-cash charge that reduces reported profit but not cash flow. Added back in EBITDA and FCF.",
                "",
                ["EBITDA", "CapEx", "UFCF"],
                [], "accounting"),

            KBEntry("Tax Shield", ["interest tax shield", "debt tax shield"],
                "Tax saving from debt: interest is deductible, reducing taxable income.",
                "Tax Shield = Interest × Tax Rate",
                ["cost of debt", "WACC", "capital structure"],
                ["$1B debt at 6%, 25% tax → $15M/yr tax shield"],
                "accounting"),

            # ── Processes & Concepts ─────────────────────────────────
            KBEntry("Due Diligence", ["DD", "buyer due diligence", "VDD"],
                "Investigation of target company: financial, legal, commercial, tax, IT, HR diligence.",
                "",
                ["SPA", "merger agreement", "QoE"],
                [], "general"),

            KBEntry("Quality of Earnings", ["QoE", "quality of earnings report"],
                "Accounting firm review of EBITDA adjustments, working capital, and one-time items.",
                "",
                ["due diligence", "EBITDA", "normalisation"],
                [], "general"),

            KBEntry("SPA", ["share purchase agreement", "stock purchase agreement"],
                "Legal contract governing a M&A transaction: reps & warranties, conditions, purchase price.",
                "",
                ["merger agreement", "earnout", "break fee"],
                [], "general"),

            KBEntry("Recapitalisation", ["recap", "leveraged recap"],
                "Restructuring a company's debt/equity mix, often by taking on debt to pay a special dividend.",
                "",
                ["leverage", "dividend recap", "capital structure"],
                [], "structuring"),

            KBEntry("Spin-off", ["spinout", "carve-out"],
                "Separation of a business unit into an independent public company.",
                "",
                ["divestiture", "carve-out", "tax-free spinoff"],
                [], "structuring"),

            KBEntry("ROIC", ["return on invested capital"],
                "Efficiency metric: how much return a company generates on capital deployed.",
                "ROIC = NOPAT / Invested Capital",
                ["WACC", "value creation", "NOPAT"],
                ["ROIC > WACC → value creation; ROIC < WACC → value destruction"],
                "valuation"),

            KBEntry("NAV", ["net asset value"],
                "Book value of assets minus liabilities. Used in real estate, financials, and asset-heavy sectors.",
                "NAV = Total Assets − Total Liabilities",
                ["equity value", "book value"],
                [], "valuation"),

            KBEntry("PIK", ["payment in kind", "PIK interest", "PIK toggle"],
                "Debt where interest accrues to principal rather than being paid in cash. Common in LBOs.",
                "",
                ["mezzanine", "LBO", "HY", "debt schedule"],
                [], "credit"),

            KBEntry("Mezzanine", ["mezz debt", "mezzanine financing"],
                "Subordinated debt between senior secured and equity. Higher return, subordinate claim.",
                "",
                ["LBO", "PIK", "second lien", "capital structure"],
                [], "credit"),

            KBEntry("Management Buyout", ["MBO", "management led buyout"],
                "LBO where incumbent management team leads the acquisition, typically with PE backing.",
                "",
                ["LBO", "sponsor", "equity rollover"],
                [], "lbo"),

            KBEntry("SPAC", ["special purpose acquisition company", "blank check company"],
                "Shell company that raises capital via IPO to acquire a private company within a set period.",
                "",
                ["merger", "de-SPAC", "IPO"],
                [], "structuring"),

            KBEntry("Staple Financing", ["stapled financing", "staple"],
                "Debt financing arranged by the sell-side bank and offered to all potential buyers.",
                "",
                ["LBO", "sell-side", "process"],
                [], "structuring"),

            KBEntry("Fairness Opinion", ["FO"],
                "Written opinion from an investment bank that deal consideration is fair from a financial POV.",
                "",
                ["board of directors", "valuation", "litigation"],
                [], "general"),

            KBEntry("Teaser", ["deal teaser", "executive summary"],
                "1-2 page anonymous marketing document sent to potential buyers to gauge interest.",
                "",
                ["CIM", "process letter", "NDA"],
                [], "general"),

            KBEntry("CIM", ["confidential information memorandum", "offering memorandum", "OM"],
                "Detailed marketing document sent to qualified buyers after NDA. Describes business, financials, thesis.",
                "",
                ["teaser", "NDA", "due diligence"],
                [], "general"),

            KBEntry("Management Presentation", ["mgmt pres", "management roadshow"],
                "In-person or virtual meeting where target company management presents to potential buyers.",
                "",
                ["CIM", "due diligence", "process"],
                [], "general"),

            KBEntry("LOI", ["letter of intent", "indication of interest", "IOI"],
                "Non-binding document outlining key deal terms before a binding offer.",
                "",
                ["SPA", "process", "exclusivity"],
                [], "general"),
        ]

        for entry in entries:
            self.add(entry)
