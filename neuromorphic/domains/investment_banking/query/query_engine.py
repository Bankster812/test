"""
QueryEngine — Natural Language → IB Intent Parser
==================================================
Pure regex + keyword matching.  No ML dependencies.
Converts a free-text IB question into a structured QueryVector that
IBBrain uses to route computation and select financial models.

Supported intents
-----------------
  valuation   — DCF, comps, multiples, price target
  lbo         — LBO, leveraged buyout, returns, IRR, MOIC
  structuring — M&A mechanics, accretion/dilution, synergies, exchange ratio
  credit      — leverage, coverage, covenants, rating
  model_build — "build me a ...", "run a ...", "calculate ..."
  risk        — risk analysis, flags, red flags, stress test
  knowledge   — definition, explain, what is, how does
  general     — catch-all

Usage
-----
engine = QueryEngine()
qv = engine.parse("What WACC should I use for a tech company LBO?")
print(qv.intent, qv.model_type, qv.entities)
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field

logger = logging.getLogger("ib_brain.query_engine")


# ── QueryVector ──────────────────────────────────────────────────────────────

@dataclass
class QueryVector:
    raw_text:         str
    intent:           str                    # valuation | lbo | structuring | credit | model_build | risk | knowledge | general
    model_type:       str | None             # dcf | lbo | merger | comps | precedents | credit | None
    entities:         list[str]              # IB terms detected
    numerical_values: dict[str, float]       # param_name → value extracted from text
    sector:           str | None             # tech | healthcare | energy | …
    confidence:       float                  # 0.0 → 1.0

    def __str__(self):
        nums = ", ".join(f"{k}={v}" for k, v in self.numerical_values.items())
        return (f"QueryVector(intent={self.intent}, model={self.model_type}, "
                f"entities={self.entities[:3]}, nums={{{nums}}}, conf={self.confidence:.2f})")


# ── Pattern tables ────────────────────────────────────────────────────────────

# Intent keywords — higher specificity listed first
_INTENT_PATTERNS = [
    ("model_build",  r"\b(build|run|calculate|compute|model|create|generate|make)\b.*\b(dcf|lbo|comps?|merger|accretion|precedent|credit|valuation|model)\b"),
    ("model_build",  r"\b(dcf|lbo|comps?|merger|precedent|credit)\s+(model|analysis|valuation|run|calc)\b"),
    ("lbo",          r"\b(lbo|leveraged\s+buyout|private\s+equity|buyout|irr|moic|money.on.money|exit\s+multiple|entry\s+multiple|sponsor|pe\s+deal)\b"),
    ("structuring",  r"\b(accretion|dilution|synerg|exchange\s+ratio|acquisition\s+premium|merger\s+of\s+equals|break.?fee|earnout|collar|consideration)\b"),
    ("credit",       r"\b(leverage|covenant|interest\s+coverage|dscr|debt\s+service|credit\s+rating|high\s+yield|investment\s+grade|net\s+debt)\b"),
    ("valuation",    r"\b(wacc|dcf|terminal\s+value|enterprise\s+value|equity\s+value|ev.ebitda|ev.revenue|comps?|trading\s+comps?|precedent|football\s+field|multiple|valuation)\b"),
    ("risk",         r"\b(risk|red\s+flag|stress\s+test|downside|worst\s+case|covenant\s+breach|default|danger|concern|warning)\b"),
    ("knowledge",    r"\b(what\s+is|what\s+are|define|explain|how\s+does|how\s+do|meaning\s+of|tell\s+me\s+about|definition)\b"),
]

_MODEL_PATTERNS = {
    "dcf":         r"\b(dcf|discounted\s+cash\s+flow|intrinsic\s+value|terminal\s+value)\b",
    "lbo":         r"\b(lbo|leveraged\s+buyout|buyout\s+model|pe\s+model)\b",
    "merger":      r"\b(accretion|dilution|merger\s+model|a/?d\s+analysis|pro.?forma\s+eps)\b",
    "comps":       r"\b(comps?|trading\s+comps?|comparable\s+compan|public\s+comps?)\b",
    "precedents":  r"\b(precedent|deal\s+comps?|transaction\s+comps?|m.?a\s+comps?)\b",
    "credit":      r"\b(credit\s+model|leverage\s+analysis|debt\s+capacity|credit\s+analysis)\b",
}

_SECTOR_MAP = {
    "tech":         r"\b(tech|technology|software|saas|cloud|semiconductor|ai|artificial\s+intelligence)\b",
    "healthcare":   r"\b(health\s*care|healthcare|hospital|pharma|biotech|medical|device)\b",
    "energy":       r"\b(energy|oil|gas|petroleum|renewable|power|utility|utilities)\b",
    "financials":   r"\b(bank|financial|insurance|fintech|asset\s+management|brokerage)\b",
    "industrials":  r"\b(industrial|manufacturing|aerospace|defence|defense|logistics)\b",
    "consumer":     r"\b(consumer|retail|food|beverage|cpg|fmcg)\b",
    "real_estate":  r"\b(real\s+estate|reit|property|commercial|residential)\b",
    "media":        r"\b(media|entertainment|gaming|streaming|publishing|telecom)\b",
}

# Named numerical extractors: (param_name, regex, unit_divisor)
_NUM_PATTERNS = [
    ("revenue_m",        r"revenue\s+(?:of\s+)?[\$]?([\d,.]+)\s*([mbk]|million|billion)?", 1.0),
    ("ebitda_m",         r"ebitda\s+(?:of\s+)?[\$]?([\d,.]+)\s*([mbk]|million|billion)?", 1.0),
    ("ebitda_margin",    r"ebitda\s+margin\s+(?:of\s+)?(\d+(?:\.\d+)?)\s*%", 1.0),
    ("wacc",             r"wacc\s+(?:of\s+)?(\d+(?:\.\d+)?)\s*%", 100.0),
    ("cost_of_equity",   r"cost\s+of\s+equity\s+(?:of\s+)?(\d+(?:\.\d+)?)\s*%", 100.0),
    ("cost_of_debt",     r"cost\s+of\s+debt\s+(?:of\s+)?(\d+(?:\.\d+)?)\s*%", 100.0),
    ("tax_rate",         r"tax\s+(?:rate\s+)?(?:of\s+)?(\d+(?:\.\d+)?)\s*%", 100.0),
    ("leverage",         r"(?:leverage|debt)\s+(?:of\s+)?(\d+(?:\.\d+)?)\s*x", 1.0),
    ("entry_multiple",   r"entry\s+(?:multiple\s+)?(?:of\s+)?(\d+(?:\.\d+)?)\s*x", 1.0),
    ("exit_multiple",    r"exit\s+(?:multiple\s+)?(?:of\s+)?(\d+(?:\.\d+)?)\s*x", 1.0),
    ("terminal_growth",  r"terminal\s+(?:growth\s+)?(?:rate\s+)?(?:of\s+)?(\d+(?:\.\d+)?)\s*%", 100.0),
    ("equity_pct",       r"equity\s+(?:of\s+)?(\d+(?:\.\d+)?)\s*%", 100.0),
    ("debt_pct",         r"debt\s+(?:of\s+)?(\d+(?:\.\d+)?)\s*%", 100.0),
    ("premium_pct",      r"premium\s+(?:of\s+)?(\d+(?:\.\d+)?)\s*%", 1.0),
    ("irr",              r"irr\s+(?:of\s+)?(\d+(?:\.\d+)?)\s*%", 100.0),
    ("deal_size_m",      r"deal\s+(?:size\s+)?(?:of\s+)?[\$]?([\d,.]+)\s*([mbk]|million|billion)?", 1.0),
    ("projection_years", r"(\d+)[\-\s]year\s+(?:projection|horizon|forecast)", 1.0),
    ("ev_ebitda",        r"(\d+(?:\.\d+)?)\s*x\s+ebitda", 1.0),
]

# IB entity vocabulary (order matters — longer phrases first)
_IB_ENTITIES = [
    "wacc", "irr", "moic", "lbo", "dcf", "ebitda", "ebit", "ufcf", "nopat",
    "enterprise value", "equity value", "terminal value", "net debt",
    "cost of equity", "cost of debt", "equity risk premium", "beta",
    "accretion", "dilution", "synergies", "exchange ratio", "acquisition premium",
    "break fee", "earnout", "collar", "go-shop", "no-shop",
    "leverage", "coverage", "covenant", "dscr", "credit rating",
    "ev/ebitda", "ev/revenue", "p/e", "football field",
    "comps", "precedents", "trading comps", "deal comps",
    "tax shield", "nwc", "capex", "d&a",
    "sponsor", "carried interest", "hurdle rate",
    "spin-off", "carve-out", "recapitalisation", "mbo", "spac",
    "high yield", "investment grade", "mezzanine", "pik",
    "fairness opinion", "due diligence", "cim", "spa", "loi",
]


# ── Core Engine ───────────────────────────────────────────────────────────────

class QueryEngine:
    """
    Parses free-text IB questions into structured QueryVector objects.
    Stateless — safe to call from multiple threads.
    """

    def parse(self, text: str) -> QueryVector:
        """Main entry point.  Returns QueryVector."""
        norm = text.lower().strip()

        intent     = self._detect_intent(norm)
        model_type = self._detect_model(norm)
        entities   = self._extract_entities(norm)
        numerics   = self._extract_numerics(norm)
        sector     = self._detect_sector(norm)
        confidence = self._score_confidence(intent, model_type, entities, numerics)

        qv = QueryVector(
            raw_text         = text,
            intent           = intent,
            model_type       = model_type,
            entities         = entities,
            numerical_values = numerics,
            sector           = sector,
            confidence       = confidence,
        )
        logger.debug(f"QueryEngine: {qv}")
        return qv

    # ------------------------------------------------------------------
    # Detection methods
    # ------------------------------------------------------------------

    def _detect_intent(self, norm: str) -> str:
        for intent, pattern in _INTENT_PATTERNS:
            if re.search(pattern, norm):
                return intent
        return "general"

    def _detect_model(self, norm: str) -> str | None:
        for model, pattern in _MODEL_PATTERNS.items():
            if re.search(pattern, norm):
                return model
        return None

    def _detect_sector(self, norm: str) -> str | None:
        for sector, pattern in _SECTOR_MAP.items():
            if re.search(pattern, norm):
                return sector
        return None

    def _extract_entities(self, norm: str) -> list[str]:
        found = []
        for entity in _IB_ENTITIES:
            # match whole word/phrase
            pat = r"\b" + re.escape(entity) + r"\b"
            if re.search(pat, norm):
                found.append(entity)
        return found

    def _extract_numerics(self, norm: str) -> dict[str, float]:
        result = {}
        for param_name, pattern, divisor in _NUM_PATTERNS:
            m = re.search(pattern, norm)
            if not m:
                continue
            raw_val = m.group(1).replace(",", "")
            try:
                val = float(raw_val)
            except ValueError:
                continue

            # Handle unit suffix for monetary values (M/B/K)
            if m.lastindex and m.lastindex >= 2:
                try:
                    unit = m.group(2)
                except IndexError:
                    unit = None
                if unit:
                    unit = unit.lower()
                    if unit in ("b", "billion"):
                        val *= 1000.0
                    elif unit in ("k",):
                        val /= 1000.0
                    # "m" or "million" → already in millions, no change

            result[param_name] = val / divisor if divisor != 1.0 else val

        return result

    def _score_confidence(
        self,
        intent: str,
        model_type: str | None,
        entities: list[str],
        numerics: dict,
    ) -> float:
        score = 0.3  # base
        if intent != "general":
            score += 0.2
        if model_type:
            score += 0.2
        score += min(len(entities) * 0.05, 0.2)
        score += min(len(numerics) * 0.05, 0.1)
        return min(score, 1.0)
