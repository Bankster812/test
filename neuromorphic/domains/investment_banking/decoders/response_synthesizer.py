"""
ResponseSynthesizer — Brain state to structured IB analysis
============================================================
Deterministic template engine. The brain provides decoded parameters;
this formats them into investment-bank-grade written analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from neuromorphic.domains.investment_banking.decoders.financial_decoder import FinancialParams


@dataclass
class IBResponse:
    """Complete response to an IB query."""
    answer_text:    str
    parameters:     FinancialParams
    model_result:   object | None = None   # ModelResult if a model was computed
    risk_flags:     list[str]     = field(default_factory=list)
    confidence:     float         = 0.0
    sources:        list[str]     = field(default_factory=list)
    reasoning_trace: list[str]   = field(default_factory=list)

    def __str__(self):
        return self.answer_text


class ResponseSynthesizer:
    """
    Converts decoded brain state into structured IB analysis text.
    """

    def __init__(self, config):
        self.cfg = config

    def synthesize(
        self,
        params: FinancialParams,
        query,                   # ParsedQuery
        brain_state: dict,
        model_result=None,
        risk_flags: list[str] | None = None,
    ) -> IBResponse:
        """Build a complete IBResponse from brain output."""
        risk_flags   = risk_flags or []
        confidence   = params.overall_confidence()
        query_type   = getattr(query, "query_type", "general")
        sector       = getattr(query, "target_sector", None) or "general"
        target_model = getattr(query, "target_model", None)

        if query_type == "valuation" or target_model in ("dcf", "comps", "precedents"):
            answer = self._valuation_response(params, sector, confidence)
        elif query_type == "structuring":
            answer = self._structuring_response(params, sector, confidence)
        elif query_type == "model_request":
            answer = self._model_response(params, target_model, model_result, confidence)
        elif query_type == "risk":
            answer = self._risk_response(params, risk_flags, confidence)
        elif query_type == "lbo" or target_model == "lbo":
            answer = self._lbo_response(params, sector, confidence)
        else:
            answer = self._general_response(params, query.raw_text, confidence)

        if risk_flags:
            answer += "\n\n**Risk flags:**\n" + "\n".join(f"  ⚠  {f}" for f in risk_flags)

        if confidence < 0.3:
            answer += (
                "\n\n*[Low confidence — the brain has limited exposure to this deal type. "
                "Ingest more relevant precedents or documents to improve accuracy.]*"
            )

        return IBResponse(
            answer_text    = answer,
            parameters     = params,
            model_result   = model_result,
            risk_flags     = risk_flags,
            confidence     = confidence,
        )

    # ------------------------------------------------------------------
    # Response templates
    # ------------------------------------------------------------------

    def _valuation_response(self, params, sector, confidence) -> str:
        wacc,    wc  = params.get("wacc")
        tg,      tgc = params.get("terminal_growth")
        eveb,    ec  = params.get("ev_ebitda")
        evrev,   erc = params.get("ev_revenue")
        adj,     _   = params.get("sector_adjustment")
        lp,      _   = params.get("liquidity_premium")
        sa,      _   = params.get("size_adjustment")

        wacc_adj = wacc + adj + lp + sa
        eveb_lo  = round(eveb * 0.85, 1)
        eveb_hi  = round(eveb * 1.15, 1)

        # Look up sector benchmarks
        bm = self.cfg.SECTOR_BENCHMARKS.get(sector.lower(), {})
        bm_ev = bm.get("ev_ebitda", (None, None))
        bm_str = ""
        if bm_ev[0]:
            bm_str = (
                f"Sector median EV/EBITDA for {sector}: **{bm_ev[0]}x–{bm_ev[1]}x**. "
            )

        return (
            f"## Valuation — {sector.title()} ({confidence*100:.0f}% confidence)\n\n"
            f"**WACC / Discount Rate:** {wacc_adj*100:.1f}% "
            f"(base {wacc*100:.1f}% + sector adj {adj*100:+.1f}% + "
            f"size premium {sa*100:+.1f}% + liquidity {lp*100:+.1f}%)\n\n"
            f"**Terminal Growth Rate:** {tg*100:.1f}%\n\n"
            f"**EV/EBITDA range:** {eveb_lo}x – {eveb_hi}x  |  "
            f"**EV/Revenue:** {evrev:.1f}x\n\n"
            f"{bm_str}"
            f"**Methodology note:** For a {sector} deal, the WACC should reflect "
            f"the sector's capital structure, regulatory risk, and current rates. "
            f"Terminal growth should not exceed long-run GDP growth ({tg*100:.1f}% is "
            f"{'appropriate' if tg < 0.04 else 'on the high side — consider justification'}). "
            f"Run sensitivity analysis across WACC ±{100:.0f}bp and terminal growth ±{50:.0f}bp."
        )

    def _structuring_response(self, params, sector, confidence) -> str:
        tax,  _ = params.get("tax_rate")
        prem, _ = params.get("premium_pct")
        de,   _ = params.get("debt_equity_ratio")
        ad,   _ = params.get("accretion_dilution")

        return (
            f"## Deal Structure Analysis ({confidence*100:.0f}% confidence)\n\n"
            f"**Recommended structure:** "
            f"{'Cash deal' if tax > 0.25 else 'Stock deal'} "
            f"({'tax step-up advantageous' if tax > 0.25 else 'pooling benefits'})\n\n"
            f"**Implied acquisition premium:** {prem*100:.1f}%\n"
            f"**Debt/Equity ratio:** {de:.1f}x\n"
            f"**EPS impact (accretion/dilution):** {ad*100:+.2f}%\n\n"
            f"**Key structuring considerations:**\n"
            f"  1. At {tax*100:.0f}% effective tax rate, a cash deal provides "
            f"step-up in asset basis — typically worth 3–8% in NPV terms\n"
            f"  2. {prem*100:.0f}% premium is "
            f"{'within normal range (20–35%)' if 0.20 <= prem <= 0.40 else 'outside typical range — requires strong synergy justification'}\n"
            f"  3. Pro-forma EPS is "
            f"{'accretive — supportive for stock deal' if ad > 0 else 'dilutive — cash deal preferred to avoid EPS drag'}\n"
            f"  4. Consider collar mechanism to protect exchange ratio if stock deal pursued"
        )

    def _lbo_response(self, params, sector, confidence) -> str:
        lev,  _ = params.get("leverage_ratio")
        irr,  _ = params.get("irr")
        moic, _ = params.get("moic")
        em,   _ = params.get("exit_multiple")
        hp,   _ = params.get("hold_period")
        eq,   _ = params.get("equity_contribution")

        return (
            f"## LBO Analysis — {sector.title()} ({confidence*100:.0f}% confidence)\n\n"
            f"**Entry leverage:** {lev:.1f}x Debt/EBITDA\n"
            f"**Equity contribution:** {eq*100:.0f}% of TEV "
            f"({'standard PE structure' if 0.30 <= eq <= 0.50 else 'non-standard — review debt capacity'})\n"
            f"**Hold period:** {hp:.0f} years\n"
            f"**Exit multiple assumption:** {em:.1f}x EV/EBITDA\n\n"
            f"**Returns:**\n"
            f"  IRR: **{irr*100:.1f}%** "
            f"({'strong' if irr > 0.20 else 'marginal' if irr > 0.15 else 'below hurdle'})\n"
            f"  MOIC: **{moic:.2f}x** "
            f"({'excellent' if moic > 3.0 else 'acceptable' if moic > 2.0 else 'weak'})\n\n"
            f"**PE hurdle:** Most sponsors require >20% IRR and >2.5x MOIC. "
            f"{'This deal clears the hurdle.' if irr > 0.20 and moic > 2.5 else 'This deal does NOT clear standard hurdles — revisit assumptions.'}\n\n"
            f"**Sensitivity note:** Run IRR sensitivity on ±1x entry/exit multiple and "
            f"±1x leverage. Key value drivers: entry price and EBITDA growth."
        )

    def _model_response(self, params, model_type, model_result, confidence) -> str:
        if model_result is None:
            return (
                f"## {(model_type or 'Financial').upper()} Model Request\n\n"
                f"Parameters decoded (confidence {confidence*100:.0f}%):\n"
                + "\n".join(
                    f"  {k}: {v:.4f}" for k, v in params.high_confidence(0.3).items()
                )
                + "\n\nProvide company financials to compute the full model."
            )
        return (
            f"## {(model_type or 'Financial').upper()} Model Output\n\n"
            f"{model_result}\n\n"
            f"*Model confidence: {confidence*100:.0f}%*"
        )

    def _risk_response(self, params, risk_flags, confidence) -> str:
        risk, _ = params.get("deal_risk")
        return (
            f"## Risk Assessment ({confidence*100:.0f}% confidence)\n\n"
            f"**Overall deal risk score:** {risk*100:.0f}/100 "
            f"({'Low' if risk < 0.3 else 'Moderate' if risk < 0.6 else 'High'})\n\n"
            f"**Specific risk factors:**\n"
            + ("\n".join(f"  • {f}" for f in risk_flags) if risk_flags
               else "  • No critical risk flags identified")
        )

    def _general_response(self, params, question, confidence) -> str:
        hc = params.high_confidence(0.35)
        param_str = "\n".join(
            f"  {k}: {v:.4f}" for k, v in list(hc.items())[:8]
        ) if hc else "  [No high-confidence parameters — more ingestion needed]"
        return (
            f"## IB Analysis ({confidence*100:.0f}% confidence)\n\n"
            f"**Query:** {question}\n\n"
            f"**Brain output (top parameters):**\n{param_str}\n\n"
            f"*Tip: Ask a more specific question (DCF, LBO, deal structure, valuation, "
            f"precedents) for more targeted analysis. Ingest relevant deal documents to "
            f"improve domain accuracy.*"
        )
