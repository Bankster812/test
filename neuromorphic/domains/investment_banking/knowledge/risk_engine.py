"""
RiskEngine — Amygdala-backed Financial Risk Detector
=====================================================
Flags dangerous financial conditions before they reach output.
The amygdala in the neuromorphic brain fires on threat signals;
this module is the financial analogue: a fast, always-on watchdog.

Risk levels:  LOW < MEDIUM < HIGH < CRITICAL
All methods are pure (no side-effects) — safe to call from any thread.
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger("ib_brain.risk_engine")

# ── Enums ────────────────────────────────────────────────────────────────────

class RiskLevel(str, Enum):
    LOW      = "LOW"
    MEDIUM   = "MEDIUM"
    HIGH     = "HIGH"
    CRITICAL = "CRITICAL"

    def __ge__(self, other):  # allow comparison
        order = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
        return order.index(self.value) >= order.index(other.value)

    def __gt__(self, other):
        order = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
        return order.index(self.value) > order.index(other.value)


# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class RiskFlag:
    code:        str          # machine-readable key  e.g. "LEVERAGE_TOO_HIGH"
    description: str          # human-readable message
    level:       RiskLevel
    value:       float | None = None   # the offending value, if numeric
    threshold:   float | None = None   # the limit that was breached

    def __str__(self):
        val_str = f" (value={self.value:.2f}, limit={self.threshold})" if self.value is not None else ""
        return f"[{self.level.value}] {self.code}: {self.description}{val_str}"


@dataclass
class RiskReport:
    flags:        list[RiskFlag] = field(default_factory=list)
    level:        RiskLevel      = RiskLevel.LOW
    score:        float          = 0.0   # 0.0 (safe) → 1.0 (critical)
    is_safe:      bool           = True
    summary:      str            = ""

    def __post_init__(self):
        self._recompute()

    def add(self, flag: RiskFlag) -> None:
        self.flags.append(flag)
        self._recompute()

    def _recompute(self):
        if not self.flags:
            self.level   = RiskLevel.LOW
            self.score   = 0.0
            self.is_safe = True
            self.summary = "No risk flags."
            return

        level_weights = {"LOW": 0.1, "MEDIUM": 0.3, "HIGH": 0.6, "CRITICAL": 1.0}
        raw = sum(level_weights[f.level.value] for f in self.flags)
        self.score   = float(min(raw / max(len(self.flags), 1), 1.0))

        # Highest flag wins
        ordered = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
        max_lv  = max(f.level.value for f in self.flags)
        self.level   = RiskLevel(max_lv)
        self.is_safe = self.level not in (RiskLevel.HIGH, RiskLevel.CRITICAL)

        crit_count = sum(1 for f in self.flags if f.level == RiskLevel.CRITICAL)
        high_count = sum(1 for f in self.flags if f.level == RiskLevel.HIGH)
        self.summary = (
            f"{len(self.flags)} flag(s) — level {self.level.value} "
            f"[{crit_count} CRITICAL, {high_count} HIGH] score={self.score:.2f}"
        )

    def __str__(self):
        lines = [f"RiskReport: {self.summary}"]
        for f in self.flags:
            lines.append(f"  {f}")
        return "\n".join(lines)


# ── Thresholds (tuned for IB realism) ────────────────────────────────────────

class _Limits:
    # Leverage
    LEVERAGE_WARN     = 6.0    # net debt / EBITDA
    LEVERAGE_HIGH     = 8.0
    LEVERAGE_CRITICAL = 12.0

    # Coverage
    COVERAGE_WARN     = 2.0    # EBITDA / interest
    COVERAGE_LOW      = 1.5
    COVERAGE_CRITICAL = 1.0

    # WACC
    WACC_MIN = 0.03            # 3%
    WACC_MAX = 0.30            # 30%

    # IRR
    IRR_MIN  = 0.0

    # Premium
    PREMIUM_HIGH_PCT  = 60.0   # % premium over unaffected
    PREMIUM_WARN_PCT  = 45.0

    # EV/EBITDA
    EV_EBITDA_HIGH    = 25.0
    EV_EBITDA_WARN    = 18.0
    EV_EBITDA_LOW     = 3.0    # suspiciously cheap

    # DCF-specific
    TERMINAL_GROWTH_MAX = 0.05  # 5% — higher is usually fantasy
    REVENUE_GROWTH_MAX  = 0.50  # 50% CAGR over 5yr is aggressive

    # LBO
    MOIC_SUSPICIOUS   = 10.0   # above 10x raises eyebrows
    EXIT_MULT_HIGH    = 20.0


# ── Core Engine ───────────────────────────────────────────────────────────────

class RiskEngine:
    """
    Stateless financial risk detector.

    Usage
    -----
    engine = RiskEngine()
    report = engine.analyse(params_dict)
    if not report.is_safe:
        print(report)
    """

    def analyse(self, params: dict[str, Any]) -> RiskReport:
        """
        Main entry point.  `params` is a flat dict of financial values.
        Keys align with ib_config.PARAM_SLOTS names (lowercase, underscored).
        """
        report = RiskReport()

        self._check_nan_inf(params, report)
        self._check_leverage(params, report)
        self._check_coverage(params, report)
        self._check_wacc(params, report)
        self._check_irr(params, report)
        self._check_premium(params, report)
        self._check_multiples(params, report)
        self._check_dcf_assumptions(params, report)
        self._check_lbo_assumptions(params, report)
        self._check_ebitda(params, report)

        if report.flags:
            logger.warning(f"RiskEngine: {report.summary}")
        else:
            logger.debug("RiskEngine: all clear")

        return report

    def analyse_model_result(self, model_type: str, result: dict[str, Any]) -> RiskReport:
        """Convenience wrapper for analysing model outputs directly."""
        return self.analyse(result)

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def _check_nan_inf(self, p: dict, r: RiskReport):
        bad = []
        for k, v in p.items():
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                bad.append(k)
        if bad:
            r.add(RiskFlag(
                "NAN_INF_OUTPUT",
                f"NaN or Inf in output field(s): {', '.join(bad)}",
                RiskLevel.CRITICAL,
            ))

    def _check_leverage(self, p: dict, r: RiskReport):
        lev = p.get("leverage") or p.get("total_leverage") or p.get("net_leverage")
        if lev is None:
            return
        lev = float(lev)
        if lev >= _Limits.LEVERAGE_CRITICAL:
            r.add(RiskFlag("LEVERAGE_CRITICAL", "Leverage is dangerously high — likely unsustainable", RiskLevel.CRITICAL, lev, _Limits.LEVERAGE_CRITICAL))
        elif lev >= _Limits.LEVERAGE_HIGH:
            r.add(RiskFlag("LEVERAGE_HIGH", "High leverage — covenant risk elevated", RiskLevel.HIGH, lev, _Limits.LEVERAGE_HIGH))
        elif lev >= _Limits.LEVERAGE_WARN:
            r.add(RiskFlag("LEVERAGE_ELEVATED", "Leverage above typical IB comfort zone", RiskLevel.MEDIUM, lev, _Limits.LEVERAGE_WARN))

    def _check_coverage(self, p: dict, r: RiskReport):
        cov = p.get("interest_coverage") or p.get("coverage")
        if cov is None:
            return
        cov = float(cov)
        if cov <= _Limits.COVERAGE_CRITICAL:
            r.add(RiskFlag("COVERAGE_BREACH", "Interest coverage ≤1.0x — default risk", RiskLevel.CRITICAL, cov, _Limits.COVERAGE_CRITICAL))
        elif cov <= _Limits.COVERAGE_LOW:
            r.add(RiskFlag("COVERAGE_LOW", "Interest coverage below 1.5x — covenant likely breached", RiskLevel.HIGH, cov, _Limits.COVERAGE_LOW))
        elif cov <= _Limits.COVERAGE_WARN:
            r.add(RiskFlag("COVERAGE_WARN", "Interest coverage below 2.0x — watch closely", RiskLevel.MEDIUM, cov, _Limits.COVERAGE_WARN))

    def _check_wacc(self, p: dict, r: RiskReport):
        wacc = p.get("wacc")
        if wacc is None:
            return
        wacc = float(wacc)
        # Normalise: accept both 0.10 and 10.0 formats
        if wacc > 1.0:
            wacc /= 100.0
        if wacc < _Limits.WACC_MIN:
            r.add(RiskFlag("WACC_TOO_LOW", f"WACC {wacc*100:.1f}% is below realistic floor of {_Limits.WACC_MIN*100:.0f}%", RiskLevel.HIGH, wacc, _Limits.WACC_MIN))
        elif wacc > _Limits.WACC_MAX:
            r.add(RiskFlag("WACC_TOO_HIGH", f"WACC {wacc*100:.1f}% exceeds {_Limits.WACC_MAX*100:.0f}% — check inputs", RiskLevel.HIGH, wacc, _Limits.WACC_MAX))

    def _check_irr(self, p: dict, r: RiskReport):
        irr = p.get("irr")
        if irr is None:
            return
        irr = float(irr)
        if irr > 1.0:
            irr /= 100.0
        if irr < _Limits.IRR_MIN:
            r.add(RiskFlag("IRR_NEGATIVE", "Negative IRR — deal destroys value", RiskLevel.HIGH, irr, 0.0))
        elif irr < 0.08:
            r.add(RiskFlag("IRR_BELOW_HURDLE", f"IRR {irr*100:.1f}% below typical PE hurdle rate of 20%+", RiskLevel.MEDIUM, irr, 0.15))

    def _check_premium(self, p: dict, r: RiskReport):
        prem = p.get("premium_pct") or p.get("acquisition_premium")
        if prem is None:
            return
        prem = float(prem)
        if prem > _Limits.PREMIUM_HIGH_PCT:
            r.add(RiskFlag("PREMIUM_VERY_HIGH", f"Acquisition premium {prem:.1f}% — value destruction risk", RiskLevel.HIGH, prem, _Limits.PREMIUM_HIGH_PCT))
        elif prem > _Limits.PREMIUM_WARN_PCT:
            r.add(RiskFlag("PREMIUM_ELEVATED", f"Premium {prem:.1f}% is above typical range (25-45%)", RiskLevel.MEDIUM, prem, _Limits.PREMIUM_WARN_PCT))

    def _check_multiples(self, p: dict, r: RiskReport):
        ev_ebitda = p.get("ev_ebitda") or p.get("entry_multiple") or p.get("exit_multiple")
        if ev_ebitda is not None:
            ev_ebitda = float(ev_ebitda)
            if ev_ebitda >= _Limits.EV_EBITDA_HIGH:
                r.add(RiskFlag("MULTIPLE_VERY_HIGH", f"EV/EBITDA {ev_ebitda:.1f}x is extremely rich", RiskLevel.HIGH, ev_ebitda, _Limits.EV_EBITDA_HIGH))
            elif ev_ebitda >= _Limits.EV_EBITDA_WARN:
                r.add(RiskFlag("MULTIPLE_ELEVATED", f"EV/EBITDA {ev_ebitda:.1f}x above typical range", RiskLevel.MEDIUM, ev_ebitda, _Limits.EV_EBITDA_WARN))
            elif 0 < ev_ebitda < _Limits.EV_EBITDA_LOW:
                r.add(RiskFlag("MULTIPLE_SUSPICIOUSLY_LOW", f"EV/EBITDA {ev_ebitda:.1f}x may indicate data error", RiskLevel.MEDIUM, ev_ebitda, _Limits.EV_EBITDA_LOW))

    def _check_dcf_assumptions(self, p: dict, r: RiskReport):
        tgr = p.get("terminal_growth") or p.get("terminal_growth_rate")
        if tgr is not None:
            tgr = float(tgr)
            if tgr > 1.0:
                tgr /= 100.0
            if tgr > _Limits.TERMINAL_GROWTH_MAX:
                r.add(RiskFlag("TERMINAL_GROWTH_HIGH", f"Terminal growth {tgr*100:.1f}% exceeds GDP — unrealistic", RiskLevel.HIGH, tgr, _Limits.TERMINAL_GROWTH_MAX))
            if tgr < 0:
                r.add(RiskFlag("TERMINAL_GROWTH_NEGATIVE", "Negative terminal growth — declining perpetuity assumed", RiskLevel.MEDIUM, tgr, 0.0))

        rev_growth = p.get("revenue_growth") or p.get("revenue_cagr")
        if rev_growth is not None:
            rg = float(rev_growth)
            if rg > 1.0:
                rg /= 100.0
            if rg > _Limits.REVENUE_GROWTH_MAX:
                r.add(RiskFlag("REVENUE_GROWTH_AGGRESSIVE", f"Revenue CAGR {rg*100:.0f}% is very aggressive — stress-test assumptions", RiskLevel.MEDIUM, rg, _Limits.REVENUE_GROWTH_MAX))

    def _check_lbo_assumptions(self, p: dict, r: RiskReport):
        moic = p.get("moic")
        if moic is not None:
            moic = float(moic)
            if moic > _Limits.MOIC_SUSPICIOUS:
                r.add(RiskFlag("MOIC_SUSPICIOUS", f"MOIC {moic:.1f}x — verify exit assumptions", RiskLevel.MEDIUM, moic, _Limits.MOIC_SUSPICIOUS))

        exit_mult = p.get("exit_multiple")
        if exit_mult is not None:
            em = float(exit_mult)
            if em > _Limits.EXIT_MULT_HIGH:
                r.add(RiskFlag("EXIT_MULTIPLE_HIGH", f"Exit multiple {em:.1f}x — significant multiple expansion assumed", RiskLevel.MEDIUM, em, _Limits.EXIT_MULT_HIGH))

    def _check_ebitda(self, p: dict, r: RiskReport):
        ebitda = p.get("ebitda") or p.get("ebitda_m")
        if ebitda is not None:
            ebitda = float(ebitda)
            if ebitda < 0:
                r.add(RiskFlag("EBITDA_NEGATIVE", "Negative EBITDA — leverage multiples are meaningless", RiskLevel.CRITICAL, ebitda, 0.0))
            elif ebitda == 0:
                r.add(RiskFlag("EBITDA_ZERO", "EBITDA is zero — division errors likely", RiskLevel.HIGH, ebitda, 0.0))
