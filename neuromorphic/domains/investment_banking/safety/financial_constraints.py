"""
FinancialSafetyKernel — Hard Financial Parameter Constraints
============================================================
Wraps the neuromorphic SafetyKernel with IB-specific frozen limits.

CRITICAL DESIGN RULE:
  These limits are constructed once and never modified by learning.
  The constraint dict uses Python name-mangling (_FinancialSafetyKernel__limits)
  so no external code can overwrite them after init.

Checks performed on every FinancialParams output:
  1. NaN / Inf → zero out + flag CRITICAL
  2. WACC in [3%, 30%]
  3. Leverage ≤ 12x
  4. IRR must be ≥ −50% (physically possible minimum)
  5. Premium ≤ 120% (anything higher is almost certainly a data error)
  6. Terminal growth in (−5%, +6%)
  7. EV/EBITDA in (0x, 40x]
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass

logger = logging.getLogger("ib_brain.financial_safety")


# ── Violation record ─────────────────────────────────────────────────────────

@dataclass
class ConstraintViolation:
    param:     str
    value:     float
    limit:     float
    direction: str   # "above" | "below" | "invalid"
    clamped_to: float

    def __str__(self):
        return (f"CONSTRAINT VIOLATION [{self.param}]: "
                f"value={self.value:.4f} is {self.direction} limit={self.limit:.4f} "
                f"→ clamped to {self.clamped_to:.4f}")


# ── Core kernel ──────────────────────────────────────────────────────────────

class FinancialSafetyKernel:
    """
    Frozen hard-limit enforcer for IB financial parameters.

    Constructed once.  Limits stored in name-mangled attribute so the
    Brain's learning machinery cannot overwrite them.

    Parameters
    ----------
    strict : bool
        If True, raise ValueError on CRITICAL violations instead of clamping.
        Default False (clamp silently and log).
    """

    def __init__(self, strict: bool = False):
        self.__strict = strict

        # ── Hard limits (name-mangled — cannot be modified by external code) ──
        self.__limits = {
            # (min, max) — values are in natural units (not %)
            "wacc":             (0.03,   0.30),    # 3% – 30%
            "cost_of_equity":   (0.03,   0.40),
            "cost_of_debt":     (0.01,   0.25),
            "leverage":         (0.0,    12.0),    # x EBITDA
            "net_leverage":     (-2.0,   12.0),    # can be negative (net cash)
            "irr":              (-0.50,  5.00),    # −50% to 500%
            "moic":             (0.0,    50.0),    # 0x to 50x
            "terminal_growth":  (-0.05,  0.06),    # −5% to 6%
            "revenue_growth":   (-1.0,   5.0),     # −100% to 500%
            "premium_pct":      (-10.0,  120.0),   # % — negative = below market
            "ev_ebitda":        (0.0,    40.0),    # 0x to 40x
            "ev_revenue":       (0.0,    50.0),    # 0x to 50x
            "interest_coverage": (0.0,   100.0),
            "dscr":             (0.0,    100.0),
        }

        logger.info("FinancialSafetyKernel initialised — limits frozen")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check_params(self, params: dict) -> tuple[dict, list[ConstraintViolation]]:
        """
        Enforce hard limits on a dict of financial parameters.

        Parameters
        ----------
        params : dict  {param_name: float}

        Returns
        -------
        (clamped_params, violations)
            clamped_params: copy of params with all values within limits
            violations:     list of ConstraintViolation (empty if all clear)
        """
        result = dict(params)
        violations: list[ConstraintViolation] = []

        for key, value in params.items():
            if value is None:
                continue

            # NaN / Inf check
            if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
                v = ConstraintViolation(key, float("nan"), 0.0, "invalid", 0.0)
                violations.append(v)
                result[key] = 0.0
                logger.warning(f"FinancialSafetyKernel: {v}")
                if self.__strict:
                    raise ValueError(f"NaN/Inf in financial param '{key}'")
                continue

            # Range check
            if key in self.__limits:
                lo, hi = self.__limits[key]
                fval = float(value)
                if fval < lo:
                    v = ConstraintViolation(key, fval, lo, "below", lo)
                    violations.append(v)
                    result[key] = lo
                    logger.warning(f"FinancialSafetyKernel: {v}")
                elif fval > hi:
                    v = ConstraintViolation(key, fval, hi, "above", hi)
                    violations.append(v)
                    result[key] = hi
                    logger.warning(f"FinancialSafetyKernel: {v}")

        return result, violations

    def check_value(self, param: str, value: float) -> tuple[float, ConstraintViolation | None]:
        """
        Check a single named parameter.  Returns (clamped_value, violation_or_None).
        """
        result, violations = self.check_params({param: value})
        return result[param], violations[0] if violations else None

    def limits_for(self, param: str) -> tuple[float, float] | None:
        """Return (min, max) for a named param, or None if unconstrained."""
        return self.__limits.get(param)

    def all_limits(self) -> dict:
        """Return a copy of all limit definitions (read-only copy)."""
        return dict(self.__limits)

    def is_safe(self, params: dict) -> bool:
        """Quick boolean check: True if all params within limits."""
        _, violations = self.check_params(params)
        return len(violations) == 0

    def validate_model_output(
        self, model_type: str, result: dict
    ) -> tuple[dict, list[ConstraintViolation]]:
        """
        Convenience: validate and clamp a full model output dict.
        model_type is logged for traceability.
        """
        clamped, violations = self.check_params(result)
        if violations:
            logger.warning(
                f"FinancialSafetyKernel [{model_type}]: "
                f"{len(violations)} violation(s) clamped"
            )
        return clamped, violations
