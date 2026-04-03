"""
ResponseFormatter — IBResponse → human-readable output
=======================================================
Three output modes:
  format_terminal(response)  → ANSI-coloured terminal string
  format_markdown(response)  → GitHub-flavoured markdown
  format_dict(response)      → JSON-serialisable dict

No external dependencies.
"""

from __future__ import annotations

import json
from typing import Any


# ── ANSI colour helpers ───────────────────────────────────────────────────────

class _C:
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    DIM    = "\033[2m"
    RED    = "\033[91m"
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    BLUE   = "\033[94m"
    CYAN   = "\033[96m"
    WHITE  = "\033[97m"
    GREY   = "\033[90m"


def _c(text: str, *codes: str) -> str:
    return "".join(codes) + text + _C.RESET


# ── Formatter ─────────────────────────────────────────────────────────────────

class ResponseFormatter:
    """
    Stateless formatter.  All methods are static — no instance needed.

    Usage
    -----
    fmt = ResponseFormatter()
    print(fmt.format_terminal(response))
    """

    # ------------------------------------------------------------------
    # Terminal (ANSI)
    # ------------------------------------------------------------------

    def format_terminal(self, response: Any) -> str:
        """Pretty-print IBResponse to terminal with ANSI colours."""
        lines = []

        # Header
        conf_pct = getattr(response, "confidence", 0.0) * 100
        conf_col = _C.GREEN if conf_pct >= 70 else (_C.YELLOW if conf_pct >= 40 else _C.RED)
        lines.append(_c("=" * 70, _C.BOLD, _C.BLUE))
        lines.append(_c("  NEUROMORPHIC IB ASSISTANT", _C.BOLD, _C.WHITE))
        lines.append(_c(f"  Confidence: {conf_pct:.0f}%", _C.BOLD, conf_col))
        lines.append(_c("=" * 70, _C.BOLD, _C.BLUE))
        lines.append("")

        # Answer text
        answer = getattr(response, "answer_text", str(response))
        lines.append(_c("ANSWER:", _C.BOLD, _C.CYAN))
        for line in answer.split("\n"):
            lines.append("  " + line)
        lines.append("")

        # Model result
        model_result = getattr(response, "model_result", None)
        if model_result:
            lines.append(_c("MODEL OUTPUT:", _C.BOLD, _C.CYAN))
            lines.extend(self._format_dict_terminal(model_result, indent=2))
            lines.append("")

        # Parameters
        params = getattr(response, "parameters", None)
        if params:
            lines.append(_c("FINANCIAL PARAMETERS:", _C.BOLD, _C.CYAN))
            param_dict = params if isinstance(params, dict) else {}
            if hasattr(params, "get"):
                # FinancialParams object
                from ..ib_config import PARAM_SLOTS
                for name in list(PARAM_SLOTS.keys())[:12]:  # top 12
                    try:
                        val, conf = params.get(name)
                        if val != 0.0:
                            c_str = _c(f"(conf {conf*100:.0f}%)", _C.DIM, _C.GREY)
                            lines.append(f"  {_c(name, _C.WHITE)}: {val:.4f} {c_str}")
                    except Exception:
                        pass
            lines.append("")

        # Risk flags
        risk_flags = getattr(response, "risk_flags", [])
        if risk_flags:
            lines.append(_c("RISK FLAGS:", _C.BOLD, _C.RED))
            for flag in risk_flags:
                lvl = str(flag) if hasattr(flag, "__str__") else str(flag)
                colour = _C.RED if "CRITICAL" in lvl or "HIGH" in lvl else _C.YELLOW
                lines.append("  " + _c(lvl, colour))
            lines.append("")

        lines.append(_c("=" * 70, _C.DIM, _C.BLUE))
        return "\n".join(lines)

    def _format_dict_terminal(self, d: dict, indent: int = 0) -> list[str]:
        lines = []
        prefix = " " * indent
        for k, v in d.items():
            if isinstance(v, dict):
                lines.append(f"{prefix}{_c(str(k)+':', _C.WHITE)}")
                lines.extend(self._format_dict_terminal(v, indent + 2))
            elif isinstance(v, (list, tuple)):
                lines.append(f"{prefix}{_c(str(k)+':', _C.WHITE)} [{len(v)} items]")
            elif isinstance(v, float):
                lines.append(f"{prefix}{_c(str(k)+':', _C.WHITE)} {_c(f'{v:.4f}', _C.GREEN)}")
            else:
                lines.append(f"{prefix}{_c(str(k)+':', _C.WHITE)} {v}")
        return lines

    # ------------------------------------------------------------------
    # Markdown
    # ------------------------------------------------------------------

    def format_markdown(self, response: Any) -> str:
        """Format IBResponse as GitHub-flavoured markdown."""
        sections = []

        conf_pct = getattr(response, "confidence", 0.0) * 100
        sections.append(f"# Neuromorphic IB Assistant\n\n> Confidence: **{conf_pct:.0f}%**\n")

        # Answer
        answer = getattr(response, "answer_text", str(response))
        sections.append(f"## Answer\n\n{answer}\n")

        # Model result
        model_result = getattr(response, "model_result", None)
        if model_result:
            sections.append("## Model Output\n")
            sections.append(self._dict_to_markdown_table(model_result))

        # Risk flags
        risk_flags = getattr(response, "risk_flags", [])
        if risk_flags:
            sections.append("## Risk Flags\n")
            for flag in risk_flags:
                lvl_str = str(flag)
                emoji = "🔴" if "CRITICAL" in lvl_str else ("🟠" if "HIGH" in lvl_str else "🟡")
                sections.append(f"- {emoji} {lvl_str}")
            sections.append("")

        return "\n".join(sections)

    def _dict_to_markdown_table(self, d: dict) -> str:
        flat = self._flatten(d)
        if not flat:
            return ""
        lines = ["| Parameter | Value |", "|-----------|-------|"]
        for k, v in flat.items():
            val_str = f"{v:.4f}" if isinstance(v, float) else str(v)
            lines.append(f"| `{k}` | {val_str} |")
        return "\n".join(lines) + "\n"

    def _flatten(self, d: dict, prefix: str = "") -> dict:
        out = {}
        for k, v in d.items():
            full_key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                out.update(self._flatten(v, full_key))
            elif not isinstance(v, (list, tuple)):
                out[full_key] = v
        return out

    # ------------------------------------------------------------------
    # Dict / JSON
    # ------------------------------------------------------------------

    def format_dict(self, response: Any) -> dict:
        """Return a JSON-serialisable dict of the response."""
        result: dict[str, Any] = {}

        result["confidence"] = getattr(response, "confidence", 0.0)
        result["answer_text"] = getattr(response, "answer_text", str(response))

        model_result = getattr(response, "model_result", None)
        if model_result:
            result["model_result"] = self._serialise(model_result)

        risk_flags = getattr(response, "risk_flags", [])
        result["risk_flags"] = [str(f) for f in risk_flags]

        return result

    def format_json(self, response: Any, indent: int = 2) -> str:
        """Return a formatted JSON string."""
        return json.dumps(self.format_dict(response), indent=indent, default=str)

    def _serialise(self, obj: Any) -> Any:
        """Recursively make obj JSON-serialisable."""
        if isinstance(obj, dict):
            return {k: self._serialise(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._serialise(v) for v in obj]
        if isinstance(obj, float):
            import math
            if math.isnan(obj) or math.isinf(obj):
                return None
            return obj
        return obj
