"""
agents — the autonomous workforce
=================================

One specialist per stage of the wholesale value chain. Each is Claude-API
powered for the "soft" work (judgement, seller messaging, summaries) and
deterministic for the "hard" numbers (underwriting, buyer matching), so
financial outputs are never hallucinated.
"""

from __future__ import annotations

from .base import BaseAgent
from .sourcing import ScoutAgent
from .underwriting import AnalystAgent
from .negotiation import CloserAgent
from .disposition import DispoAgent
from .transaction import CoordinatorAgent
from .compliance import ComplianceAgent

__all__ = [
    "BaseAgent", "ScoutAgent", "AnalystAgent", "CloserAgent",
    "DispoAgent", "CoordinatorAgent", "ComplianceAgent", "build_roster",
]


def build_roster(company) -> dict[str, BaseAgent]:
    """Instantiate the full org and key it by agent code."""
    roster = [
        ScoutAgent(company),
        AnalystAgent(company),
        CloserAgent(company),
        DispoAgent(company),
        CoordinatorAgent(company),
        ComplianceAgent(company),
    ]
    return {a.code: a for a in roster}
