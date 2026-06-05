"""
base — common agent machinery
=============================

A `BaseAgent` owns one or more pipeline stages. The company calls
`agent.handle(deal)` for every deal sitting in a stage the agent owns.
Agents publish their live status to the event bus so the dashboard can
show "who is doing what" at any instant.
"""

from __future__ import annotations

import time
from typing import Any

from ..llm import default_client


class BaseAgent:
    # Subclasses set these.
    code: str = "AGT"
    name: str = "Agent"
    role: str = "Generic"
    color: str = "#8aa"
    team: str = "Operations"         # org grouping under the Chief of Staff
    desc: str = "A specialist agent."  # one-line "what I do" for the UI
    owns: tuple = ()                 # Stage values this agent processes

    def __init__(self, company) -> None:
        self.company = company
        self.bus = company.bus
        self.llm = default_client()
        self.status = "idle"         # "idle" | "thinking" | "acting"
        self.current_task = "Standing by"
        self.current_deal: int | None = None
        self.handled = 0
        self.last_active = time.time()

    # -- status plumbing ----------------------------------------------------
    def _set(self, status: str, task: str, deal_id: int | None = None) -> None:
        self.status = status
        self.current_task = task
        self.current_deal = deal_id
        self.last_active = time.time()

    def say(self, message: str, level: str = "info", deal_id: int | None = None) -> None:
        self.bus.publish(self.name, message, level=level, deal_id=deal_id)

    def idle(self) -> None:
        self._set("idle", "Standing by", None)

    # -- LLM helper ---------------------------------------------------------
    def reason(self, system: str, user: str, *, max_tokens: int = 400,
               temperature: float = 0.7, fallback: str = "") -> str:
        """Ask Claude; fall back to a deterministic string if unavailable."""
        out = self.llm.complete(system, user, max_tokens=max_tokens,
                                temperature=temperature)
        return out if out else fallback

    # -- contract -----------------------------------------------------------
    def handle(self, deal) -> None:  # pragma: no cover - overridden
        raise NotImplementedError

    def snapshot(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "name": self.name,
            "role": self.role,
            "team": self.team,
            "desc": self.desc,
            "color": self.color,
            "status": self.status,
            "task": self.current_task,
            "deal": self.current_deal,
            "handled": self.handled,
            "last_active": self.last_active,
        }
