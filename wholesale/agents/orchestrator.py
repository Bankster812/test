"""
Orchestrator — "Atlas", the Chief of Staff (central accountable agent)
======================================================================

The single agent that takes accountability for the whole org. Every tick it
reviews the subagents and the deal book, rolls up an accountability report by
team, surfaces QA failures / policy blocks / stalls, and reports one-line org
health to the CEO. It owns no pipeline stage — it owns OUTCOMES.

This mirrors the playbook's governance principle: every part has an owner,
and nothing reaches the human without an accountable review layer.
"""

from __future__ import annotations

from typing import Any

from .base import BaseAgent
from ..core.models import Stage


class Orchestrator(BaseAgent):
    code = "CHIEF"
    name = "Atlas (Chief of Staff)"
    role = "Central oversight & accountability for all agents"
    team = "Executive"
    desc = ("Accountable for the whole org: reviews every subagent each cycle, "
            "rolls up health by team, surfaces QA failures, policy blocks and "
            "stalls, and reports status to the CEO.")
    color = "#4cc9f0"
    owns = ()

    def oversee(self) -> dict[str, Any]:
        self._set("acting", "Reviewing org & subagents", None)
        agents = [a for a in self.company.agents.values() if a.code != self.code]
        deals = self.company.deals

        # Roll up subagents by team.
        teams: dict[str, dict] = {}
        for a in agents:
            t = teams.setdefault(a.team, {"agents": [], "actions": 0, "active": 0})
            t["agents"].append(a.code)
            t["actions"] += a.handled
            if a.status in ("thinking", "acting"):
                t["active"] += 1

        qa_fail = [d.id for d in deals if d.qa.get("pass_fail") == "fail"]
        blocked = sum(1 for d in deals
                      if d.stage == Stage.DEAD and any("Blocked market" in f for f in d.flags))
        needs_human = len(self.company.action_queue())

        # Health: red if QA failures, amber if a backlog is piling up, else green.
        if qa_fail:
            health, note = "red", f"{len(qa_fail)} deal(s) failed QA — review required."
        elif needs_human > 8:
            health, note = "amber", f"{needs_human} actions awaiting the CEO."
        else:
            health, note = "green", "All teams nominal; agents executing."

        self.handled += 1
        report = {
            "health": health,
            "note": note,
            "teams": teams,
            "qa_failures": qa_fail,
            "policy_blocked": blocked,
            "needs_human": needs_human,
            "accountable_for": [a.code for a in agents],
        }
        self.current_task = f"Org {health.upper()} — {note}"
        return report
