"""
cockpit — the CEO Action Queue
==============================

Collapses the whole autonomous company into ONE prioritised list of the
actions that legally/operationally require a human: outreach ready to send,
deals to approve, contracts to sign, deals to list with buyers. The agents
do everything up to each action; you execute the last mile.

Draft-only by default: completing a "send" action marks it done and logs
it; nothing leaves the machine unless integrations are armed and you've
authorised it.
"""

from __future__ import annotations

from .models import Stage


def build_action_queue(company) -> list[dict]:
    """Return prioritised human actions across the company. Pure read."""
    actions: list[dict] = []

    # 1) CEO approvals — highest priority, money is committed.
    for d in company.deals:
        if d.needs_ceo and d.ceo_decision is None and d.stage == Stage.UNDER_CONTRACT:
            actions.append({
                "id": f"ceo:{d.id}",
                "kind": "approve",
                "priority": 1,
                "title": f"Approve/reject {d.label}",
                "detail": f"Contract ${d.contract_price:,}, fee ${d.assignment_fee:,}. "
                          f"{' | '.join(d.flags)}",
                "value": d.assignment_fee,
                "deal_id": d.id,
            })

    # 2) Contracts ready to sign / send (under contract or assigned).
    for d in company.deals:
        if d.stage in (Stage.UNDER_CONTRACT, Stage.ASSIGNED) and not d.needs_ceo:
            who = d.buyer.name if d.buyer else "seller"
            actions.append({
                "id": f"sign:{d.id}",
                "kind": "sign",
                "priority": 2,
                "title": f"Review & sign documents — {d.label}",
                "detail": f"PSA/assignment + disclosure ready ({who}). "
                          f"Generate: company.contracts.preforeclosure_packet(deal).",
                "value": d.assignment_fee,
                "deal_id": d.id,
            })

    # 3) Deals needing a buyer — list on a disposition channel.
    for d in company.deals:
        if d.stage == Stage.DISPOSITION and d.dispo_rounds >= 1 and d.buyer is None:
            actions.append({
                "id": f"dispo:{d.id}",
                "kind": "list",
                "priority": 3,
                "title": f"List {d.label} to cash buyers",
                "detail": f"Fee ${d.assignment_fee:,}. Post to OfferMarket / "
                          f"DispoBridge / your buyer list (see disposition panel).",
                "value": d.assignment_fee,
                "deal_id": d.id,
            })

    # 4) Outreach drafted and ready to send (B2B, dry-run).
    for i, c in enumerate(company.contacts):
        if c.status == "drafted":
            actions.append({
                "id": f"send:{i}",
                "kind": "send",
                "priority": 4,
                "title": f"Send {c.kind} outreach — {c.name}",
                "detail": (f"Submit via {c.url} " if c.url else "") +
                          "or copy the draft. (Dry-run until armed.)",
                "value": 0,
                "contact": c.name,
            })

    actions.sort(key=lambda a: (a["priority"], -a["value"]))
    return actions
