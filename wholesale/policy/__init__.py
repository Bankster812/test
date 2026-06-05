"""
policy — machine-readable, default-deny operating policy
========================================================

Loads the JSON policy files (market / outreach / payout) and answers the
questions the agents must check before acting. Default posture is BLOCKED:
any state not explicitly allowed/caution is blocked.

This turns compliance from scattered heuristics into one auditable policy
layer (per the playbook's Compliance Gatekeeper design).
"""

from __future__ import annotations

import json
import os
from functools import lru_cache

_DIR = os.path.dirname(__file__)


@lru_cache(maxsize=None)
def _load(name: str) -> dict:
    with open(os.path.join(_DIR, name)) as f:
        return json.load(f)


def market_rules() -> dict:
    return _load("market_rules.json")


def outreach_rules() -> dict:
    return _load("outreach_rules.json")


def payout_rules() -> dict:
    return _load("payout_rules.json")


def state_rule(state: str) -> dict:
    mr = market_rules()
    return mr["states"].get(state.upper(),
                            {"status": mr.get("default_status", "blocked"),
                             "reason": "unreviewed — default deny"})


def state_status(state: str) -> str:
    """'allowed' | 'caution' | 'blocked'."""
    return state_rule(state).get("status", "blocked")


def is_operable(state: str) -> bool:
    """May the company work deals here at all? (allowed or caution)."""
    return state_status(state) in ("allowed", "caution")


def outreach_allowed(state: str, channel: str) -> bool | str:
    """Per-state channel permission. Returns True/False or a gating string
    like 'after_legal_review' / 'caution'."""
    rule = state_rule(state)
    key = {"mail": "allow_direct_mail", "call": "allow_calling",
           "sms": "allow_sms"}.get(channel, channel)
    return rule.get(key, False)


def summary() -> dict:
    """Compact view for the dashboard."""
    mr = market_rules()
    return {
        "default_status": mr.get("default_status", "blocked"),
        "states": {s: r.get("status") for s, r in mr["states"].items()},
        "outreach_global": outreach_rules()["global_rules"],
        "payout_global": payout_rules()["global_rules"],
    }
