"""Adapter base — dry-run by default, never sends unless armed."""

from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class DispatchResult:
    channel: str          # "crm" | "email" | "slack" | "dispo"
    action: str           # "upsert_deal" | "send" | "post" | "submit"
    target: str           # recipient / object id
    armed: bool           # is the channel armed for live sends?
    delivered: bool       # did a live transport actually fire?
    summary: str          # one-line human description
    payload: dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Adapter:
    """Common arming gate. Subclasses implement live transport in `_live_*`.

    Arming never crashes the company: if armed but no transport is wired
    (`_live_*` raises NotImplementedError) the action is recorded as
    not-delivered, so going live degrades gracefully until you wire a real
    transport.
    """

    channel = "base"

    def __init__(self, armed: bool = False) -> None:
        self.armed = armed

    def _dispatch(self, live_fn, payload: dict[str, Any]) -> bool:
        if not self.armed:
            return False
        try:
            live_fn(payload)
            return True
        except NotImplementedError:
            return False

    def _result(self, action: str, target: str, summary: str,
                payload: dict[str, Any], delivered: bool = False) -> DispatchResult:
        return DispatchResult(
            channel=self.channel, action=action, target=target,
            armed=self.armed, delivered=delivered, summary=summary, payload=payload,
        )
