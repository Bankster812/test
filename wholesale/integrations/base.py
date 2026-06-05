"""Adapter base — dry-run by default, never sends unless armed."""

from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class DispatchResult:
    channel: str          # "crm" | "email" | "slack"
    action: str           # "upsert_deal" | "send" | "post"
    target: str           # recipient / object id
    armed: bool           # was a live send actually performed?
    summary: str          # one-line human description
    payload: dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Adapter:
    """Common arming gate. Subclasses implement live transport in `_live_*`."""

    channel = "base"

    def __init__(self, armed: bool = False) -> None:
        self.armed = armed

    def _result(self, action: str, target: str, summary: str,
                payload: dict[str, Any]) -> DispatchResult:
        return DispatchResult(
            channel=self.channel, action=action, target=target,
            armed=self.armed, summary=summary, payload=payload,
        )
