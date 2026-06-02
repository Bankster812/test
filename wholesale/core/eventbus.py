"""
eventbus — the company's nervous system
=======================================

A tiny thread-safe activity log + counters. Every agent action publishes
an `Activity`. The dashboard polls `snapshot()` to render what each agent
is doing. Kept deliberately simple: an in-memory ring buffer guarded by a
lock, no external broker.
"""

from __future__ import annotations

import itertools
import threading
import time
from dataclasses import dataclass, asdict
from typing import Any

from .. import config

_seq = itertools.count(1)


@dataclass
class Activity:
    agent: str
    level: str        # "info" | "win" | "warn" | "escalate"
    deal_id: int | None
    message: str
    ts: float
    seq: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class EventBus:
    def __init__(self, maxlen: int = config.MAX_ACTIVITY_LOG) -> None:
        self._lock = threading.Lock()
        self._log: list[Activity] = []
        self._maxlen = maxlen
        self.counters: dict[str, int] = {}

    def publish(
        self,
        agent: str,
        message: str,
        level: str = "info",
        deal_id: int | None = None,
    ) -> Activity:
        with self._lock:
            act = Activity(agent, level, deal_id, message, time.time(), next(_seq))
            self._log.append(act)
            if len(self._log) > self._maxlen:
                self._log = self._log[-self._maxlen:]
            self.counters[level] = self.counters.get(level, 0) + 1
            return act

    def recent(self, limit: int = 60, after_seq: int = 0) -> list[Activity]:
        with self._lock:
            items = [a for a in self._log if a.seq > after_seq]
            return items[-limit:]

    def snapshot(self, limit: int = 60) -> list[dict[str, Any]]:
        return [a.to_dict() for a in self.recent(limit)]
