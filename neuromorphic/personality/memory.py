"""
Conversation Memory — persistent, brain-encoded
================================================
Stores every conversation turn and encodes it into the neuromorphic
brain via STDP spike injection.  Persists to data/memory.json.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

_DEFAULT_PATH = Path(__file__).parent.parent.parent / "data" / "memory.json"


class ConversationMemory:
    """
    Thread-safe conversation history with JSON persistence.
    Each entry: {role, content, timestamp, brain_step}
    """

    MAX_CONTEXT  = 20   # messages kept in live context window
    MAX_STORED   = 500  # messages stored on disk

    def __init__(self, path: Path | str = _DEFAULT_PATH) -> None:
        self.path  = Path(path)
        self._lock = threading.Lock()
        self._turns: list[dict] = []
        self._load()

    # ── persistence ───────────────────────────────────────────────────────────

    def _load(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text())
                self._turns = data.get("turns", [])
            except Exception:
                self._turns = []

    def _save(self) -> None:
        try:
            self.path.write_text(json.dumps({
                "turns": self._turns[-self.MAX_STORED:],
                "saved": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }, indent=2, ensure_ascii=False))
        except Exception:
            pass

    # ── public API ────────────────────────────────────────────────────────────

    def add(self, role: str, content: str, brain_step: int = 0) -> None:
        with self._lock:
            self._turns.append({
                "role":       role,
                "content":    content,
                "timestamp":  time.strftime("%Y-%m-%dT%H:%M:%S"),
                "brain_step": brain_step,
            })
            self._save()

    def context_messages(self) -> list[dict]:
        """Return last MAX_CONTEXT turns as {role, content} dicts for LLM."""
        with self._lock:
            recent = self._turns[-self.MAX_CONTEXT:]
        return [{"role": t["role"], "content": t["content"]} for t in recent]

    def all_turns(self) -> list[dict]:
        with self._lock:
            return list(self._turns)

    def last_n(self, n: int = 6) -> list[dict]:
        with self._lock:
            return list(self._turns[-n:])

    @property
    def total_turns(self) -> int:
        with self._lock:
            return len(self._turns)

    @property
    def user_turns(self) -> int:
        with self._lock:
            return sum(1 for t in self._turns if t["role"] == "user")

    def summary_text(self, n: int = 5) -> str:
        """One-line summary of the last n exchanges for the system prompt."""
        recent = self.last_n(n * 2)
        if not recent:
            return "No prior conversations."
        lines = []
        for t in recent:
            who   = "Human" if t["role"] == "user" else "You"
            snip  = t["content"][:80].replace("\n", " ")
            lines.append(f"{who}: {snip}…")
        return "\n".join(lines)
