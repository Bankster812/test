"""
llm — Claude API client for the company's agents
================================================

Stdlib-only (urllib) Anthropic Messages client, in the same spirit as
`neuromorphic/llm/groq_client.py`. Agents call `client.complete(system,
user)` to reason, draft seller outreach, or summarise a deal.

If no `ANTHROPIC_API_KEY` is set the client reports `is_available() ==
False` and callers fall back to their deterministic heuristics, so the
company keeps operating offline.
"""

from __future__ import annotations

import json
import logging
import urllib.request
import urllib.error

from . import config

logger = logging.getLogger("wholesale.llm")

_API_URL = "https://api.anthropic.com/v1/messages"
_API_VERSION = "2023-06-01"


class ClaudeClient:
    """Minimal Anthropic Messages client (stdlib only)."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        timeout: int | None = None,
    ) -> None:
        self.api_key = api_key if api_key is not None else config.ANTHROPIC_API_KEY
        self.model = model or config.LLM_MODEL
        self.timeout = timeout or config.LLM_TIMEOUT
        self.calls = 0          # cheap usage telemetry surfaced on the dashboard
        self.failures = 0

    def is_available(self) -> bool:
        return bool(self.api_key)

    def complete(
        self,
        system: str,
        user: str,
        max_tokens: int | None = None,
        temperature: float = 0.7,
    ) -> str | None:
        """Single-turn completion. Returns text, or None on any failure."""
        if not self.is_available():
            return None

        payload = json.dumps({
            "model": self.model,
            "max_tokens": max_tokens or config.LLM_MAX_TOKENS,
            "temperature": temperature,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }).encode()

        req = urllib.request.Request(
            _API_URL,
            data=payload,
            headers={
                "content-type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": _API_VERSION,
            },
            method="POST",
        )
        self.calls += 1
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read())
                blocks = data.get("content", [])
                text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
                return text.strip() or None
        except urllib.error.HTTPError as e:
            self.failures += 1
            body = e.read().decode(errors="replace")
            logger.warning("Claude HTTP %s: %s", e.code, body[:300])
        except Exception as e:  # noqa: BLE001  — never let an agent crash the company
            self.failures += 1
            logger.warning("Claude error: %s", e)
        return None


# Process-wide singleton; agents share it.
_default: ClaudeClient | None = None


def default_client() -> ClaudeClient:
    global _default
    if _default is None:
        _default = ClaudeClient()
    return _default
