"""
llm — Claude access for the agents, three backends
==================================================

Priority order (first that's available wins):

  1. API      — HTTP to api.anthropic.com, if ANTHROPIC_API_KEY is set
                (pay-per-token; separate from a claude.ai subscription).
  2. CLI      — shells out to the local `claude` binary (Claude Code), which
                is authenticated by YOUR claude.ai subscription. No API key,
                uses your plan. This is the default for subscription users.
  3. heuristic— no LLM; agents use their deterministic fallbacks and the
                company keeps running offline.

Force a backend with WS_LLM_BACKEND = api | cli | off.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import urllib.request
import urllib.error

from . import config

logger = logging.getLogger("wholesale.llm")

_API_URL = "https://api.anthropic.com/v1/messages"
_API_VERSION = "2023-06-01"


class ClaudeClient:
    """Talks to Claude via the API, the local CLI (subscription), or not at all."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        timeout: int | None = None,
    ) -> None:
        self.api_key = api_key if api_key is not None else config.ANTHROPIC_API_KEY
        self.model = model or config.LLM_MODEL
        self.timeout = timeout or config.LLM_TIMEOUT
        self.calls = 0
        self.failures = 0
        self._cli = shutil.which("claude")
        self.backend = self._pick_backend()

    # -- backend selection --------------------------------------------------
    def _pick_backend(self) -> str:
        # Explicit override wins. CLI (subscription) is OPT-IN: each call spawns
        # a `claude` subprocess, so it is not auto-selected on the hot loop.
        forced = os.environ.get("WS_LLM_BACKEND", "").lower()
        if forced in ("api", "cli", "off"):
            if forced == "off":
                return "heuristic"
            if forced == "cli" and not self._cli:
                return "heuristic"
            if forced == "api" and not self.api_key:
                return "heuristic"
            return forced
        if self.api_key:
            return "api"
        return "heuristic"

    def is_available(self) -> bool:
        return self.backend in ("api", "cli")

    def describe(self) -> str:
        return {"api": f"Claude API ({self.model})",
                "cli": "Claude subscription (claude CLI)",
                "heuristic": "heuristic (no LLM)"}[self.backend]

    # -- public API ---------------------------------------------------------
    def complete(self, system: str, user: str, max_tokens: int | None = None,
                 temperature: float = 0.7) -> str | None:
        if self.backend == "api":
            return self._via_api(system, user, max_tokens, temperature)
        if self.backend == "cli":
            return self._via_cli(system, user)
        return None

    # -- API backend --------------------------------------------------------
    def _via_api(self, system, user, max_tokens, temperature) -> str | None:
        payload = json.dumps({
            "model": self.model,
            "max_tokens": max_tokens or config.LLM_MAX_TOKENS,
            "temperature": temperature,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }).encode()
        req = urllib.request.Request(_API_URL, data=payload, method="POST", headers={
            "content-type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": _API_VERSION,
        })
        self.calls += 1
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read())
                blocks = data.get("content", [])
                text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
                return text.strip() or None
        except urllib.error.HTTPError as e:
            self.failures += 1
            logger.warning("Claude API HTTP %s: %s", e.code, e.read()[:300])
        except Exception as e:  # noqa: BLE001
            self.failures += 1
            logger.warning("Claude API error: %s", e)
        return None

    # -- CLI backend (subscription) ----------------------------------------
    def _via_cli(self, system, user) -> str | None:
        prompt = f"{system}\n\n{user}"
        self.calls += 1
        try:
            proc = subprocess.run(
                [self._cli, "-p", prompt, "--output-format", "json"],
                capture_output=True, text=True, timeout=self.timeout,
            )
            if proc.returncode != 0:
                self.failures += 1
                logger.warning("claude CLI exit %s: %s", proc.returncode, proc.stderr[:200])
                return None
            data = json.loads(proc.stdout)
            text = (data.get("result") or "").strip()
            return text or None
        except (subprocess.TimeoutExpired, json.JSONDecodeError, OSError) as e:
            self.failures += 1
            logger.warning("claude CLI error: %s", e)
        return None


_default: ClaudeClient | None = None


def default_client() -> ClaudeClient:
    global _default
    if _default is None:
        _default = ClaudeClient()
    return _default
