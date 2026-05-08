"""
NVIDIA NIM Client — Free Tier, Strongest Available Model
=========================================================
Uses the NVIDIA NIM API (OpenAI-compatible) with no local compute needed.
Responses come from NVIDIA's GPU cloud in 2-5 seconds.

Setup:
    1. Go to https://build.nvidia.com
    2. Sign up free → get API key (starts with nvapi-)
    3. Set env var:  NVIDIA_API_KEY=nvapi-xxxxxxxxxxxx
    Or pass key=    NvidiaClient(api_key="nvapi-...")

Free tier: ~1,000 credits/month — plenty for hundreds of conversations.

Strongest free model: nvidia/llama-3.1-nemotron-70b-instruct
  - 70 billion parameters
  - GPT-4 class reasoning
  - Runs on NVIDIA H100 GPUs (2-5s response time)
"""

from __future__ import annotations

import json
import logging
import os
import urllib.request
import urllib.error
from typing import Iterator

logger = logging.getLogger("neuromorphic.llm.nvidia")

_NIM_BASE   = "https://integrate.api.nvidia.com/v1"
_BEST_MODEL = "meta/llama-3.1-405b-instruct"   # frontier model — free tier on NVIDIA NIM


class NvidiaClient:
    """
    NVIDIA NIM API client — OpenAI-compatible, stdlib only.
    Falls back gracefully if no API key is configured.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model:   str        = _BEST_MODEL,
        timeout: int        = 60,
    ) -> None:
        self.api_key = api_key or os.environ.get("NVIDIA_API_KEY", "")
        self.model   = model
        self.timeout = timeout

    # ── status ────────────────────────────────────────────────────────────────

    def is_available(self) -> bool:
        return bool(self.api_key and self.api_key.startswith("nvapi-"))

    # ── chat completion ────────────────────────────────────────────────────────

    def chat(
        self,
        messages:    list[dict],
        temperature: float = 0.8,
        max_tokens:  int   = 1024,
    ) -> str | None:
        """
        Send a list of {role, content} messages and return the reply.
        Returns None if the API key is not set or the call fails.
        """
        if not self.is_available():
            return None

        payload = json.dumps({
            "model":       self.model,
            "messages":    messages,
            "temperature": temperature,
            "max_tokens":  max_tokens,
        }).encode()

        req = urllib.request.Request(
            f"{_NIM_BASE}/chat/completions",
            data    = payload,
            headers = {
                "Content-Type":  "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method = "POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data    = json.loads(resp.read())
                choices = data.get("choices", [])
                if choices:
                    return choices[0].get("message", {}).get("content", "").strip()
                return None
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")
            logger.error(f"NVIDIA NIM HTTP {e.code}: {body[:300]}")
            return None
        except Exception as e:
            logger.error(f"NVIDIA NIM error: {e}")
            return None

    def stream(
        self,
        messages:    list[dict],
        temperature: float = 0.8,
        max_tokens:  int   = 1024,
    ) -> Iterator[str]:
        """Streaming version — yields tokens as they arrive."""
        if not self.is_available():
            return

        payload = json.dumps({
            "model":       self.model,
            "messages":    messages,
            "temperature": temperature,
            "max_tokens":  max_tokens,
            "stream":      True,
        }).encode()

        req = urllib.request.Request(
            f"{_NIM_BASE}/chat/completions",
            data    = payload,
            headers = {
                "Content-Type":  "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method = "POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                for line in resp:
                    line = line.decode().strip()
                    if not line.startswith("data:"):
                        continue
                    line = line[5:].strip()
                    if line == "[DONE]":
                        break
                    try:
                        chunk   = json.loads(line)
                        delta   = chunk["choices"][0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            yield content
                    except Exception:
                        continue
        except Exception as e:
            logger.error(f"NVIDIA NIM stream error: {e}")
