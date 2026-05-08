"""
GenericLLMClient — works with any OpenAI-compatible API
========================================================
Supports any provider that speaks the /chat/completions format.
No third-party libraries — stdlib only.

Pre-configured providers (all free):
─────────────────────────────────────────────────────────
GOOGLE GEMINI (recommended — genuinely free, no card needed)
  url:   https://aistudio.google.com  → Get API Key
  key:   starts with AIza...
  model: gemini-2.0-flash   (free: 1500 req/day)

  client = GenericLLMClient.gemini("AIza-yourkey")

CEREBRAS (free, Llama 70B on fast custom hardware)
  url:   https://cloud.cerebras.ai   → sign up → API Keys
  key:   starts with csk-...
  model: llama3.1-70b

  client = GenericLLMClient.cerebras("csk-yourkey")

OPENROUTER (free models available — no card for free tier)
  url:   https://openrouter.ai  → sign up → API Keys
  key:   starts with sk-or-...
  model: meta-llama/llama-3.1-8b-instruct:free  (free)
         google/gemma-3-12b-it:free              (free)

  client = GenericLLMClient.openrouter("sk-or-yourkey")

GROQ (free tier with rate limits)
  url:   https://console.groq.com  → API Keys
  key:   starts with gsk_...
  model: llama-3.3-70b-versatile

  client = GenericLLMClient.groq("gsk_yourkey")

CUSTOM (any OpenAI-compatible endpoint)
  client = GenericLLMClient(base_url=..., api_key=..., model=...)
─────────────────────────────────────────────────────────
"""

from __future__ import annotations

import json
import logging
import os
import urllib.request
import urllib.error
from typing import Iterator

logger = logging.getLogger("neuromorphic.llm.generic")


class GenericLLMClient:
    """Universal OpenAI-compatible LLM client — stdlib only."""

    def __init__(
        self,
        base_url: str,
        api_key:  str,
        model:    str,
        timeout:  int = 90,
        name:     str = "",      # human-readable provider name for logs
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key  = api_key
        self.model    = model
        self.timeout  = timeout
        self.name     = name or base_url.split("//")[-1].split("/")[0]

    # ── factory methods ────────────────────────────────────────────────────────

    @classmethod
    def gemini(cls, api_key: str | None = None) -> "GenericLLMClient":
        """Google Gemini 2.0 Flash — free 1500 req/day, no credit card."""
        return cls(
            base_url = "https://generativelanguage.googleapis.com/v1beta/openai",
            api_key  = api_key or os.environ.get("GEMINI_API_KEY", ""),
            model    = "gemini-2.0-flash",
            name     = "Google Gemini 2.0 Flash",
        )

    @classmethod
    def cerebras(cls, api_key: str | None = None) -> "GenericLLMClient":
        """Cerebras Cloud — free tier, Llama 3.1 70B on fast hardware."""
        return cls(
            base_url = "https://api.cerebras.ai/v1",
            api_key  = api_key or os.environ.get("CEREBRAS_API_KEY", ""),
            model    = "llama3.1-70b",
            name     = "Cerebras Llama 3.1 70B",
        )

    @classmethod
    def openrouter(cls, api_key: str | None = None, model: str = "meta-llama/llama-3.1-8b-instruct:free") -> "GenericLLMClient":
        """OpenRouter — has several completely free models (`:free` suffix)."""
        return cls(
            base_url = "https://openrouter.ai/api/v1",
            api_key  = api_key or os.environ.get("OPENROUTER_API_KEY", ""),
            model    = model,
            name     = f"OpenRouter {model}",
        )

    @classmethod
    def groq(cls, api_key: str | None = None) -> "GenericLLMClient":
        """Groq LPU — free tier, Llama 3.3 70B at ~800 tok/s."""
        return cls(
            base_url = "https://api.groq.com/openai/v1",
            api_key  = api_key or os.environ.get("GROQ_API_KEY", ""),
            model    = "llama-3.3-70b-versatile",
            name     = "Groq Llama 3.3 70B",
        )

    @classmethod
    def nvidia(cls, api_key: str | None = None) -> "GenericLLMClient":
        """NVIDIA NIM — Llama 3.3 70B Instruct."""
        return cls(
            base_url = "https://integrate.api.nvidia.com/v1",
            api_key  = api_key or os.environ.get("NVIDIA_API_KEY", ""),
            model    = "meta/llama-3.3-70b-instruct",
            name     = "NVIDIA Llama 3.3 70B",
        )

    # ── availability ──────────────────────────────────────────────────────────

    def is_available(self) -> bool:
        return bool(self.api_key and len(self.api_key) > 8)

    # ── chat completion ────────────────────────────────────────────────────────

    def chat(
        self,
        messages:    list[dict],
        temperature: float = 0.8,
        max_tokens:  int   = 1024,
    ) -> str | None:
        if not self.is_available():
            return None

        payload = json.dumps({
            "model":       self.model,
            "messages":    messages,
            "temperature": temperature,
            "max_tokens":  max_tokens,
        }).encode()

        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data    = payload,
            headers = {
                "Content-Type":  "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "User-Agent":    "neuromorphic-brain/1.0",
            },
            method  = "POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data    = json.loads(resp.read())
                choices = data.get("choices", [])
                if choices:
                    text = choices[0].get("message", {}).get("content", "").strip()
                    logger.debug(f"{self.name}: {len(text)} chars")
                    return text
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")
            logger.error(f"{self.name} HTTP {e.code}: {body[:300]}")
        except Exception as e:
            logger.error(f"{self.name} error: {e}")
        return None

    def stream(
        self,
        messages:    list[dict],
        temperature: float = 0.8,
        max_tokens:  int   = 1024,
    ) -> Iterator[str]:
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
            f"{self.base_url}/chat/completions",
            data    = payload,
            headers = {
                "Content-Type":  "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "User-Agent":    "neuromorphic-brain/1.0",
            },
            method  = "POST",
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
            logger.error(f"{self.name} stream error: {e}")
