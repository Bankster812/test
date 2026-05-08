"""
GroqClient — fastest free LLM (Llama 3.3 70B at ~800 tok/s)
=============================================================
Groq runs LLMs on custom LPU hardware — responses feel near-instant.

Setup (30 seconds):
    1. Go to https://console.groq.com
    2. Sign up free (no credit card)
    3. API Keys → Create API Key
    4. Set env var:  GROQ_API_KEY=gsk_xxxxxxxxxxxx

Free tier: very generous — hundreds of queries per day.
Model: llama-3.3-70b-versatile (70B parameters, excellent quality)

API is OpenAI-compatible — same format as NVIDIA NIM.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.request
import urllib.error
from typing import Iterator

logger = logging.getLogger("neuromorphic.llm.groq")

_GROQ_BASE   = "https://api.groq.com/openai/v1"
_BEST_MODEL  = "llama-3.3-70b-versatile"   # 70B, ~800 tok/s on Groq LPU hardware


class GroqClient:
    """
    Groq LPU inference client — OpenAI-compatible, stdlib only.
    Llama 3.3 70B runs at ~800 tokens/second — effectively instant replies.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model:   str        = _BEST_MODEL,
        timeout: int        = 60,
    ) -> None:
        self.api_key = api_key or os.environ.get("GROQ_API_KEY", "")
        self.model   = model
        self.timeout = timeout

    def is_available(self) -> bool:
        return bool(self.api_key and self.api_key.startswith("gsk_"))

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
            f"{_GROQ_BASE}/chat/completions",
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
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")
            logger.error(f"Groq HTTP {e.code}: {body[:300]}")
        except Exception as e:
            logger.error(f"Groq error: {e}")
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
            f"{_GROQ_BASE}/chat/completions",
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
            logger.error(f"Groq stream error: {e}")
