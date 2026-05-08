"""
OllamaClient — local LLM bridge (no API key needed)
====================================================
Connects to a locally-running Ollama instance at localhost:11434.
Falls back gracefully if Ollama is not available.

Recommended models (run: ollama pull <name>):
  llama3.2:3b   — fastest on CPU, ~2GB,  good for queries
  llama3.1:8b   — better quality,  ~4.7GB
  mistral:7b    — strong reasoning, ~4.1GB
  qwen2.5:7b    — multilingual,     ~4.4GB
"""

from __future__ import annotations

import json
import logging
import threading
import urllib.request
import urllib.error
from typing import Iterator

logger = logging.getLogger("neuromorphic.llm")

_DEFAULT_URL   = "http://localhost:11434"
_DEFAULT_MODEL = "llama3.1:8b"    # best speed/quality on CPU-only (4.7 GB)
                                   # deepseek-r1:14b available via --model but needs GPU for speed


def _strip_thinking(text: str) -> str:
    """Remove DeepSeek-R1 <think>...</think> chain-of-thought blocks."""
    import re
    # Remove everything between <think> and </think> (including the tags)
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return cleaned.strip()


class OllamaClient:
    """
    Thin wrapper around the Ollama REST API.
    All network calls use stdlib only — no extra deps.
    """

    def __init__(
        self,
        base_url: str  = _DEFAULT_URL,
        model: str     = _DEFAULT_MODEL,
        timeout: int   = 180,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model    = model
        self.timeout  = timeout
        self._lock    = threading.Lock()

    # ── availability ──────────────────────────────────────────────────────────

    def is_available(self) -> bool:
        """Return True if Ollama is reachable and the model is loaded."""
        try:
            with urllib.request.urlopen(
                f"{self.base_url}/api/tags", timeout=2
            ) as resp:
                data   = json.loads(resp.read())
                models = [m["name"].split(":")[0] for m in data.get("models", [])]
                target = self.model.split(":")[0]
                return target in models
        except Exception:
            return False

    def available_models(self) -> list[str]:
        try:
            with urllib.request.urlopen(
                f"{self.base_url}/api/tags", timeout=2
            ) as resp:
                data = json.loads(resp.read())
                return [m["name"] for m in data.get("models", [])]
        except Exception:
            return []

    # ── generation ────────────────────────────────────────────────────────────

    def generate(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.7,
        max_tokens: int    = 1024,
    ) -> str | None:
        """
        Blocking single-shot generation.
        Returns the response string, or None on failure.
        For DeepSeek-R1 style models the <think>...</think> chain-of-thought
        block is automatically stripped so only the final answer is returned.
        """
        payload: dict = {
            "model":  self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature":   temperature,
                "num_predict":   max_tokens,
                "num_ctx":       8192,
            },
        }
        if system:
            payload["system"] = system

        body = json.dumps(payload).encode()
        req  = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data    = body,
            headers = {"Content-Type": "application/json"},
            method  = "POST",
        )

        with self._lock:
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    result = json.loads(resp.read())
                    text   = result.get("response", "").strip()
                    return _strip_thinking(text)
            except urllib.error.URLError as e:
                logger.warning(f"Ollama unavailable: {e}")
                return None
            except Exception as e:
                logger.error(f"Ollama generate error: {e}")
                return None

    def stream(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.7,
        max_tokens: int    = 512,
    ) -> Iterator[str]:
        """
        Streaming generation — yields text chunks as they arrive.
        Caller should handle StopIteration or break the loop.
        """
        payload: dict = {
            "model":  self.model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        if system:
            payload["system"] = system

        body = json.dumps(payload).encode()
        req  = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data    = body,
            headers = {"Content-Type": "application/json"},
            method  = "POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                for line in resp:
                    if not line.strip():
                        continue
                    try:
                        chunk = json.loads(line)
                        token = chunk.get("response", "")
                        if token:
                            yield token
                        if chunk.get("done"):
                            break
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            logger.error(f"Ollama stream error: {e}")
            return

    # ── model management ──────────────────────────────────────────────────────

    def pull(self, model: str | None = None) -> bool:
        """Pull a model from Ollama registry. Blocking."""
        target = model or self.model
        payload = json.dumps({"name": target}).encode()
        req = urllib.request.Request(
            f"{self.base_url}/api/pull",
            data    = payload,
            headers = {"Content-Type": "application/json"},
            method  = "POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=600) as resp:
                for line in resp:
                    try:
                        chunk = json.loads(line)
                        if chunk.get("status") == "success":
                            return True
                    except Exception:
                        pass
            return True
        except Exception as e:
            logger.error(f"Ollama pull failed: {e}")
            return False
