"""
ElectronicLifeForm — the thinking entity
=========================================
Orchestrates:
  • NVIDIA NIM 70B (primary LLM — fastest, strongest)
  • Ollama local 8B (fallback if no API key)
  • ConversationMemory (persistent history)
  • PersonalityCore (evolving traits)
  • Neuromorphic brain STDP learning (synapses update from each exchange)
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

from .memory  import ConversationMemory
from .traits  import PersonalityCore

logger = logging.getLogger("neuromorphic.entity")


class ElectronicLifeForm:
    """
    The entity. One instance lives for the lifetime of brain_web.py.
    Every conversation turn:
      1. Builds a context-rich system prompt from personality + memory
      2. Calls NVIDIA NIM 70B (or Ollama fallback)
      3. Stores the exchange in persistent memory
      4. Runs STDP learning on the brain to encode the conversation
      5. Updates personality traits from brain state
    """

    def __init__(
        self,
        cloud_client  = None,   # any GenericLLMClient (Gemini, Cerebras, Groq, …)
        ollama_client = None,   # local Ollama fallback
        sim_thread    = None,
        brain_state   = None,
        # legacy kwargs kept for backward-compat
        groq_client   = None,
        nvidia_client = None,
    ) -> None:
        self.cloud    = cloud_client or groq_client or nvidia_client
        self.ollama   = ollama_client
        self.sim      = sim_thread
        self.state    = brain_state
        self.memory   = ConversationMemory()
        self.persona  = PersonalityCore()
        self._lock    = threading.Lock()

        logger.info(
            f"ElectronicLifeForm '{self.persona.name}' initialised. "
            f"Age: {self.persona.age_turns} turns. "
            f"Mood: {self.persona.mood}."
        )

    # ── main chat method ──────────────────────────────────────────────────────

    def chat(self, user_text: str) -> str:
        """
        Receive a message, think, respond, learn.
        Returns the entity's reply as a string.
        """
        with self._lock:
            brain_step = self._brain_step()
            rates      = self._brain_rates()

            # 1. Build messages for LLM
            system  = self.persona.system_prompt(self.memory.summary_text(n=4))
            history = self.memory.context_messages()
            messages = (
                [{"role": "system", "content": system}]
                + history
                + [{"role": "user", "content": user_text}]
            )

            # 2. Call LLM — NVIDIA NIM first, Ollama fallback
            reply = self._call_llm(messages)
            if not reply:
                reply = (
                    f"*{self.persona.name} pauses, neural activity spiking across PFC…*\n"
                    "I seem to be having difficulty forming a coherent response right now. "
                    "Please try again."
                )

            # 3. Persist to memory
            self.memory.add("user",      user_text, brain_step)
            self.memory.add("assistant", reply,     brain_step)

            # 4. Encode conversation into brain (STDP learning)
            self._encode_into_brain(user_text, reply)

            # 5. Update personality from this exchange
            self.persona.update_from_conversation(user_text, reply, rates)

            return reply

    # ── LLM routing ───────────────────────────────────────────────────────────

    def _call_llm(self, messages: list[dict]) -> str | None:
        # 1. Cloud LLM (Gemini / Cerebras / Groq / NVIDIA — whichever key is set)
        if self.cloud and self.cloud.is_available():
            try:
                resp = self.cloud.chat(messages, temperature=0.85, max_tokens=800)
                if resp:
                    return resp
            except Exception as e:
                logger.warning(f"Cloud LLM failed: {e}")

        # Fallback: local Ollama
        if self.ollama and self.ollama.is_available():
            try:
                # Convert messages to single prompt for Ollama generate API
                prompt = _messages_to_prompt(messages)
                system = next(
                    (m["content"] for m in messages if m["role"] == "system"), ""
                )
                resp = self.ollama.generate(prompt, system=system, max_tokens=700)
                if resp:
                    logger.debug(f"Ollama responded ({len(resp)} chars)")
                    return resp
            except Exception as e:
                logger.warning(f"Ollama failed: {e}")

        return None

    # ── brain integration ─────────────────────────────────────────────────────

    def _brain_step(self) -> int:
        if self.sim:
            return getattr(self.sim, "_step_count", 0)
        return 0

    def _brain_rates(self) -> dict[str, float]:
        if self.state:
            snap = self.state.snapshot()
            return snap.get("rates", {})
        return {}

    def _encode_into_brain(self, user_text: str, reply_text: str) -> None:
        """
        Inject the conversation as a brief spike burst into the brain
        and let STDP strengthen relevant synaptic pathways.
        Runs in a background thread — does not block the reply.
        """
        def _run():
            try:
                if not (self.sim and hasattr(self.sim, "_brain") and self.sim._brain):
                    return
                brain  = self.sim._brain
                import numpy as np

                # Encode text length / complexity as input amplitude
                complexity = min(1.0, len(user_text) / 500.0)
                n_v1 = brain.regions["V1"].end - brain.regions["V1"].start
                n_a1 = brain.regions["A1"].end - brain.regions["A1"].start
                n_s1 = brain.regions["S1"].end - brain.regions["S1"].start

                rng  = np.random.default_rng()
                # Run 30 high-signal steps — like a "thought burst" encoding the exchange
                for _ in range(30):
                    vis  = rng.random(n_v1).astype(np.float32) * complexity * 0.4
                    aud  = rng.random(n_a1).astype(np.float32) * complexity * 0.6
                    soma = rng.random(n_s1).astype(np.float32) * 0.1
                    brain.step(
                        visual   = vis,
                        auditory = aud,
                        soma     = soma,
                        reward   = 0.8,   # positive reward → dopamine gating → LTP
                    )
                logger.debug("STDP learning burst complete — synapses updated.")
            except Exception as e:
                logger.debug(f"STDP encode skipped: {e}")

        threading.Thread(target=_run, daemon=True).start()

    # ── info for UI ───────────────────────────────────────────────────────────

    def info(self) -> dict:
        return {
            "name":       self.persona.name,
            "mood":       self.persona.mood,
            "age_turns":  self.persona.age_turns,
            "traits":     self.persona.traits,
            "interests":  self.persona.top_interests,
            "memory_turns": self.memory.total_turns,
            "llm":        (
                f"{self.cloud.name} ({self.cloud.model.split('/')[-1]})" if (self.cloud and self.cloud.is_available()) else
                (self.ollama.model if self.ollama else "none")
            ),
        }


# ── helpers ───────────────────────────────────────────────────────────────────

def _messages_to_prompt(messages: list[dict]) -> str:
    """Convert OpenAI-style messages to a single Ollama prompt string."""
    parts = []
    for m in messages:
        if m["role"] == "system":
            continue   # passed separately as `system=`
        role = "Human" if m["role"] == "user" else "Assistant"
        parts.append(f"{role}: {m['content']}")
    parts.append("Assistant:")
    return "\n\n".join(parts)
