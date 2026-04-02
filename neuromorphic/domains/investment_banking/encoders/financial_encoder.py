"""
FinancialEncoder — Financial data to spike trains
==================================================
Maps all IB-domain data to the three sensory channels Brain.step() expects.

Channel assignments
-------------------
V1 (visual)    : Deal context patterns — sector, deal type, size tier.
                 Encoded as a 2D spatial activation map (like a "scene").
A1 (auditory)  : Concept/token sequences — sequential injection of IB terms.
                 Each concept fires a dedicated A1 population; tokens spread
                 across timesteps for temporal processing.
S1 (soma)      : Numerical magnitudes — financial metrics rate-coded as
                 intensity arrays. High EBITDA margin → high S1 firing rate.

All outputs are compatible with Brain.step(visual=..., auditory=..., soma=...).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re

import numpy as np

from neuromorphic.io.encoder import SensoryEncoder


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class FinancialChunk:
    """A digestible unit of financial knowledge for ingestion."""
    concepts:          list[str]
    numerical_values:  dict[str, float]
    relationships:     list[tuple[str, str, str]]   # (subject, relation, object)
    table_data:        np.ndarray | None
    source:            str
    chunk_type:        str    # "text","table","model","deal_summary","web","transcript"


@dataclass
class QueryVector:
    """Parsed representation of a user IB question."""
    raw_text:         str
    query_type:       str                   # "valuation","structuring","model_request",
                                            # "precedent","risk","general"
    concepts:         list[str]
    numerical_values: dict[str, float]
    target_sector:    str | None
    target_model:     str | None            # "dcf","lbo","merger","comps","precedents","credit"
    constraints:      dict[str, float]


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

class FinancialTokenizer:
    """
    Maps IB domain terms to A1 neuron population indices.
    Each term maps to a contiguous block of neurons.
    Block size = n_a1_neurons / vocabulary_size.
    """

    def __init__(self, vocabulary: dict[str, int]):
        self.vocab = vocabulary                        # term → block_index
        self._reverse = {v: k for k, v in vocabulary.items()}

    def tokenize(self, text: str) -> list[int]:
        """Convert text to list of vocabulary indices (known terms only)."""
        text_lower = text.lower()
        # Replace punctuation with spaces
        text_lower = re.sub(r"[^a-z0-9_\s]", " ", text_lower)
        words  = text_lower.split()
        tokens = []
        for w in words:
            # Exact match
            if w in self.vocab:
                tokens.append(self.vocab[w])
                continue
            # Partial match (prefix)
            for term, idx in self.vocab.items():
                if w.startswith(term) or term.startswith(w):
                    tokens.append(idx)
                    break
        return tokens

    def get_neuron_range(self, token_id: int, n_a1: int) -> tuple[int, int]:
        """Return [start, end) within A1 for this token's population."""
        vocab_size = max(len(self.vocab), 1)
        block_size = max(1, n_a1 // vocab_size)
        start = (token_id % (n_a1 // block_size)) * block_size
        end   = min(start + block_size, n_a1)
        return start, end


# ---------------------------------------------------------------------------
# Main encoder
# ---------------------------------------------------------------------------

class FinancialEncoder:
    """
    Converts financial domain data into spike-train arrays for Brain.step().

    Parameters
    ----------
    config : ib_config module
    rng    : np.random.Generator
    """

    def __init__(self, config, rng: np.random.Generator):
        self.cfg       = config
        self.rng       = rng
        self.base      = SensoryEncoder(rng=rng)
        self.tokenizer = FinancialTokenizer(config.IB_VOCABULARY)

    # ------------------------------------------------------------------
    # Query encoding
    # ------------------------------------------------------------------

    def encode_query(
        self,
        query: QueryVector,
        n_v1: int,
        n_a1: int,
        n_s1: int,
        dt: float,
    ) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
        """
        Encode a parsed query into spike arrays.

        Returns
        -------
        visual    : np.ndarray bool, shape (n_v1,)   — context pattern
        auditory  : list[np.ndarray bool], len=QUERY_STEPS — token sequence
        soma      : np.ndarray bool, shape (n_s1,)   — numerical magnitudes
        """
        visual   = self._encode_context(query, n_v1, dt)
        auditory = self._encode_concepts(query.concepts, n_a1, dt,
                                         n_steps=self.cfg.QUERY_STEPS)
        soma     = self._encode_numerics(query.numerical_values, n_s1, dt)
        return visual, auditory, soma

    # ------------------------------------------------------------------
    # Document chunk encoding
    # ------------------------------------------------------------------

    def encode_chunk(
        self,
        chunk: FinancialChunk,
        n_v1: int,
        n_a1: int,
        n_s1: int,
        dt: float,
    ) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
        """Encode a document chunk for ingestion (same signature as encode_query)."""
        # Context: chunk type as a sparse 2D pattern
        ctx_vec = np.zeros(n_v1, dtype=np.float32)
        type_hash = hash(chunk.chunk_type) % n_v1
        ctx_vec[type_hash::max(1, n_v1 // 8)] = 1.0
        visual   = self.base.rate_encode(ctx_vec, n_v1, dt, max_rate_hz=60.0)
        auditory = self._encode_concepts(chunk.concepts, n_a1, dt,
                                         n_steps=self.cfg.INGESTION_STEPS)
        soma     = self._encode_numerics(chunk.numerical_values, n_s1, dt)
        return visual, auditory, soma

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _encode_context(self, query: QueryVector, n_v1: int, dt: float) -> np.ndarray:
        """
        Encode deal context (sector, query type) as a V1 spatial pattern.
        Different sectors produce reliably different spatial activations.
        """
        context_vec = np.zeros(n_v1, dtype=np.float32)
        # Query type occupies first quarter of V1
        qt_hash = abs(hash(query.query_type)) % (n_v1 // 4)
        context_vec[qt_hash:qt_hash + max(1, n_v1 // 20)] = 0.8
        # Sector occupies second quarter
        if query.target_sector:
            sec_hash = abs(hash(query.target_sector)) % (n_v1 // 4) + n_v1 // 4
            context_vec[sec_hash:sec_hash + max(1, n_v1 // 20)] = 0.7
        # Model type occupies third quarter
        if query.target_model:
            mod_hash = abs(hash(query.target_model)) % (n_v1 // 4) + n_v1 // 2
            context_vec[mod_hash:mod_hash + max(1, n_v1 // 20)] = 0.9
        return self.base.rate_encode(context_vec, n_v1, dt, max_rate_hz=80.0)

    def _encode_concepts(
        self,
        concepts: list[str],
        n_a1: int,
        dt: float,
        n_steps: int,
    ) -> list[np.ndarray]:
        """
        Temporal-code concept tokens into sequential A1 spike patterns.
        Each concept fires its dedicated A1 population for a window of steps.
        Concepts are spread across the full n_steps window.
        """
        result = []
        tokens = []
        for c in concepts:
            toks = self.tokenizer.tokenize(c)
            tokens.extend(toks)

        if not tokens:
            # No concepts: return low-noise baseline
            for _ in range(n_steps):
                result.append(self.rng.random(n_a1) < 0.01)
            return result

        steps_per_token = max(1, n_steps // len(tokens))
        token_idx = 0
        for step in range(n_steps):
            spikes = np.zeros(n_a1, dtype=np.bool_)
            if token_idx < len(tokens):
                tok = tokens[token_idx]
                s, e = self.tokenizer.get_neuron_range(tok, n_a1)
                # Rate-code: population fires for its window
                probs = np.zeros(n_a1, dtype=np.float32)
                probs[s:e] = 0.7
                spikes = self.rng.random(n_a1) < probs
                if (step + 1) % steps_per_token == 0:
                    token_idx += 1
            else:
                spikes = self.rng.random(n_a1) < 0.01
            result.append(spikes)
        return result

    def _encode_numerics(
        self,
        values: dict[str, float],
        n_s1: int,
        dt: float,
    ) -> np.ndarray:
        """
        Rate-code financial numerical values into S1.
        Each named value is normalized to [0,1] using PARAM_RANGES priors.
        """
        stimulus = np.zeros(n_s1, dtype=np.float32)
        param_ranges = self.cfg.PARAM_RANGES
        slots        = self.cfg.PARAM_SLOTS
        n_params     = self.cfg.N_FINANCIAL_PARAMS
        block_size   = max(1, n_s1 // n_params)

        for name, value in values.items():
            if name not in slots:
                continue
            idx   = slots[name]
            lo, hi = param_ranges.get(name, (0.0, 1.0))
            norm  = float(np.clip((value - lo) / max(hi - lo, 1e-9), 0.0, 1.0))
            start = idx * block_size
            end   = min(start + block_size, n_s1)
            stimulus[start:end] = norm

        return self.base.rate_encode(stimulus, n_s1, dt, max_rate_hz=120.0)
