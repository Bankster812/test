"""
Hippocampus (HPC) — Hippocampus
=================================
80,000 neurons at full scale.

Episodic memory encoding and retrieval. Pattern completion (auto-associator):
given a partial input pattern, reconstruct the full stored pattern.
Also critical for spatial navigation and sequence learning.

Biologically: CA3 acts as an auto-associator via recurrent Schaffer
collaterals; CA1 performs pattern separation. Here we implement a
simplified version: high recurrent density + STDP creates Hopfield-like
attractor dynamics over time.

Unique property: HPC has the highest recurrent density (0.008) after PFC,
enabling stable attractor states that represent stored memories.

Outgoing: HPC→HPC (recurrent, auto-associator), HPC→PFC (retrieval to WM)
"""

from __future__ import annotations

import numpy as np
from neuromorphic.regions.base_region import BrainRegion
from neuromorphic.core.neuron_group import NeuronGroup


class Hippocampus(BrainRegion):

    LABEL = "HPC"

    def __init__(
        self,
        name: str,
        global_start: int,
        global_end: int,
        neuron_group: NeuronGroup,
        config,
    ):
        super().__init__(name, global_start, global_end, neuron_group, config)
        # Novelty signal: how different is current input from recent history?
        self.novelty: float = 0.0
        # Exponential moving average of activity for novelty detection
        self._activity_ema = np.zeros(self.n_neurons, dtype=np.float32)
        self._ema_alpha    = np.float32(0.01)

    def step(
        self,
        t: float,
        dt: float,
        external_input: np.ndarray | None = None,
    ):
        super().step(t, dt, external_input)
        # Update novelty: high divergence between current and expected activity
        current = self.local_spikes.astype(np.float32)
        delta   = np.abs(current - self._activity_ema)
        self.novelty = float(delta.mean()) * 100.0   # scale to [0, ~1]
        self._activity_ema += self._ema_alpha * (current - self._activity_ema)

    def encode_input(self, stimulus: np.ndarray) -> np.ndarray:
        return np.zeros(self.n_neurons, dtype=np.float32)
