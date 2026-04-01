"""
PrefrontalCortex (PFC) — Prefrontal Cortex
===========================================
150,000 neurons at full scale.

Working memory, executive control, and action planning.
Maintains sustained activity (persistent firing) representing current
goals and context. The most recurrently connected region.

Biologically: PFC sustains task-relevant information via recurrent
excitation. Here the high recurrent density (0.008) implements this.

Outgoing: PFC→PFC (strong recurrent), PFC→M1 (voluntary movement),
          PFC→BG (action selection), PFC→HPC (directed retrieval)
"""

from __future__ import annotations

import numpy as np
from neuromorphic.regions.base_region import BrainRegion
from neuromorphic.core.neuron_group import NeuronGroup


class PrefrontalCortex(BrainRegion):

    LABEL = "PFC"

    def __init__(
        self,
        name: str,
        global_start: int,
        global_end: int,
        neuron_group: NeuronGroup,
        config,
    ):
        super().__init__(name, global_start, global_end, neuron_group, config)
        # PFC has persistent activity — slight extra depolarisation bias
        self._bias_current = np.float32(0.02)

    def step(
        self,
        t: float,
        dt: float,
        external_input: np.ndarray | None = None,
    ):
        # Inject small tonic bias to sustain working memory activity
        tonic = np.full(self.n_neurons, self._bias_current, dtype=np.float32)
        self.neurons.inject_current_slice(self.start, self.end, tonic)
        super().step(t, dt, external_input)

    def encode_input(self, stimulus: np.ndarray) -> np.ndarray:
        return np.zeros(self.n_neurons, dtype=np.float32)
