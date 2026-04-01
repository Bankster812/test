"""
AssociationCortex (IT) — Inferior Temporal / Multimodal Association Cortex
===========================================================================
100,000 neurons at full scale.

Binds visual, auditory, and somatosensory signals into unified object
and event representations. The first stage of high-level cognition.

Biologically: IT cortex in the ventral stream responds to complex objects,
faces, and scenes. Here it serves as the multimodal convergence zone where
cross-modal associations form through STDP.

Outgoing: IT→IT (recurrent), IT→PFC (working memory), IT→HPC (encoding),
          IT→AMY (threat/reward tagging)
"""

from __future__ import annotations

import numpy as np
from neuromorphic.regions.base_region import BrainRegion
from neuromorphic.core.neuron_group import NeuronGroup


class AssociationCortex(BrainRegion):

    LABEL = "IT"

    def __init__(
        self,
        name: str,
        global_start: int,
        global_end: int,
        neuron_group: NeuronGroup,
        config,
    ):
        super().__init__(name, global_start, global_end, neuron_group, config)

    def step(
        self,
        t: float,
        dt: float,
        external_input: np.ndarray | None = None,
    ):
        super().step(t, dt, external_input)

    def encode_input(self, stimulus: np.ndarray) -> np.ndarray:
        return np.zeros(self.n_neurons, dtype=np.float32)
