"""
AuditoryCortex (A1) — Primary Auditory Cortex
===============================================
50,000 neurons at full scale.

Receives auditory input (spectrograms, frequency bands) encoded as
population-coded spike trains. Organised tonotopically — neurons are
tuned to specific frequencies.

Outgoing: A1→A1 (recurrent), A1→IT (auditory feature integration)
"""

from __future__ import annotations

import numpy as np
from neuromorphic.regions.base_region import BrainRegion
from neuromorphic.core.neuron_group import NeuronGroup


class AuditoryCortex(BrainRegion):

    LABEL = "A1"

    def __init__(
        self,
        name: str,
        global_start: int,
        global_end: int,
        neuron_group: NeuronGroup,
        config,
    ):
        super().__init__(name, global_start, global_end, neuron_group, config)
        # Tonotopic preferred frequencies: log-spaced over human hearing range
        self._preferred_freq = np.logspace(
            np.log10(20), np.log10(20_000), self.n_neurons, dtype=np.float32
        )

    def step(
        self,
        t: float,
        dt: float,
        external_input: np.ndarray | None = None,
    ):
        super().step(t, dt, external_input)

    def encode_input(self, stimulus: np.ndarray) -> np.ndarray:
        """
        Convert a frequency-domain representation (mel spectrogram bands,
        values in [0,1]) to synaptic currents using tonotopic population coding.
        """
        flat = np.asarray(stimulus, dtype=np.float32).ravel()
        n    = self.n_neurons
        tiles  = (n + len(flat) - 1) // len(flat)
        scaled = np.tile(flat, tiles)[:n]
        return scaled * 1.5
