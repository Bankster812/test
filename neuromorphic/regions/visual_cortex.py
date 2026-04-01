"""
VisualCortex (V1) — Primary Visual Cortex
==========================================
150,000 neurons at full scale.

Receives visual input (images, video frames) encoded as Poisson spike
trains by SensoryEncoder.encode_visual(). Responds to oriented edges,
spatial frequencies, and colour contrasts.

Biologically: primary visual cortex organises into orientation columns,
ocular dominance columns, and retinotopic maps. Here we approximate this
with a flat population of rate-coded neurons, each with a Gaussian
receptive field over the input feature space.

Outgoing: V1→V1 (recurrent), V1→IT (ventral stream, object recognition)
"""

from __future__ import annotations

import numpy as np
from neuromorphic.regions.base_region import BrainRegion
from neuromorphic.core.neuron_group import NeuronGroup


class VisualCortex(BrainRegion):

    LABEL = "V1"

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
        """
        Convert a flattened visual frame (values in [0,1]) to a
        synaptic current array for this region.

        A strong stimulus pixel → high current to corresponding neurons.
        """
        flat = np.asarray(stimulus, dtype=np.float32).ravel()
        n    = self.n_neurons
        # Tile/trim to fit population
        tiles  = (n + len(flat) - 1) // len(flat)
        scaled = np.tile(flat, tiles)[:n]
        # Scale to biologically plausible current amplitude
        return scaled * 2.0   # nA (normalised)
