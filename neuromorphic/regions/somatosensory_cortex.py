"""
SomatosensoryCortex (S1) — Somatosensory Cortex
=================================================
100,000 neurons at full scale.

Processes touch, pressure, temperature, and proprioception.
Organised somatotopically (body map / homunculus).
Receives input from touch sensors and joint encoders.

Outgoing: S1→S1 (recurrent), S1→IT (multisensory integration)
"""

from __future__ import annotations

import numpy as np
from neuromorphic.regions.base_region import BrainRegion
from neuromorphic.core.neuron_group import NeuronGroup


class SomatosensoryCortex(BrainRegion):

    LABEL = "S1"

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
        Convert touch / proprioception data (values in [0,1]) to
        synaptic currents. High pressure → high current.
        """
        flat = np.asarray(stimulus, dtype=np.float32).ravel()
        n    = self.n_neurons
        tiles  = (n + len(flat) - 1) // len(flat)
        scaled = np.tile(flat, tiles)[:n]
        return scaled * 2.5   # slightly higher amplitude for touch
