"""
MotorCortex (M1) — Primary Motor Cortex
=========================================
100,000 neurons at full scale.

Generates motor commands from descending inputs (PFC voluntary,
CB cerebellar correction, BG action gating).

IMPORTANT: M1 output passes directly to SafetyKernel before decoding.
           This region provides get_motor_spikes() which Brain uses.

Biologically: M1 neurons encode movement direction via population vectors.
The linear population-vector decoder in MotorDecoder approximates this.

Outgoing: M1→M1 (recurrent), M1 → SafetyKernel (not a SynapsePool —
          it goes through io.decoder then safety.kernel)
"""

from __future__ import annotations

import numpy as np
from neuromorphic.regions.base_region import BrainRegion
from neuromorphic.core.neuron_group import NeuronGroup


class MotorCortex(BrainRegion):

    LABEL = "M1"

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

    def get_motor_spikes(self) -> np.ndarray:
        """
        Return the current spike array for this region.
        Called by Brain.step() → passed to SafetyKernel.check_and_gate().
        Returns a copy to prevent in-place modification by the kernel.
        """
        return self.local_spikes.copy()

    def encode_input(self, stimulus: np.ndarray) -> np.ndarray:
        return np.zeros(self.n_neurons, dtype=np.float32)
