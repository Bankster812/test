"""
Brainstem (BS) — Brainstem
============================
100,000 neurons at full scale.

Homeostasis, arousal, and global neuromodulatory broadcast.
The brainstem drives tonic baseline activity across all regions —
equivalent to the ascending arousal system (locus coeruleus → NA,
raphe → 5HT, basal forebrain → ACh).

Key functions:
  - Maintains wakefulness / arousal level
  - Broadcasts arousal signal to all other regions
  - Responds to AMY threat signals with autonomic arousal

Unique property: BS projects to ALL other regions (11 BS→X pools),
allowing global gain modulation. Brainstem activity sets the overall
excitability of the network.

Outgoing: BS→BS (recurrent), BS→all 10 other regions
"""

from __future__ import annotations

import numpy as np
from neuromorphic.regions.base_region import BrainRegion
from neuromorphic.core.neuron_group import NeuronGroup


class Brainstem(BrainRegion):

    LABEL = "BS"

    def __init__(
        self,
        name: str,
        global_start: int,
        global_end: int,
        neuron_group: NeuronGroup,
        config,
    ):
        super().__init__(name, global_start, global_end, neuron_group, config)
        # Tonic drive to maintain baseline arousal
        self._tonic_current    = np.float32(0.05)
        self._arousal_ema      = np.float32(0.5)
        self._ema_alpha        = np.float32(0.02)

    def step(
        self,
        t: float,
        dt: float,
        external_input: np.ndarray | None = None,
    ):
        # Tonic drive ensures the brainstem stays moderately active
        tonic = np.full(self.n_neurons, self._tonic_current, dtype=np.float32)
        self.neurons.inject_current_slice(self.start, self.end, tonic)
        super().step(t, dt, external_input)

        # Update arousal EMA
        activity = float(self.local_spikes.mean()) * 200.0
        self._arousal_ema = np.float32(
            (1.0 - self._ema_alpha) * float(self._arousal_ema)
            + self._ema_alpha * float(np.clip(activity, 0.0, 1.0))
        )

    @property
    def arousal(self) -> float:
        """
        Arousal level [0, 1]. Fed into NeuromodulationSystem as ACh signal.
        """
        return float(self._arousal_ema)

    def encode_input(self, stimulus: np.ndarray) -> np.ndarray:
        return np.zeros(self.n_neurons, dtype=np.float32)
