"""
Amygdala (AMY) — Amygdala
===========================
30,000 neurons at full scale.

Threat detection, emotional valence tagging, and fear/reward conditioning.
Fast-path processing: responds to sensory stimuli before cortical analysis
completes (the "low road" in LeDoux's dual-pathway model).

Key outputs:
  - threat signal → Brainstem (autonomic arousal)
  - valence signal → PFC (emotional context for decisions)
  - reward signal → BG (modulates dopamine)

The AMY integrates inputs from IT (recognised objects) and sends
fast inhibitory/excitatory signals to downstream areas.

Outgoing: AMY→AMY (recurrent), AMY→PFC (valence/context),
          AMY→BG (reward/threat to action selection),
          AMY→BS (autonomic arousal)
"""

from __future__ import annotations

import numpy as np
from neuromorphic.regions.base_region import BrainRegion
from neuromorphic.core.neuron_group import NeuronGroup


class Amygdala(BrainRegion):

    LABEL = "AMY"

    def __init__(
        self,
        name: str,
        global_start: int,
        global_end: int,
        neuron_group: NeuronGroup,
        config,
    ):
        super().__init__(name, global_start, global_end, neuron_group, config)
        self._threat_ema = np.float32(0.0)
        self._ema_alpha  = np.float32(0.05)

    def step(
        self,
        t: float,
        dt: float,
        external_input: np.ndarray | None = None,
    ):
        super().step(t, dt, external_input)
        # Update threat estimate from population activity
        activity = float(self.local_spikes.mean())
        self._threat_ema = (
            (1.0 - self._ema_alpha) * self._threat_ema
            + self._ema_alpha * activity * 200.0
        )

    @property
    def threat_signal(self) -> float:
        """
        Threat level [0, 1]. High activity = threat detected.
        Feeds into NeuromodulationSystem and SafetyKernel indirectly.
        """
        return float(np.clip(self._threat_ema, 0.0, 1.0))

    def encode_input(self, stimulus: np.ndarray) -> np.ndarray:
        return np.zeros(self.n_neurons, dtype=np.float32)
