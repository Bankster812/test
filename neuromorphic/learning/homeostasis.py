"""
HomeostaticScaling — Adaptive threshold regulation
====================================================
Adjusts each neuron's firing threshold to maintain a target firing rate.
This prevents run-away excitation or complete silence as synaptic weights
change through learning.

Mechanism (intrinsic plasticity):
    rate_error = firing_rate_est - target_rate
    v_thresh  += eta * rate_error

    → Neuron fires too fast: threshold rises  → harder to spike
    → Neuron fires too slow: threshold falls  → easier to spike

Applied every HOMEOSTASIS_INTERVAL timesteps (default: 1000 = 1 second
of simulated time), making it much slower than STDP. This separation of
timescales is biologically realistic.
"""

from __future__ import annotations

import numpy as np
from neuromorphic.core.neuron_group import NeuronGroup


class HomeostaticScaling:
    """
    Slow sliding threshold adaptation for all neurons.

    Parameters
    ----------
    neuron_group : NeuronGroup
    config : module
    """

    def __init__(self, neuron_group: NeuronGroup, config):
        self.ng              = neuron_group
        self.target_rate     = np.float32(config.TARGET_RATE)
        self.eta             = np.float32(config.ETA_HOMEOSTASIS)
        self.interval        = config.HOMEOSTASIS_INTERVAL
        self.v_thresh_min    = np.float32(config.V_THRESH_MIN)
        self.v_thresh_max    = np.float32(config.V_THRESH_MAX)

    def step(self, step_count: int):
        """
        Called every timestep; only applies the update every `interval` steps.
        """
        if step_count % self.interval != 0:
            return
        self._apply()

    def _apply(self):
        """
        Adjust all thresholds based on deviation from target firing rate.
        Clamps thresholds to physiologically plausible range.
        """
        rate_error = self.ng.firing_rate_est - self.target_rate
        self.ng.v_thresh += self.eta * rate_error
        np.clip(
            self.ng.v_thresh,
            self.v_thresh_min,
            self.v_thresh_max,
            out=self.ng.v_thresh,
        )

    def force_apply(self):
        """Apply homeostasis immediately regardless of step counter."""
        self._apply()
