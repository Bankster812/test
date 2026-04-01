"""
Cerebellum (CB) — Cerebellum
==============================
100,000 neurons at full scale.

Motor learning and timing. Computes forward models: given a motor command,
predict the sensory consequence and pre-emptively correct for it.

Biologically: the cerebellum learns to predict and cancel predictable
disturbances (internal models). Here it operates as a feedforward
corrector: its output to M1 modifies commands based on learned experience.

Unique property: Cerebellum uses a different learning rule internally
(perceptron-like, not STDP). Its corrections are delivered to M1 via the
standard CB→M1 SynapsePool, but the CB→M1 weights are trained on
motor error rather than pure STDP.

Outgoing: CB→CB (recurrent), CB→M1 (feedforward correction, strong)
"""

from __future__ import annotations

import numpy as np
from neuromorphic.regions.base_region import BrainRegion
from neuromorphic.core.neuron_group import NeuronGroup


class Cerebellum(BrainRegion):

    LABEL = "CB"

    def __init__(
        self,
        name: str,
        global_start: int,
        global_end: int,
        neuron_group: NeuronGroup,
        config,
    ):
        super().__init__(name, global_start, global_end, neuron_group, config)
        # Motor error trace (updated externally by Brain when actual ≠ predicted)
        self.motor_error: np.ndarray = np.zeros(self.n_neurons, dtype=np.float32)

    def step(
        self,
        t: float,
        dt: float,
        external_input: np.ndarray | None = None,
    ):
        # Inject motor error signal as extra driving current
        if self.motor_error.any():
            self.neurons.inject_current_slice(
                self.start, self.end, self.motor_error * 0.5
            )
        super().step(t, dt, external_input)

    def update_motor_error(self, error: np.ndarray):
        """
        Provide motor prediction error to CB (called by Brain after
        comparing actual vs. expected sensory feedback).
        """
        n = min(len(error), self.n_neurons)
        self.motor_error[:n] = error[:n].astype(np.float32)
        self.motor_error[n:] = 0.0

    def encode_input(self, stimulus: np.ndarray) -> np.ndarray:
        return np.zeros(self.n_neurons, dtype=np.float32)
