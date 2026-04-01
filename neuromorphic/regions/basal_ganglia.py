"""
BasalGanglia (BG) — Basal Ganglia
===================================
40,000 neurons at full scale.

Action selection and reward-based learning. The primary source of
dopamine in the network — specifically, the reward prediction error (RPE)
signal that gates all STDP weight updates (three-factor learning rule).

Dopamine computation (simplified temporal-difference RPE):
    prediction  ← EMA of BG population activity
    RPE         ← current_activity - prediction
    dopamine    ← clip(DA_decay * dopamine + RPE,  0, 1)

High BG activity after a reward → DA spike → synapses active during
that period get strengthened (reward-modulated STDP).

BG projections are inhibitory (as in biological direct/indirect pathways).

Outgoing: BG→BG (recurrent), BG→M1 (disinhibition of actions),
          BG→PFC (action gating)
"""

from __future__ import annotations

import numpy as np
from neuromorphic.regions.base_region import BrainRegion
from neuromorphic.core.neuron_group import NeuronGroup


class BasalGanglia(BrainRegion):

    LABEL = "BG"

    def __init__(
        self,
        name: str,
        global_start: int,
        global_end: int,
        neuron_group: NeuronGroup,
        config,
    ):
        super().__init__(name, global_start, global_end, neuron_group, config)
        self._dopamine         = np.float32(config.DA_BASELINE)
        self._reward_pred      = np.float32(0.0)
        self._da_decay         = np.float32(config.DA_DECAY)
        self._pred_lr          = np.float32(0.05)   # prediction update rate
        self._external_reward  = np.float32(0.0)

    def step(
        self,
        t: float,
        dt: float,
        external_input: np.ndarray | None = None,
    ):
        super().step(t, dt, external_input)

        # Compute reward prediction error from neural activity + external reward
        bg_activity = float(self.local_spikes.mean()) * 200.0  # scale to [0,1]
        combined    = float(np.clip(bg_activity + self._external_reward, 0.0, 1.0))

        rpe = combined - float(self._reward_pred)

        # Update slow prediction trace
        self._reward_pred = np.float32(
            float(self._reward_pred) + self._pred_lr * rpe
        )

        # Update dopamine level with decay + RPE injection
        self._dopamine = np.float32(np.clip(
            float(self._dopamine) * float(self._da_decay) + max(0.0, rpe),
            0.0, 1.0,
        ))

        # Reset external reward (consumed this step)
        self._external_reward = np.float32(0.0)

    def inject_reward(self, reward: float):
        """
        Provide an external reward signal (e.g. task success).
        Called by Brain.step() before the BG step.
        """
        self._external_reward = np.float32(float(reward))

    @property
    def dopamine(self) -> float:
        """Current dopamine concentration [0, 1]."""
        return float(self._dopamine)

    def encode_input(self, stimulus: np.ndarray) -> np.ndarray:
        return np.zeros(self.n_neurons, dtype=np.float32)
