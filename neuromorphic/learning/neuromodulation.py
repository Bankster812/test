"""
NeuromodulationSystem — Global neuromodulator dynamics
=======================================================
Tracks three neuromodulators that gate plasticity and modulate
neural dynamics:

  DA  (Dopamine)       — reward prediction error from Basal Ganglia
                         Gates STDP weight updates (three-factor rule)
  ACh (Acetylcholine)  — arousal / attention from Brainstem
                         Scales effective learning rate
  5HT (Serotonin)      — valence / safety from Amygdala + Brainstem
                         Modulates risk-aversion and threshold

All levels are normalised to [0, 1].
"""

from __future__ import annotations

import numpy as np


class NeuromodulationSystem:
    """
    Integrates neuromodulator signals from BG, BS, and AMY.

    Parameters
    ----------
    config : module
    """

    def __init__(self, config):
        self.cfg = config
        self.da  = np.float32(config.DA_BASELINE)
        self.ach = np.float32(config.ACH_INIT)
        self.sht = np.float32(config.SHT_INIT)  # serotonin (5-HT)

        self._da_decay = np.float32(config.DA_DECAY)
        self._alpha    = np.float32(config.NM_ALPHA)

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(
        self,
        bg_dopamine: float,
        bs_arousal: float,
        amy_threat: float,
        dt: float,
    ):
        """
        Update all neuromodulator levels.

        Parameters
        ----------
        bg_dopamine : float  — dopamine signal from BG (RPE, [0,1])
        bs_arousal  : float  — arousal signal from Brainstem ([0,1])
        amy_threat  : float  — threat signal from Amygdala ([0,1])
        dt          : float  — timestep (unused here; decay is per-step)
        """
        # DA: direct from BG reward prediction error, exponential decay
        self.da = float(np.clip(
            self.da * self._da_decay + float(bg_dopamine),
            0.0, 1.0,
        ))

        # ACh: driven by BS arousal (slow exponential smoothing)
        self.ach = float(np.clip(
            (1.0 - self._alpha) * self.ach + self._alpha * float(bs_arousal),
            0.0, 1.0,
        ))

        # 5HT: inversely related to threat (high threat suppresses serotonin)
        target_sht = 1.0 - float(amy_threat)
        self.sht = float(np.clip(
            (1.0 - self._alpha) * self.sht + self._alpha * target_sht,
            0.0, 1.0,
        ))

    # ------------------------------------------------------------------
    # Derived signals
    # ------------------------------------------------------------------

    @property
    def learning_gate(self) -> float:
        """
        Combined modulation factor for STDP weight updates.
        High dopamine + high ACh = fast learning.
        """
        return float(np.clip(self.da * (1.0 + 0.3 * self.ach), 0.0, 2.0))

    @property
    def arousal(self) -> float:
        return self.ach

    @property
    def safety_tone(self) -> float:
        """
        5HT level — used to modulate risk-aversion in action selection.
        High serotonin = calm, low serotonin = agitated / risk-averse.
        """
        return self.sht

    def as_dict(self) -> dict[str, float]:
        return {"DA": self.da, "ACh": self.ach, "5HT": self.sht,
                "learning_gate": self.learning_gate}
