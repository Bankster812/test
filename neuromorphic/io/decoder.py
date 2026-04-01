"""
MotorDecoder — Convert M1 spike trains to motor commands
=========================================================
Stateless by design: decode() is a pure function of (spikes, readout_weights).
This is required because SafetyKernel calls decode() internally — keeping
the decoder stateless prevents any shared mutable state between the safety
layer and the neural simulation.

Population vector decoding
    A linear readout matrix W_read (n_dof × n_m1) projects the M1 spike
    vector to a motor command. Biologically, this represents the motor
    population vector (Georgopoulos, 1986).

    command = W_read @ spike_rates

The readout weights can be calibrated offline (e.g. matched to a specific
robot kinematic model) but are NOT modified by STDP.
"""

from __future__ import annotations

import numpy as np
from neuromorphic.safety.constraints import MotorCommand


class MotorDecoder:
    """
    Linear population-vector decoder for M1 spike trains.

    Parameters
    ----------
    n_m1_neurons : int
    n_dof        : int  — degrees of freedom (joint angles + velocities)
    seed         : int  — for reproducible weight initialisation
    """

    def __init__(self, n_m1_neurons: int, n_dof: int, seed: int = 0):
        self.n_m1   = n_m1_neurons
        self.n_dof  = n_dof
        rng         = np.random.default_rng(seed)

        # Linear readout: (2*n_dof, n_m1) — first half = angles, second = velocities
        self.W_read = rng.standard_normal(
            (2 * n_dof, n_m1_neurons)
        ).astype(np.float32) * 0.01

    # ------------------------------------------------------------------
    # Decode
    # ------------------------------------------------------------------

    def decode(
        self,
        spikes: np.ndarray,
        angle_scale: float = 0.1,
        velocity_scale: float = 0.5,
    ) -> MotorCommand:
        """
        Project M1 spike vector → MotorCommand via linear readout.

        Parameters
        ----------
        spikes         : bool/float array, shape (n_m1,)
        angle_scale    : float — scales decoded angle deltas (rad)
        velocity_scale : float — scales decoded velocities (rad/s)

        Returns
        -------
        MotorCommand
        """
        activation = spikes.astype(np.float32)
        raw        = self.W_read @ activation          # shape (2*n_dof,)

        joint_angles = np.tanh(raw[:self.n_dof])  * angle_scale
        velocities   = np.tanh(raw[self.n_dof:])  * velocity_scale

        return MotorCommand(
            joint_angles = joint_angles.astype(np.float32),
            velocities   = velocities.astype(np.float32),
        )

    # ------------------------------------------------------------------
    # Calibration (optional, offline only)
    # ------------------------------------------------------------------

    def set_readout_weights(self, W: np.ndarray):
        """
        Override the readout weights (for offline calibration).
        Shape must be (2*n_dof, n_m1).
        Not called during simulation; never called by STDP.
        """
        assert W.shape == (2 * self.n_dof, self.n_m1), (
            f"Expected ({2*self.n_dof}, {self.n_m1}), got {W.shape}"
        )
        self.W_read = W.astype(np.float32)
