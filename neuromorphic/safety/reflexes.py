"""
ReflexLibrary — Pre-programmed withdrawal reflexes
====================================================
Reflexes are fixed lookup tables: violation type → motor command.
They are NOT learned by the neural simulation. They represent
hard-wired protective behaviors analogous to spinal cord reflexes
in biology — they bypass cortical processing entirely.

ReflexLibrary is immutable after construction and holds no reference
to the Brain or NeuronGroup.
"""

from __future__ import annotations

import numpy as np
from neuromorphic.safety.constraints import MotorCommand


# ---------------------------------------------------------------------------
# Reflex patterns (normalised joint angle deltas + velocities)
# ---------------------------------------------------------------------------
# All values are in [-1, 1] relative to the safety constraint range.
# SafetyKernel scales them to actual physical units when building MotorCommand.

_REFLEX_PATTERNS: dict[str, dict[str, list[float]]] = {
    # Joint limit hit: hold still, zero velocity, zero force
    "joint_limit": {
        "joint_angles": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "velocities":   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "forces":       [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    },
    # Velocity too high: brake all joints (negative velocity to decelerate)
    "velocity_limit": {
        "joint_angles": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "velocities":   [-0.5, -0.5, -0.5, -0.5, -0.5, -0.5],
        "forces":       [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    },
    # Force too high: immediate zero torque (gravity compensation only)
    "force_limit": {
        "joint_angles": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "velocities":   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "forces":       [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    },
    # Collision detected: retract all joints toward neutral position
    "collision": {
        "joint_angles": [-0.2, -0.2, -0.2, -0.2, -0.2, -0.2],
        "velocities":   [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
        "forces":       [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    },
}


class ReflexLibrary:
    """
    Provides pre-programmed reflex motor commands for each violation type.

    Parameters
    ----------
    n_dof : int
        Degrees of freedom (default 6). Patterns are tiled/truncated to fit.
    """

    def __init__(self, n_dof: int = 6):
        self.n_dof = n_dof
        # Build MotorCommand objects for each reflex, scaled to n_dof
        self._reflexes: dict[str, MotorCommand] = {}
        for vtype, pattern in _REFLEX_PATTERNS.items():
            self._reflexes[vtype] = self._build(pattern, n_dof)

    @staticmethod
    def _build(pattern: dict, n_dof: int) -> MotorCommand:
        def resize(lst: list[float], n: int) -> np.ndarray:
            arr = np.array(lst, dtype=np.float32)
            if len(arr) >= n:
                return arr[:n]
            return np.tile(arr, n // len(arr) + 1)[:n]

        return MotorCommand(
            joint_angles = resize(pattern["joint_angles"], n_dof),
            velocities   = resize(pattern["velocities"],   n_dof),
            forces       = resize(pattern["forces"],        n_dof),
        )

    def get_reflex(self, violation_type: str) -> MotorCommand:
        """
        Retrieve the reflex motor command for a given violation type.

        Falls back to "joint_limit" (hold still) if type not recognised.
        Returns a copy so the library's data cannot be mutated by callers.
        """
        pattern = self._reflexes.get(violation_type,
                                     self._reflexes["joint_limit"])
        return MotorCommand(
            joint_angles = pattern.joint_angles.copy(),
            velocities   = pattern.velocities.copy(),
            forces       = pattern.forces.copy(),
        )

    def known_types(self) -> list[str]:
        return list(self._reflexes.keys())
