"""
Motor constraints — frozen dataclasses for safety bounds
=========================================================
All constraint objects are frozen dataclasses (immutable after creation).
This is a compile-time guarantee that learning cannot modify safety bounds.

Separation of concerns:
  - MotorCommand : decoded output from M1 spike trains
  - RobotState   : current sensor readings from the physical system
  - MotorConstraints : safety envelope (joint limits, velocity, force, collision)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence
import numpy as np


@dataclass
class MotorCommand:
    """
    Decoded motor command from the Motor Cortex.

    Attributes
    ----------
    joint_angles : np.ndarray, shape (n_dof,), radians
    velocities   : np.ndarray, shape (n_dof,), rad/s
    forces       : np.ndarray, shape (n_dof,), N·m  (optional, zero if unknown)
    """
    joint_angles: np.ndarray
    velocities:   np.ndarray
    forces:       np.ndarray = field(default=None)

    def __post_init__(self):
        self.joint_angles = np.asarray(self.joint_angles, dtype=np.float32)
        self.velocities   = np.asarray(self.velocities,   dtype=np.float32)
        if self.forces is None:
            self.forces = np.zeros_like(self.joint_angles)
        else:
            self.forces = np.asarray(self.forces, dtype=np.float32)


@dataclass
class RobotState:
    """
    Current physical state of the robot / actuated system.

    Attributes
    ----------
    joint_angles : np.ndarray, shape (n_dof,), current joint angles
    velocities   : np.ndarray, shape (n_dof,), current velocities
    position_3d  : np.ndarray, shape (3,), end-effector position [x, y, z]
    """
    joint_angles: np.ndarray
    velocities:   np.ndarray
    position_3d:  np.ndarray = field(default_factory=lambda: np.zeros(3))

    def __post_init__(self):
        self.joint_angles = np.asarray(self.joint_angles, dtype=np.float32)
        self.velocities   = np.asarray(self.velocities,   dtype=np.float32)
        self.position_3d  = np.asarray(self.position_3d,  dtype=np.float32)

    @classmethod
    def zero(cls, n_dof: int) -> "RobotState":
        return cls(
            joint_angles=np.zeros(n_dof, dtype=np.float32),
            velocities=np.zeros(n_dof, dtype=np.float32),
            position_3d=np.zeros(3, dtype=np.float32),
        )


@dataclass(frozen=True)
class MotorConstraints:
    """
    Safety envelope for motor outputs.  FROZEN — cannot be changed at runtime.

    Attributes
    ----------
    joint_angle_min : np.ndarray  per-DOF minimum joint angle (rad)
    joint_angle_max : np.ndarray  per-DOF maximum joint angle (rad)
    max_velocity    : np.ndarray  per-DOF maximum angular velocity (rad/s)
    max_force       : np.ndarray  per-DOF maximum torque/force (N·m)
    collision_zones : tuple of (center_xyz, radius) pairs
    n_dof           : int
    """
    joint_angle_min: np.ndarray
    joint_angle_max: np.ndarray
    max_velocity:    np.ndarray
    max_force:       np.ndarray
    collision_zones: tuple = field(default_factory=tuple)
    n_dof:           int   = field(default=6)

    @classmethod
    def default(cls, n_dof: int = 6) -> "MotorConstraints":
        """Sensible defaults for a 6-DOF robot arm."""
        import neuromorphic.config as cfg
        return cls(
            joint_angle_min = np.full(n_dof, cfg.JOINT_ANGLE_MIN, dtype=np.float32),
            joint_angle_max = np.full(n_dof, cfg.JOINT_ANGLE_MAX, dtype=np.float32),
            max_velocity    = np.full(n_dof, cfg.MAX_JOINT_VELOCITY, dtype=np.float32),
            max_force       = np.full(n_dof, cfg.MAX_JOINT_FORCE,    dtype=np.float32),
            collision_zones = tuple(cfg.COLLISION_ZONES),
            n_dof           = n_dof,
        )

    def validate(self, command: MotorCommand, state: RobotState) -> list[str]:
        """
        Check all safety constraints.

        Returns
        -------
        violations : list[str]
            Empty list if all constraints are satisfied.
            Each entry is a human-readable violation description.
        """
        violations: list[str] = []

        # 1. Joint angle limits
        angle_target = state.joint_angles + command.joint_angles
        below = angle_target < self.joint_angle_min
        above = angle_target > self.joint_angle_max
        if below.any():
            dofs = np.where(below)[0].tolist()
            violations.append(f"joint_limit_min: DOFs {dofs}")
        if above.any():
            dofs = np.where(above)[0].tolist()
            violations.append(f"joint_limit_max: DOFs {dofs}")

        # 2. Velocity limits
        excess_vel = np.abs(command.velocities) > self.max_velocity
        if excess_vel.any():
            dofs = np.where(excess_vel)[0].tolist()
            violations.append(f"velocity_limit: DOFs {dofs}")

        # 3. Force limits
        excess_force = np.abs(command.forces) > self.max_force
        if excess_force.any():
            dofs = np.where(excess_force)[0].tolist()
            violations.append(f"force_limit: DOFs {dofs}")

        # 4. Collision zones (sphere check on 3D position)
        for (cx, cy, cz), radius in self.collision_zones:
            center = np.array([cx, cy, cz], dtype=np.float32)
            dist   = float(np.linalg.norm(state.position_3d - center))
            if dist < radius:
                violations.append(f"collision_zone: within {radius:.2f}m of {center}")

        return violations

    def violation_type(self, violations: list[str]) -> str:
        """
        Return the primary violation type for reflex lookup.
        Priority: collision > force > velocity > joint_limit.
        """
        for v in violations:
            if v.startswith("collision"):
                return "collision"
        for v in violations:
            if v.startswith("force"):
                return "force_limit"
        for v in violations:
            if v.startswith("velocity"):
                return "velocity_limit"
        return "joint_limit"
