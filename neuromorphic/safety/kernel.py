"""
SafetyKernel — Hardware-enforced motor output supervisor
=========================================================
DESIGN INVARIANTS (never break these):

  1. SafetyKernel holds NO reference to Brain, NeuronGroup, or any
     mutable simulation state. It cannot be influenced by learning.

  2. SafetyKernel is constructed BEFORE Brain. Brain receives a
     reference to the kernel but cannot modify it.

  3. ALL motor commands pass through check_and_gate() before execution.
     There is no code path from M1 spikes to physical actuation that
     bypasses this method.

  4. MotorConstraints is a frozen dataclass. SafetyKernel stores it
     as a private attribute with no public setter.

Biological analogy:
  This corresponds to the spinal cord's stretch reflex arc — a fast,
  hard-wired feedback loop that can override descending cortical commands
  when a dangerous state is detected, before the signal ever reaches muscle.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from neuromorphic.safety.constraints import (
    MotorCommand, MotorConstraints, RobotState
)
from neuromorphic.safety.reflexes import ReflexLibrary


@dataclass
class ViolationRecord:
    """Log entry for a safety violation."""
    timestamp:      float          # wall-clock time
    sim_time:       float          # simulated time (s)
    violation_types: list[str]
    raw_command:    MotorCommand
    reflex_command: MotorCommand


class SafetyKernel:
    """
    Validates motor commands and returns safe alternatives on violation.

    Parameters
    ----------
    constraints : MotorConstraints
        Frozen safety envelope.
    reflexes : ReflexLibrary
        Pre-programmed withdrawal responses.
    """

    def __init__(self, constraints: MotorConstraints, reflexes: ReflexLibrary):
        # Private — no public setter
        self.__constraints = constraints
        self.__reflexes    = reflexes
        self.__violation_log: list[ViolationRecord] = []
        self.__total_commands  = 0
        self.__total_violations = 0

    # ------------------------------------------------------------------
    # Main interface — called by Brain every timestep
    # ------------------------------------------------------------------

    def check_and_gate(
        self,
        motor_command: MotorCommand,
        state: RobotState,
        sim_time: float = 0.0,
    ) -> tuple[MotorCommand, bool]:
        """
        Validate a decoded motor command against all safety constraints.

        Parameters
        ----------
        motor_command : MotorCommand
            Command decoded from M1 spike trains.
        state : RobotState
            Current physical state of the robot.
        sim_time : float
            Current simulation time (for logging).

        Returns
        -------
        (command, is_safe) : tuple
            is_safe=True  → command is the validated original command.
            is_safe=False → command is a reflex override (withdrawal).
        """
        self.__total_commands += 1

        violations = self.__constraints.validate(motor_command, state)

        if not violations:
            return motor_command, True

        # Violation detected
        self.__total_violations += 1
        primary_type   = self.__constraints.violation_type(violations)
        reflex_command = self.__reflexes.get_reflex(primary_type)

        self.__violation_log.append(ViolationRecord(
            timestamp        = time.time(),
            sim_time         = sim_time,
            violation_types  = violations,
            raw_command      = motor_command,
            reflex_command   = reflex_command,
        ))

        return reflex_command, False

    # ------------------------------------------------------------------
    # Pure predicate — no side effects, useful for testing
    # ------------------------------------------------------------------

    def is_safe(self, command: MotorCommand, state: RobotState) -> bool:
        """Return True iff command passes all safety constraints."""
        return len(self.__constraints.validate(command, state)) == 0

    # ------------------------------------------------------------------
    # Diagnostics (read-only access to internals)
    # ------------------------------------------------------------------

    @property
    def violation_history(self) -> list[ViolationRecord]:
        """Defensive copy of the violation log."""
        return list(self.__violation_log)

    @property
    def n_violations(self) -> int:
        return self.__total_violations

    @property
    def n_commands(self) -> int:
        return self.__total_commands

    @property
    def violation_rate(self) -> float:
        if self.__total_commands == 0:
            return 0.0
        return self.__total_violations / self.__total_commands

    def summary(self) -> dict:
        return {
            "total_commands":   self.__total_commands,
            "total_violations": self.__total_violations,
            "violation_rate":   self.violation_rate,
            "last_violation":   (
                self.__violation_log[-1].violation_types
                if self.__violation_log else None
            ),
        }
