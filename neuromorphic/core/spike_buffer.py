"""
SpikeBuffer — Circular delay buffer for axonal transmission delays
===================================================================
Implements a fixed-size circular buffer indexed by timestep modulo
max_delay_steps. Each slot holds a list of (pool_name, spike_array)
tuples scheduled for delivery at that future step.

Usage inside Brain.step():
    # Enqueue spikes with 2ms delay
    spike_buffer.push(delay_steps=2, pool_name="V1->IT", spikes=fired_v1)

    # At start of each step, deliver what's due
    for pool_name, spikes in spike_buffer.pop_current():
        pools[pool_name].propagate(spikes, neuron_group)

    spike_buffer.advance()
"""

from __future__ import annotations

import numpy as np


class SpikeBuffer:
    """
    Circular buffer for delayed spike delivery.

    Parameters
    ----------
    max_delay_ms : float
        Maximum axonal delay in milliseconds.
    dt : float
        Simulation timestep in seconds.
    """

    def __init__(self, max_delay_ms: float, dt: float):
        self.max_steps   = int(max_delay_ms / (dt * 1e3)) + 2  # +2 for safety
        self.current_step = 0
        # Each slot: list of (pool_name, spikes_copy)
        self._buf: list[list[tuple[str, np.ndarray]]] = [
            [] for _ in range(self.max_steps)
        ]

    def push(self, delay_steps: int, pool_name: str, spikes: np.ndarray):
        """
        Queue a spike array for delivery `delay_steps` steps in the future.
        Copies the spike array to prevent aliasing.
        """
        delay_steps = max(1, delay_steps)  # minimum 1-step delay
        target = (self.current_step + delay_steps) % self.max_steps
        self._buf[target].append((pool_name, spikes.copy()))

    def pop_current(self) -> list[tuple[str, np.ndarray]]:
        """
        Return and clear all events scheduled for the current timestep.
        """
        slot = self.current_step % self.max_steps
        events = self._buf[slot]
        self._buf[slot] = []
        return events

    def advance(self):
        """Increment the internal step counter."""
        self.current_step += 1

    def clear(self):
        """Flush all pending events (e.g. on reset)."""
        for slot in self._buf:
            slot.clear()
        self.current_step = 0
