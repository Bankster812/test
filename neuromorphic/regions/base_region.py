"""
BrainRegion — Abstract base class for all 11 brain regions
===========================================================
Each region:
  - Holds a (start, end) slice into the global NeuronGroup — NO data copies.
  - Knows which SynapsePools it is the source of (outgoing projections).
  - Has a step() method called once per simulation timestep.
  - Optionally accepts external sensory input via encode_input().

Subclasses override encode_input() and may override step() for
region-specific behaviour (e.g. dopamine computation in BasalGanglia).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from neuromorphic.core.neuron_group import NeuronGroup
from neuromorphic.core.synapse_pool import SynapsePool


class BrainRegion(ABC):
    """
    Abstract base for all brain regions.

    Parameters
    ----------
    name          : str  — short identifier, e.g. "V1"
    global_start  : int  — first global neuron index (inclusive)
    global_end    : int  — last global neuron index (exclusive)
    neuron_group  : NeuronGroup  — shared reference, not a copy
    config        : module       — neuromorphic.config
    """

    def __init__(
        self,
        name: str,
        global_start: int,
        global_end: int,
        neuron_group: NeuronGroup,
        config,
    ):
        self.name        = name
        self.start       = global_start
        self.end         = global_end
        self.n_neurons   = global_end - global_start
        self.neurons     = neuron_group   # shared, no copy
        self.cfg         = config

        # Outgoing SynapsePools keyed by pool name ("V1->IT")
        self.pools: dict[str, SynapsePool] = {}

    # ------------------------------------------------------------------
    # Convenience views into the shared NeuronGroup
    # ------------------------------------------------------------------

    @property
    def local_spikes(self) -> np.ndarray:
        """Boolean spike sub-array for this region (view, not copy)."""
        return self.neurons.spikes[self.start:self.end]

    @property
    def local_ids(self) -> np.ndarray:
        """Global neuron IDs for this region as int32 array."""
        return np.arange(self.start, self.end, dtype=np.int32)

    @property
    def mean_rate(self) -> float:
        """Estimated mean firing rate (Hz) for this region."""
        return self.neurons.mean_rate(self.start, self.end)

    # ------------------------------------------------------------------
    # Pool management
    # ------------------------------------------------------------------

    def add_outgoing_pool(self, pool: SynapsePool):
        self.pools[pool.name] = pool

    # ------------------------------------------------------------------
    # Simulation step
    # ------------------------------------------------------------------

    def step(
        self,
        t: float,
        dt: float,
        external_input: np.ndarray | None = None,
    ):
        """
        Default step: inject external sensory input (if provided) then
        propagate local spikes through all outgoing pools.

        Parameters
        ----------
        t              : float — current simulation time (s)
        dt             : float — timestep (s)
        external_input : np.ndarray | None — shape (n_neurons,), dtype float32
                         Synaptic current to add before the step.
        """
        if external_input is not None:
            n = min(len(external_input), self.n_neurons)
            self.neurons.inject_current_slice(
                self.start, self.start + n, external_input[:n]
            )

    # ------------------------------------------------------------------
    # Sensory input encoding (override in sensory regions)
    # ------------------------------------------------------------------

    def encode_input(self, stimulus: np.ndarray) -> np.ndarray:
        """
        Convert raw stimulus data to a current-injection array.
        Default: return zeros (no sensory input to this region).

        Subclasses of sensory regions override this.
        """
        return np.zeros(self.n_neurons, dtype=np.float32)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def diagnostics(self) -> dict:
        return {
            "region":    self.name,
            "n_neurons": self.n_neurons,
            "mean_rate": round(self.mean_rate, 2),
            "n_spikes":  int(self.local_spikes.sum()),
            "n_pools":   len(self.pools),
        }
