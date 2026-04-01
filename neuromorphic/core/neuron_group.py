"""
NeuronGroup — Vectorized Leaky Integrate-and-Fire neurons
==========================================================
All 1M neurons are stored as contiguous NumPy arrays. Regions are
identified by (start, end) slice indices into these arrays — no data
copies, no per-neuron Python objects.

LIF dynamics (Euler integration):
    dV/dt = (-(V - V_rest) + R * I_syn) / tau_mem
    V[t+dt] = V[t] + dt/tau_mem * (-(V[t]-V_rest) + R * I_syn[t])

Spike detection:
    spikes = V >= V_thresh  (and not in refractory)

Post-spike:
    V[spikes] = V_reset
    refrac_counter[spikes] = T_REFRAC / DT
    I_syn decays: I_syn *= exp(-dt / tau_syn)
"""

import numpy as np


class NeuronGroup:
    """
    Vectorized LIF neuron state for all neurons in the network.

    Parameters
    ----------
    n_neurons : int
        Total number of neurons (all regions combined).
    config : module
        neuromorphic.config module (or any object with the required attributes).
    """

    def __init__(self, n_neurons: int, config):
        self.n = n_neurons
        self.cfg = config

        # ---- State arrays (float32) ----
        self.v              = np.full(n_neurons, config.V_REST,   dtype=np.float32)
        self.v_thresh       = np.full(n_neurons, config.V_THRESH, dtype=np.float32)
        self.i_syn          = np.zeros(n_neurons, dtype=np.float32)

        # Spike history
        self.t_last_spike   = np.full(n_neurons, -1e6, dtype=np.float32)
        self.spikes         = np.zeros(n_neurons, dtype=np.bool_)

        # Refractory countdown in timesteps (int16 saves memory; max ~200 steps)
        self.refrac_counter = np.zeros(n_neurons, dtype=np.int16)

        # Per-neuron firing rate estimate (exponential moving average over time)
        self.firing_rate_est = np.zeros(n_neurons, dtype=np.float32)

        # Pre-computed decay factors (constant for the life of the simulation)
        self._decay_mem = np.float32(np.exp(-config.DT / config.TAU_MEM))
        self._decay_syn = np.float32(np.exp(-config.DT / config.TAU_SYN))
        self._dt_over_tau = np.float32(config.DT / config.TAU_MEM)
        self._r_mem       = np.float32(config.R_MEM)
        self._v_rest      = np.float32(config.V_REST)
        self._v_reset     = np.float32(config.V_RESET)
        self._refrac_steps = int(round(config.T_REFRAC / config.DT))
        self._rate_alpha   = np.float32(config.RATE_EMA_ALPHA)

        self.t = 0.0
        self.step_count = 0

    # ------------------------------------------------------------------
    # Current injection  (called by SynapsePool.propagate)
    # ------------------------------------------------------------------

    def inject_current(self, neuron_ids: np.ndarray, currents: np.ndarray):
        """
        Add synaptic current to i_syn at given global indices.
        Safe for duplicate indices (uses np.add.at).
        """
        np.add.at(self.i_syn, neuron_ids, currents)

    def inject_current_slice(self, start: int, end: int, currents: np.ndarray):
        """
        Add currents to a contiguous slice [start:end] — faster than inject_current
        for dense region-level injections (e.g. sensory input).
        """
        self.i_syn[start:end] += currents

    # ------------------------------------------------------------------
    # Main simulation step
    # ------------------------------------------------------------------

    def step(self, dt: float, t: float) -> np.ndarray:
        """
        Advance all neurons by one timestep.

        Returns
        -------
        spikes : np.ndarray, dtype=bool, shape (n,)
            True for every neuron that fired this step.
        """
        # 1. Zero out current injection for neurons in refractory
        in_refrac = self.refrac_counter > 0
        self.i_syn[in_refrac] = 0.0

        # 2. LIF membrane update (Euler)
        #    dv = dt/tau * (-(v - v_rest) + R * i_syn)
        self.v += self._dt_over_tau * (
            -(self.v - self._v_rest) + self._r_mem * self.i_syn
        )

        # 3. Detect spikes (threshold crossing, not in refractory)
        fired = (self.v >= self.v_thresh) & (~in_refrac)
        self.spikes[:] = fired

        # 4. Reset fired neurons
        if fired.any():
            self.v[fired]              = self._v_reset
            self.refrac_counter[fired] = self._refrac_steps
            self.t_last_spike[fired]   = np.float32(t)

        # 5. Decay synaptic current
        self.i_syn *= self._decay_syn

        # 6. Decrement refractory counters (floor at 0)
        np.subtract(self.refrac_counter, 1, out=self.refrac_counter,
                    where=in_refrac)

        # 7. Update firing rate EMA
        self.firing_rate_est += self._rate_alpha * (
            fired.astype(np.float32) / dt - self.firing_rate_est
        )

        self.t = t
        self.step_count += 1
        return fired

    # ------------------------------------------------------------------
    # Diagnostic helpers
    # ------------------------------------------------------------------

    def mean_rate(self, start: int, end: int) -> float:
        """Mean estimated firing rate (Hz) for a region slice."""
        return float(self.firing_rate_est[start:end].mean())

    def spike_count(self, start: int, end: int) -> int:
        """Number of neurons that fired last step in region slice."""
        return int(self.spikes[start:end].sum())

    def get_spikes_local(self, start: int, end: int) -> np.ndarray:
        """Boolean spike sub-array for a region (view, not copy)."""
        return self.spikes[start:end]
