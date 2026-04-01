"""
Brain — Main Neuromorphic Platform Orchestrator
================================================
Owns and coordinates:
  - Global NeuronGroup (1M neurons)
  - 11 BrainRegions
  - All 31 SynapsePools (1.2B synapses at full scale)
  - Learning: STDPRule, HomeostaticScaling, NeuromodulationSystem
  - I/O: SensoryEncoder, MotorDecoder, SpikeBuffer
  - Safety: SafetyKernel reference (not owned — constructed externally)

Per-timestep sequence
---------------------
1.  Encode sensory inputs → inject currents into V1, A1, S1
2.  Deliver queued spike events from SpikeBuffer
3.  NeuronGroup.step() → LIF integration → spikes for all 1M neurons
4.  Propagate spikes through all 31 SynapsePools → inject post currents
5.  Collect neuromodulatory signals (BG dopamine, BS arousal, AMY threat)
6.  NeuromodulationSystem.update()
7.  STDPRule.apply_all() — gated by dopamine
8.  HomeostaticScaling.step() — every 1000 steps
9.  Extract M1 spikes → SafetyKernel.check_and_gate() → motor command
10. Advance SpikeBuffer, increment time
"""

from __future__ import annotations

import types
from typing import TYPE_CHECKING

import numpy as np

import neuromorphic.config as _default_cfg
from neuromorphic.core.neuron_group import NeuronGroup
from neuromorphic.core.synapse_pool import SynapsePool
from neuromorphic.core.spike_buffer import SpikeBuffer
from neuromorphic.learning.stdp import STDPRule
from neuromorphic.learning.homeostasis import HomeostaticScaling
from neuromorphic.learning.neuromodulation import NeuromodulationSystem
from neuromorphic.io.encoder import SensoryEncoder
from neuromorphic.io.decoder import MotorDecoder
from neuromorphic.safety.kernel import SafetyKernel
from neuromorphic.safety.constraints import MotorCommand, RobotState
from neuromorphic.regions.visual_cortex import VisualCortex
from neuromorphic.regions.auditory_cortex import AuditoryCortex
from neuromorphic.regions.somatosensory_cortex import SomatosensoryCortex
from neuromorphic.regions.association_cortex import AssociationCortex
from neuromorphic.regions.prefrontal_cortex import PrefrontalCortex
from neuromorphic.regions.motor_cortex import MotorCortex
from neuromorphic.regions.cerebellum import Cerebellum
from neuromorphic.regions.hippocampus import Hippocampus
from neuromorphic.regions.amygdala import Amygdala
from neuromorphic.regions.basal_ganglia import BasalGanglia
from neuromorphic.regions.brainstem import Brainstem


# Map region names to their classes
_REGION_CLASSES = {
    "V1":  VisualCortex,
    "A1":  AuditoryCortex,
    "S1":  SomatosensoryCortex,
    "IT":  AssociationCortex,
    "PFC": PrefrontalCortex,
    "M1":  MotorCortex,
    "CB":  Cerebellum,
    "HPC": Hippocampus,
    "AMY": Amygdala,
    "BG":  BasalGanglia,
    "BS":  Brainstem,
}


class Brain:
    """
    Neuromorphic AI Platform.

    Parameters
    ----------
    config : module, optional
        neuromorphic.config or a module with the same attributes.
        Defaults to neuromorphic.config with whatever SCALE is set there.
    safety_kernel : SafetyKernel
        Must be constructed before Brain (invariant: safety layer is
        independent of simulation).
    seed : int
        Random seed for reproducibility.
    verbose : bool
        Print progress during initialisation.
    """

    def __init__(
        self,
        config=None,
        safety_kernel: SafetyKernel | None = None,
        seed: int = 42,
        verbose: bool = True,
    ):
        self.cfg     = config if config is not None else _default_cfg
        self.safety  = safety_kernel
        self.rng     = np.random.default_rng(seed)
        self.t       = 0.0
        self.step_count = 0

        region_sizes = self.cfg.get_region_sizes()
        id_ranges    = self.cfg.compute_id_ranges(region_sizes)
        total_n      = sum(region_sizes.values())

        if verbose:
            print(f"[Brain] Initialising with SCALE={self.cfg.SCALE}")
            print(f"[Brain] Total neurons: {total_n:,}")

        # ---- 1. Global NeuronGroup ----
        self.neurons = NeuronGroup(total_n, self.cfg)

        # ---- 2. Brain regions ----
        self.regions: dict[str, object] = {}
        for name, cls in _REGION_CLASSES.items():
            start, end = id_ranges[name]
            self.regions[name] = cls(name, start, end, self.neurons, self.cfg)

        # ---- 3. Synapse pools ----
        if verbose:
            print("[Brain] Building synaptic connectivity...")
        self.pools: dict[str, SynapsePool] = self._build_connectivity(
            region_sizes, id_ranges, verbose
        )

        # Register pools with their source regions
        for pool_name, pool in self.pools.items():
            pre_name = pool_name.split("->")[0]
            self.regions[pre_name].add_outgoing_pool(pool)

        # ---- 4. Learning components ----
        self.stdp        = STDPRule(self.cfg)
        self.homeostasis = HomeostaticScaling(self.neurons, self.cfg)
        self.neuromod    = NeuromodulationSystem(self.cfg)

        # ---- 5. Spike delay buffer ----
        self.spike_buffer = SpikeBuffer(
            max_delay_ms=self.cfg.MAX_DELAY_MS,
            dt=self.cfg.DT,
        )

        # ---- 6. I/O ----
        self.encoder = SensoryEncoder(rng=self.rng)
        self.decoder = MotorDecoder(
            n_m1_neurons=region_sizes["M1"],
            n_dof=self.cfg.N_DOF,
            seed=seed,
        )

        if verbose:
            total_syn = sum(p.n_synapses for p in self.pools.values())
            print(f"[Brain] Total synapses: {total_syn:,}")
            print("[Brain] Ready.")

    # ------------------------------------------------------------------
    # Connectivity construction
    # ------------------------------------------------------------------

    def _build_connectivity(
        self,
        region_sizes: dict[str, int],
        id_ranges: dict[str, tuple[int, int]],
        verbose: bool,
    ) -> dict[str, SynapsePool]:
        pools: dict[str, SynapsePool] = {}
        inhibitory = self.cfg.INHIBITORY_REGIONS

        for (pre_name, post_name), density in self.cfg.CONNECTIVITY.items():
            pool_name = f"{pre_name}->{post_name}"

            pre_start, pre_end   = id_ranges[pre_name]
            post_start, post_end = id_ranges[post_name]

            pre_ids  = np.arange(pre_start,  pre_end,  dtype=np.int32)
            post_ids = np.arange(post_start, post_end, dtype=np.int32)

            pool = SynapsePool(
                name=pool_name,
                pre_ids=pre_ids,
                post_ids=post_ids,
                density=density,
                config=self.cfg,
                rng=self.rng,
                is_inhibitory=(pre_name in inhibitory),
            )
            pools[pool_name] = pool

        if verbose:
            synapse_count = sum(p.n_synapses for p in pools.values())
            print(f"[Brain]   {len(pools)} pools, {synapse_count:,} synapses")

        return pools

    # ------------------------------------------------------------------
    # Main simulation step
    # ------------------------------------------------------------------

    def step(
        self,
        visual:    np.ndarray | None = None,
        auditory:  np.ndarray | None = None,
        soma:      np.ndarray | None = None,
        reward:    float = 0.0,
        robot_state: RobotState | None = None,
    ) -> tuple[MotorCommand | None, bool]:
        """
        Advance the simulation by one timestep (DT seconds).

        Parameters
        ----------
        visual    : np.ndarray — visual input (any shape; will be flattened)
        auditory  : np.ndarray — auditory/spectral input
        soma      : np.ndarray — touch / proprioception input
        reward    : float      — external reward signal [0, 1]
        robot_state : RobotState — current physical state for safety check

        Returns
        -------
        (motor_command, is_safe) : tuple
            motor_command is None if there is no Safety kernel configured.
        """
        t  = self.t
        dt = self.cfg.DT

        # -- Step 1: Encode sensory input → inject currents --
        self._inject_sensory(visual, auditory, soma, dt)

        # -- Step 2: Deliver delayed spikes from buffer --
        for pool_name, queued_spikes in self.spike_buffer.pop_current():
            if pool_name in self.pools:
                self.pools[pool_name].propagate(queued_spikes, self.neurons)

        # -- Step 3: Region-specific pre-spike logic --
        # BG gets reward signal before its step
        bg: BasalGanglia = self.regions["BG"]
        bg.inject_reward(reward)

        # All regions run their step (injects tonic currents, etc.)
        for region in self.regions.values():
            region.step(t, dt)

        # -- Step 4: LIF integration → spikes --
        self.neurons.step(dt, t)

        # -- Step 5: Propagate spikes through all pools --
        spikes_by_region = self._collect_region_spikes()
        for pool_name, pool in self.pools.items():
            pre_name = pool_name.split("->")[0]
            pre_spikes = spikes_by_region[pre_name]
            # Queue with 1-step delay (can be extended to model axonal delays)
            self.spike_buffer.push(1, pool_name, pre_spikes)

        # Also propagate immediately (delay=0 approximation for dense local pools)
        for pool_name, pool in self.pools.items():
            pre_name = pool_name.split("->")[0]
            pool.propagate(spikes_by_region[pre_name], self.neurons)

        # -- Step 6: Collect neuromodulatory signals --
        bs:  Brainstem    = self.regions["BS"]
        amy: Amygdala     = self.regions["AMY"]

        self.neuromod.update(
            bg_dopamine = bg.dopamine,
            bs_arousal  = bs.arousal,
            amy_threat  = amy.threat_signal,
            dt          = dt,
        )

        # -- Step 7: STDP weight updates --
        self.stdp.apply_all(
            pools=self.pools,
            spikes_by_region=spikes_by_region,
            dopamine=self.neuromod.learning_gate,
            dt=dt,
        )

        # -- Step 8: Homeostatic scaling --
        self.homeostasis.step(self.step_count)

        # -- Step 9: Motor output → safety check --
        motor_result = self._get_motor_output(robot_state, t)

        # -- Step 10: Advance time --
        self.spike_buffer.advance()
        self.t += dt
        self.step_count += 1

        return motor_result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _inject_sensory(
        self,
        visual:   np.ndarray | None,
        auditory: np.ndarray | None,
        soma:     np.ndarray | None,
        dt: float,
    ):
        """Encode and inject sensory inputs into V1, A1, S1."""
        v1: VisualCortex       = self.regions["V1"]
        a1: AuditoryCortex     = self.regions["A1"]
        s1: SomatosensoryCortex = self.regions["S1"]

        if visual is not None:
            spikes = self.encoder.encode_visual(visual, v1.n_neurons, dt)
            # Convert bool spikes to current (1.0 nA per spike)
            currents = spikes.astype(np.float32) * 1.5
            self.neurons.inject_current_slice(v1.start, v1.end, currents)

        if auditory is not None:
            spikes = self.encoder.encode_auditory(auditory, a1.n_neurons, dt)
            currents = spikes.astype(np.float32) * 1.5
            self.neurons.inject_current_slice(a1.start, a1.end, currents)

        if soma is not None:
            spikes = self.encoder.encode_touch(soma, s1.n_neurons, dt)
            currents = spikes.astype(np.float32) * 2.0
            self.neurons.inject_current_slice(s1.start, s1.end, currents)

    def _collect_region_spikes(self) -> dict[str, np.ndarray]:
        """Return {region_name: bool spike array (local)} for all regions."""
        result = {}
        for name, region in self.regions.items():
            result[name] = self.neurons.get_spikes_local(region.start, region.end)
        return result

    def _get_motor_output(
        self,
        robot_state: RobotState | None,
        sim_time: float,
    ) -> tuple[MotorCommand | None, bool]:
        """Decode M1 spikes and pass through safety kernel."""
        m1: MotorCortex = self.regions["M1"]
        motor_spikes    = m1.get_motor_spikes()
        raw_command     = self.decoder.decode(motor_spikes)

        if self.safety is None:
            return raw_command, True

        if robot_state is None:
            robot_state = RobotState.zero(self.cfg.N_DOF)

        return self.safety.check_and_gate(raw_command, robot_state, sim_time)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_diagnostics(self) -> dict:
        """
        Return a snapshot of key metrics.

        Keys
        ----
        firing_rates    : {region_name: Hz}
        neuromodulators : {DA, ACh, 5HT, learning_gate}
        safety          : {total_commands, total_violations, violation_rate}
        total_synapses  : int
        sim_time_ms     : float
        """
        firing_rates = {
            name: round(region.mean_rate, 2)
            for name, region in self.regions.items()
        }
        diag = {
            "firing_rates":    firing_rates,
            "neuromodulators": self.neuromod.as_dict(),
            "sim_time_ms":     round(self.t * 1000, 1),
            "step":            self.step_count,
            "total_synapses":  sum(p.n_synapses for p in self.pools.values()),
        }
        if self.safety is not None:
            diag["safety"] = self.safety.summary()
        return diag

    def region_spike_counts(self) -> dict[str, int]:
        """Number of neurons that fired last step in each region."""
        return {
            name: self.neurons.spike_count(r.start, r.end)
            for name, r in self.regions.items()
        }
