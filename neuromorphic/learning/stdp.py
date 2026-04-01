"""
STDPRule — Spike-Timing Dependent Plasticity
=============================================
Three-factor reward-modulated STDP:

    ΔW_ij = η · DA(t) · [A+ · x_pre_i · δ(post_j fires)
                        - A- · x_post_j · δ(pre_i fires)]

where:
  x_pre_i  : pre-synaptic eligibility trace (decays with τ+)
  x_post_j : post-synaptic eligibility trace (decays with τ-)
  DA(t)    : dopamine concentration at time t (from Basal Ganglia)

Biologically:
  - Pre fires before post → synapse strengthens (LTP)
  - Post fires before pre → synapse weakens (LTD)
  - Dopamine gates whether the change actually sticks (reward modulation)

This class is stateless — all trace state lives in SynapsePool.
"""

from __future__ import annotations

import numpy as np
from neuromorphic.core.synapse_pool import SynapsePool


class STDPRule:
    """
    Applies STDP updates to a SynapsePool.

    Parameters
    ----------
    config : module
        neuromorphic.config or equivalent.
    """

    def __init__(self, config):
        self.cfg     = config
        self.A_plus  = config.A_PLUS
        self.A_minus = config.A_MINUS

    def apply(
        self,
        pool: SynapsePool,
        pre_spikes_local: np.ndarray,
        post_spikes_local: np.ndarray,
        dopamine: float,
        dt: float,
    ):
        """
        Apply one STDP step to a SynapsePool.

        Delegates to pool.update_stdp() which implements the three-tier
        lazy evaluation for CPU efficiency.

        Parameters
        ----------
        pool : SynapsePool
        pre_spikes_local : bool array (n_pre,)
        post_spikes_local : bool array (n_post,)
        dopamine : float  — three-factor gate, typically in [0, 1]
        dt : float        — simulation timestep in seconds
        """
        pool.update_stdp(pre_spikes_local, post_spikes_local, dopamine, dt)

    def apply_all(
        self,
        pools: dict[str, SynapsePool],
        spikes_by_region: dict[str, np.ndarray],
        dopamine: float,
        dt: float,
    ):
        """
        Apply STDP to every pool in the network in one call.

        Parameters
        ----------
        pools : dict  — {pool_name: SynapsePool}, keyed as "PRE->POST"
        spikes_by_region : dict  — {region_name: bool array (n_local,)}
        dopamine : float
        dt : float
        """
        for pool_name, pool in pools.items():
            pre_name, post_name = pool_name.split("->")
            pre_spikes  = spikes_by_region.get(pre_name)
            post_spikes = spikes_by_region.get(post_name)
            if pre_spikes is None or post_spikes is None:
                continue
            self.apply(pool, pre_spikes, post_spikes, dopamine, dt)
