"""
SynapsePool — Sparse connectivity with online STDP
====================================================
One SynapsePool represents a directed projection from one brain region
(or sub-population) to another. Weights are stored in scipy CSR format
for efficient spike propagation (sparse matrix-vector products).

Efficient STDP via three-tier laziness
---------------------------------------
Tier 1: Per-neuron trace arrays  x_pre[n_pre], x_post[n_post]
        are decayed with a single vectorized multiply each step — O(N).

Tier 2: Weight updates only happen when a neuron actually fires.
        At 5 Hz average across 1M neurons, only ~5K neurons fire/step.

Tier 3: For each fired pre-neuron i, the STDP potentiation update is
        a pure CSR row-slice operation (NumPy, no Python loop over synapses).
        For depression (post fires), the CSC view of the same matrix is used.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


class SynapsePool:
    """
    Directed projection between two neuron populations.

    Parameters
    ----------
    name : str
        Human-readable label, e.g. "V1->IT".
    pre_ids : np.ndarray[int32]
        Global neuron IDs of the pre-synaptic population.
    post_ids : np.ndarray[int32]
        Global neuron IDs of the post-synaptic population.
    density : float
        Connection probability in (0, 1).
    config : module
        neuromorphic.config or equivalent.
    rng : np.random.Generator
    is_inhibitory : bool
        If True, propagated currents are negated.
    """

    def __init__(
        self,
        name: str,
        pre_ids: np.ndarray,
        post_ids: np.ndarray,
        density: float,
        config,
        rng: np.random.Generator,
        is_inhibitory: bool = False,
    ):
        self.name          = name
        self.pre_ids       = pre_ids.astype(np.int32)
        self.post_ids      = post_ids.astype(np.int32)
        self.is_inhibitory = is_inhibitory
        self.cfg           = config

        n_pre  = len(pre_ids)
        n_post = len(post_ids)
        self.n_pre  = n_pre
        self.n_post = n_post

        # ---- Build sparse connectivity ----
        n_syn = max(1, int(n_pre * n_post * density))
        pre_local  = rng.integers(0, n_pre,  size=n_syn, dtype=np.int32)
        post_local = rng.integers(0, n_post, size=n_syn, dtype=np.int32)
        weights    = rng.uniform(0.0, config.W_INIT_SCALE, size=n_syn).astype(np.float32)

        # CSR: rows = pre (local), cols = post (local)
        self.W: sp.csr_matrix = sp.csr_matrix(
            (weights, (pre_local, post_local)),
            shape=(n_pre, n_post),
            dtype=np.float32,
        )
        self.W.sum_duplicates()   # merge duplicate (i,j) entries by summing
        self.W.eliminate_zeros()
        np.clip(self.W.data, config.W_MIN, config.W_MAX, out=self.W.data)

        # CSC view for efficient column-wise access during LTD STDP
        self._W_csc: sp.csc_matrix = self.W.tocsc()

        # ---- Per-neuron STDP traces ----
        self.x_pre  = np.zeros(n_pre,  dtype=np.float32)  # pre-synaptic trace
        self.x_post = np.zeros(n_post, dtype=np.float32)  # post-synaptic trace

        # Pre-computed decay factors
        self._pre_decay  = np.float32(np.exp(-config.DT / config.TAU_PLUS))
        self._post_decay = np.float32(np.exp(-config.DT / config.TAU_MINUS))

        self._A_plus  = np.float32(config.A_PLUS)
        self._A_minus = np.float32(config.A_MINUS)
        self._w_min   = np.float32(config.W_MIN)
        self._w_max   = np.float32(config.W_MAX)

    # ------------------------------------------------------------------
    # Spike propagation
    # ------------------------------------------------------------------

    def propagate(
        self,
        pre_spikes_local: np.ndarray,
        neuron_group,
    ) -> np.ndarray:
        """
        Propagate spikes from pre to post population.

        Fired pre-neurons index into CSR rows of W; the corresponding
        post-neuron currents are accumulated into neuron_group.i_syn
        via neuron_group.inject_current().

        Parameters
        ----------
        pre_spikes_local : np.ndarray, bool, shape (n_pre,)
        neuron_group : NeuronGroup

        Returns
        -------
        post_indices : np.ndarray, int32
            Global IDs of post-neurons that received input (for active_mask).
        """
        firing_local = np.where(pre_spikes_local)[0].astype(np.int32)
        if len(firing_local) == 0:
            return np.empty(0, dtype=np.int32)

        # Slice rows of W corresponding to fired pre-neurons
        W_active = self.W[firing_local, :]      # shape (n_fired, n_post), sparse
        # Sum incoming weights per post-neuron
        i_post_dense = np.asarray(W_active.sum(axis=0)).ravel()  # (n_post,)

        if self.is_inhibitory:
            i_post_dense = -i_post_dense

        # Find post-neurons that actually receive any current
        nonzero_post = np.where(i_post_dense != 0.0)[0].astype(np.int32)
        if len(nonzero_post) == 0:
            return np.empty(0, dtype=np.int32)

        global_post = self.post_ids[nonzero_post]
        neuron_group.inject_current(global_post, i_post_dense[nonzero_post])

        return global_post

    # ------------------------------------------------------------------
    # STDP weight update
    # ------------------------------------------------------------------

    def update_stdp(
        self,
        pre_spikes_local: np.ndarray,
        post_spikes_local: np.ndarray,
        dopamine: float,
        dt: float,
    ):
        """
        Online STDP: called every timestep.

        Tier 1  — decay all traces (vectorized, O(N)):
            x_pre  *= exp(-dt/tau_plus)
            x_post *= exp(-dt/tau_minus)

        Tier 2  — process only neurons that fired this step:
            pre spikes  → LTP: dw += A_plus  * x_post[connected posts]
            post spikes → LTD: dw -= A_minus * x_pre[connected pres]

        Tier 3  — weight update is a CSR/CSC slice, no Python loop.

        Three-factor gating: all dw are multiplied by (1 + dopamine).
        """
        gate = np.float32(1.0 + float(dopamine))

        # -- Tier 1: decay traces --
        self.x_pre  *= self._pre_decay
        self.x_post *= self._post_decay

        fired_pre  = np.where(pre_spikes_local)[0]
        fired_post = np.where(post_spikes_local)[0]

        # Guard against out-of-bounds indices (can occur on Windows/numpy 2.x
        # when the local spike array has a different dtype or stride interpretation)
        n_pre, n_post = self.W.shape
        if len(fired_pre) > 0:
            fired_pre = fired_pre[fired_pre < n_pre]
        if len(fired_post) > 0:
            fired_post = fired_post[fired_post < n_post]

        # -- Tier 2+3: LTP (pre fires → strengthen post-synaptic response) --
        if len(fired_pre) > 0:
            for i in fired_pre:
                # Indices into W.data for row i (CSR)
                row_start = self.W.indptr[i]
                row_end   = self.W.indptr[i + 1]
                if row_start == row_end:
                    continue
                post_cols = self.W.indices[row_start:row_end]
                dw = self._A_plus * self.x_post[post_cols] * gate
                self.W.data[row_start:row_end] += dw
            # Increment pre trace
            self.x_pre[fired_pre] += 1.0

        # -- Tier 2+3: LTD (post fires → weaken synapses from slow pre) --
        if len(fired_post) > 0:
            # Sync CSC data pointer to current W.data
            # (CSC was built from the original W; we update its data in-place
            #  keeping the same sparsity pattern — only data values change)
            self._sync_csc()
            for j in fired_post:
                col_start = self._W_csc.indptr[j]
                col_end   = self._W_csc.indptr[j + 1]
                if col_start == col_end:
                    continue
                pre_rows = self._W_csc.indices[col_start:col_end]
                dw = self._A_minus * self.x_pre[pre_rows] * gate
                self._W_csc.data[col_start:col_end] -= dw
            # Increment post trace
            self.x_post[fired_post] += 1.0

        # Clip weights (in-place on the underlying data array)
        np.clip(self.W.data, self._w_min, self._w_max, out=self.W.data)
        np.clip(self._W_csc.data, self._w_min, self._w_max, out=self._W_csc.data)

    def _sync_csc(self):
        """
        Keep CSC data array pointing to the same values as CSR.
        Since both share the same sparsity pattern, we can just copy data.
        Called only when LTD updates are needed (post fires).
        """
        # CSR and CSC have the same nnz but different orderings.
        # Rebuild CSC from current CSR to keep them consistent.
        self._W_csc = self.W.tocsc()

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def n_synapses(self) -> int:
        return int(self.W.nnz)

    def mean_weight(self) -> float:
        if self.W.nnz == 0:
            return 0.0
        return float(self.W.data.mean())
