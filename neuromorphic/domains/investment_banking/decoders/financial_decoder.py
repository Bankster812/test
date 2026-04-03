"""
FinancialDecoder — M1 spikes to financial parameters
=====================================================
Wraps MotorDecoder to decode M1 spike trains into named financial
parameters instead of robot joint angles/velocities.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from neuromorphic.io.decoder import MotorDecoder


@dataclass
class FinancialParams:
    """Named access to decoded financial parameters with per-parameter confidence."""
    raw_values:  np.ndarray       # shape (N_FINANCIAL_PARAMS,)
    confidence:  np.ndarray       # shape (N_FINANCIAL_PARAMS,)
    param_names: list[str]        # PARAM_NAMES list from config
    param_ranges: dict[str, tuple[float, float]]

    def get(self, name: str) -> tuple[float, float]:
        """Return (value, confidence) for a named parameter."""
        try:
            idx = self.param_names.index(name)
            return float(self.raw_values[idx]), float(self.confidence[idx])
        except (ValueError, IndexError):
            return 0.0, 0.0

    def value(self, name: str, default: float = 0.0) -> float:
        v, _ = self.get(name)
        return v if v != 0.0 else default

    def as_dict(self) -> dict[str, float]:
        return {
            name: float(self.raw_values[i])
            for i, name in enumerate(self.param_names)
            if name is not None
        }

    def high_confidence(self, threshold: float = 0.4) -> dict[str, float]:
        return {
            name: float(self.raw_values[i])
            for i, name in enumerate(self.param_names)
            if name is not None and float(self.confidence[i]) >= threshold
        }

    def overall_confidence(self) -> float:
        return float(self.confidence.mean())


class FinancialDecoder:
    """
    Decodes M1 spike trains into financial model parameters.

    Wraps MotorDecoder (composition). MotorDecoder is configured with
    n_dof = N_FINANCIAL_PARAMS so its readout matrix is (2*N, n_m1).
    First N outputs = parameter values; second N = confidence scores.

    Parameters
    ----------
    n_m1_neurons : int
    config       : ib_config module
    seed         : int
    """

    def __init__(self, n_m1_neurons: int, config, seed: int = 0):
        self.cfg         = config
        self.n_params    = config.N_FINANCIAL_PARAMS
        self.base_decoder = MotorDecoder(n_m1_neurons, config.N_FINANCIAL_PARAMS, seed)

    def decode(self, m1_spikes: np.ndarray) -> FinancialParams:
        """
        Decode a single timestep of M1 spikes into FinancialParams.
        """
        cmd = self.base_decoder.decode(m1_spikes, angle_scale=1.0, velocity_scale=1.0)
        return self._build_params(cmd.joint_angles, cmd.velocities)

    def decode_over_window(self, spike_history: list[np.ndarray]) -> FinancialParams:
        """
        Decode from a window of M1 spike snapshots.
        Averages spike rates over the window for stable output.
        Uses the last min(50, len(history)) snapshots.
        """
        if not spike_history:
            return self._zero_params()
        n_avg = min(50, len(spike_history))
        stack = np.stack(spike_history[-n_avg:], axis=0).astype(np.float32)
        mean_spikes = stack.mean(axis=0)
        return self.decode(mean_spikes)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_params(
        self,
        raw_angles: np.ndarray,
        raw_conf:   np.ndarray,
    ) -> FinancialParams:
        """
        Map tanh-scaled raw values through PARAM_RANGES to real-world values.
        """
        n = self.n_params
        values = np.zeros(n, dtype=np.float32)
        conf   = np.zeros(n, dtype=np.float32)

        for i, name in enumerate(self.cfg.PARAM_NAMES):
            if name is None:
                continue
            lo, hi = self.cfg.PARAM_RANGES.get(name, (0.0, 1.0))
            # raw_angles[i] is in [-1, 1] (tanh). Map to [lo, hi].
            norm      = (float(raw_angles[i]) + 1.0) / 2.0   # → [0, 1]
            values[i] = lo + norm * (hi - lo)
            # Confidence is also tanh-scaled; sigmoid it to [0, 1]
            conf[i]   = float(1.0 / (1.0 + np.exp(-raw_conf[i])))

        return FinancialParams(
            raw_values  = values,
            confidence  = conf,
            param_names = self.cfg.PARAM_NAMES,
            param_ranges = self.cfg.PARAM_RANGES,
        )

    def _zero_params(self) -> FinancialParams:
        n = self.n_params
        return FinancialParams(
            raw_values  = np.zeros(n, dtype=np.float32),
            confidence  = np.zeros(n, dtype=np.float32),
            param_names = self.cfg.PARAM_NAMES,
            param_ranges = self.cfg.PARAM_RANGES,
        )
