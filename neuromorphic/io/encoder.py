"""
SensoryEncoder — Convert raw sensory data to spike trains
==========================================================
Implements three biologically-inspired encoding schemes:

Rate coding (Poisson)
    Each stimulus feature is mapped to a firing probability per timestep.
    p_spike = rate * dt  where  rate = stimulus_value * max_rate_hz
    Suitable for slowly-varying stimuli (luminance, volume).

Temporal coding (first-spike latency)
    Stronger stimuli cause earlier spikes within a coding window.
    More information-efficient than rate coding; used in V1, A1.

Population coding (Gaussian tuning curves)
    Neurons are assigned preferred values along the stimulus dimension.
    Each neuron fires probabilistically based on how close the stimulus
    is to its preferred value.
    Nearest biological analogue: orientation columns in V1, tonotopy in A1.

All encoders return a bool numpy array of shape (n_neurons,).
"""

from __future__ import annotations

import numpy as np


class SensoryEncoder:
    """
    Stateless sensory encoder. All methods are pure functions of inputs.

    Parameters
    ----------
    rng : np.random.Generator, optional
        Shared RNG for reproducibility.
    """

    def __init__(self, rng: np.random.Generator | None = None):
        self.rng = rng if rng is not None else np.random.default_rng()

    # ------------------------------------------------------------------
    # Rate coding (Poisson spike trains)
    # ------------------------------------------------------------------

    def rate_encode(
        self,
        stimulus: np.ndarray,
        n_neurons: int,
        dt: float,
        max_rate_hz: float = 100.0,
    ) -> np.ndarray:
        """
        Poisson rate coding.

        Parameters
        ----------
        stimulus   : np.ndarray, shape (n_features,), values in [0, 1]
        n_neurons  : int  — target population size
        dt         : float — simulation timestep (seconds)
        max_rate_hz: float — firing rate at stimulus=1.0

        Returns
        -------
        spikes : np.ndarray, bool, shape (n_neurons,)
        """
        stimulus = np.asarray(stimulus, dtype=np.float32).ravel()
        n_feat   = len(stimulus)

        # Tile or trim stimulus to population size (each feature maps to
        # a block of n_neurons // n_feat neurons)
        tiles  = (n_neurons + n_feat - 1) // n_feat
        rates  = np.tile(stimulus, tiles)[:n_neurons]  # (n_neurons,)

        spike_probs = rates * (max_rate_hz * dt)
        spike_probs = np.clip(spike_probs, 0.0, 1.0)
        return self.rng.random(n_neurons) < spike_probs

    # ------------------------------------------------------------------
    # Temporal coding (first-spike latency within a window)
    # ------------------------------------------------------------------

    def temporal_encode(
        self,
        stimulus: np.ndarray,
        n_neurons: int,
        t: float,
        window_start: float,
        window_ms: float = 20.0,
        dt: float = 1e-3,
    ) -> np.ndarray:
        """
        First-spike latency coding.

        Stronger stimulus → earlier spike within the coding window.
        Each neuron fires at most once per window.

        Returns bool array — True if this neuron fires at time t.
        """
        stimulus = np.asarray(stimulus, dtype=np.float32).ravel()
        n_feat   = len(stimulus)
        tiles    = (n_neurons + n_feat - 1) // n_feat
        s        = np.tile(stimulus, tiles)[:n_neurons]
        s        = np.clip(s, 1e-6, 1.0)

        window_s = window_ms * 1e-3
        # Spike time within window: t_spike = window_start + window*(1 - s)
        spike_time = window_start + window_s * (1.0 - s)
        # Fire if current time t falls within [spike_time, spike_time + dt)
        return (t >= spike_time) & (t < spike_time + dt)

    # ------------------------------------------------------------------
    # Population coding (Gaussian tuning curves)
    # ------------------------------------------------------------------

    def population_encode(
        self,
        stimulus: np.ndarray,
        n_neurons: int,
        preferred_values: np.ndarray | None = None,
        sigma: float = 0.15,
        dt: float = 1e-3,
        max_rate_hz: float = 100.0,
    ) -> np.ndarray:
        """
        Gaussian tuning curve population code.

        Each neuron has a preferred stimulus value; it fires with probability
        proportional to exp(-0.5 * ((s - pref) / sigma)²).

        Parameters
        ----------
        stimulus         : np.ndarray shape (n_features,), values in [0, 1]
        n_neurons        : int
        preferred_values : np.ndarray shape (n_neurons,) or None
                           (if None, evenly spaced over [0, 1])
        sigma            : float — tuning width
        dt               : float — timestep
        max_rate_hz      : float

        Returns
        -------
        spikes : bool array, shape (n_neurons,)
        """
        stimulus = np.asarray(stimulus, dtype=np.float32).ravel()

        if preferred_values is None:
            preferred_values = np.linspace(0.0, 1.0, n_neurons, dtype=np.float32)
        else:
            preferred_values = np.asarray(preferred_values, dtype=np.float32)

        # Average stimulus across features (or use first feature if 1D)
        s_mean = float(stimulus.mean())

        # Gaussian response
        response  = np.exp(-0.5 * ((preferred_values - s_mean) / sigma) ** 2)
        rates     = response * max_rate_hz
        spike_probs = rates * dt
        spike_probs = np.clip(spike_probs, 0.0, 1.0)
        return self.rng.random(n_neurons) < spike_probs.astype(np.float32)

    # ------------------------------------------------------------------
    # Multi-modal convenience wrappers
    # ------------------------------------------------------------------

    def encode_visual(
        self,
        frame: np.ndarray,
        n_neurons: int,
        dt: float,
    ) -> np.ndarray:
        """
        Encode a 2D visual frame (H×W or H×W×C) as a V1 spike train.
        Normalises to [0, 1], applies population coding.
        """
        flat = frame.ravel().astype(np.float32)
        flat /= (flat.max() + 1e-8)
        return self.rate_encode(flat, n_neurons, dt, max_rate_hz=80.0)

    def encode_auditory(
        self,
        spectrum: np.ndarray,
        n_neurons: int,
        dt: float,
    ) -> np.ndarray:
        """
        Encode a spectral frame (e.g. mel spectrogram band) as A1 spike train.
        Uses population coding over frequency bands (tonotopy).
        """
        flat = spectrum.ravel().astype(np.float32)
        flat /= (flat.max() + 1e-8)
        return self.population_encode(flat, n_neurons, dt=dt, sigma=0.1)

    def encode_touch(
        self,
        touch_data: np.ndarray,
        n_neurons: int,
        dt: float,
    ) -> np.ndarray:
        """
        Encode touch / proprioception data as S1 spike train.
        Uses rate coding (contact pressure is a scalar per sensor).
        """
        flat = touch_data.ravel().astype(np.float32)
        flat /= (flat.max() + 1e-8)
        return self.rate_encode(flat, n_neurons, dt, max_rate_hz=120.0)
