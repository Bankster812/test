"""
Neuromorphic AI Platform — Configuration
=========================================
All constants live here. Set SCALE before importing anything else.

  SCALE = 0.01  → 10K neurons,  ~120K synapses  (laptop demo)
  SCALE = 0.1   → 100K neurons, ~12M synapses   (workstation)
  SCALE = 1.0   → 1M neurons,   ~1.2B synapses  (full target)

Synapse count scales as SCALE² because both pre and post populations shrink,
preserving biological density ratios.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Scale factor — override this before constructing Brain
# ---------------------------------------------------------------------------
SCALE: float = 0.01

# ---------------------------------------------------------------------------
# Timestep
# ---------------------------------------------------------------------------
DT: float = 1e-3          # 1 ms simulation timestep
N_DOF: int = 6             # degrees of freedom for motor output

# ---------------------------------------------------------------------------
# Region sizes at full scale (SCALE = 1.0)
# ---------------------------------------------------------------------------
_BASE_REGION_SIZES: dict[str, int] = {
    "V1":  150_000,   # Primary Visual Cortex
    "A1":   50_000,   # Primary Auditory Cortex
    "S1":  100_000,   # Somatosensory Cortex
    "IT":  100_000,   # Inferior Temporal / Association Cortex
    "PFC": 150_000,   # Prefrontal Cortex
    "M1":  100_000,   # Motor Cortex
    "CB":  100_000,   # Cerebellum
    "HPC":  80_000,   # Hippocampus
    "AMY":  30_000,   # Amygdala
    "BG":   40_000,   # Basal Ganglia
    "BS":  100_000,   # Brainstem
}
assert sum(_BASE_REGION_SIZES.values()) == 1_000_000, "Region sizes must sum to 1M"

REGION_ORDER: list[str] = list(_BASE_REGION_SIZES.keys())  # canonical order


def get_region_sizes() -> dict[str, int]:
    """Return region sizes scaled by SCALE, minimum 1 neuron each."""
    return {k: max(1, int(v * SCALE)) for k, v in _BASE_REGION_SIZES.items()}


def get_total_neurons() -> int:
    return sum(get_region_sizes().values())


def compute_id_ranges(region_sizes: dict[str, int]) -> dict[str, tuple[int, int]]:
    """
    Assign contiguous global neuron ID ranges in REGION_ORDER.
    Returns {region_name: (global_start, global_end)}.
    """
    ranges: dict[str, tuple[int, int]] = {}
    cursor = 0
    for name in REGION_ORDER:
        n = region_sizes[name]
        ranges[name] = (cursor, cursor + n)
        cursor += n
    return ranges


# ---------------------------------------------------------------------------
# LIF neuron parameters
# ---------------------------------------------------------------------------
V_REST: float    = -65e-3   # V — resting potential
V_RESET: float   = -65e-3   # V — post-spike reset
V_THRESH: float  = -50e-3   # V — spike threshold (before homeostasis)
V_THRESH_MIN: float = -55e-3
V_THRESH_MAX: float = -45e-3
TAU_MEM: float   = 20e-3    # s — membrane time constant
TAU_SYN: float   =  5e-3    # s — synaptic current decay
T_REFRAC: float  =  2e-3    # s — absolute refractory period
R_MEM: float     =  1.0     # MΩ (normalised) — membrane resistance

# ---------------------------------------------------------------------------
# STDP parameters
# ---------------------------------------------------------------------------
A_PLUS: float    = 0.01     # LTP amplitude
A_MINUS: float   = 0.0105   # LTD amplitude
TAU_PLUS: float  = 20e-3    # s — pre-synaptic trace decay
TAU_MINUS: float = 20e-3    # s — post-synaptic trace decay
W_MIN: float     = 0.0
W_MAX: float     = 1.0
W_INIT_SCALE: float = 0.05  # initial weight ~ Uniform(0, W_INIT_SCALE)

# ---------------------------------------------------------------------------
# Neuromodulation
# ---------------------------------------------------------------------------
DA_DECAY: float     = 0.995   # dopamine decay per step
DA_BASELINE: float  = 0.0
ACH_INIT: float     = 0.5     # acetylcholine initial level
SHT_INIT: float     = 0.5     # serotonin initial level
NM_ALPHA: float     = 0.1     # neuromodulator update rate

# ---------------------------------------------------------------------------
# Homeostasis
# ---------------------------------------------------------------------------
TARGET_RATE: float        = 5.0    # Hz — desired mean firing rate per neuron
ETA_HOMEOSTASIS: float    = 5e-5   # threshold adjustment step size
HOMEOSTASIS_INTERVAL: int = 1000   # steps between homeostatic updates
RATE_EMA_ALPHA: float     = 0.001  # EMA coefficient for firing rate estimator

# ---------------------------------------------------------------------------
# Spike delay buffer
# ---------------------------------------------------------------------------
MAX_DELAY_MS: float = 20.0   # ms — maximum axonal delay

# ---------------------------------------------------------------------------
# Connectivity densities  (pre_region, post_region) → density
# All connections listed here are instantiated as SynapsePools.
# Values chosen so that at SCALE=1.0 the total sums to ~1.2B synapses.
# ---------------------------------------------------------------------------
CONNECTIVITY: dict[tuple[str, str], float] = {
    # Recurrent within-region (stabilises activity)
    ("V1",  "V1"):  0.005,
    ("A1",  "A1"):  0.005,
    ("S1",  "S1"):  0.005,
    ("IT",  "IT"):  0.008,
    ("PFC", "PFC"): 0.008,
    ("M1",  "M1"):  0.005,
    ("CB",  "CB"):  0.005,
    ("HPC", "HPC"): 0.008,
    ("AMY", "AMY"): 0.005,
    ("BG",  "BG"):  0.005,
    ("BS",  "BS"):  0.003,
    # Sensory → Association
    ("V1",  "IT"):  0.020,
    ("A1",  "IT"):  0.020,
    ("S1",  "IT"):  0.015,
    # Association → higher areas
    ("IT",  "PFC"): 0.020,
    ("IT",  "HPC"): 0.015,
    ("IT",  "AMY"): 0.020,
    # PFC projections
    ("PFC", "M1"):  0.015,
    ("PFC", "BG"):  0.020,
    ("PFC", "HPC"): 0.015,
    # Basal ganglia
    ("BG",  "M1"):  0.020,
    ("BG",  "PFC"): 0.015,
    # Cerebellum
    ("CB",  "M1"):  0.025,
    # Hippocampus
    ("HPC", "PFC"): 0.020,
    # Amygdala — threat / valence
    ("AMY", "PFC"): 0.025,
    ("AMY", "BG"):  0.020,
    ("AMY", "BS"):  0.025,
    # Brainstem → all (arousal / neuromodulation broadcast)
    ("BS",  "V1"):  0.004,
    ("BS",  "A1"):  0.004,
    ("BS",  "S1"):  0.004,
    ("BS",  "IT"):  0.004,
    ("BS",  "PFC"): 0.004,
    ("BS",  "M1"):  0.004,
    ("BS",  "CB"):  0.004,
    ("BS",  "HPC"): 0.004,
    ("BS",  "AMY"): 0.004,
    ("BS",  "BG"):  0.004,
}

# Regions whose efferent synapses are inhibitory
INHIBITORY_REGIONS: set[str] = {"BG"}


def estimate_synapse_count(region_sizes: dict[str, int] | None = None) -> int:
    """Compute total synapses given region sizes (default: current SCALE)."""
    if region_sizes is None:
        region_sizes = get_region_sizes()
    total = 0
    for (pre, post), density in CONNECTIVITY.items():
        n_pre  = region_sizes[pre]
        n_post = region_sizes[post]
        total += int(n_pre * n_post * density)
    return total


# ---------------------------------------------------------------------------
# Safety bounds (hardware-enforced, not modifiable by learning)
# ---------------------------------------------------------------------------
JOINT_ANGLE_MIN: float  = -np.pi          # rad
JOINT_ANGLE_MAX: float  =  np.pi          # rad
MAX_JOINT_VELOCITY: float = 2.0           # rad/s
MAX_JOINT_FORCE: float    = 50.0          # N·m
# Collision zones: list of (center_xyz, radius) — empty by default
COLLISION_ZONES: list[tuple[tuple[float, float, float], float]] = []
