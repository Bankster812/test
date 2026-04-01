"""
Neuromorphic AI Platform
========================
A biologically-inspired spiking neural network with:
  - 1,000,000 LIF neurons across 11 brain regions
  - 1.2 billion sparse synaptic connections
  - Continuous online STDP learning (no batch training)
  - Reward-modulated three-factor plasticity
  - Homeostatic regulation
  - Hardware-enforced safety kernel on all motor outputs

Quick start
-----------
    import neuromorphic.config as cfg
    cfg.SCALE = 0.01          # demo scale: 10K neurons
    from neuromorphic.brain import Brain
    from neuromorphic.safety.kernel import SafetyKernel
    from neuromorphic.safety.constraints import MotorConstraints
    from neuromorphic.safety.reflexes import ReflexLibrary

    safety = SafetyKernel(MotorConstraints.default(), ReflexLibrary())
    brain  = Brain(cfg, safety_kernel=safety)

    cmd, is_safe = brain.step(visual=frame, auditory=spectrum, reward=0.0)
"""

from neuromorphic.brain import Brain
from neuromorphic.safety.kernel import SafetyKernel
from neuromorphic.safety.constraints import MotorConstraints
from neuromorphic.safety.reflexes import ReflexLibrary

__all__ = ["Brain", "SafetyKernel", "MotorConstraints", "ReflexLibrary"]
__version__ = "0.1.0"
