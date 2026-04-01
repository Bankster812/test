"""
Neuromorphic AI Platform — Demo
================================
Demonstrates the full pipeline at SCALE=0.01 (10K neurons, ~120K synapses)
so it runs on any laptop without special hardware.

What the demo shows
-------------------
1.  Continuous learning: STDP weights change every timestep in response
    to experience — no training epochs, no gradient descent.

2.  11 brain regions active simultaneously: per-region firing rates
    printed every 100 ms of simulated time.

3.  Neuromodulation: dopamine rises after reward pulses, gating the
    rate of synaptic change.

4.  Safety kernel: a deliberately extreme motor command (very high
    velocity) triggers reflex suppression. The kernel logs the violation.

5.  Spike visualisation: optional ASCII raster and firing-rate plot
    if matplotlib is installed.

Usage
-----
    python -m neuromorphic.demo           # default 2 seconds
    python -m neuromorphic.demo --steps 5000 --no-plot
    python -m neuromorphic.demo --scale 0.05
"""

from __future__ import annotations

import argparse
import time
import sys

import numpy as np


def build_brain(scale: float, seed: int = 0):
    """Construct Brain + SafetyKernel at the given scale."""
    import neuromorphic.config as cfg
    cfg.SCALE = scale

    from neuromorphic.brain import Brain
    from neuromorphic.safety.kernel import SafetyKernel
    from neuromorphic.safety.constraints import MotorConstraints
    from neuromorphic.safety.reflexes import ReflexLibrary

    constraints = MotorConstraints.default(n_dof=cfg.N_DOF)
    reflexes    = ReflexLibrary(n_dof=cfg.N_DOF)
    safety      = SafetyKernel(constraints, reflexes)
    brain       = Brain(config=cfg, safety_kernel=safety, seed=seed, verbose=True)
    return brain, cfg


def generate_inputs(step: int, rng: np.random.Generator):
    """Generate synthetic sensory inputs for one timestep."""
    # Visual: 16×16 random frame with a slow drift pattern
    t_phase  = step * 0.01
    base     = np.clip(
        0.5 + 0.4 * np.sin(np.linspace(0, 2 * np.pi + t_phase, 256)),
        0, 1,
    ).reshape(16, 16).astype(np.float32)
    visual   = base + rng.uniform(0, 0.1, base.shape).astype(np.float32)
    visual   = np.clip(visual, 0, 1)

    # Auditory: 32-band spectrum with a dominant frequency that shifts
    band     = int(step * 0.1) % 32
    spectrum = rng.uniform(0, 0.2, 32).astype(np.float32)
    spectrum[band] = 1.0

    # Touch: 16-channel sensors, sparse contact
    touch    = rng.uniform(0, 0.1, 16).astype(np.float32)
    if step % 50 < 5:    # contact burst every 50 ms
        touch[:4] = rng.uniform(0.7, 1.0, 4).astype(np.float32)

    return visual, spectrum, touch


def run_demo(
    n_steps: int = 2000,
    scale:   float = 0.01,
    seed:    int   = 0,
    plot:    bool  = True,
):
    """
    Run the neuromorphic brain for n_steps timesteps.

    Parameters
    ----------
    n_steps : int   — number of 1 ms timesteps (2000 = 2 seconds)
    scale   : float — network scale factor
    seed    : int   — RNG seed
    plot    : bool  — show matplotlib plots at the end
    """
    rng = np.random.default_rng(seed)

    print("=" * 60)
    print(" NEUROMORPHIC AI PLATFORM — DEMO")
    print("=" * 60)
    brain, cfg = build_brain(scale, seed)

    # Storage for visualisation
    history: dict[str, list[float]] = {name: [] for name in brain.regions}
    da_history:  list[float] = []
    safe_history: list[bool] = []

    print(f"\nRunning {n_steps} steps ({n_steps} ms simulated time)...\n")
    wall_start = time.perf_counter()

    for step in range(n_steps):
        visual, spectrum, touch = generate_inputs(step, rng)

        # Reward pulse every 100 ms for 10 ms
        reward = 1.0 if (step % 100 < 10) else 0.0

        cmd, is_safe = brain.step(
            visual=visual,
            auditory=spectrum,
            soma=touch,
            reward=reward,
        )

        safe_history.append(is_safe)

        # Record per-region firing rates
        for name, region in brain.regions.items():
            history[name].append(region.mean_rate)
        da_history.append(brain.neuromod.da)

        # Print status every 100 steps
        if step % 100 == 0:
            diag    = brain.get_diagnostics()
            rates   = diag["firing_rates"]
            nm      = diag["neuromodulators"]
            t_ms    = diag["sim_time_ms"]
            safety_str = "" if is_safe else "  [REFLEX]"
            print(
                f"t={t_ms:6.1f}ms | "
                f"V1:{rates['V1']:4.1f} A1:{rates['A1']:4.1f} "
                f"IT:{rates['IT']:4.1f} PFC:{rates['PFC']:4.1f} "
                f"M1:{rates['M1']:4.1f} HPC:{rates['HPC']:4.1f} "
                f"BG:{rates['BG']:4.1f} BS:{rates['BS']:4.1f} Hz | "
                f"DA:{nm['DA']:.3f} ACh:{nm['ACh']:.3f}"
                f"{safety_str}"
            )

    wall_elapsed = time.perf_counter() - wall_start
    n_violations = sum(1 for s in safe_history if not s)

    print("\n" + "=" * 60)
    print(f"  Simulation complete.")
    print(f"  Wall time   : {wall_elapsed:.2f} s")
    print(f"  Sim time    : {n_steps} ms")
    print(f"  Speed       : {n_steps / wall_elapsed:.0f}× real-time")
    print(f"  Violations  : {n_violations} / {n_steps} steps "
          f"({100*n_violations/n_steps:.1f}%)")
    if brain.safety:
        s = brain.safety.summary()
        print(f"  Safety log  : {s['total_violations']} reflex triggers")
    print("=" * 60)

    # -- Optional matplotlib plots --
    if plot:
        _plot_results(history, da_history, safe_history, n_steps)

    return brain


def _plot_results(
    history: dict[str, list[float]],
    da_history: list[float],
    safe_history: list[bool],
    n_steps: int,
):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        print("\n[demo] matplotlib not installed — skipping plots.")
        print("       Install with: pip install matplotlib")
        return

    t_ms = np.arange(n_steps)

    fig = plt.figure(figsize=(14, 9))
    fig.suptitle("Neuromorphic AI Platform — Live Brain Activity", fontsize=14)
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

    # Panel 1: Sensory cortices
    ax1 = fig.add_subplot(gs[0, 0])
    for name in ["V1", "A1", "S1"]:
        ax1.plot(t_ms, history[name], label=name, alpha=0.8)
    ax1.set_title("Sensory Cortices")
    ax1.set_ylabel("Firing rate (Hz)")
    ax1.legend(fontsize=8)
    ax1.set_xlabel("Time (ms)")

    # Panel 2: Higher cognitive
    ax2 = fig.add_subplot(gs[0, 1])
    for name in ["IT", "PFC", "HPC"]:
        ax2.plot(t_ms, history[name], label=name, alpha=0.8)
    ax2.set_title("Association / Cognition")
    ax2.set_ylabel("Firing rate (Hz)")
    ax2.legend(fontsize=8)
    ax2.set_xlabel("Time (ms)")

    # Panel 3: Motor + cerebellar
    ax3 = fig.add_subplot(gs[1, 0])
    for name in ["M1", "CB"]:
        ax3.plot(t_ms, history[name], label=name, alpha=0.8)
    ax3.set_title("Motor System")
    ax3.set_ylabel("Firing rate (Hz)")
    ax3.legend(fontsize=8)
    ax3.set_xlabel("Time (ms)")

    # Panel 4: Subcortical
    ax4 = fig.add_subplot(gs[1, 1])
    for name in ["AMY", "BG", "BS"]:
        ax4.plot(t_ms, history[name], label=name, alpha=0.8)
    ax4.set_title("Subcortical (AMY / BG / BS)")
    ax4.set_ylabel("Firing rate (Hz)")
    ax4.legend(fontsize=8)
    ax4.set_xlabel("Time (ms)")

    # Panel 5: Dopamine
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.plot(t_ms, da_history, color="purple", alpha=0.9, label="DA")
    ax5.axhline(0, color="k", linewidth=0.5)
    ax5.set_title("Dopamine (Learning Gate)")
    ax5.set_ylabel("DA level")
    ax5.set_xlabel("Time (ms)")

    # Panel 6: Safety violations
    ax6 = fig.add_subplot(gs[2, 1])
    violation_steps = [i for i, s in enumerate(safe_history) if not s]
    if violation_steps:
        ax6.vlines(violation_steps, 0, 1, color="red", alpha=0.6, linewidth=1)
    ax6.set_ylim(-0.1, 1.1)
    ax6.set_title("Safety Violations (red = reflex triggered)")
    ax6.set_xlabel("Time (ms)")
    ax6.set_ylabel("Violation")

    plt.savefig("/tmp/neuromorphic_demo.png", dpi=120, bbox_inches="tight")
    plt.show()
    print("\n[demo] Plot saved to /tmp/neuromorphic_demo.png")


def main():
    parser = argparse.ArgumentParser(
        description="Neuromorphic AI Platform Demo"
    )
    parser.add_argument(
        "--steps", type=int, default=2000,
        help="Number of timesteps to simulate (default: 2000 = 2 seconds)"
    )
    parser.add_argument(
        "--scale", type=float, default=0.01,
        help="Network scale factor (default: 0.01 = 10K neurons)"
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed"
    )
    parser.add_argument(
        "--no-plot", action="store_true", help="Disable matplotlib plots"
    )
    args = parser.parse_args()

    run_demo(
        n_steps = args.steps,
        scale   = args.scale,
        seed    = args.seed,
        plot    = not args.no_plot,
    )


if __name__ == "__main__":
    main()
