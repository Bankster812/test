#!/usr/bin/env python3
"""
Neuromorphic IB Brain — Live Visual Dashboard
==============================================
Run:  python neuromorphic/visualise.py

Shows:
  1. Brain region connectivity map (static network graph)
  2. Live firing rates per region (updates every 50ms)
  3. Spike raster — M1 neurons over last 200 steps
  4. WACC / IRR / Leverage decoded from M1 in real-time
  5. Query input box — type a question and watch the brain respond
"""

import sys, os, threading, time, collections
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("TkAgg")           # works on Mac/Windows/Linux
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import TextBox, Button

# ── Colour palette ────────────────────────────────────────────────────────────
REGION_COLOURS = {
    "V1":  "#4A90D9",   # blue       — visual cortex
    "A1":  "#7B68EE",   # purple     — auditory
    "S1":  "#5BA85A",   # green      — somatosensory
    "IT":  "#50C8C8",   # teal       — association
    "PFC": "#E8A838",   # amber      — prefrontal
    "M1":  "#E05C5C",   # red        — motor output
    "CB":  "#A0C878",   # lime       — cerebellum
    "HPC": "#9B7FD4",   # lavender   — hippocampus
    "AMY": "#E87050",   # orange     — amygdala
    "BG":  "#D4A0C8",   # pink       — basal ganglia
    "BS":  "#888888",   # grey       — brainstem
}

# Region positions for the network diagram (x, y) in [0,1]
REGION_POS = {
    "V1":  (0.15, 0.75),
    "A1":  (0.15, 0.50),
    "S1":  (0.15, 0.25),
    "IT":  (0.35, 0.62),
    "PFC": (0.55, 0.80),
    "M1":  (0.85, 0.50),
    "CB":  (0.70, 0.20),
    "HPC": (0.50, 0.50),
    "AMY": (0.40, 0.30),
    "BG":  (0.65, 0.65),
    "BS":  (0.85, 0.20),
}

# Key connections to draw (subset — most visually important)
KEY_CONNECTIONS = [
    ("V1","IT"), ("A1","IT"), ("S1","IT"),
    ("IT","PFC"), ("IT","HPC"), ("IT","AMY"),
    ("PFC","BG"), ("PFC","M1"),
    ("BG","M1"),  ("HPC","PFC"),
    ("AMY","BG"), ("AMY","PFC"),
    ("CB","M1"),  ("BS","M1"),
]

RASTER_WINDOW = 200   # timesteps shown in spike raster
N_RASTER_NEURONS = 60 # how many M1 neurons to show

# ── Brain init (background thread) ───────────────────────────────────────────

class BrainRunner:
    def __init__(self):
        self.brain       = None
        self.ready       = False
        self.error       = None
        self.rates       = {r: 0.0 for r in REGION_COLOURS}
        self.raster      = collections.deque(maxlen=RASTER_WINDOW)  # list of spike arrays
        self.params      = {}
        self.query_text  = ""
        self.query_result= ""
        self.querying    = False
        self._lock       = threading.Lock()

    def init_brain(self):
        try:
            import neuromorphic.config as base_cfg
            base_cfg.SCALE = 0.01
            from neuromorphic.domains.investment_banking import ib_config
            ib_config.SCALE = 0.01
            from neuromorphic.domains.investment_banking.ib_brain import IBBrain
            self.brain = IBBrain(config=ib_config, verbose=False)
            self.ready = True
        except Exception as e:
            self.error = str(e)

    def step_loop(self):
        """Background thread: continuously steps the brain."""
        while not self.ready:
            time.sleep(0.05)
        brain = self.brain
        cfg   = brain.cfg
        n_v1  = brain.regions["V1"].end - brain.regions["V1"].start

        while True:
            # Random tonic input (idle activity)
            vis = (np.random.rand(n_v1) < 0.02).astype(np.float32)
            brain.step(visual=vis, reward=0.0)

            # Collect rates and raster
            with self._lock:
                for name, region in brain.regions.items():
                    spikes = brain.neurons.spikes[region.start:region.end]
                    self.rates[name] = float(spikes.mean() / cfg.DT)

                m1s = brain.regions["M1"].start
                m1e = brain.regions["M1"].end
                self.raster.append(
                    brain.neurons.spikes[m1s:m1s + N_RASTER_NEURONS].copy()
                )

                # Decode params from M1
                avg = brain.neurons.firing_rate_est[m1s:m1e]
                try:
                    fp = brain.financial_decoder.decode(avg)
                    self.params = {
                        "WACC":     fp.get("wacc")[0] * 100,
                        "IRR":      fp.get("irr")[0] * 100,
                        "Leverage": fp.get("leverage_ratio")[0],
                        "EV/EBITDA":fp.get("ev_ebitda")[0],
                    }
                except Exception:
                    pass

            time.sleep(0.001)  # ~1ms per step

    def run_query(self, text):
        if not self.ready or self.querying:
            return
        self.querying = True
        self.query_result = "Thinking..."
        def _do():
            try:
                r = self.brain.query(text, verbose=False)
                self.query_result = r.answer_text[:300]
            except Exception as e:
                self.query_result = f"Error: {e}"
            self.querying = False
        threading.Thread(target=_do, daemon=True).start()


# ── Dashboard ─────────────────────────────────────────────────────────────────

def build_dashboard(runner: BrainRunner):
    fig = plt.figure(figsize=(16, 9), facecolor="#0D0D0D")
    fig.suptitle("Neuromorphic IB Brain — Live Dashboard",
                 color="white", fontsize=14, fontweight="bold", y=0.98)

    gs = GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35,
                  left=0.06, right=0.97, top=0.93, bottom=0.12)

    ax_net   = fig.add_subplot(gs[0:2, 0])   # network graph
    ax_rates = fig.add_subplot(gs[0,   1])   # firing rates bar
    ax_rast  = fig.add_subplot(gs[1,   1])   # spike raster
    ax_param = fig.add_subplot(gs[0,   2])   # decoded parameters
    ax_query = fig.add_subplot(gs[1,   2])   # query result text
    ax_time  = fig.add_subplot(gs[2,   :])   # time-series firing rates

    for ax in [ax_net, ax_rates, ax_rast, ax_param, ax_query, ax_time]:
        ax.set_facecolor("#1A1A2E")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333355")

    # ── 1. Network graph (static) ─────────────────────────────────────────
    ax_net.set_title("Brain Region Connectivity", color="white", fontsize=9)
    ax_net.set_xlim(-0.05, 1.05); ax_net.set_ylim(-0.05, 1.05)
    ax_net.axis("off")

    for src, dst in KEY_CONNECTIONS:
        x0, y0 = REGION_POS[src]
        x1, y1 = REGION_POS[dst]
        ax_net.annotate("", xy=(x1, y1), xytext=(x0, y0),
                        arrowprops=dict(arrowstyle="-|>", color="#334466",
                                        lw=0.8, mutation_scale=8))

    region_circles = {}
    for name, (x, y) in REGION_POS.items():
        col = REGION_COLOURS[name]
        circ = plt.Circle((x, y), 0.075, color=col, alpha=0.85, zorder=3)
        ax_net.add_patch(circ)
        ax_net.text(x, y, name, ha="center", va="center",
                    color="white", fontsize=7.5, fontweight="bold", zorder=4)
        region_circles[name] = circ

    # ── 2. Firing rates bar ───────────────────────────────────────────────
    ax_rates.set_title("Firing Rates (Hz)", color="white", fontsize=9)
    regions_list = list(REGION_COLOURS.keys())
    bar_colours  = [REGION_COLOURS[r] for r in regions_list]
    bars = ax_rates.barh(regions_list,
                         [0.0] * len(regions_list),
                         color=bar_colours, alpha=0.85)
    ax_rates.set_xlim(0, 1200)
    ax_rates.tick_params(colors="white", labelsize=7)
    ax_rates.xaxis.label.set_color("white")
    ax_rates.set_xlabel("Hz", color="#AAAAAA", fontsize=7)

    # ── 3. Spike raster ───────────────────────────────────────────────────
    ax_rast.set_title(f"M1 Spike Raster (last {RASTER_WINDOW} steps, {N_RASTER_NEURONS} neurons)",
                      color="white", fontsize=9)
    raster_img = ax_rast.imshow(
        np.zeros((N_RASTER_NEURONS, RASTER_WINDOW), dtype=np.float32),
        aspect="auto", cmap="hot", vmin=0, vmax=1,
        origin="lower"
    )
    ax_rast.set_xlabel("Time (steps)", color="#AAAAAA", fontsize=7)
    ax_rast.set_ylabel("Neuron", color="#AAAAAA", fontsize=7)
    ax_rast.tick_params(colors="white", labelsize=7)

    # ── 4. Decoded parameters ─────────────────────────────────────────────
    ax_param.set_title("Brain → IB Parameters", color="white", fontsize=9)
    ax_param.axis("off")
    param_texts = {}
    param_names = ["WACC", "IRR", "Leverage", "EV/EBITDA"]
    for i, pname in enumerate(param_names):
        y_pos = 0.85 - i * 0.22
        ax_param.text(0.05, y_pos, pname + ":", color="#AAAAAA",
                      fontsize=9, transform=ax_param.transAxes)
        t = ax_param.text(0.55, y_pos, "—", color=REGION_COLOURS["M1"],
                          fontsize=11, fontweight="bold",
                          transform=ax_param.transAxes)
        param_texts[pname] = t

    # ── 5. Query result ───────────────────────────────────────────────────
    ax_query.set_title("Query Response", color="white", fontsize=9)
    ax_query.axis("off")
    query_text_obj = ax_query.text(
        0.03, 0.95, "Type a question below and press Enter...",
        color="#CCCCCC", fontsize=7.5, transform=ax_query.transAxes,
        va="top", wrap=True, multialignment="left"
    )

    # ── 6. Time-series ────────────────────────────────────────────────────
    ax_time.set_title("Region Firing Rates — Time Series", color="white", fontsize=9)
    ax_time.set_facecolor("#1A1A2E")
    ax_time.set_xlim(0, 200); ax_time.set_ylim(0, 1200)
    ax_time.tick_params(colors="white", labelsize=7)
    ax_time.set_xlabel("Steps", color="#AAAAAA", fontsize=7)
    ax_time.set_ylabel("Hz", color="#AAAAAA", fontsize=7)
    HISTORY = 200
    rate_history = {r: collections.deque([0.0] * HISTORY, maxlen=HISTORY)
                    for r in regions_list}
    time_lines = {}
    for name in regions_list:
        line, = ax_time.plot([], [], color=REGION_COLOURS[name],
                             alpha=0.75, lw=1.0, label=name)
        time_lines[name] = line
    ax_time.legend(loc="upper right", fontsize=6, ncol=6,
                   facecolor="#0D0D0D", labelcolor="white", framealpha=0.7)

    # ── Query input widget ────────────────────────────────────────────────
    ax_input = fig.add_axes([0.06, 0.02, 0.76, 0.045])
    ax_input.set_facecolor("#1A1A2E")
    text_box = TextBox(ax_input, "Query: ", color="#1A1A2E",
                       hovercolor="#252550",
                       label_pad=0.03)
    text_box.label.set_color("white")
    text_box.text_disp.set_color("#00FF88")

    ax_btn = fig.add_axes([0.84, 0.02, 0.12, 0.045])
    btn = Button(ax_btn, "Ask Brain", color="#334488", hovercolor="#4466BB")
    btn.label.set_color("white")

    submitted_query = {"text": ""}

    def submit_query(text):
        q = text.strip()
        if q and runner.ready:
            submitted_query["text"] = q
            query_text_obj.set_text(f"Q: {q}\n\nThinking...")
            runner.run_query(q)
        text_box.set_val("")

    text_box.on_submit(submit_query)
    btn.on_clicked(lambda e: submit_query(text_box.text))

    # ── Loading overlay ───────────────────────────────────────────────────
    loading_text = ax_net.text(
        0.5, 0.5, "Initialising\nbrain...",
        ha="center", va="center", color="white",
        fontsize=12, fontweight="bold",
        transform=ax_net.transAxes,
        bbox=dict(boxstyle="round", fc="#0D0D0D", alpha=0.8)
    )

    # ── Animation update ──────────────────────────────────────────────────
    def update(frame):
        if not runner.ready:
            if runner.error:
                loading_text.set_text(f"Error:\n{runner.error[:60]}")
            return

        loading_text.set_visible(False)

        with runner._lock:
            rates  = dict(runner.rates)
            raster = list(runner.raster)
            params = dict(runner.params)
            qr     = runner.query_result

        # Update network circles (brightness = firing rate)
        for name, circ in region_circles.items():
            rate = rates.get(name, 0)
            alpha = 0.3 + min(rate / 800.0, 0.7)
            circ.set_alpha(alpha)

        # Update bars
        max_rate = max(rates.values(), default=1)
        for bar, name in zip(bars, regions_list):
            bar.set_width(rates.get(name, 0))
        ax_rates.set_xlim(0, max(max_rate * 1.1, 100))

        # Update raster
        if len(raster) > 0:
            mat = np.array(raster).T  # (N_RASTER, time)
            if mat.shape[1] < RASTER_WINDOW:
                pad = np.zeros((N_RASTER_NEURONS, RASTER_WINDOW - mat.shape[1]))
                mat = np.hstack([pad, mat])
            raster_img.set_data(mat[:, -RASTER_WINDOW:].astype(np.float32))

        # Update decoded parameters
        fmt = {"WACC": "{:.1f}%", "IRR": "{:.1f}%",
               "Leverage": "{:.1f}x", "EV/EBITDA": "{:.1f}x"}
        for pname, t in param_texts.items():
            val = params.get(pname, 0.0)
            try:
                t.set_text(fmt[pname].format(val))
            except Exception:
                t.set_text("—")

        # Update query result
        if qr and qr != "Thinking...":
            query_text_obj.set_text(qr[:350])
        elif qr == "Thinking...":
            query_text_obj.set_text("Thinking... (200 neural steps)")

        # Update time-series
        for name in regions_list:
            rate_history[name].append(rates.get(name, 0))
            y = list(rate_history[name])
            time_lines[name].set_data(range(len(y)), y)
        max_h = max(max(list(h)) for h in rate_history.values() if h)
        ax_time.set_ylim(0, max(max_h * 1.1, 100))

        return list(region_circles.values()) + list(bars) + [raster_img] + \
               list(param_texts.values()) + [query_text_obj] + list(time_lines.values())

    ani = animation.FuncAnimation(
        fig, update, interval=50, blit=False, cache_frame_data=False
    )

    plt.show()
    return ani


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Starting Neuromorphic IB Brain Dashboard...")
    print("Initialising brain in background (takes ~5s at SCALE=0.01)...\n")

    runner = BrainRunner()

    # Init brain in background so GUI opens immediately
    threading.Thread(target=runner.init_brain, daemon=True).start()
    threading.Thread(target=runner.step_loop,  daemon=True).start()

    build_dashboard(runner)
