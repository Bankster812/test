# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository overview

A biologically-inspired spiking neural network platform with a specialised Investment Banking domain. There is no build system and no formal package config — it is run directly with `python -m`. Dependencies are listed in `requirements.txt`. A `pytest` suite under `tests/` covers the deterministic IB financial models (the neural platform itself is still validated by running the demos).

## Common commands

```bash
# Install dependencies (numpy, scipy, matplotlib, openpyxl, pdfplumber, beautifulsoup4)
pip install -r requirements.txt

# Run the core neuromorphic demo (10K neurons, ~120K synapses, ~2 s sim time)
python -m neuromorphic.demo
python -m neuromorphic.demo --steps 5000 --no-plot
python -m neuromorphic.demo --scale 0.05      # 50K neurons

# Run the Investment Banking end-to-end demo (KB lookup → DCF/LBO/Credit → NL query)
python -m neuromorphic.domains.investment_banking.demo

# Live matplotlib dashboard (firing rates, M1 raster, query box)
python neuromorphic/visualise.py

# Three.js HTTP brain viewer (optional LLM narration via llm/ clients)
python -m neuromorphic.brain_web --port 8000 --demo
python -m neuromorphic.brain_web --scale 0.15 --no-llm          # skip LLM clients
python -m neuromorphic.brain_web --model llama3.1:8b --groq-key KEY   # other flags: --nvidia-key

# Run the IB financial-model test suite (deterministic, fast, no network)
pytest tests/ -q

# Worked end-to-end walkthrough of all six IB models (no network, no Brain)
python -m neuromorphic.domains.investment_banking.examples.worked_models
python -m neuromorphic.domains.investment_banking.examples.worked_models --excel out/
```

The neural platform has no automated tests — validate changes by running the relevant demo and checking the diagnostics output (per-region firing rates, neuromodulator levels, safety violation counts). The IB financial models *are* covered by `pytest tests/`; run it after touching anything under `domains/investment_banking/models/`. CI (`.github/workflows`) runs flake8 (non-blocking style + blocking syntax/undefined-name checks) and `pytest`.

## High-level architecture

### Two layers
1. **`neuromorphic/`** — generic spiking-network platform. `Brain` orchestrates 11 brain regions, 37 synapse pools, learning, I/O, and a hardware-style safety kernel for motor output.
2. **`neuromorphic/domains/investment_banking/`** — `IBBrain(Brain)` adds financial encoders/decoders, a knowledge base, financial model code (DCF/LBO/Merger/Comps/Precedents/Credit), Excel auditor, and a 24/7 learning daemon.

### Core package layout (`neuromorphic/`)
- `core/` — the hot path: `neuron_group.py` (vectorised LIF over contiguous arrays), `synapse_pool.py` (sparse CSR/CSC connectivity + STDP), `spike_buffer.py` (circular axonal-delay buffer).
- `regions/` — one module per region plus `base_region.py` (abstract `BrainRegion`); files map to the codes below (`visual_cortex.py`→V1, `motor_cortex.py`→M1, `basal_ganglia.py`→BG, …).
- `learning/` — `stdp.py`, `homeostasis.py`, `neuromodulation.py`.
- `io/` — `encoder.py` (`SensoryEncoder`), `decoder.py` (`MotorDecoder`).
- `safety/` — `kernel.py`, `constraints.py`, `reflexes.py` (see Safety architecture).
- `llm/` (`groq_client.py`, `nvidia_nim.py`, `ollama_client.py`) and `personality/` (`entity.py`, `memory.py`, `traits.py`) are **peripheral**: optional narration/persona scaffolding used only by `brain_web.py`, not part of the `Brain.step()` pipeline.

### `Brain.step()` pipeline (in `neuromorphic/brain.py`)
The per-timestep order is load-bearing — most subtle bugs come from violating it:
1. Encode sensory inputs → inject currents into V1, A1, S1.
2. Deliver delayed spikes from `SpikeBuffer`.
3. Region pre-spike hooks (e.g. `BasalGanglia.inject_reward`) and `region.step()`.
4. `NeuronGroup.step()` — vectorised LIF integration → spikes for all neurons.
5. Propagate spikes through every `SynapsePool` (also queued with delay).
6. `NeuromodulationSystem.update()` reads `BG.dopamine`, `BS.arousal`, `AMY.threat_signal`.
7. `STDPRule.apply_all()` — gated by dopamine (three-factor rule).
8. `HomeostaticScaling.step()` — every `HOMEOSTASIS_INTERVAL` steps only.
9. Decode M1 spikes → `SafetyKernel.check_and_gate()` → motor command.
10. `SpikeBuffer.advance()` and `t += dt`.

### Memory layout — single source of truth
All neurons live in **one** `NeuronGroup` with contiguous NumPy arrays (`v`, `i_syn`, `spikes`, …). Regions are just `(start, end)` slice views — never copies. Pool connectivity uses local indices into per-region populations, with `pre_ids`/`post_ids` translating back to global IDs. Do not introduce per-neuron Python objects; do not duplicate spike state.

### Synapse pool naming and inhibition
Pools are keyed `"PRE->POST"` everywhere (`brain.pools`, STDP iteration, the `cfg.CONNECTIVITY` dict, which has 37 entries). `cfg.INHIBITORY_REGIONS` is a set of source-region names whose efferent currents are negated inside `SynapsePool.propagate` — currently just `{"BG"}`. To add a connection, add an entry to `CONNECTIVITY` in `neuromorphic/config.py` — it is auto-instantiated by `Brain._build_connectivity`.

### STDP performance contract (`core/synapse_pool.py`)
Three-tier laziness — preserve this when editing:
- Tier 1: vectorised trace decays each step, O(N).
- Tier 2: weight updates only run for neurons that actually fired this step.
- Tier 3: per-fired-neuron update is a CSR row slice (LTP) or CSC column slice (LTD), no Python loop over individual synapses. `_sync_csc()` rebuilds the CSC view when LTD is needed.

### Configuration discipline
`neuromorphic/config.py` is a module of constants. **`SCALE` must be set before `Brain` is constructed** — region sizes are computed at init from `cfg.SCALE` (0.01 → 10K, 0.1 → 100K, 1.0 → 1M neurons; synapses scale as SCALE²). The module default is `SCALE = 0.15` (~150K neurons), but the demos pass smaller values (`demo.py` defaults to `--scale 0.01`). The IB layer ships `domains/investment_banking/ib_config.py`, which duck-types as `neuromorphic.config` (re-exports every base attribute and overrides `N_DOF=32`, `TARGET_RATE=8.0`, etc.). Pass it as `Brain(config=ib_config, …)`.

### Safety architecture (do not weaken)
`neuromorphic/safety/` enforces hardware-style invariants on motor output. From `safety/kernel.py`:
- `SafetyKernel` is constructed **before** `Brain` and passed in. It holds no reference to `Brain`/`NeuronGroup` and cannot be influenced by learning.
- `MotorConstraints` is a frozen dataclass; `SafetyKernel` stores it in a private (name-mangled) attribute with no setter.
- Every motor command must pass through `check_and_gate()` — no decode→actuator path may bypass it. Violations return a reflex command from `ReflexLibrary` and `is_safe=False`.

The IB layer mirrors this with `FinancialSafetyKernel` (`domains/investment_banking/safety/financial_constraints.py`): name-mangled `__limits` dict clamps params (WACC ∈ [3%, 30%], leverage ≤ 12×, etc.) before any `IBResponse` is returned.

### IBBrain query lifecycle (`domains/investment_banking/ib_brain.py`)
`IBBrain.query(text)` flow:
1. `QueryEngine.parse` → `QueryVector` (intent, entities, numerics, sector, model_type).
2. **Knowledge-base shortcut** — for `intent in ("knowledge", "explanation")`, return text from `KnowledgeBase` directly without running the network.
3. `FinancialEncoder.encode_query` → V1/A1/S1 spike trains.
4. Run `cfg.QUERY_STEPS` (default 200) brain steps with reward injected for the first half; M1 spikes pushed into a 50-step ring buffer (`_SpikeHistory`) for stable decoding.
5. `FinancialDecoder.decode(avg_m1)` → `FinancialParams`; clamp through `FinancialSafetyKernel`; analyse via `RiskEngine`.
6. If intent maps to a model (`dcf`/`lbo`/`merger`/`comps`/`precedents`/`credit`), call `_run_model` — this is plain numerical code, not neural settling. Inputs use `_m` suffix for millions; `_run_model` multiplies by `M = 1e6`.
7. `ResponseSynthesizer.synthesize` → `IBResponse` (text + params + risk flags).

`IBBrain.build_model(model_type, inputs)` skips neural settling and runs a financial model directly — use this for deterministic numerical work.

### Continuous learning daemon
`start_continuous_learning()` spawns a background thread (`learning/continuous_daemon.py`, `ContinuousLearningDaemon`) that periodically pulls from `learning/web_learner.py` (`WebLearner`) / `learning/youtube_learner.py` (`YouTubeLearner`) and feeds chunks into `_ingest_chunks`, which runs short (~20 step) STDP bursts with `reward=0.6`. Always `stop_continuous_learning()` before exit. Daemon access is guarded by `self._lock`.

### IB domain component map (`domains/investment_banking/`)
- **Query path** — `query/query_engine.py` (`QueryEngine` → `QueryVector`), `encoders/financial_encoder.py` (`FinancialEncoder`, `FinancialTokenizer`, and a *second* `QueryVector`), `decoders/financial_decoder.py` (`FinancialDecoder` → `FinancialParams`), `decoders/response_synthesizer.py` (`ResponseSynthesizer` → `IBResponse`), `query/response_formatter.py`. Note the two `QueryVector` classes: `ib_brain.py` imports the encoder one as `EncoderQueryVector` and the parser one as `QueryVector`.
- **Models** (`models/`, deterministic numerical code, no neural settling) — `dcf.py`, `lbo.py`, `merger.py`, `comps.py`, `precedents.py`, `credit.py`, each exposing `<Name>Model` / `<Name>Inputs` / `<Name>Result`. Dispatched by `IBBrain._run_model`. `compute()` validates inputs and raises `ValueError` on nonsense (e.g. DCF `terminal_growth >= wacc`); DCF supports mid-year convention and perpetuity-vs-exit-multiple terminal value with a cross-check; LBO uses one consistent levered-FCF/cash-sweep loop with a bisection IRR. Adding result fields is safe; `_run_model` only reads a fixed subset. Covered by `tests/`.
- **Worked examples** (`examples/worked_models.py`) — runnable, network-free walkthrough of all six models on one mid-market target (`python -m ...examples.worked_models [--excel DIR]`); see `examples/README.md`.
- **Knowledge & memory** (`knowledge/`) — `knowledge_base.py` (`KnowledgeBase`/`KBEntry`, backs the KB shortcut), `deal_memory.py` (`DealMemory`, 128-d precedent embeddings with cosine retrieval), `risk_engine.py` (`RiskEngine` → `RiskReport`/`RiskLevel`/`RiskFlag`).
- **Ingestion & learning** — `encoders/document_ingestion.py` (`DocumentIngestion` for PDF/Excel/text → `FinancialChunk`), plus the learning daemon above.
- **Safety & Excel** — `safety/financial_constraints.py` (`FinancialSafetyKernel`) and `excel/` (`auditor.py` `ExcelAuditor`, `reader.py`, `writer.py`), which back `IBBrain.correct_excel()`.

## Conventions specific to this codebase

- Region codes are fixed three-letter labels: `V1, A1, S1, IT, PFC, M1, CB, HPC, AMY, BG, BS`. They are used as dict keys in `Brain.regions`, `cfg.CONNECTIVITY`, and pool names. Don't rename.
- Spike arrays are `bool`; current arrays are `float32`. STDP traces and weights are `float32`. Refractory counters are `int16`. Stay in these dtypes when adding code in the hot path.
- `MAX_DELAY_MS` bounds the `SpikeBuffer` window; `inject_current` uses `np.add.at` because the same post-neuron may receive multiple injections per step.
- The `data/` directory contains reference IB PDFs/Excel models used for ingestion demos, not source code.
- `__pycache__/`, `*.pyc`, `*.pyo` are ignored (`.gitignore`).
