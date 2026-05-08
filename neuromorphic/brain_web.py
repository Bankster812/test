#!/usr/bin/env python3
"""
brain_web.py — Neuromorphic Brain: 3D Visualization + Electronic Life Form
===========================================================================
Run:  python -m neuromorphic.brain_web [--port 8000] [--demo]
      NVIDIA_API_KEY=nvapi-xxx python -m neuromorphic.brain_web --demo

Set NVIDIA_API_KEY to enable Llama-3.1-Nemotron-70B (GPT-4 class, free tier).
Without it, falls back to local Ollama llama3.1:8b.
"""

import argparse
import json
import math
import os
import random
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Region / connection metadata (mirrored in JS too)
# ---------------------------------------------------------------------------

REGIONS: Dict[str, dict] = {
    'V1':  {'pos': [  0,   0.5, -7.5], 'color': '#00e5ff', 'size': 2.0, 'neurons': 150000},
    'A1':  {'pos': [  4.5, 0.5, -1.5], 'color': '#ff6d00', 'size': 1.3, 'neurons':  50000},
    'S1':  {'pos': [  0,   5.5, -0.5], 'color': '#ffff00', 'size': 1.7, 'neurons': 100000},
    'IT':  {'pos': [  4,  -2,    1  ], 'color': '#d500f9', 'size': 1.6, 'neurons': 100000},
    'PFC': {'pos': [  0,   3.5,  7.5], 'color': '#2979ff', 'size': 2.0, 'neurons': 150000},
    'M1':  {'pos': [  0,   5.5,  2  ], 'color': '#ff1744', 'size': 1.7, 'neurons': 100000},
    'CB':  {'pos': [  0,  -5,   -5.5], 'color': '#00e676', 'size': 1.7, 'neurons': 100000},
    'HPC': {'pos': [  2,  -1,    0.5], 'color': '#ff80ab', 'size': 1.5, 'neurons':  80000},
    'AMY': {'pos': [  2.5,-2.5,  3  ], 'color': '#ea80fc', 'size': 0.9, 'neurons':  30000},
    'BG':  {'pos': [  0,   0,    0.5], 'color': '#1de9b6', 'size': 1.1, 'neurons':  40000},
    'BS':  {'pos': [  0,  -6.5, -3  ], 'color': '#bdbdbd', 'size': 1.6, 'neurons': 100000},
}

CONNECTIONS: List[List[str]] = [
    ['V1','IT'],  ['V1','S1'],  ['IT','PFC'],  ['IT','HPC'],
    ['S1','M1'],  ['S1','PFC'], ['PFC','M1'],  ['A1','PFC'],
    ['A1','IT'],  ['HPC','PFC'],['HPC','AMY'], ['AMY','PFC'],
    ['BG','M1'],  ['BG','PFC'], ['CB','M1'],   ['BS','CB'],
    ['BS','BG'],  ['PFC','BG'], ['M1','CB'],
]

IB_CONCEPTS = [
    'WACC', 'DCF', 'EBITDA', 'IRR', 'Enterprise Value',
    'LBO', 'Synergies', 'Terminal Value', 'Net Debt', 'MOIC',
    'Accretion/Dilution',
]


# ---------------------------------------------------------------------------
# BrainState
# ---------------------------------------------------------------------------

class BrainState:
    """Thread-safe container for brain simulation state."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.rates: Dict[str, float] = {r: 0.0 for r in REGIONS}
        self.concepts: List[dict] = []
        self.params: dict = {}
        self.sim_time: float = 0.0
        self.step: int = 0
        self.last_query: str = ''
        self.last_response: str = ''

    def update(self, rates: Dict[str, float], concepts: List[dict],
               sim_time: float, step: int) -> None:
        with self._lock:
            self.rates = dict(rates)
            self.concepts = list(concepts)
            self.sim_time = sim_time
            self.step = step

    def set_response(self, query: str, response: str) -> None:
        with self._lock:
            self.last_query = query
            self.last_response = response

    def snapshot(self) -> dict:
        with self._lock:
            return {
                'rates': dict(self.rates),
                'concepts': list(self.concepts),
                'sim_time': self.sim_time,
                'step': self.step,
                'last_query': self.last_query,
                'last_response': self.last_response,
            }


# ---------------------------------------------------------------------------
# SimulationThread
# ---------------------------------------------------------------------------

class SimulationThread(threading.Thread):
    """Background thread: tries IBBrain, falls back to demo mode."""

    def __init__(self, state: BrainState, demo: bool = False,
                 scale: float = 1.0) -> None:
        super().__init__(daemon=True)
        self.state = state
        self.force_demo = demo
        self.scale = scale
        self._stop_event = threading.Event()

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        if not self.force_demo:
            try:
                from neuromorphic.ib_brain import IBBrain  # type: ignore
                brain = IBBrain(scale=self.scale)
                self._run_brain(brain)
                return
            except Exception:
                pass
        self._run_demo()

    def _run_brain(self, brain) -> None:
        self._brain = brain   # expose for query handler
        step = 0
        while not self._stop_event.is_set():
            t0 = time.perf_counter()
            result = brain.step()
            rates = {r: float(result.get('rates', {}).get(r, 0)) for r in REGIONS}
            concepts = _decode_concepts(result.get('m1_spikes', []))
            self.state.update(rates, concepts, brain.sim_time, step)
            step += 1
            elapsed = time.perf_counter() - t0
            time.sleep(max(0, 0.01 - elapsed))

    def _run_demo(self) -> None:
        rng = random.Random(42)
        base_rates = {r: rng.uniform(2, 15) for r in REGIONS}
        freqs      = {r: rng.uniform(0.3, 2.0) for r in REGIONS}
        phases     = {r: rng.uniform(0, 2 * math.pi) for r in REGIONS}

        concept_pool = list(IB_CONCEPTS)
        active_concepts: List[dict] = []
        next_concept_change = 0.0
        t = 0.0
        step = 0

        while not self._stop_event.is_set():
            t0 = time.perf_counter()

            rates: Dict[str, float] = {}
            for r in REGIONS:
                burst = rng.gauss(0, 0.2) if rng.random() < 0.03 else 0
                val = base_rates[r] * (
                    1 + 0.5 * math.sin(2 * math.pi * freqs[r] * t + phases[r])
                    + burst ** 2
                )
                rates[r] = max(0.0, val)

            if t >= next_concept_change:
                n = rng.randint(2, 5)
                chosen = rng.sample(concept_pool, n)
                confidences = sorted([rng.random() for _ in chosen], reverse=True)
                active_concepts = [
                    {'label': lbl, 'confidence': round(conf, 3)}
                    for lbl, conf in zip(chosen, confidences)
                ]
                next_concept_change = t + rng.uniform(1.5, 4.0)

            self.state.update(rates, active_concepts, t, step)

            elapsed = time.perf_counter() - t0
            time.sleep(max(0, 0.01 - elapsed))
            t += 0.01
            step += 1


def _decode_concepts(m1_spikes) -> List[dict]:
    """Map M1 spike chunks to IB concept labels (stub for real IBBrain)."""
    if not m1_spikes:
        return []
    rng = random.Random(int(sum(m1_spikes[:5]) * 1000) if m1_spikes else 0)
    n = rng.randint(1, 4)
    chosen = rng.sample(IB_CONCEPTS, min(n, len(IB_CONCEPTS)))
    return [{'label': lbl, 'confidence': round(rng.random(), 3)} for lbl in chosen]


# ---------------------------------------------------------------------------
# Embedded HTML
# ---------------------------------------------------------------------------

_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Neuromorphic Brain Visualization</title>
<style>
  *{margin:0;padding:0;box-sizing:border-box}
  body{background:#050510;overflow:hidden;font-family:'Segoe UI',monospace;color:#e0e8ff}
  #c{display:block;width:100vw;height:100vh;position:relative;z-index:1}
  #ui{position:fixed;top:0;left:0;width:100%;height:100%;pointer-events:none;z-index:10}
  .panel{pointer-events:auto;background:rgba(5,5,25,0.75);border:1px solid rgba(100,150,255,0.2);
         border-radius:8px;padding:10px 14px;backdrop-filter:blur(6px)}
  #info{position:absolute;top:14px;left:14px;min-width:180px;font-size:12px;line-height:1.8}
  #info .title{font-size:16px;font-weight:700;color:#7eb8ff;letter-spacing:1px;margin-bottom:6px}
  #info .row{color:#8899cc}
  #info .val{color:#cce4ff;font-weight:600}
  #thoughts{position:absolute;top:14px;right:14px;min-width:220px;max-width:260px;font-size:12px}
  #thoughts h3{color:#7eb8ff;font-size:13px;font-weight:700;margin-bottom:8px;letter-spacing:.5px}
  .concept{margin-bottom:7px}
  .concept-label{display:flex;justify-content:space-between;margin-bottom:3px;font-size:11px}
  .concept-name{color:#cce4ff}
  .concept-pct{color:#7eb8ff}
  .concept-bar-bg{background:rgba(100,150,255,0.15);border-radius:3px;height:5px;overflow:hidden}
  .concept-bar{background:linear-gradient(90deg,#2979ff,#7eb8ff);height:5px;transition:width .4s}
  /* ── Chat panel ── */
  #chat-panel{position:absolute;bottom:0;left:50%;transform:translateX(-50%);
              width:580px;max-width:96vw;z-index:20;display:flex;flex-direction:column;
              background:rgba(4,5,22,0.88);border:1px solid rgba(100,150,255,0.2);
              border-bottom:none;border-radius:14px 14px 0 0;backdrop-filter:blur(10px)}
  #entity-bar{display:flex;align-items:center;gap:10px;padding:10px 16px 6px;
              border-bottom:1px solid rgba(100,150,255,0.1)}
  #entity-name{font-size:13px;font-weight:700;color:#7eb8ff;letter-spacing:.05em}
  #entity-mood{font-size:11px;color:#556;background:rgba(100,150,255,0.08);
               border-radius:4px;padding:2px 8px}
  #entity-llm{margin-left:auto;font-size:10px;color:#445}
  #speak-btn{background:none;border:none;cursor:pointer;font-size:16px;padding:0 4px;
             opacity:.6;transition:opacity .2s}
  #speak-btn:hover{opacity:1}
  #chat-msgs{height:220px;overflow-y:auto;padding:10px 14px;display:flex;
             flex-direction:column;gap:8px;scroll-behavior:smooth}
  #chat-msgs::-webkit-scrollbar{width:4px}
  #chat-msgs::-webkit-scrollbar-thumb{background:rgba(100,150,255,0.2);border-radius:2px}
  .msg{max-width:88%;padding:8px 12px;border-radius:10px;font-size:12px;line-height:1.55;
       animation:fadein .3s ease}
  .msg-user{align-self:flex-end;background:rgba(41,121,255,0.18);
            border:1px solid rgba(41,121,255,0.3);color:#cce4ff;border-radius:10px 10px 2px 10px}
  .msg-ai{align-self:flex-start;background:rgba(0,230,118,0.07);
          border:1px solid rgba(0,230,118,0.2);color:#b8ffd4;border-radius:10px 10px 10px 2px}
  .msg-ai .sender{font-size:10px;color:#00e676;font-weight:700;margin-bottom:4px}
  .msg-thinking{align-self:flex-start;color:#556;font-size:11px;font-style:italic}
  @keyframes fadein{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:none}}
  #chat-input-row{display:flex;gap:8px;padding:10px 14px 12px}
  #chat-input{flex:1;background:rgba(10,12,40,0.9);border:1px solid rgba(100,150,255,0.3);
              border-radius:8px;color:#cce4ff;padding:8px 12px;font-size:13px;outline:none;
              resize:none;height:38px;font-family:inherit}
  #chat-input:focus{border-color:rgba(100,150,255,0.65);box-shadow:0 0 8px rgba(41,121,255,0.25)}
  #chat-send{background:rgba(0,230,118,0.15);border:1px solid rgba(0,230,118,0.4);
             border-radius:8px;color:#00e676;padding:8px 18px;cursor:pointer;font-size:13px;
             transition:background .2s;white-space:nowrap}
  #chat-send:hover{background:rgba(0,230,118,0.3)}
  #chat-send:disabled{opacity:.35;cursor:default}
  #legend{position:absolute;bottom:18px;right:14px;font-size:11px;min-width:120px}
  #legend h3{color:#7eb8ff;font-size:12px;font-weight:700;margin-bottom:7px}
  .leg-row{display:flex;align-items:center;gap:7px;margin-bottom:4px;color:#9aaccc}
  .leg-dot{width:10px;height:10px;border-radius:50%;flex-shrink:0}
  #lod-hint{position:absolute;bottom:18px;left:14px;font-size:11px;color:rgba(150,180,255,0.5);
            pointer-events:none;line-height:1.7}
</style>
</head>
<body>
<canvas id="c"></canvas>
<div id="ui">
  <div id="info" class="panel">
    <div class="title">&#9632; NEUROMORPHIC</div>
    <div class="row">Neurons: <span class="val" id="stat-neurons">0</span></div>
    <div class="row">Sim&nbsp;time: <span class="val" id="stat-time">0.000</span> s</div>
    <div class="row">Steps: <span class="val" id="stat-steps">0</span></div>
    <div class="row">Zoom&nbsp;dist: <span class="val" id="stat-dist">0.0</span></div>
    <div class="row">LLM: <span class="val" id="stat-llm" style="color:#888">–</span></div>
  </div>
  <div id="thoughts" class="panel">
    <h3>&#9670; Active Thoughts</h3>
    <div id="concept-list"></div>
  </div>
  <!-- Chat panel — talks to the electronic life form -->
  <div id="chat-panel">
    <div id="entity-bar">
      <span id="entity-name">…</span>
      <span id="entity-mood">initialising</span>
      <button id="speak-btn" title="Toggle voice">🔊</button>
      <span id="entity-llm">–</span>
    </div>
    <div id="chat-msgs"></div>
    <div id="chat-input-row">
      <input id="chat-input" type="text" placeholder="Say something to the life form…" autocomplete="off"/>
      <button id="chat-send">Send</button>
    </div>
  </div>
  <div id="legend" class="panel">
    <h3>&#9632; Regions</h3>
    <div id="legend-list"></div>
  </div>
  <div id="lod-hint">
    Scroll to zoom<br>
    &gt;28 → regions only<br>
    9–28 → neurons<br>
    &lt;9 → synapses
  </div>
</div>

<script type="importmap">
{
  "imports": {
    "three": "https://cdn.jsdelivr.net/npm/three@0.152.2/build/three.module.js",
    "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.152.2/examples/jsm/"
  }
}
</script>
<script type="module">
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// ── Region / connection data ──────────────────────────────────────────────
const REGIONS = {
  V1:  {pos:[  0,  0.5,-7.5],color:'#00e5ff',size:2.0,neurons:150000},
  A1:  {pos:[4.5,  0.5,-1.5],color:'#ff6d00',size:1.3,neurons: 50000},
  S1:  {pos:[  0,  5.5,-0.5],color:'#ffff00',size:1.7,neurons:100000},
  IT:  {pos:[  4,   -2,  1  ],color:'#d500f9',size:1.6,neurons:100000},
  PFC: {pos:[  0,  3.5, 7.5],color:'#2979ff',size:2.0,neurons:150000},
  M1:  {pos:[  0,  5.5,  2  ],color:'#ff1744',size:1.7,neurons:100000},
  CB:  {pos:[  0,   -5,-5.5],color:'#00e676',size:1.7,neurons:100000},
  HPC: {pos:[  2,   -1, 0.5],color:'#ff80ab',size:1.5,neurons: 80000},
  AMY: {pos:[2.5, -2.5,  3  ],color:'#ea80fc',size:0.9,neurons: 30000},
  BG:  {pos:[  0,    0, 0.5],color:'#1de9b6',size:1.1,neurons: 40000},
  BS:  {pos:[  0, -6.5, -3  ],color:'#bdbdbd',size:1.6,neurons:100000},
};
const CONNECTIONS = [
  ['V1','IT'],['V1','S1'],['IT','PFC'],['IT','HPC'],
  ['S1','M1'],['S1','PFC'],['PFC','M1'],['A1','PFC'],
  ['A1','IT'],['HPC','PFC'],['HPC','AMY'],['AMY','PFC'],
  ['BG','M1'],['BG','PFC'],['CB','M1'],['BS','CB'],['BS','BG'],
  ['PFC','BG'],['M1','CB'],
];
const TOTAL_NEURONS = Object.values(REGIONS).reduce((s,r)=>s+r.neurons,0);

// ── Scene setup ────────────────────────────────────────────────────────────
const renderer = new THREE.WebGLRenderer({canvas:document.getElementById('c'),antialias:true});
renderer.setPixelRatio(Math.min(window.devicePixelRatio,2));
renderer.setSize(window.innerWidth,window.innerHeight);
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.1;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x050510);
scene.fog = new THREE.FogExp2(0x050510, 0.022);

const camera = new THREE.PerspectiveCamera(55, window.innerWidth/window.innerHeight, 0.1, 300);
camera.position.set(0, 4, 22);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.07;
controls.minDistance = 2;
controls.maxDistance = 80;

// Ambient + point lights
scene.add(new THREE.AmbientLight(0x111133, 2));
const plight = new THREE.PointLight(0x2244ff, 3, 60);
plight.position.set(0, 8, 5);
scene.add(plight);

// ── Region meshes ──────────────────────────────────────────────────────────
const regionMeshes = {};   // name → {core, glow}
const regionRates  = {};   // name → normalized rate [0,1]
const MAX_RATE = 20;

for (const [name, rd] of Object.entries(REGIONS)) {
  const col = new THREE.Color(rd.color);
  const pos = new THREE.Vector3(...rd.pos);

  // Core sphere
  const geo  = new THREE.SphereGeometry(rd.size, 32, 32);
  const mat  = new THREE.MeshStandardMaterial({
    color: col,
    emissive: col,
    emissiveIntensity: 0.15,
    roughness: 0.4,
    metalness: 0.3,
    transparent: true,
    opacity: 0.82,
  });
  const core = new THREE.Mesh(geo, mat);
  core.position.copy(pos);
  scene.add(core);

  // Outer glow shell (BackSide)
  const glowGeo = new THREE.SphereGeometry(rd.size * 1.35, 24, 24);
  const glowMat = new THREE.MeshBasicMaterial({
    color: col,
    side: THREE.BackSide,
    transparent: true,
    opacity: 0.06,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
  });
  const glow = new THREE.Mesh(glowGeo, glowMat);
  glow.position.copy(pos);
  scene.add(glow);

  regionMeshes[name] = {core, glow, baseSize: rd.size};
  regionRates[name]  = 0;
}

// ── HTML region labels ─────────────────────────────────────────────────────
const labelDivs = {};
for (const [name] of Object.entries(REGIONS)) {
  const d = document.createElement('div');
  d.className = 'region-label';
  d.textContent = name;
  d.style.cssText = `position:fixed;color:#cce4ff;font-size:11px;font-weight:700;
    pointer-events:none;text-shadow:0 0 6px rgba(100,180,255,0.9);
    letter-spacing:.5px;transform:translate(-50%,-50%);transition:opacity .3s`;
  document.getElementById('ui').appendChild(d);
  labelDivs[name] = d;
}

// ── Connection tubes + curves ──────────────────────────────────────────────
const connCurves  = [];
const connObjects = [];

for (const [a, b] of CONNECTIONS) {
  const pa = new THREE.Vector3(...REGIONS[a].pos);
  const pb = new THREE.Vector3(...REGIONS[b].pos);
  const mid = pa.clone().lerp(pb, 0.5);
  mid.y += (Math.random() - 0.5) * 3;
  mid.x += (Math.random() - 0.5) * 2;
  const curve = new THREE.QuadraticBezierCurve3(pa, mid, pb);
  connCurves.push({curve, a, b});

  const pts  = curve.getPoints(40);
  const geo  = new THREE.BufferGeometry().setFromPoints(pts);
  const mat  = new THREE.LineBasicMaterial({
    color: 0x2244aa,
    transparent: true,
    opacity: 0.25,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
  });
  const line = new THREE.Line(geo, mat);
  scene.add(line);
  connObjects.push(line);
}

// ── Neuron point clouds (per region) ──────────────────────────────────────
const neuronClouds = {};
const POINTS_PER_REGION = 800;

for (const [name, rd] of Object.entries(REGIONS)) {
  const positions = new Float32Array(POINTS_PER_REGION * 3);
  const r = rd.size * 0.92;
  for (let i = 0; i < POINTS_PER_REGION; i++) {
    let x,y,z;
    do {
      x = (Math.random()*2-1)*r;
      y = (Math.random()*2-1)*r;
      z = (Math.random()*2-1)*r;
    } while(x*x+y*y+z*z > r*r);
    const base = new THREE.Vector3(...rd.pos);
    positions[i*3]   = base.x + x;
    positions[i*3+1] = base.y + y;
    positions[i*3+2] = base.z + z;
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  const col = new THREE.Color(rd.color);
  const mat = new THREE.PointsMaterial({
    color: col,
    size: 0.06,
    transparent: true,
    opacity: 0.55,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
    sizeAttenuation: true,
  });
  const pts = new THREE.Points(geo, mat);
  pts.visible = false;
  scene.add(pts);
  neuronClouds[name] = pts;
}

// ── Synapse lines (per region) ─────────────────────────────────────────────
const synapseMeshes = {};

for (const [name, rd] of Object.entries(REGIONS)) {
  const base = new THREE.Vector3(...rd.pos);
  const r    = rd.size * 0.85;
  const N    = 300;
  const positions = new Float32Array(N * 2 * 3);

  for (let i = 0; i < N; i++) {
    const p1 = randomInSphere(base, r);
    const p2 = p1.clone().add(new THREE.Vector3(
      (Math.random()-.5)*.4, (Math.random()-.5)*.4, (Math.random()-.5)*.4
    ));
    const idx = i * 6;
    positions[idx]   = p1.x; positions[idx+1] = p1.y; positions[idx+2] = p1.z;
    positions[idx+3] = p2.x; positions[idx+4] = p2.y; positions[idx+5] = p2.z;
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  const col = new THREE.Color(rd.color);
  const mat = new THREE.LineBasicMaterial({
    color: col,
    transparent: true,
    opacity: 0.3,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
  });
  const segs = new THREE.LineSegments(geo, mat);
  segs.visible = false;
  scene.add(segs);
  synapseMeshes[name] = segs;
}

function randomInSphere(center, r) {
  let x,y,z;
  do {
    x=(Math.random()*2-1)*r; y=(Math.random()*2-1)*r; z=(Math.random()*2-1)*r;
  } while(x*x+y*y+z*z > r*r);
  return new THREE.Vector3(center.x+x, center.y+y, center.z+z);
}

// ── Spike particles ────────────────────────────────────────────────────────
const MAX_SPIKES = 300;
const spikePositions = new Float32Array(MAX_SPIKES * 3);
const spikeGeo = new THREE.BufferGeometry();
spikeGeo.setAttribute('position', new THREE.BufferAttribute(spikePositions, 3));
const spikeMat = new THREE.PointsMaterial({
  color: 0xffffff,
  size: 0.14,
  transparent: true,
  opacity: 0.9,
  blending: THREE.AdditiveBlending,
  depthWrite: false,
  sizeAttenuation: true,
});
const spikeSystem = new THREE.Points(spikeGeo, spikeMat);
scene.add(spikeSystem);

const activeSpikes = []; // {curve, t, speed}

function spawnSpikes(dt) {
  for (const {curve, a, b} of connCurves) {
    const rate = (regionRates[a] + regionRates[b]) * 0.5;
    const prob = rate * 0.7 * dt;
    if (Math.random() < prob && activeSpikes.length < MAX_SPIKES) {
      activeSpikes.push({curve, t: 0, speed: 0.4 + Math.random() * 0.4});
    }
  }
}

function updateSpikes(dt) {
  for (let i = activeSpikes.length - 1; i >= 0; i--) {
    activeSpikes[i].t += activeSpikes[i].speed * dt;
    if (activeSpikes[i].t >= 1) activeSpikes.splice(i, 1);
  }
  const count = Math.min(activeSpikes.length, MAX_SPIKES);
  for (let i = 0; i < count; i++) {
    const p = activeSpikes[i].curve.getPoint(Math.min(activeSpikes[i].t, 1));
    spikePositions[i*3]   = p.x;
    spikePositions[i*3+1] = p.y;
    spikePositions[i*3+2] = p.z;
  }
  // Zero out unused slots
  for (let i = count; i < MAX_SPIKES; i++) {
    spikePositions[i*3] = 1e9; spikePositions[i*3+1] = 1e9; spikePositions[i*3+2] = 1e9;
  }
  spikeGeo.attributes.position.needsUpdate = true;
  spikeGeo.setDrawRange(0, count);
}

// ── Legend ─────────────────────────────────────────────────────────────────
const legendList = document.getElementById('legend-list');
for (const [name, rd] of Object.entries(REGIONS)) {
  const row = document.createElement('div');
  row.className = 'leg-row';
  row.innerHTML = `<span class="leg-dot" style="background:${rd.color}"></span>${name}`;
  legendList.appendChild(row);
}
document.getElementById('stat-neurons').textContent = TOTAL_NEURONS.toLocaleString();

// ── Data polling ───────────────────────────────────────────────────────────
let lastResponse = '';
let lastRates = {};

async function pollState() {
  try {
    const res = await fetch('/state');
    const data = await res.json();

    // Update rates
    for (const [name, rate] of Object.entries(data.rates || {})) {
      regionRates[name] = Math.min(1, rate / MAX_RATE);
      lastRates[name] = rate;
    }

    // Update stats
    document.getElementById('stat-time').textContent  = (data.sim_time||0).toFixed(3);
    document.getElementById('stat-steps').textContent = (data.step||0).toLocaleString();
    const llmEl = document.getElementById('stat-llm');
    if (data.llm_available) {
      llmEl.textContent = data.llm_model || 'ready';
      llmEl.style.color = '#00e676';
    } else if (data.llm_model) {
      llmEl.textContent = 'loading…';
      llmEl.style.color = '#ff8800';
    } else {
      llmEl.textContent = 'offline';
      llmEl.style.color = '#555';
    }

    // Update thoughts panel
    const concepts = data.concepts || [];
    const cl = document.getElementById('concept-list');
    cl.innerHTML = '';
    for (const c of concepts) {
      const pct = Math.round(c.confidence * 100);
      const div = document.createElement('div');
      div.className = 'concept';
      div.innerHTML = `
        <div class="concept-label">
          <span class="concept-name">${c.label}</span>
          <span class="concept-pct">${pct}%</span>
        </div>
        <div class="concept-bar-bg">
          <div class="concept-bar" style="width:${pct}%"></div>
        </div>`;
      cl.appendChild(div);
    }

    // Chat: incoming reply from entity
    if (data.last_response && data.last_response !== lastResponse) {
      lastResponse = data.last_response;
      removeThinking();
      waitingReply = false;
      addMessage('ai', lastResponse);
      if (voiceEnabled) speak(lastResponse);
      document.getElementById('chat-send').disabled = false;
    }
  } catch(e) {}
}
setInterval(pollState, 180);

// ── Entity info poll ───────────────────────────────────────────────────────
async function pollEntity() {
  try {
    const r = await fetch('/entity');
    if (!r.ok) return;
    const info = await r.json();
    if (info.name) {
      document.getElementById('entity-name').textContent = info.name;
      document.getElementById('entity-mood').textContent =
        `${info.mood || ''}  ·  ${info.age_turns || 0} turns`;
      document.getElementById('entity-llm').textContent = info.llm || '';
    }
  } catch(e) {}
}
pollEntity();
setInterval(pollEntity, 5000);

// ── Chat UI ───────────────────────────────────────────────────────────────
let voiceEnabled   = false;
let waitingReply   = false;

function addMessage(role, text) {
  const msgs = document.getElementById('chat-msgs');
  const div  = document.createElement('div');
  div.className = role === 'user' ? 'msg msg-user' : 'msg msg-ai';
  if (role === 'ai') {
    const sender = document.getElementById('entity-name').textContent || 'Entity';
    div.innerHTML = `<div class="sender">${sender}</div>${escapeHtml(text)}`;
  } else {
    div.textContent = text;
  }
  msgs.appendChild(div);
  msgs.scrollTop = msgs.scrollHeight;
}

function addThinking() {
  const msgs = document.getElementById('chat-msgs');
  const div  = document.createElement('div');
  div.className = 'msg msg-thinking';
  div.id        = 'thinking-indicator';
  const name = document.getElementById('entity-name').textContent || 'Entity';
  div.textContent = `${name} is thinking…`;
  msgs.appendChild(div);
  msgs.scrollTop = msgs.scrollHeight;
}

function removeThinking() {
  const el = document.getElementById('thinking-indicator');
  if (el) el.remove();
}

function escapeHtml(str) {
  return str.replace(/&/g,'&amp;').replace(/</g,'&lt;')
            .replace(/>/g,'&gt;').replace(/\n/g,'<br>');
}

async function sendChat() {
  const input = document.getElementById('chat-input');
  const btn   = document.getElementById('chat-send');
  const text  = input.value.trim();
  if (!text || waitingReply) return;

  addMessage('user', text);
  input.value   = '';
  waitingReply  = true;
  btn.disabled  = true;
  addThinking();

  try {
    await fetch('/chat', {
      method:  'POST',
      headers: {'Content-Type':'application/json'},
      body:    JSON.stringify({text}),
    });
    // Reply arrives via next /state poll → lastResponse handler above
  } catch(e) {
    removeThinking();
    addMessage('ai', 'Connection error. Please try again.');
    btn.disabled = false;
    waitingReply = false;
  }
}

document.getElementById('chat-send').addEventListener('click', sendChat);
document.getElementById('chat-input').addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendChat(); }
});

// ── Voice output (browser Web Speech API — no install needed) ────────────
const synth = window.speechSynthesis;
function speak(text) {
  if (!synth) return;
  synth.cancel();
  // Strip markdown-ish chars for cleaner speech
  const clean = text.replace(/[*_`#►]/g, '').replace(/<[^>]+>/g, '').slice(0, 600);
  const utt   = new SpeechSynthesisUtterance(clean);
  utt.rate    = 0.92;
  utt.pitch   = 0.95;
  // Prefer a deeper/neutral voice if available
  const voices = synth.getVoices();
  const pref   = voices.find(v => v.name.includes('Google') && v.lang.startsWith('en'))
              || voices.find(v => v.lang.startsWith('en'));
  if (pref) utt.voice = pref;
  synth.speak(utt);
}

document.getElementById('speak-btn').addEventListener('click', () => {
  voiceEnabled = !voiceEnabled;
  document.getElementById('speak-btn').textContent = voiceEnabled ? '🔊' : '🔇';
  if (!voiceEnabled) synth && synth.cancel();
});

// ── Resize ─────────────────────────────────────────────────────────────────
window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

// ── Render loop ────────────────────────────────────────────────────────────
const tmpV = new THREE.Vector3();
const clock = new THREE.Clock();

function animate() {
  requestAnimationFrame(animate);
  const dt = clock.getDelta();
  controls.update();

  const dist = camera.position.distanceTo(controls.target);
  document.getElementById('stat-dist').textContent = dist.toFixed(1);

  // LOD
  const showNeurons  = dist < 28;
  const showSynapses = dist < 9;
  const showLabels   = dist > 5;

  // Update region meshes
  for (const [name, rd] of Object.entries(REGIONS)) {
    const rate = regionRates[name] || 0;
    const {core, glow, baseSize} = regionMeshes[name];
    core.material.emissiveIntensity = 0.08 + rate * 1.8;
    const sc = 1 + rate * 0.12;
    core.scale.setScalar(sc);
    glow.material.opacity = 0.04 + rate * 0.25;
    glow.scale.setScalar(1 + rate * 0.3);

    // Labels
    const pos3 = new THREE.Vector3(...REGIONS[name].pos);
    pos3.project(camera);
    const sx = ( pos3.x * 0.5 + 0.5) * window.innerWidth;
    const sy = (-pos3.y * 0.5 + 0.5) * window.innerHeight - (baseSize * 28 / dist);
    const lbl = labelDivs[name];
    lbl.style.left = sx + 'px';
    lbl.style.top  = sy + 'px';
    lbl.style.opacity = showLabels ? (0.5 + rate * 0.5) : '0';

    neuronClouds[name].visible  = showNeurons;
    synapseMeshes[name].visible = showSynapses;

    if (showNeurons) {
      neuronClouds[name].material.opacity = 0.3 + rate * 0.55;
    }
    if (showSynapses) {
      synapseMeshes[name].material.opacity = 0.15 + rate * 0.45;
    }
  }

  // Spikes
  spawnSpikes(dt);
  updateSpikes(dt);

  // Pulse connection lines
  const t = performance.now() * 0.001;
  connObjects.forEach((line, i) => {
    const {a, b} = connCurves[i];
    const rate = (regionRates[a] + regionRates[b]) * 0.5;
    line.material.opacity = 0.1 + rate * 0.5 + 0.06 * Math.sin(t * 2 + i);
  });

  renderer.render(scene, camera);
}
animate();
</script>
</body>
</html>
"""

# ---------------------------------------------------------------------------
# Keyword-based IB fallback (no imports needed)
# ---------------------------------------------------------------------------

_IB_KNOWLEDGE: dict[str, str] = {
    "dcf": (
        "DCF (Discounted Cash Flow) for a private company — key steps:\n"
        "1. Project unlevered free cash flows (UFCF) for 5–10 years\n"
        "   UFCF = EBIT × (1-tax) + D&A − CapEx − ΔNWC\n"
        "2. Estimate WACC — harder for privates because no observable beta;\n"
        "   use comparable public company betas, unlever/relever for target cap structure\n"
        "3. Calculate Terminal Value: Gordon Growth (TV = FCF_n × (1+g) / (WACC−g))\n"
        "   or Exit Multiple (TV = EBITDA_n × EV/EBITDA_peer)\n"
        "4. Discount all cash flows + TV back to today at WACC\n"
        "5. Add non-operating assets, subtract net debt → Equity Value\n"
        "6. Apply private-company discounts: DLOM (10–30%) for illiquidity,\n"
        "   minority discount if valuing < 50% stake\n"
        "Key private-company adjustments: normalise owner compensation,\n"
        "add back personal expenses, adjust for related-party transactions."
    ),
    "lbo": (
        "LBO (Leveraged Buyout) — core mechanics:\n"
        "1. Acquisition funded ~60–70% debt, 30–40% sponsor equity\n"
        "2. Debt tranches: senior secured (Term Loan A/B), subordinated, sometimes mezzanine\n"
        "3. Target: strong, stable FCF to service debt; asset-light preferred\n"
        "4. Hold period typically 3–7 years\n"
        "5. Returns driven by: leverage paydown, EBITDA growth, multiple expansion\n"
        "6. Key metrics: IRR (target 20–25%+), MOIC (2.0–3.0x+)\n"
        "7. Entry multiple (EV/EBITDA) vs exit multiple → delta drives returns\n"
        "   IRR sensitivity: +1x exit multiple ≈ +3–5% IRR at 5-year hold"
    ),
    "wacc": (
        "WACC = (E/V)×Ke + (D/V)×Kd×(1−t)\n"
        "• Ke (cost of equity) = Rf + β×ERP + size premium\n"
        "  Rf: 10-yr govt bond; ERP: ~5–6% US; β: peer-levered avg, then unlever/relever\n"
        "• Kd (cost of debt) = yield on existing debt or new issue rate, pre-tax\n"
        "• E/V, D/V: target capital structure (not current)\n"
        "For private companies: add company-specific risk premium (1–3%).\n"
        "Typical WACC ranges: tech 10–14%, industrials 8–11%, utilities 6–8%."
    ),
    "ebitda": (
        "EBITDA = Earnings Before Interest, Tax, Depreciation & Amortisation\n"
        "= Net Income + Interest + Tax + D&A\n"
        "Used as a proxy for operating cash flow and for EV/EBITDA multiples.\n"
        "Adjusted EBITDA: add back non-recurring items (restructuring, one-off legal,\n"
        "stock-based comp sometimes). Bankers scrutinise adjustments carefully.\n"
        "Typical EV/EBITDA multiples by sector:\n"
        "  Tech/SaaS: 12–25×  |  Healthcare: 10–16×  |  Industrials: 7–10×"
    ),
    "merger": (
        "Merger Analysis (Accretion/Dilution):\n"
        "1. Pro-forma EPS = (Acquirer NI + Target NI + synergies − dis-synergies\n"
        "                    − incremental interest − amortisation of intangibles) \n"
        "                  / pro-forma diluted shares\n"
        "2. Accretive if pro-forma EPS > standalone acquirer EPS\n"
        "3. Stock deal: exchange ratio = offer price / acquirer share price\n"
        "4. Breakeven synergies: minimum synergies to make deal neutral\n"
        "5. Contribution analysis: what % of combined revenue/EBITDA each party brings\n"
        "   vs what % of equity each gets"
    ),
    "credit": (
        "Credit / Leverage Analysis:\n"
        "• Leverage: Net Debt / EBITDA — investment grade <2×, HY 4–6×, LBO 5–7×\n"
        "• Interest Coverage: EBITDA / Interest Expense — banks want >3×\n"
        "• DSCR (Debt Service Coverage): (EBITDA − CapEx) / (Interest + Amort)\n"
        "• Fixed Charge Coverage: EBITDA / (Interest + rent + scheduled principal)\n"
        "• Covenants: maintenance (tested quarterly) vs incurrence (tested on actions)\n"
        "• Key covenant levels: leverage ≤ 4.5×, coverage ≥ 3.0× typical"
    ),
}

def _ib_keyword_response(query: str) -> str:
    q = query.lower()
    # Match longest keyword first
    for kw, text in sorted(_IB_KNOWLEDGE.items(), key=lambda x: -len(x[0])):
        if kw in q:
            return text
    # Generic fallback
    return (
        f'Query received: "{query}"\n'
        "No specific knowledge entry matched. Try asking about:\n"
        "DCF, LBO, WACC, EBITDA, merger accretion/dilution, or credit analysis.\n"
        "Run without --demo for full neural IB responses."
    )


# ---------------------------------------------------------------------------
# HTTP Handler
# ---------------------------------------------------------------------------

class BrainHTTPHandler(BaseHTTPRequestHandler):
    """Serves the brain web UI and JSON state."""

    brain_state: BrainState                    # set by main()
    _sim_thread: 'SimulationThread | None' = None  # set by main()
    _ollama:     'object | None'           = None  # OllamaClient, set by main()
    _entity:     'object | None'           = None  # ElectronicLifeForm, set by main()

    def log_message(self, fmt, *args):  # suppress access logs
        pass

    def _send_cors(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')

    def do_OPTIONS(self):
        self.send_response(204)
        self._send_cors()
        self.end_headers()

    def do_GET(self):
        if self.path == '/' or self.path == '/index.html':
            body = _HTML.encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self._send_cors()
            self.end_headers()
            self.wfile.write(body)

        elif self.path == '/entity':
            # Return entity info (name, mood, traits, interests)
            entity = BrainHTTPHandler._entity
            info   = entity.info() if entity else {}
            body   = json.dumps(info).encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self._send_cors()
            self.end_headers()
            self.wfile.write(body)

        elif self.path == '/state':
            snap = self.brain_state.snapshot()
            # Inject LLM status (cached — don't ping Ollama on every poll)
            ollama = BrainHTTPHandler._ollama
            snap['llm_model']     = getattr(ollama, 'model', None)
            snap['llm_available'] = getattr(ollama, '_last_available', False)
            body = json.dumps(snap).encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self._send_cors()
            self.end_headers()
            self.wfile.write(body)

        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == '/chat':
            # Full entity chat — uses personality, memory, STDP learning
            length = int(self.headers.get('Content-Length', 0))
            raw    = self.rfile.read(length)
            try:
                data = json.loads(raw)
                text = data.get('text', '').strip()
            except Exception:
                text = raw.decode('utf-8', errors='replace').strip()

            if not text:
                self.send_error(400, 'No text')
                return

            entity = BrainHTTPHandler._entity
            state  = self.brain_state

            def _chat():
                if entity:
                    reply = entity.chat(text)
                else:
                    reply = "(Entity not initialised — restart with NVIDIA_API_KEY set)"
                state.set_response(text, reply)

            threading.Thread(target=_chat, daemon=True).start()

            body = json.dumps({"status": "thinking"}).encode()
            self.send_response(202)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self._send_cors()
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path == '/query':
            length = int(self.headers.get('Content-Length', 0))
            raw = self.rfile.read(length)
            try:
                data = json.loads(raw)
                query = data.get('query', '').strip()
            except Exception:
                query = raw.decode('utf-8', errors='replace')

            # Dispatch to brain in background thread
            state  = self.brain_state
            sim    = BrainHTTPHandler._sim_thread
            ollama = BrainHTTPHandler._ollama      # set by main()

            def _handle(q):
                # ── 1. Ollama local LLM (all topics, no API key) ──────────────
                if ollama is not None and ollama.is_available():
                    # Build system prompt enriched with brain's active thoughts
                    snap     = state.snapshot()
                    concepts = [c.get('label', c.get('name','')) for c in snap.get('concepts', [])]
                    top_regions = sorted(
                        snap.get('rates', {}).items(), key=lambda x: -x[1]
                    )[:3]
                    brain_ctx = ""
                    if concepts:
                        brain_ctx += f"Currently active in the neural network: {', '.join(concepts[:5])}. "
                    if top_regions:
                        brain_ctx += f"Most active brain regions: {', '.join(r for r,_ in top_regions)}."

                    system = (
                        "You are a neuromorphic AI assistant powered by a spiking neural network "
                        "with 150,000 neurons and 27 million synapses. You have deep expertise in "
                        "finance, investment banking, science, engineering, medicine, law, and any "
                        "other topic the user asks about. Be concise, accurate, and helpful. "
                        "For financial questions, give specific numbers and formulas where relevant.\n"
                        + (f"Brain context: {brain_ctx}" if brain_ctx else "")
                    )
                    response = ollama.generate(q, system=system, max_tokens=600)
                    if response:
                        state.set_response(q, response)
                        return

                # ── 2. IBBrain neural query (if loaded) ──────────────────────
                if sim is not None and getattr(sim, '_brain', None) is not None:
                    try:
                        result = sim._brain.query(q)
                        state.set_response(q, result.answer_text)
                        return
                    except Exception:
                        pass

                # ── 3. Lightweight KB lookup (IB topics, pure Python) ────────
                try:
                    from neuromorphic.domains.investment_banking.query.query_engine import QueryEngine
                    from neuromorphic.domains.investment_banking.knowledge.knowledge_base import KnowledgeBase
                    qv = QueryEngine().parse(q)
                    kb = KnowledgeBase()
                    texts = [str(kb.lookup(e)) for e in qv.entities if kb.lookup(e)]
                    if not texts:
                        texts = [str(r) for r in kb.search(q, top_k=2)]
                    if texts:
                        state.set_response(q, '\n\n'.join(texts))
                        return
                except Exception:
                    pass

                # ── 4. Keyword fallback ───────────────────────────────────────
                state.set_response(q, _ib_keyword_response(q))

            threading.Thread(target=_handle, args=(query,), daemon=True).start()

            self.send_response(202)
            self.send_header('Content-Type', 'application/json')
            self._send_cors()
            self.end_headers()
            self.wfile.write(b'{"status":"queued"}')

        else:
            self.send_response(404)
            self.end_headers()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description='Neuromorphic brain web server')
    parser.add_argument('--port',  type=int,   default=8000,          help='HTTP port (default 8000)')
    parser.add_argument('--scale', type=float, default=0.15,          help='Brain scale (default 0.15 = 150K neurons)')
    parser.add_argument('--demo',  action='store_true',               help='Force demo mode (no IBBrain init)')
    parser.add_argument('--model',      type=str, default='llama3.1:8b', help='Ollama model name')
    parser.add_argument('--no-llm',    action='store_true',             help='Disable all LLM integration')
    parser.add_argument('--groq-key',   type=str, default='',           help='Groq API key (or set GROQ_API_KEY env var)')
    parser.add_argument('--nvidia-key', type=str, default='',           help='NVIDIA NIM API key (or set NVIDIA_API_KEY env var)')
    args = parser.parse_args()

    state = BrainState()

    sim = SimulationThread(state, demo=args.demo, scale=args.scale)
    sim.start()

    # ── Ollama LLM setup ──────────────────────────────────────────────────────
    ollama_client = None
    if not args.no_llm:
        try:
            from neuromorphic.llm.ollama_client import OllamaClient
            ollama_client = OllamaClient(model=args.model)
            ollama_client._last_available = False

            def _ollama_monitor():
                """Background thread: ping Ollama every 10s, cache availability."""
                while True:
                    available = ollama_client.is_available()
                    ollama_client._last_available = available
                    if available:
                        pass  # silent once available
                    time.sleep(10)

            threading.Thread(target=_ollama_monitor, daemon=True).start()
            # Initial check
            if ollama_client.is_available():
                ollama_client._last_available = True
                print(f'[brain_web] LLM: {args.model} ready via Ollama ✓')
            else:
                print(f'[brain_web] LLM: Ollama not available yet — will retry (run: ollama serve)')
        except ImportError:
            pass

    # ── Groq setup (fastest free option — Llama 3.3 70B at 800 tok/s) ──────────
    groq_client = None
    if not args.no_llm:
        try:
            from neuromorphic.llm.groq_client import GroqClient
            groq_key    = args.groq_key or os.environ.get("GROQ_API_KEY", "")
            groq_client = GroqClient(api_key=groq_key)
            if groq_client.is_available():
                print(f'[brain_web] Groq: {groq_client.model} ✓  (70B, ~800 tok/s, free)')
            else:
                print('[brain_web] Groq: no key set  (get free key at console.groq.com)')
                groq_client = None
        except ImportError:
            pass

    # ── NVIDIA NIM setup ──────────────────────────────────────────────────────
    nvidia_client = None
    nvidia_key    = args.nvidia_key or os.environ.get("NVIDIA_API_KEY", "")
    if nvidia_key and not args.no_llm:
        try:
            from neuromorphic.llm.nvidia_nim import NvidiaClient
            nvidia_client = NvidiaClient(api_key=nvidia_key)
            if nvidia_client.is_available():
                print(f'[brain_web] NVIDIA NIM: {nvidia_client.model} ✓  (70B, GPU-hosted)')
            else:
                print('[brain_web] NVIDIA NIM: key set but API not reachable — check key')
                nvidia_client = None
        except ImportError:
            pass
    else:
        print('[brain_web] NVIDIA NIM: not configured (set NVIDIA_API_KEY for 70B model)')

    # ── Electronic Life Form ──────────────────────────────────────────────────
    entity = None
    try:
        from neuromorphic.personality.entity import ElectronicLifeForm
        entity = ElectronicLifeForm(
            groq_client   = groq_client,
            nvidia_client = nvidia_client,
            ollama_client = ollama_client,
            sim_thread    = sim,
            brain_state   = state,
        )
        print(f'[brain_web] Life form: {entity.persona.name}  '
              f'(age={entity.persona.age_turns} turns, mood={entity.persona.mood})')
    except Exception as e:
        print(f'[brain_web] Life form init failed: {e}')

    # ── Bind handler ──────────────────────────────────────────────────────────
    handler = BrainHTTPHandler
    handler.brain_state = state
    handler._sim_thread = sim
    handler._ollama     = ollama_client
    handler._entity     = entity

    server = HTTPServer(('', args.port), handler)
    url    = f'http://localhost:{args.port}/'

    neurons = int(sum(v['neurons'] for v in REGIONS.values()) * args.scale)
    print(f'[brain_web] Neurons: {neurons:,} (scale={args.scale})')
    print(f'[brain_web] Serving at {url}')
    print('[brain_web] Press Ctrl+C to stop.\n')

    def _open():
        time.sleep(1.2)
        webbrowser.open(url)
    threading.Thread(target=_open, daemon=True).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\n[brain_web] Shutting down.')
        sim.stop()
        server.server_close()


if __name__ == '__main__':
    main()
