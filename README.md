# Simulations of Particulate Flows — Project 2: Universal Physics Transformer (Tiny Operator Surrogate)

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](#) [![PyTorch](https://img.shields.io/badge/PyTorch-lightgrey.svg)](#) [![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](#)

**GitHub:** https://github.com/Riya2382


# Tiny Universal Physics Transformer (U‑PT) — Advection‑Diffusion with Rotation

**Goal:** A compact operator‑learning surrogate that maps *(state, parameters)* → *next state* for a rotating advection‑diffusion system (2D). We vary rotation rate `omega`, diffusivity `kappa`, and forcing amplitude, then train a tiny U‑Net to step the PDE. Multi‑step rollouts show generalization.

This is a *tractable* proxy for "universal physics transformers" in the JD, highlighting parameter‑conditioned surrogates with fast rollout.

## Run
```bash
python src/train.py --epochs 3
python src/rollout.py --omega 2.0 --kappa 1e-3 --steps 60
```
Artifacts in `outputs/` (loss curves, rollout frames, metrics).

## Why this helps
- Shows **operator/surrogate learning** conditioned on physics parameters.
- Demonstrates compact, robust architecture + reproducibility.
- Clear path to scale: swap toy PDE data with CFD/CFD‑DEM snapshots.
## How this maps to moving/fluidized beds & rotary kilns

- Treat the U‑Net as a **parameter‑conditioned operator** that maps (state, process parameters) → next state.
- Swap toy PDE data with coarse CFD/CFD‑DEM fields (e.g., solids fraction, gas velocity) and condition on key process knobs (gas flow, rotation rate, particle size).
- This approximates a **universal physics transformer** capable of **fast rollouts** for process‑scale digital twins and control.


> **Use in your email:** Include one of the generated plots and a 1–2 line summary:
> *“Built a small, reproducible demo aligning with recurrence/operator-learning/digital-twin ideas and showed real-time rollouts/forecasting on toy data; ready to swap in CFD/CFD‑DEM snapshots.”*
