# Project context for Claude Code

## What this repo is
HTS-Quench thrust: data-driven surrogate models for thermal-runaway (quench)
detection in High-Temperature Superconductor tapes (REBCO and Bi-2223).
The physics engine implements the analytic heat-balance solver from
Marchevsky & Prestemon, *Supercond. Sci. Technol.* 37 035012 (2024),
DOI 10.1088/1361-6668/ad20fe. Point and sequence surrogates learn the
runaway surfaces I*(T₀, λ), T*, and the adiabatic time margin from ramped
telemetry trajectories inside Gymnasium environments.

## Stack & conventions
- Language / framework: Python 3.11, PyTorch (sequence surrogates), JAX (1D FD solver), scikit-learn (point surrogates), Gymnasium (env API)
- Env manager: conda on Polaris (ALCF), venv locally. Setup: `pip install -r requirements.txt` + `pip install "jax[cpu]==0.4.29"`
- Test command: `pytest tests/`
- Lint/format command: none configured
- Data location: `data/` directory (gitignored). FEM sweeps on Eagle filesystem at `/eagle/fanc_ldrd/magnets_fem/`. Never commit data, checkpoints, `.pt`, `.joblib`, `.h5`, or credentials.

## Domain notes Claude should know
- Reference paper: Marchevsky & Prestemon (2024), DOI 10.1088/1361-6668/ad20fe
- I* = runaway current [A]: the current at which Joule heating exceeds cooling → thermal runaway
- T* = runaway temperature [K]: the temperature at the runaway point
- margin_norm = (I* − I) / I*: fractional distance to runaway (0 = at boundary, negative = past it)
- λ (lam) = cooling coefficient [W/m/K] for REBCO; η_c = fixed cooling [W/K] for Bi-2223
- Ramp trajectory: I += ramp_rate × I* each step
- Observation: [T0_K, I_A, voltage_V, T_surface_K] — the paper's four live instruments
- 5 conductor envs: REBCO no-stab, REBCO Cu/SS stab, REBCO 1D-FD, Bi-2223 no-stab, Bi-2223 Ag-stab
- 3 sequence architectures: GRU, Transformer, SSM (S4D diagonal state-space)
- Key papers live in my Obsidian vault under 03-Resources/Papers.

## How I want help
- Keep changes small and reviewable; state physics assumptions explicitly.
- Update docs/PROGRESS.md after meaningful changes (see below).
- Never commit data, checkpoints, or credentials.

## Progress logging
After a meaningful change, append a dated entry to docs/PROGRESS.md:
what changed, why, and the result/metric. I sync this into the matching
deliverable note in my vault during weekly review.
