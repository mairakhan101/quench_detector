# quench_detector

Data-driven surrogate models and Gymnasium environments for **thermal-runaway
(quench) detection** in High-Temperature Superconductor (HTS) tapes.

The physics engine implements the analytic heat-balance solver from
[Marchevsky & Prestemon, *Supercond. Sci. Technol.* **37** 035012 (2024)](https://doi.org/10.1088/1361-6668/ad20fe)
for REBCO and extends it to Bi-2223 conductors. Point surrogates
(HistGradientBoosting) and sequence surrogates (GRU / Transformer / SSM)
learn the runaway surfaces I\*(T₀, λ), T\*, and the adiabatic time margin
directly from ramped telemetry trajectories — the same noisy signals a
deployed protection loop would observe.

---

## Repository structure

```
hts_tape_env/           Core library
  physics.py              REBCO analytic solver (I*, T*, V, T_surface)
  physics_bi2223.py       Bi-2223 analytic solver
  physics_fd1d.py         1D Du Fort-Frankel finite-difference PDE solver
  physics_fd1d_jax.py     JAX-accelerated vectorized FD solver
  materials.py            Tape geometry and material properties
  thermal.py              REBCO adiabatic time-margin solver
  thermal_bi2223.py       Bi-2223 time-margin solver
  env.py                  REBCO Gymnasium environment (QuenchDetectorEnv)
  env_bi2223.py           Bi-2223 Gymnasium environment
  env_fd1d.py             1D-FD spatially-resolved Gymnasium environment
  fem_data.py             FEM sweep.h5 loader (RTD + voltage-tap telemetry)
  fem_env.py              FEM replay Gymnasium environment
  datagen.py              Static (i.i.d.) dataset generation
  rollout.py              Trajectory dataset generation (env rollouts)
  rollout_bi2223_env.py   Bi-2223 trajectory rollouts
  bi2223_rollout.py       Bi-2223 rollout helpers
  surrogates.py           Point surrogate registry (HistGBR specs A/two_d/B/raw)
  seq_data.py             Sequence dataset and normalization
  seq_surrogates.py       Sequence surrogate architectures (GRU/Transformer/SSM)
  adaptive_sampling.py    Boundary-aware adaptive sampling strategies
  benchmark.py            Unified benchmark metrics and diagnostic figures
  controllers.py          Rule-based and surrogate-augmented controllers
  physics_diagnostics.py  Physics layer sanity checks

scripts/                Training and evaluation scripts
  benchmark_all_envs.py   Run all 5 conductor benchmarks (entry point)
  train_eval_rebco_nostab.py
  train_eval_rebco_stab.py
  train_eval_rebco_fd1d.py
  train_eval_bi2223_nostab.py
  train_eval_bi2223_agstab.py
  train_seq_surrogate.py  Standalone sequence model training
  cross_conductor_benchmark.py
  diffusion_map_analysis.py
  adaptive_sampling_diagnostics.py
  _bootstrap.py           Adds repo root to sys.path for scripts

scripts/polaris/        ALCF Polaris HPC job scripts
  setup_env.sh            One-time conda environment setup
  run_benchmark_quick.pbs         Debug queue, 1 node, uniform
  run_benchmark_quick_adaptive.pbs  Debug queue, 1 node, adaptive
  run_benchmark_prod.pbs          Production, 10 nodes, pbsdsh dispatch

tests/                  pytest test suite
models/                 Saved surrogates and figures (gitignored binaries)
docs/                   Progress log
legacy/                 Earlier exploration code (event_selection, etc.)
```

---

## Installation

Python 3.11+. Clone and install dependencies:

```bash
git clone https://github.com/mairakhan101/quench_detector.git
cd quench_detector
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Optional: JAX for the 1D finite-difference solver
pip install "jax[cpu]==0.4.29"
```

On **ALCF Polaris**, use the provided setup script instead:

```bash
bash scripts/polaris/setup_env.sh
```

---

## Conductor environments

Five Gymnasium environments cover two HTS materials with different physics
and stabilization models. All share the same obs/action/reward API:

| Name | Class | Physics | Key parameter |
|------|-------|---------|---------------|
| REBCO no-stab | `QuenchDetectorEnv` | `physics.py` | λ (cooling W/m/K) |
| REBCO Cu/SS stab | `QuenchDetectorEnv(stabilizer=...)` | `physics.py` | λ + redistribution |
| REBCO 1D-FD | `FD1DQuenchDetectorEnv` | `physics_fd1d.py` | Spatially-resolved T(x) |
| Bi-2223 no-stab | `Bi2223QuenchDetectorEnv(mode='no_stab')` | `physics_bi2223.py` | Fixed η_c cooling |
| Bi-2223 Ag-stab | `Bi2223QuenchDetectorEnv(mode='ag_stab')` | `physics_bi2223.py` | Ag-matrix redistribution |

**Observation** (Box, float32): `[T0_K, I_A, voltage_V, T_surface_K]` — the
paper's four live instruments.

**Action** (Discrete): `0` = continue ramping, `1` = protective dump.

**Episode**: `reset()` samples (T₀, λ) → solves for I\*, T\*. Each step
ramps `I += ramp_rate × I*`. The episode ends on dump (action 1) or when
I exceeds the runaway threshold.

---

## Surrogate pipeline

### Point surrogates

`hts_tape_env/surrogates.py` defines four HistGradientBoosting specs:

| Spec | Inputs | Target | Use case |
|------|--------|--------|----------|
| `A` | T₀ | I\* | 1D runaway curve (Fig. 2) |
| `two_d` | T₀, λ | I\* | 2D runaway surface (Fig. 3) |
| `B` | T₀, I | margin | Protection-loop quantity |
| `raw` | T₀, I, V, T_surface | I\* | Full instrument vector |

```python
from hts_tape_env.datagen import generate_dataset
from hts_tape_env.surrogates import train_spec

df = generate_dataset(5000, lam_range=(20, 200))
result = train_spec("two_d", df)
print(result.metrics)  # {'mae': ..., 'r2': ...}
result.save("models/surr_two_d.joblib")
```

### Sequence surrogates

`hts_tape_env/seq_surrogates.py` trains GRU, Transformer, or SSM (S4D)
architectures on ramped telemetry trajectories. Five prediction heads:

- **margin** `[B,T]` — running (I\*−I)/I\* at every step
- **breach_logit** `[B,T]` — logit of margin < dump threshold (BCE loss)
- **istar_log** `[B]` — log I\* from final hidden state
- **tstar** `[B]` — T\* from final hidden state
- **time_margin_log** `[B]` — log(adiabatic time to 300 K)

```python
from hts_tape_env.rollout import generate_rollout_dataset
from hts_tape_env.seq_surrogates import train_seq_surrogate

df = generate_rollout_dataset(1500, seed=7)
result = train_seq_surrogate(df, arch="gru")  # or "transformer", "ssm"
result.save("models/seq_gru.pt")
```

### Adaptive sampling

`hts_tape_env/adaptive_sampling.py` concentrates training samples near the
runaway boundary (I/I\* ≈ 1) and in high-curvature regions of the I\*(T₀)
surface using:

1. **Boundary-concentrated I/I\* sampling** — truncated normal at 1.0
2. **Curvature-weighted T₀ sampling** — |d²I\*/dT₀²| importance weights

Use `--adaptive --blend 0.5` on benchmark scripts to mix 50% adaptive /
50% uniform samples.

---

## Running benchmarks

The benchmark system trains point + sequence surrogates for each conductor
and evaluates against the analytic solver, producing metrics JSON, diagnostic
figures, and error surface plots.

### Local (quick smoke test)

```bash
python scripts/benchmark_all_envs.py --scale quick --seed 42
```

### Local (single conductor)

```bash
python scripts/benchmark_all_envs.py --scale quick --only "REBCO no-redist" --seed 42
```

### All flags

```
--scale {quick,production}   Sample counts and epoch budget
--only "name"                Run a single conductor (use exact name from table)
--adaptive                   Enable boundary-aware adaptive sampling
--blend 0.5                  Fraction of adaptive vs uniform samples
--out-dir DIR                Output directory for models and figures
--seed N                     Random seed
```

---

## Running on ALCF Polaris

### One-time setup

```bash
# On a Polaris login node:
bash scripts/polaris/setup_env.sh
```

This creates a `quench_surr` conda environment under
`/soft/applications/conda/2024-04-29/mconda3/` with all dependencies
including JAX CPU.

### Debug queue (quick tests)

```bash
# Uniform sampling, all 5 conductors, ~25 min
qsub scripts/polaris/run_benchmark_quick.pbs

# Adaptive sampling, all 5 conductors, ~25 min
qsub scripts/polaris/run_benchmark_quick_adaptive.pbs
```

Debug queue: 1 node, 1 hour walltime, immediate scheduling.

### Production (multi-node)

```bash
qsub scripts/polaris/run_benchmark_prod.pbs
```

This requests **10 nodes** from the `prod` queue (minimum for that queue)
and dispatches 10 tasks via `pbsdsh`:

| Node | Conductor | Sampling |
|------|-----------|----------|
| 0 | Bi-2223 no-stab | uniform |
| 1 | Bi-2223 Ag-stab | uniform |
| 2 | REBCO no-redist | uniform |
| 3 | REBCO Cu/SS redist | uniform |
| 4 | REBCO 1D-FD | uniform |
| 5 | Bi-2223 no-stab | adaptive |
| 6 | Bi-2223 Ag-stab | adaptive |
| 7 | REBCO no-redist | adaptive |
| 8 | REBCO Cu/SS redist | adaptive |
| 9 | REBCO 1D-FD | adaptive |

Production scale: 2000 samples, 500 episodes, 200 epochs per conductor.
Results go to `/eagle/fanc_ldrd/magnets_fem/benchmark_prod_{uniform,adaptive}/`.

### Monitoring jobs

```bash
qstat -u $USER          # Job status
cat ~/bench_prod_all.o*  # Stdout/stderr (merged)
```

### JAX / PyTorch GPU coexistence

JAX and PyTorch compete for CUDA context within a single process. The PBS
scripts handle this:

- **Debug queue** (CPU only): `CUDA_VISIBLE_DEVICES=""`
- **Production** (PyTorch on GPU, JAX on CPU): `JAX_PLATFORMS=cpu`

Thread limits are also set to avoid pthread exhaustion on compute nodes:
`OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`.

---

## Tests

```bash
pytest tests/
```

The test suite covers physics solvers, environments, FEM data loading, and
thermal calculations for both REBCO and Bi-2223 conductors.

---

## Progress log

Development history is tracked in [`docs/PROGRESS.md`](docs/PROGRESS.md).

---

## References

- Marchevsky, M. & Prestemon, S. "Quench detection and protection of HTS
  conductors for high-field magnets," *Supercond. Sci. Technol.* **37**
  035012 (2024). [DOI: 10.1088/1361-6668/ad20fe](https://doi.org/10.1088/1361-6668/ad20fe)
