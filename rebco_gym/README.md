# REBCO-Gym

Farama Gymnasium environments for reinforcement learning-based quench protection and current-ramp control in REBCO (Rare-Earth Barium Copper Oxide) high-temperature superconducting tapes.

This package is a physics-approximation testbed, not a certified conductor model. Every module below states what it actually solves and what it deliberately simplifies — read the assumptions before trusting a number out of it for anything beyond RL training dynamics.

## Quick Start

```bash
cd rebco_gym
pip install -e .                    # core deps: gymnasium, numpy, scipy
pip install -e ".[train]"           # + stable-baselines3, torch
pip install -e ".[viz]"             # + matplotlib

python run_demo.py                  # sanity-checks all 3 envs + runs heuristic-agent demos
```

`run_demo.py` does three things, in order:
1. Runs `gymnasium.utils.env_checker.check_env` against all three registered envs (catches API/shape bugs).
2. Runs a hand-written heuristic policy through each env for a few episodes and prints an ASCII render + episode outcome (safe shutdown / quench / timeout).
3. Exits nonzero if anything raises — safe to use as a CI smoke test.

Minimal usage of any env:

```python
import gymnasium as gym
import rebco_gym  # registers ThermalRunawayEnv-v0, CurrentRedistEnv-v0, REBCOFullEnv-v0

env = gym.make("REBCOFullEnv-v0")
obs, info = env.reset(seed=0)
for _ in range(1000):
    action = env.action_space.sample()          # replace with your policy / RL agent
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
env.close()
```

To train with `stable-baselines3` (after `pip install -e ".[train]"`):

```python
from stable_baselines3 import PPO
import gymnasium as gym
import rebco_gym

env = gym.make("REBCOFullEnv-v0")
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=200_000)
```

Each env also has kwargs for every physics/episode/reward parameter shown in this README (e.g. `gym.make("REBCOFullEnv-v0", B_ext_range=(0.0, 0.3), dropout_severity_range=(1.0, 2.0))`) — see each env's `__init__` signature for the full list.

## Repository Layout

```
rebco_gym/
├── __init__.py                      # gym.register() calls for all 3 envs
├── run_demo.py                      # API check + heuristic-agent demos for all 3 envs
├── physics/                         # pure numpy/scipy, no gym dependency — usable standalone
│   ├── thermal.py                   # Marchevsky & Prestemon thermal runaway model (0-D, single tape)
│   ├── quench_antenna.py            # QA voltage response, complex-potential dipole model
│   ├── current_redistribution.py    # 2D cross-section resistor-mesh (COMSOL-style, across width)
│   ├── electromagnetic.py           # Self/external B-field, Jc(B,θ) anisotropy, Ic dropout sampling
│   └── spatial_tape.py              # Segmented tape along length, EM-coupled local Ic
└── envs/
    ├── thermal_runaway_env.py       # ThermalRunawayEnv-v0
    ├── current_redist_env.py        # CurrentRedistEnv-v0
    └── full_env.py                  # REBCOFullEnv-v0
```

Three physics resolution axes are in play and it matters which one each module resolves:
- `thermal.py` — **0-D**, single lumped node, no spatial resolution at all.
- `current_redistribution.py` — resolves across tape **width** (x) and through-**thickness** (7 material layers, y); each (x,y) node has one lumped temperature.
- `spatial_tape.py` — resolves along tape **length** (z); each z-segment is a lumped 0-D node internally (no width/thickness resolution).

Nothing in this package resolves all three axes at once (width × thickness × length) — that would be the COMSOL model these are approximating.

## Physics Models & Assumptions

### 1. Thermal runaway model — `physics/thermal.py`

Implements the analytic criterion from Marchevsky & Prestemon (2024): a lumped-node energy balance `C dT/dt = Q_in(I,T) - Q_out(T)`. Below a **runaway current I\***, a stable equilibrium `T*` exists where Joule heating balances cooling; above it, no equilibrium exists and the conductor runs away. This is the same physics `mdp_surr_tests.ipynb` validates against Bi-2223 data (paper §experimental, Fig. 7); here it's parameterized with REBCO-typical defaults (`Ic0=500 A`, `Tc=85 K`).

- `Ic(T) = Ic0 · (1 − (T/Tc)^1.7)` — paper Eq. 5, a simple power-law temperature dependence, **no field dependence** (that's what `electromagnetic.py` adds separately, see below — the two aren't wired together in this module).
- **Case 1** (no stabilizer): power-law `E(J) = U0·(J/Jc)^n` with linear cooling `Q_out = η(T-T0)`.
- **Case 2** (with Cu/SS stabilizer): current sharing between SC and a parallel Cu+SS stabilizer, solved via bisection for the SC-branch current `Ix` (paper Eqs. 7–8). `ThermalRunawayEnv-v0` always uses Case 2 (`stabilized=True`).
- `η` (linear cooling coefficient) comes from a brass thermal-conductivity NIST polynomial fit (`lambda_brass`), scaled by geometry (`eta_from_geometry`).
- `find_runaway_current()` locates I\* by bisecting on whether a stable equilibrium `T*` exists in `(T0, Tc)` — this is the iterative solve `mdp_surr_tests.ipynb`'s surrogate model exists to replace; it is *not* used inside the gym envs' hot loop (too slow per-step), only as a standalone analysis/validation tool.

**Known assumptions / caveats:**
- **Sign discrepancy vs. the notebook's corrected fit.** `lambda_brass`'s polynomial coefficient `g = -0.174057` here matches the *paper's* published value. `mdp_surr_tests.ipynb` explicitly identifies this as a sign error against the NIST C26000 reference data ("makes the polynomial collapse above 25 K") and uses `g = +0.174057` instead. This module was ported with the paper's original (uncorrected) sign — if you need brass λ(T) to match NIST above ~25 K, use the notebook's corrected value.
- 0-D lumped: no spatial resolution within the element — assumes uniform T across the whole conductor length modeled. `ThermalRunawayEnv-v0` is built on this, so it cannot represent a localized hot spot; that's what `spatial_tape.py`/`REBCOFullEnv-v0` are for.
- Geometry-derived cooling (`eta_from_geometry`) is a simple `2wL/d · λ` slab formula, not a real cryogenic heat-transfer correlation (no boiling curve, no forced-flow convection).

### 2. Quench antenna (QA) voltage sensor — `physics/quench_antenna.py`

Models the voltage a quench-antenna winding would pick up from a local normal-zone flux disturbance, using 2D complex-potential theory: the disturbance is a pair of closely-spaced point current sources (a dipole), and the antenna winding vertices sample the complex potential and sum with alternating signs (the standard multipole-cancellation trick real QA windings use to reject uniform background field while staying sensitive to local disturbances). Adapted from `qa_sim.py`'s MQXFA-style QA geometry.

**Known assumptions / caveats:**
- Uses a **simplified circular winding** with configurable pole count (`n_poles`), not the actual MQXFA coil vertex geometry from the `.dat` files `qa_sim.py` reads — good for qualitative sensitivity trends, not for reproducing a specific antenna's calibration.
- **Quasi-static**: no propagation delay / retarded potentials, no eddy-current shielding by nearby conductors.
- `amplitude` is a dimensionless proxy the caller supplies (in the envs, scaled by `dT/dT_crit` or by local E-field) — it is **not** derived from an actual `dI/dt` of the normal zone, so the absolute voltage scale is not physically calibrated; only relative/trend behavior should be trusted.

### 3. Cross-sectional current redistribution mesh — `physics/current_redistribution.py`

A 2D resistor-network model of the tape **cross-section** (width × through-thickness), matching COMSOL's 0.5 mm mesh discretization, with 7 material layers (Cu / Ag / REBCO / buffer / Hastelloy / Ag / Cu). Models how current redistributes across the width when part of the REBCO layer goes locally normal.

**Known assumptions / caveats:**
- **Two-stage, not fully self-consistent per step.** Each `step()` first redistributes current by a *conductance-weighted* split (`_distribute_current`), which treats REBCO resistivity as a fixed near-zero constant (`1e-14 Ω·m`) whenever `T < Tc` — i.e. this stage assumes REBCO is either "perfectly superconducting" or fully normal, with no power-law region. The power-law `E = E0·(J/Jc)^n` heating is only applied *afterward*, as a post-hoc dissipation calculation using the `J` from that linear split. This means the current split itself doesn't feel the power-law nonlinearity within a step — `spatial_tape.py`'s current-sharing solve is more physically self-consistent on this point (see below).
- Buffer layer resistivity (`1e-2 Ω·m`) is a rough "effectively insulating" placeholder, not a measured value.
- Per-column heat capacity blends a simplified `(T/150)²`-type temperature scaling capped at the 300 K value — not real layer-resolved `Cv(T)` curves (contrast with `rebco_quench_full.py`'s `cv_cu/cv_ag/cv_hast/cv_rebco`, which are linear-in-T fits and more standard).
- No longitudinal (along-length) resolution — this module is purely a width/thickness cross-section; combining it with `spatial_tape.py`'s along-length resolution (full width × thickness × length) is on the roadmap, not implemented.

### 4. Electromagnetic model — `physics/electromagnetic.py`

New in this pass. Three independent pieces:

**Self-field** — `self_field(I, width)`: edge self-field of a thin tape carrying transport current `I`, using the standard critical-state-model thin-strip approximation `B_edge ≈ μ0·I/(π·w)`. This is a scalar magnitude, not a field map — it does not distinguish near-edge vs. tape-center field, and ignores contributions from neighboring tapes/turns in a real coil.

**Anisotropic `Jc(B,θ,T)`** — `AnisotropicJcModel`: the standard "anisotropic ellipse" engineering fit used for coated-conductor `Ic(B,θ)` characterization curves:
```
B_eff = sqrt(B_perp² + (B_parallel/g)²)
Jc_fraction(B) = 1 / (1 + B_eff/B0)^alpha
```
`g > 1` makes the conductor much less sensitive to field applied parallel to the tape face (ab-plane) than perpendicular to it (c-axis) — the well-documented REBCO field-angle anisotropy. **Default parameters (`B0=0.2 T`, `alpha=0.7`, `g=5`) are representative placeholders, not a fit to any specific conductor or temperature** — override them for your tape's real Ic(B,θ) characterization data.

**Ic dropout field** — `IcDropoutField.sample()`: a *synthetic* spatially-correlated multiplicative Ic map along tape length, combining (a) a smooth Ornstein–Uhlenbeck/AR(1) background term (correlation length + std-dev knobs) representing gradual Ic-homogeneity drift, and (b) a compound-Poisson term for rare, deep, localized defects (rate + depth-range + width knobs). This is **not fit to any measured Ic-homogeneity scan** — it's a plausible-shaped stochastic process chosen so RL training sees realistic-looking weak-spot variability episode to episode, nothing more.

**Known assumptions / caveats:**
- **Parameter regime mismatch is easy to hit.** `AnisotropicJcModel`'s defaults (`B0=0.2 T`) are more representative of ~77 K behavior; the gym envs' default bath temperatures span 15–50 K, where real REBCO tolerates much higher fields before comparable Jc suppression. Left at defaults, a modest `B_ext` (~0.3–0.5 T) at a mostly-perpendicular angle can suppress Jc by >50%, which will look like an aggressively field-sensitive conductor rather than a realistic 20 K REBCO tape. This was observed empirically while validating `SpatialTape` — see the `B_ext_range`/`B_angle_range` defaults in `REBCOFullEnv-v0`, which were kept deliberately modest (`0.0–0.15 T`) to avoid this regime by default. If you widen those ranges, expect much more aggressive quenching unless you also retune `B0`/`alpha`/`g`.
- Self-field is computed from **total transport current only** — it does not vary along the tape even where local current-sharing has diverted some current into the stabilizer, and does not model field contributions from adjacent tapes/turns in a coil.
- The dropout map is **sampled once per episode reset and held fixed** — it represents a fixed manufacturing defect map, not evolving degradation/damage during the episode.

### 5. Spatially-resolved tape — `physics/spatial_tape.py`

`nz` segments along a tape of length `L`, each a lumped thermal node, coupled by longitudinal heat diffusion — physics pattern ported from the validated `SpatialTape` prototype in `test_rebco.ipynb` §1, but with the uniform `Ic0` there replaced by a per-segment value from `IcDropoutField`, and `Ic(T)` extended to `Ic(T,B,θ)` via `AnisotropicJcModel`.

Per segment: current-sharing fraction `f_rebco` relaxes toward its steady-state value `Ic(T)/I` with time constant `tau_share` (models the delay of real current transferring into the Cu stabilizer, rather than being instantaneous); E-field follows the power law `E = Ec·(I·f/Ic)^n` below `Tc`, or a resistive `I·R_cu/dx` term above it; Joule heating, surface cooling, and longitudinal diffusion (`κ·A·d²T/dx²`, Neumann boundary conditions at both tape ends) update each segment's temperature by explicit Euler.

**Known assumptions / caveats:**
- **`tau_share` is a phenomenological relaxation constant**, not a solved electromagnetic diffusion equation — it stands in for the real current-redistribution transient (which `current_redistribution.py`'s conductance-weighted split addresses differently, and neither module solves the actual transient RL-circuit / inductive diffusion problem).
- **No width or through-thickness resolution** — each segment is a single lumped node; layer-resolved effects (7-material stack, differential cooling top vs. bottom) from `current_redistribution.py` are not present here. Combining the two (a true 3D-resolved model) is on the roadmap.
- Single uniform `C_vol` (volumetric heat capacity) and `κ` (longitudinal conductivity) per tape, not layer-resolved or strongly temperature-dependent — a simplification relative to `rebco_quench_full.py`'s per-material `Cv(T)` fits.
- Self-field is uniform along the tape (same caveat as above): local current-sharing state doesn't feed back into the local B-field.
- `qa_signal()` sums per-segment QA responses weighted by local E-field normalized to the segment's own peak — a proxy for "hotter segments dominate the antenna signal," not a rigorously derived superposition of each segment's actual disturbance amplitude.

## Sensor / Observability Design

The three environments differ in how "honest" their observations are about what a real instrumentation system could measure — worth knowing before comparing results across them:

- **`ThermalRunawayEnv-v0` / `CurrentRedistEnv-v0`** expose **ground-truth physics state** directly in the observation: `Ic(T)` at the current temperature, full `Jc`/`Ic` distributions across the mesh, exact `margin = I/Ic`. A real sensor system cannot measure `Ic(T)` directly — this is the "God view" convenient for a first RL pass, but it's an easier problem than the notebook's paradigm.
- **`REBCOFullEnv-v0`** follows the four-instrument philosophy `mdp_surr_tests.ipynb` explicitly validates against (bath thermometer `T0`, current shunt `I`, voltage taps `U`, surface thermometer `T_surface`), plus the QA voltage signal already used elsewhere in this package. Ground-truth-only quantities — the per-segment `Ic0_local` dropout map, exact worst-case margin as derived from that map — are returned in `info`, not `obs`, so the agent must infer risk from realistic sensors (`worst_local_margin` in `obs` is computed from the *tape's* local `Ic`, which is a modeling convenience already available inside `step()`, not literally "measured" — treat it as an idealized margin estimate an operator's live `η`-extraction technique, per the paper's Discussion section, would only approximate).

## Environments

### `ThermalRunawayEnv-v0` — Binary Protection Decision

| | |
|---|---|
| **Action** | `Discrete(2)`: 0 = HOLD (maintain current), 1 = RAMP_DOWN |
| **Observation** | `Box(7)`: `[T, T0, I_op, Ic(T), dT/dt, V_qa, margin]` |
| **Reward** | +1/step operating, +5 safe ramp-down, −100 quench, −10 unnecessary ramp-down |
| **Termination** | Quench (T > T0 + ΔT_crit) or successful ramp-down |

Built on `ThermalRunawayModel` (Case 2, stabilized). The agent observes the thermal state and QA voltage signal, and must decide when to initiate a protective current ramp-down. Random flux disturbances inject local heating (`disturbance_prob` per step, magnitude from `disturbance_dT_range`).

```python
import gymnasium as gym
import rebco_gym

env = gym.make("ThermalRunawayEnv-v0")
obs, info = env.reset()

for _ in range(500):
    T, T0, I_op, Ic_T, dTdt, V_qa, margin = obs
    action = 1 if margin > 0.85 else 0

    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
```

### `CurrentRedistEnv-v0` — Continuous Current Control with Mesh Observability

| | |
|---|---|
| **Action** | `Box(1)`: current setpoint fraction [0, 1] |
| **Observation** | `Box(4·nx+3)`: Ic, E-field, Jc, T profiles + V_qa + I_total + T_max |
| **Reward** | Operational reward − temperature penalty − quench penalty + uniformity bonus |
| **Termination** | Quench (T_max > T0 + ΔT_crit) |

Built on `CurrentRedistributionMesh` (width × thickness cross-section). The agent observes the full COMSOL-style mesh distributions across the tape width and controls a single global current setpoint. Future versions will support per-element MOSFET current redistribution (see Roadmap).

```python
env = gym.make("CurrentRedistEnv-v0")
obs, info = env.reset()

for _ in range(1000):
    nx = env.unwrapped.nx
    T_max = obs[4*nx + 2]
    T0 = info["T0"]

    frac = max(1.0 - (T_max - T0) / 12.0, 0.1)   # proportional controller
    action = np.array([frac], dtype=np.float32)

    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
```

### `REBCOFullEnv-v0` — Ramp-Rate Control with Thermal + Electromagnetic Coupling

| | |
|---|---|
| **Action** | `Box(1)`: fraction of the requested ramp rate to actually apply [0, 1] |
| **Observation** | `Box(10)`: `[I_op, I_requested_target, T0, T_surface, V_terminal, V_qa, dT/dt, worst_local_margin, B_self, B_ext]` |
| **Reward** | + ramp progress − lag penalty − overheat penalty − quench penalty + safe-completion bonus |
| **Termination** | Quench (T_max > T0 + ΔT_crit) → truncated; ramp target reached & held → terminated |

Built on `SpatialTape` (along-length resolution, EM-coupled `Ic`). Each episode samples: a bath temperature, a *requested* ramp rate (A/s — "how fast the operator wants the magnet charged"), an external field magnitude/angle, and an `IcDropoutField` severity. The agent doesn't have to track the requested ramp at full speed — its action is how much of that rate to actually apply each step, trading charge speed for local thermal/current-sharing margin. This is the "ramp current rate is given, control parameter is the current you actually load" framing.

```python
env = gym.make("REBCOFullEnv-v0")
obs, info = env.reset()

for _ in range(1000):
    I_op, I_req, T0, T_max, V, V_qa, dTdt, margin, B_self, B_ext = obs

    frac = 0.1 if (T_max - T0 > 2.0 or margin > 3.0) else 1.0   # back off if risky
    action = np.array([frac], dtype=np.float32)

    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
```

Useful `gym.make(..., **kwargs)` overrides for exploring the physics regime (see caveats above before widening field ranges):

| kwarg | default | meaning |
|---|---|---|
| `nz` | 30 | number of length-segments in the tape |
| `Ic0`, `Tc` | 400 A, 90 K | nominal (zero-field) critical current & temperature |
| `T0_range` | (15, 50) K | sampled bath temperature per episode |
| `I_target_fraction_range` | (0.6, 0.9) | ramp target as a fraction of `Ic0` |
| `ramp_rate_range` | (500, 4000) A/s | requested charge rate per episode |
| `B_ext_range` | (0.0, 0.15) T | sampled external field magnitude |
| `B_angle_range` | (0, π/2) rad | sampled field angle (0=parallel/ab-plane, π/2=perpendicular/c-axis) |
| `dropout_severity_range` | (0.5, 1.5) | scales `IcDropoutField`'s background std & defect rate |
| `dT_crit` | 15 K | quench threshold (T_max above T0) |

## Standalone Physics Usage

The physics models have no gym dependency and can be driven directly for analysis/validation:

```python
from rebco_gym.physics import ThermalRunawayModel

model = ThermalRunawayModel(Ic0=500, Tc=85, n_value=20)
model.reset(T0=20.0, I_op=400.0)

eta = model.eta(20.0)
I_star, T_star = model.find_runaway_current(T0=20.0, eta_val=eta)
print(f"Runaway current: {I_star:.1f} A at T*={T_star:.1f} K")
```

```python
from rebco_gym.physics import CurrentRedistributionMesh

mesh = CurrentRedistributionMesh(tape_width=4e-3, dx=0.5e-3, I_total=400, T_base=20)
mesh.set_hotspot(x_center=2e-3, width=1e-3, T_hot=60)
mesh.step(dt=0.001)

print("REBCO current fraction:", mesh.get_rebco_current_fraction())
print("Max temperature:", mesh.max_temperature())
```

```python
from rebco_gym.physics import SpatialTape, SpatialTapeParams, IcDropoutField
import numpy as np

tape = SpatialTape(
    nz=40,
    params=SpatialTapeParams(Ic0=400, T_bath=20, Tc=90),
    dropout_field=IcDropoutField(background_std=0.03, defect_rate=3.0),
    B_ext=0.05, B_angle_rad=0.3,
    rng=np.random.default_rng(0),
)

for _ in range(2000):
    tape.step(I=0.8 * 400, dt=1e-4)

print("Weakest segment Ic0:", tape.Ic0_local.min(), "A (nominal 400 A)")
print("Max temperature:", tape.max_temperature(), "K")
print("Worst local I/Ic margin:", tape.min_local_margin())
```

```python
from rebco_gym.physics import self_field, AnisotropicJcModel

jc_model = AnisotropicJcModel(B0=0.2, alpha=0.7, g=5.0)
B_self = self_field(I=400.0, width=4e-3)

B_par, B_perp = jc_model.resolve_components(B_mag=0.3, theta_rad=1.2)
print("Jc suppression factor:", jc_model.Jc_fraction(B_par, B_perp))
```

## Known Limitations (consolidated)

- **`lambda_brass` uses the paper's original (uncorrected) polynomial sign**, not the notebook's NIST-corrected fit — see §1 above.
- No module resolves width × thickness × length simultaneously; `current_redistribution.py` (width/thickness) and `spatial_tape.py` (length) are complementary, not combined.
- Field/Ic-anisotropy default parameters (`B0`, `alpha`, `g` in `AnisotropicJcModel`) are engineering-fit placeholders, not calibrated to any specific conductor or temperature — easy to land in an unrealistic suppression regime if you widen field ranges without retuning them (see §4 caveat above).
- Self-field is always computed from total transport current alone — no per-segment or per-layer local field feedback, no mutual field from neighboring tapes/turns.
- The `IcDropoutField` map is a synthetic stochastic process, not fit to measured Ic-homogeneity scans, and is fixed for the duration of an episode.
- QA voltage in all three envs is a relative/qualitative proxy (see §2) — do not treat absolute `V_qa` values as calibrated volts.
- `ThermalRunawayEnv-v0`/`CurrentRedistEnv-v0` observations include ground-truth `Ic`/`Jc` values a real sensor system wouldn't have direct access to; `REBCOFullEnv-v0` is the more sensor-realistic design (see "Sensor / Observability Design" above).
- Reward shaping in all three envs is hand-tuned/heuristic, not derived from a real operating-cost model — treat the default weights as a reasonable starting point for RL, not a ground-truth objective.

## Roadmap

- [x] `ThermalRunawayEnv-v0` — binary ramp-down decision
- [x] `CurrentRedistEnv-v0` — continuous current control with mesh obs
- [x] Coupled thermal + electromagnetic environment (`REBCOFullEnv-v0`)
- [x] Field-angle-dependent `Jc(B,θ)` (anisotropic Kim-type scaling) and spatially-correlated `Ic` dropout sampling
- [ ] Per-element MOSFET action space (Marchevsky active current redistribution)
- [ ] `n(B,θ)` field dependence (currently `n_eff` is constant)
- [ ] Fix `lambda_brass` sign to match the notebook's NIST-corrected fit (or expose both as a `mode` flag)
- [ ] Combine `current_redistribution.py` (width/thickness) and `spatial_tape.py` (length) into one 3D-resolved model
- [ ] COMSOL LiveLink integration for high-fidelity mesh import
- [ ] Multi-agent: one agent per tape in a cable stack
- [ ] Calibrate `AnisotropicJcModel` and `IcDropoutField` defaults against a real conductor's Ic(B,θ) and Ic-homogeneity data

## References

1. M. Marchevsky, S. Prestemon, "Thermal runaway criterion as a basis for the protection of high-temperature superconductor magnets", Supercond. Sci. Technol. 37 (2024)
2. M. Marchevsky, S. Prestemon, "Quench protection for high-temperature superconductor cables using active control of current distribution", Supercond. Sci. Technol. 37 085026 (2024)
3. Farama Foundation, Gymnasium, https://gymnasium.farama.org
