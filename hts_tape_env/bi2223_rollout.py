"""
bi2223_rollout.py

Trajectory generator for physics_bi2223.py, mirroring rollout.py's
QuenchDetectorEnv rollouts but for the Bi-2223 conductor -- there is no
Gymnasium env for Bi-2223 (yet; physics_bi2223.py is solver-only, see its
module docstring for why: no Cu/SS current-redistribution model, no
heat-capacity/time-margin model, and only validated up to T0~80-81 K), so
this is a plain function rather than an env.reset()/step() loop.

Each row mirrors rollout.py's column names where a Bi-2223 analog exists
(T0_K, I_true_A, voltage_true_V, T_surface_true_K -- i.e. seq_data.
TRUE_FEATURES exactly, so a REBCO-trained sequence surrogate can be run
against these trajectories with zero adapter code) plus Istar_A/Tstar_K/
margin_norm as labels. No noise model (no *_A/*_V/*_K "observed" pair the
way rollout.py has both I_A and I_true_A as distinct noisy/noiseless
channels) -- these are already noiseless, so I_A==I_true_A etc, matching
fem_data.py's convention for the same reason.

Typical use
-----------
    from hts_tape_env.bi2223_rollout import generate_bi2223_grid_dataset
    df = generate_bi2223_grid_dataset(T0_grid=np.linspace(77.16, 80.5, 12))
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from hts_tape_env.physics_bi2223 import solve_runaway, stable_surface_T, tape_voltage


def rollout_bi2223_episode(T0: float, *, ramp_rate: float = 0.01,
                            I_start_frac: float = 0.30, I_max_frac: float = 1.05,
                            episode_id: int = 0) -> list[dict]:
    """
    Ramp I from I_start_frac*I* to I_max_frac*I* in ramp_rate*I* steps at
    fixed T0, recording (I, V, T_surface) at each step via the Bi-2223
    equilibrium solvers -- the Bi-2223 analog of rollout.rollout_episode's
    always-continue trajectory.

    I_max_frac defaults to 1.05 (not physics.py's 1.10): stable_surface_T's
    brentq root can fail right at I* given how tightly the reconstructed
    Ic(T)/n(T) anchors bend near the boundary (see physics_bi2223.py's
    docstring) -- rows where it fails are simply not appended, so an episode
    may end a little short of I_max_frac rather than raise.
    """
    Istar, Tstar = solve_runaway(T0)
    if not np.isfinite(Istar) or Istar <= 0:
        return []

    rows = []
    frac = I_start_frac
    step = 0
    while frac <= I_max_frac:
        I = frac * Istar
        Tsurf = stable_surface_T(I, T0, T_hi=Tstar)
        if not np.isfinite(Tsurf):
            frac += ramp_rate
            continue
        V = tape_voltage(I, min(Tsurf, Tstar))
        rows.append(dict(
            episode_id=episode_id, step=step, T0_K=T0,
            I_A=float(I), I_true_A=float(I),
            voltage_V=float(V), voltage_true_V=float(V),
            T_surface_K=float(Tsurf), T_surface_true_K=float(Tsurf),
            Istar_A=Istar, Tstar_K=Tstar,
            margin_norm=(Istar - I) / Istar,
            runaway=int(I >= Istar),
        ))
        step += 1
        frac += ramp_rate
    return rows


def generate_bi2223_grid_dataset(T0_grid: np.ndarray) -> pd.DataFrame:
    """One rollout_bi2223_episode() per T0 in T0_grid, concatenated with a
    fresh episode_id per grid point. Points where solve_runaway finds no
    interior maximum (outside physics_bi2223's validated range -- see its
    docstring) are silently skipped."""
    rows = []
    for episode_id, T0 in enumerate(T0_grid):
        rows.extend(rollout_bi2223_episode(float(T0), episode_id=episode_id))
    return pd.DataFrame(rows)
