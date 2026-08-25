"""
rollout.py

Generate labelled *trajectory* datasets by rolling out QuenchDetectorEnv itself,
rather than sweeping the analytic solver directly (datagen.py). Each row is one
environment step, tagged with an episode_id, so the same table trains sequence
models (see seq_surrogates.py) on exactly the noisy, ramped telemetry the env
feeds a policy at deployment time -- not an i.i.d. sweep over the input space.

Columns
-------
    episode_id      int    groups rows into trajectories
    step            int    0-indexed step within the episode
    T0_K, I_A, voltage_V, T_surface_K   noisy observation (env's obs vector)
    I_true_A, voltage_true_V, T_surface_true_K   noiseless channels (env's info)
                    -- this is what seq_data.TRUE_FEATURES trains the sequence
                    surrogate against, for now (see env.set_surrogate())
    lam             cooling lambda for this episode [W/m/K]        (label)
    Istar_A         runaway current for this episode [A]           (label, constant per episode)
    Tstar_K         runaway temperature [K]                        (label, constant per episode)
    time_margin_ms  adiabatic time from (I*,T*) to 300 K [ms]      (label, constant per episode;
                    the paper's Fig. 4 quantity -- NaN if not reached within thermal.time_margin_to_safe's
                    t_max, which happens for some low-T0/low-Istar cells)
    margin_norm     (I* - I_true) / I*  using the *true* (noiseless) I  (label)
    runaway         1 on/after the step where margin_norm <= 0

These three "label" columns are the three runaway *surfaces* from Marchevsky &
Prestemon (2024): I*(T0,lambda) (Fig. 2/3), the runaway temperature T*/deltaT
(also Fig. 2/3), and the adiabatic time margin (Fig. 4). All three are
constant within an episode and available as sequence-model targets (see
seq_surrogates.py) -- not just I*.

Typical use
-----------
    from hts_tape_env.rollout import generate_rollout_dataset
    df = generate_rollout_dataset(n_episodes=1500, seed=7)
    df.to_parquet("data/rollout_main.parquet")
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from hts_tape_env.env import QuenchDetectorEnv
from hts_tape_env.thermal import time_margin_to_safe


def _row(episode_id: int, step: int, obs: np.ndarray, info: dict) -> dict:
    return dict(
        episode_id=episode_id,
        step=step,
        T0_K=float(obs[0]),
        I_A=float(obs[1]),
        voltage_V=float(obs[2]),
        T_surface_K=float(obs[3]),
        I_true_A=info["I_true_A"],
        voltage_true_V=info["voltage_true_V"],
        T_surface_true_K=info["T_surface_true_K"],
        lam=info["lam"],
        Istar_A=info["I_star"],
        Tstar_K=info["T_star"],
        margin_norm=info["margin"],
        runaway=int(info["runaway"]),
    )


def rollout_episode(env: QuenchDetectorEnv, episode_id: int, *,
                     obs=None, info=None, action_fn=None) -> list[dict]:
    """
    Roll out one already-reset episode to termination/truncation, recording
    every step. `obs`/`info` are the (obs, info) pair from the reset() (or
    reset_at()) call that started the episode; `action_fn(obs, info) -> int`
    defaults to always-continue (action 0), which is what we want here: full
    trajectories that cross the runaway boundary, not ones cut short by a
    dump decision.
    """
    if action_fn is None:
        action_fn = lambda obs, info: 0  # noqa: E731

    rows = []
    step = 0
    terminated = truncated = False
    while True:
        rows.append(_row(episode_id, step, obs, info))
        if terminated or truncated:
            break
        action = action_fn(obs, info)
        obs, reward, terminated, truncated, info = env.step(action)
        step += 1

    # I*/T* are constant within an episode, so time_margin is too -- compute
    # once and broadcast rather than re-solving it every step.
    tm_s = time_margin_to_safe(rows[0]["Istar_A"], rows[0]["Tstar_K"])
    tm_ms = float(tm_s * 1000.0) if np.isfinite(tm_s) else np.nan
    for r in rows:
        r["time_margin_ms"] = tm_ms
    return rows


def generate_rollout_dataset(
    n_episodes: int = 1500,
    *,
    env_kwargs: dict | None = None,
    seed: int = 7,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Roll out `n_episodes` episodes of QuenchDetectorEnv with an always-continue
    policy, each with a freshly-sampled (T0, lambda), recording every step's
    noisy observation plus the episode's ground-truth labels.
    """
    env = QuenchDetectorEnv(**(env_kwargs or {}))

    rows = []
    n_runaway = 0
    for ep in range(n_episodes):
        if verbose and ep and ep % 200 == 0:
            print(f"  rolled out {ep}/{n_episodes} episodes, {len(rows)} steps so far")

        obs, info = env.reset(seed=seed + ep)
        ep_rows = rollout_episode(env, ep, obs=obs, info=info)
        rows.extend(ep_rows)
        n_runaway += int(ep_rows[-1]["runaway"])

    df = pd.DataFrame(rows)
    if verbose:
        print(f"Generated {len(df)} steps over {n_episodes} episodes "
              f"({n_runaway}/{n_episodes} reached runaway = {n_runaway / n_episodes:.0%}).")
    return df


def generate_grid_dataset(
    T0_grid: np.ndarray,
    lam_grid: np.ndarray,
    *,
    env_kwargs: dict | None = None,
    seed: int = 0,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Like generate_rollout_dataset, but at exact (T0, lambda) grid points
    instead of random samples -- for evaluating a trained surrogate against a
    specific slice of the analytic runaway surface (e.g. reproducing Fig. 3a).

    episode_id encodes the flattened (T0_idx, lam_idx) grid position.
    """
    env = QuenchDetectorEnv(**(env_kwargs or {}))

    rows = []
    ep = 0
    n_total = len(T0_grid) * len(lam_grid)
    for i, T0 in enumerate(T0_grid):
        for j, lam in enumerate(lam_grid):
            if verbose and ep and ep % 100 == 0:
                print(f"  grid point {ep}/{n_total}")
            obs, info = env.reset_at(float(T0), float(lam), seed=seed + ep)
            if not np.isfinite(info["I_star"]) or info["I_star"] <= 0:
                ep += 1
                continue
            ep_rows = rollout_episode(env, ep, obs=obs, info=info)
            for r in ep_rows:
                r["T0_idx"], r["lam_idx"] = i, j
            rows.extend(ep_rows)
            ep += 1

    df = pd.DataFrame(rows)
    if verbose:
        print(f"Generated {len(df)} steps over {df['episode_id'].nunique()} grid episodes "
              f"({n_total - df['episode_id'].nunique()} skipped: no interior runaway).")
    return df
