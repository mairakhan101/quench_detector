"""
rollout_bi2223_env.py

Generate labelled trajectory datasets by rolling out Bi2223QuenchDetectorEnv,
mirroring rollout.py for the Bi-2223 conductor. Each row is one env step,
tagged with an episode_id.

Column schema matches rollout.py minus the `lam` column (Bi-2223 cooling is
a measured constant, not a swept parameter).

Typical use
-----------
    from hts_tape_env.rollout_bi2223_env import generate_grid_dataset
    import numpy as np
    df = generate_grid_dataset(np.linspace(77.5, 80.5, 25), seed=42)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from hts_tape_env.env_bi2223 import Bi2223QuenchDetectorEnv


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
        Istar_A=info["I_star"],
        Tstar_K=info["T_star"],
        margin_norm=info["margin"],
        runaway=int(info["runaway"]),
        time_margin_ms=info["time_margin_ms"],
    )


def rollout_episode(env: Bi2223QuenchDetectorEnv, episode_id: int, *,
                     obs=None, info=None, action_fn=None) -> list[dict]:
    """Roll out one already-reset episode to termination/truncation."""
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
    return rows


def generate_rollout_dataset(
    n_episodes: int = 1500,
    *,
    env_kwargs: dict | None = None,
    seed: int = 7,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Roll out n_episodes of Bi2223QuenchDetectorEnv with an always-continue
    policy, each with a freshly-sampled T0.
    """
    env = Bi2223QuenchDetectorEnv(**(env_kwargs or {}))

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
    *,
    env_kwargs: dict | None = None,
    seed: int = 0,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Like generate_rollout_dataset, but at exact T0 grid points instead of
    random samples. episode_id encodes the T0_idx.
    """
    env = Bi2223QuenchDetectorEnv(**(env_kwargs or {}))

    rows = []
    ep = 0
    for i, T0 in enumerate(T0_grid):
        if verbose and ep and ep % 10 == 0:
            print(f"  grid point {ep}/{len(T0_grid)}")
        obs, info = env.reset_at(float(T0), seed=seed + ep)
        if not np.isfinite(info["I_star"]) or info["I_star"] <= 0:
            ep += 1
            continue
        ep_rows = rollout_episode(env, ep, obs=obs, info=info)
        for r in ep_rows:
            r["T0_idx"] = i
        rows.extend(ep_rows)
        ep += 1

    df = pd.DataFrame(rows)
    if verbose:
        print(f"Generated {len(df)} steps over {df['episode_id'].nunique()} grid episodes "
              f"({len(T0_grid) - df['episode_id'].nunique()} skipped: no interior runaway).")
    return df
