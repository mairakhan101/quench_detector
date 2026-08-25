"""
fem_env.py

A Gymnasium quench-detection environment whose "physics" is a *replay* of the
FEM sweep in fem_data.py, rather than the analytic single-conductor solver in
physics.py. Where QuenchDetectorEnv observes one lumped [T0, I, V, T_surface]
point, FemReplayQuenchEnv observes the real distributed sensor layout -- a
"grid" along the tape's length: I0 plus every RTD and voltage-tap channel,
ordered by physical position. There is no PDE/boundary-condition solve here:
each step just indexes into the FEM solver's own precomputed trajectory, so
the (already-correct) cooling/current-lead boundary conditions baked into
sweep.h5 come along for free.

Caveat: the sweep has exactly 6 runs (ratio = 0.5, 0.7, 0.8, 0.9, 0.95, 0.99),
so reset() is choosing among 6 fixed trajectories (+ optional sensor noise
for variety), not sampling a continuous scenario space like
QuenchDetectorEnv.reset() does. Treat this as a validation/spatial-prototyping
env, not (yet) a large-scale RL training env -- see docs/PROGRESS.md.

Action/info contract matches QuenchDetectorEnv (Discrete(2): 0=continue,
1=dump; info["margin"]/info["runaway"]; env.dump_threshold) so
hts_tape_env.controllers.{oracle_controller, evaluate} work against this env
unmodified.

Observation (Box, float32), length 1 + 16 + 20 = 37:
    [I0_A, RTD1_1..RTD1_8, RTD2_1..RTD2_8, V1_1..V1_10, V2_1..V2_10]
RTD/V-tap ordering is fixed by x-position in the reference (r0p5) run's
sensor table (see .rtd_order / .vtap_order / .rtd_x_m / .vtap_x_m) so the
observation layout is identical across episodes even though positions shift
slightly run-to-run in the raw FEM mesh.

"margin" is defined as (QUENCH_T_K - Tmax) / QUENCH_T_K -- distance to the
300 K damage threshold (thermal.py's convention), NOT the analytic env's
(I* - I)/I*. It is a different physical quantity that happens to share the
same "positive=safe, crosses zero at the bad event" shape, which is what the
reward function and controllers.py actually rely on.
"""

from __future__ import annotations

import numpy as np
import gymnasium

from hts_tape_env.fem_data import (
    DEFAULT_PATH,
    QUENCH_T_K,
    RTD_CHANNELS,
    VTAP_CHANNELS,
    load_fem_positions,
    load_fem_sweep,
)


class FemReplayQuenchEnv(gymnasium.Env):
    """Quench-detection env that replays FEM sweep trajectories with a real
    RTD + voltage-tap sensor layout as the observation."""

    metadata: dict = {"render_modes": []}

    def __init__(
        self,
        path: str = DEFAULT_PATH,
        noise_std: dict | None = None,
        dump_threshold: float = 0.05,
        reward_hit: float = 10.0,
        reward_false_alarm: float = -2.0,
        reward_missed: float = -50.0,
        reward_step: float = 0.01,
        quench_T_K: float = QUENCH_T_K,
    ):
        super().__init__()

        self.noise_std = noise_std or {"I": 0.5, "RTD": 0.05, "V": 1e-6}
        self.dump_threshold = dump_threshold
        self.reward_hit = reward_hit
        self.reward_false_alarm = reward_false_alarm
        self.reward_missed = reward_missed
        self.reward_step = reward_step
        self.quench_T_K = quench_T_K

        self._episodes = self._load_episodes(path)
        self.n_episodes = len(self._episodes)

        n_ch = 1 + len(RTD_CHANNELS) + len(VTAP_CHANNELS)
        obs_lo = np.zeros(n_ch, dtype=np.float32)
        obs_hi = np.concatenate([[500.0], np.full(len(RTD_CHANNELS), 1000.0),
                                  np.full(len(VTAP_CHANNELS), 10.0)]).astype(np.float32)
        self.observation_space = gymnasium.spaces.Box(obs_lo, obs_hi, dtype=np.float32)
        self.action_space = gymnasium.spaces.Discrete(2)

        self._rng = np.random.default_rng()
        self._ep = None       # current episode's arrays (dict)
        self._idx = 0         # current step index within the episode
        self._true_I = 0.0
        self._true_rtd = np.zeros(len(RTD_CHANNELS), dtype=np.float32)
        self._true_vtap = np.zeros(len(VTAP_CHANNELS), dtype=np.float32)

    # ── Setup ─────────────────────────────────────────────────────────────

    def _load_episodes(self, path: str) -> list[dict]:
        df = load_fem_sweep(path)
        pos = load_fem_positions(path)

        # Fix channel ordering (by x-position) once, from the lowest-ratio
        # run -- positions shift slightly run-to-run, but this keeps the
        # observation layout identical across episodes.
        ref = pos[pos["ratio_label"] == df["ratio_label"].iloc[0]]
        rtd_pos = ref[ref["channel"].isin(RTD_CHANNELS)].set_index("channel")
        vtap_pos = ref[ref["channel"].isin(VTAP_CHANNELS)].set_index("channel")
        self.rtd_order = sorted(RTD_CHANNELS, key=lambda c: rtd_pos.loc[c, "x_m"])
        self.vtap_order = sorted(VTAP_CHANNELS, key=lambda c: vtap_pos.loc[c, "x_m"])
        self.rtd_x_m = np.array([rtd_pos.loc[c, "x_m"] for c in self.rtd_order])
        self.vtap_x_m = np.array([vtap_pos.loc[c, "x_m"] for c in self.vtap_order])

        episodes = []
        for episode_id, g in df.groupby("episode_id", sort=True):
            g = g.sort_values("step")
            episodes.append(dict(
                ratio_label=g["ratio_label"].iloc[0],
                ratio=float(g["ratio"].iloc[0]),
                status=g["status"].iloc[0],
                t_s=g["t_s"].to_numpy(dtype=np.float32),
                I_A=g["I_A"].to_numpy(dtype=np.float32),
                Tmax=g["T_surface_K"].to_numpy(dtype=np.float32),
                rtd=g[self.rtd_order].to_numpy(dtype=np.float32),
                vtap=g[self.vtap_order].to_numpy(dtype=np.float32),
                quench_step=g["quench_step"].iloc[0],
                n_steps=len(g),
            ))
        return episodes

    # ── Gymnasium API ────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        episode_id = None
        if options is not None and "ratio_label" in options:
            episode_id = next(i for i, ep in enumerate(self._episodes)
                               if ep["ratio_label"] == options["ratio_label"])
        return self._begin_episode(episode_id)

    def reset_at(self, ratio_label: str, seed=None):
        """Deterministic reset onto one specific FEM run (e.g. 'r0p95'), bypassing
        random episode sampling -- mirrors QuenchDetectorEnv.reset_at()."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        episode_id = next(i for i, ep in enumerate(self._episodes)
                           if ep["ratio_label"] == ratio_label)
        return self._begin_episode(episode_id)

    def _begin_episode(self, episode_id: int | None):
        if episode_id is None:
            episode_id = int(self._rng.integers(self.n_episodes))
        self._ep = self._episodes[episode_id]
        self._idx = 0

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action: int):
        self._idx = min(self._idx + 1, self._ep["n_steps"] - 1)
        at_end = self._idx >= self._ep["n_steps"] - 1

        obs = self._get_obs()
        info = self._get_info()
        margin = info["margin"]

        terminated = False
        truncated = False

        if action == 1:  # protective dump
            terminated = True
            if margin <= 0.0:
                reward = self.reward_missed
            elif margin < self.dump_threshold:
                lead = (self.dump_threshold - margin) / self.dump_threshold
                reward = self.reward_hit * (1.0 + lead)
            else:
                reward = self.reward_false_alarm

        elif margin <= 0.0:
            terminated = True
            reward = self.reward_missed

        elif at_end:
            truncated = True
            reward = self.reward_step

        else:
            reward = self.reward_step

        return obs, float(reward), terminated, truncated, info

    # ── Internal helpers ─────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        ep, i = self._ep, self._idx
        self._true_I = float(ep["I_A"][i])
        self._true_rtd = ep["rtd"][i].copy()
        self._true_vtap = ep["vtap"][i].copy()

        noise = self._rng.standard_normal(1 + len(self.rtd_order) + len(self.vtap_order))
        I_obs = self._true_I + noise[0] * self.noise_std.get("I", 0.5)
        rtd_obs = self._true_rtd + noise[1:1 + len(self.rtd_order)] * self.noise_std.get("RTD", 0.05)
        vtap_obs = self._true_vtap + noise[1 + len(self.rtd_order):] * self.noise_std.get("V", 1e-6)

        obs = np.concatenate([[I_obs], rtd_obs, vtap_obs]).astype(np.float32)
        return np.clip(obs, self.observation_space.low, self.observation_space.high)

    def _get_info(self) -> dict:
        ep, i = self._ep, self._idx
        Tmax = float(ep["Tmax"][i])
        margin = (self.quench_T_K - Tmax) / self.quench_T_K
        return dict(
            margin=margin,
            runaway=margin <= 0.0,
            T0=float(ep["Tmax"][0]),
            ratio=ep["ratio"],
            ratio_label=ep["ratio_label"],
            status=ep["status"],
            step=i,
            t_s=float(ep["t_s"][i]),
            quench_step=ep["quench_step"],
            I_true_A=self._true_I,
            T_surface_true_K=Tmax,
            RTD_true=self._true_rtd,
            Vtap_true=self._true_vtap,
        )
