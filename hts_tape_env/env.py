"""
env.py

Gymnasium environment for HTS magnet quench detection.

The agent observes four live instrument channels — (T0, I, V, T_surface) —
and must decide at each current step whether to trigger a protective dump or
continue ramping. The physics engine is the analytic heat-balance solver from
Marchevsky & Prestemon (2024); the surrogate trained in the notebook can later
replace the per-step brentq calls via env.set_surrogate().

Episode structure
-----------------
  reset() : sample (T0, lambda) → solve for I*, T*
  step(0) : continue ramping (I += ramp_rate * I*)
  step(1) : protective dump → episode ends

Observation (Box, float32)
--------------------------
  [T0_K, I_A, voltage_V, T_surface_K]  — the paper's four live instruments

Action (Discrete)
-----------------
  0 : continue
  1 : dump

Reward
------
  Correct dump (0 < margin < dump_threshold) : +reward_hit  (scales with lead time)
  False alarm  (margin >= dump_threshold)     : reward_false_alarm
  Missed quench (margin <= 0, no dump)        : reward_missed
  Safe step (continue, margin > 0)            : reward_step

Info keys
---------
  margin, I_star, T_star, T0, lam, I_frac, runaway (bool)
"""

from __future__ import annotations

import numpy as np
import gymnasium

from hts_tape_env.physics import (
    solve_runaway,
    stable_surface_T,
    tape_voltage,
)


class QuenchDetectorEnv(gymnasium.Env):
    """
    Single-conductor quench-detection environment.

    Parameters
    ----------
    n : float
        Power-law n-value for the conductor.
    Ic0 : float
        Critical current at 0 K [A].
    Tc : float
        Critical temperature [K].
    mode : {'stab', 'no_stab'}
        Whether the conductor has a Cu/SS stabilizer layer.
    T0_range : (float, float)
        Uniform sampling range for bath temperature [K].
    lam_range : (float, float)
        Uniform sampling range for cooling lambda [W/m/K].
    ramp_rate : float
        Current increment per step as a fraction of I*.
    I_start_frac : float
        Starting I/I* at episode onset.
    I_max_frac : float
        I/I* at which the episode truncates (no quench reached).
    noise_std : dict
        Gaussian noise standard deviations: keys 'I' [A], 'V' [V], 'T' [K].
    dump_threshold : float
        margin below which a dump is considered "timely" (paper uses ~0.05).
    reward_hit : float
        Base reward for a timely dump (scaled by lead time).
    reward_false_alarm : float
        Reward for dumping too early.
    reward_missed : float
        Reward for missing a quench.
    reward_step : float
        Per-step reward for safe continuation.
    n_branch : int
        Resolution for the runaway solver (lower = faster, less accurate).
    """

    metadata: dict = {"render_modes": []}

    def __init__(
        self,
        n: float = 20.0,
        Ic0: float = 500.0,
        Tc: float = 85.0,
        mode: str = "stab",
        T0_range: tuple[float, float] = (5.0, 75.0),
        lam_range: tuple[float, float] = (20.0, 200.0),
        ramp_rate: float = 0.005,
        I_start_frac: float = 0.30,
        I_max_frac: float = 1.10,
        noise_std: dict | None = None,
        dump_threshold: float = 0.05,
        reward_hit: float = 10.0,
        reward_false_alarm: float = -2.0,
        reward_missed: float = -50.0,
        reward_step: float = 0.01,
        n_branch: int = 150,
    ):
        super().__init__()

        self.n = n
        self.Ic0 = Ic0
        self.Tc = Tc
        self.mode = mode
        self.T0_range = T0_range
        self.lam_range = lam_range
        self.ramp_rate = ramp_rate
        self.I_start_frac = I_start_frac
        self.I_max_frac = I_max_frac
        self.noise_std = noise_std or {"I": 0.5, "V": 1e-6, "T": 0.05}
        self.dump_threshold = dump_threshold
        self.reward_hit = reward_hit
        self.reward_false_alarm = reward_false_alarm
        self.reward_missed = reward_missed
        self.reward_step = reward_step
        self.n_branch = n_branch

        # Optional surrogate: callable (obs_array) -> I_star, replaces per-step brentq
        self._surrogate = None

        # Observation: [T0, I, V, T_surface]
        obs_lo = np.array([0.0,   0.0, 0.0,   0.0], dtype=np.float32)
        obs_hi = np.array([200.0, 2000.0, 10.0, 300.0], dtype=np.float32)
        self.observation_space = gymnasium.spaces.Box(obs_lo, obs_hi, dtype=np.float32)
        self.action_space = gymnasium.spaces.Discrete(2)

        # Episode state (set in reset)
        self._T0: float = 0.0
        self._lam: float = 0.0
        self._I_star: float = 0.0
        self._T_star: float = 0.0
        self._I_frac: float = 0.0
        self._last_obs: np.ndarray = np.zeros(4, dtype=np.float32)
        self._rng = np.random.default_rng()

    # ── Public API ─────────────────────────────────────────────────────────────

    def set_surrogate(self, surrogate):
        """
        Attach a trained surrogate to replace per-step brentq in _get_obs().

        surrogate must implement predict(X: np.ndarray) -> np.ndarray,
        where X has columns [T0, I, V, T_surface] and output is I_star [A].
        """
        self._surrogate = surrogate

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Sample a (T0, lam) pair with a valid runaway point
        for _ in range(50):
            T0 = float(self._rng.uniform(*self.T0_range))
            lam = float(self._rng.uniform(*self.lam_range))
            I_star, T_star = solve_runaway(
                T0, self.n, mode=self.mode, lam_override=lam,
                Ic0=self.Ic0, Tc=self.Tc, n_branch=self.n_branch,
            )
            if np.isfinite(I_star) and I_star > 0:
                break
        else:
            # Fallback to a known-good operating point
            T0, lam = 40.0, 80.0
            I_star, T_star = solve_runaway(
                T0, self.n, mode=self.mode, lam_override=lam,
                Ic0=self.Ic0, Tc=self.Tc, n_branch=self.n_branch,
            )

        self._T0 = T0
        self._lam = lam
        self._I_star = float(I_star)
        self._T_star = float(T_star)
        self._I_frac = self.I_start_frac

        obs = self._get_obs()
        self._last_obs = obs
        return obs, self._get_info()

    def step(self, action: int):
        self._I_frac += self.ramp_rate
        margin = 1.0 - self._I_frac

        obs = self._get_obs()
        self._last_obs = obs
        info = self._get_info()

        terminated = False
        truncated = False

        if action == 1:  # protective dump
            terminated = True
            if margin <= 0.0:
                # Already past runaway when dump triggered
                reward = self.reward_missed
            elif margin < self.dump_threshold:
                # Timely dump — bonus scales with lead time (earlier = better)
                lead = (self.dump_threshold - margin) / self.dump_threshold
                reward = self.reward_hit * (1.0 + lead)
            else:
                reward = self.reward_false_alarm

        elif margin <= 0.0:
            # Runaway reached without a dump
            terminated = True
            reward = self.reward_missed

        elif self._I_frac >= self.I_max_frac:
            # Ramp completed safely — episode truncates
            truncated = True
            reward = self.reward_step

        else:
            reward = self.reward_step

        return obs, float(reward), terminated, truncated, info

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        I = self._I_frac * self._I_star
        T_surf, V = self._physics_obs(I)

        noise = self._rng.standard_normal(3)
        I_obs = float(I)      + noise[0] * self.noise_std.get("I", 0.5)
        V_obs = float(V)      + noise[1] * self.noise_std.get("V", 1e-6)
        T_obs = float(T_surf) + noise[2] * self.noise_std.get("T", 0.05)

        obs = np.array([self._T0, I_obs, V_obs, T_obs], dtype=np.float32)
        return np.clip(obs, self.observation_space.low, self.observation_space.high)

    def _physics_obs(self, I: float) -> tuple[float, float]:
        """Compute (T_surface, voltage) for current I."""
        T_surf = stable_surface_T(
            I, self._T0, self.n, mode=self.mode,
            lam_override=self._lam, Ic0=self.Ic0, Tc=self.Tc,
        )
        if not np.isfinite(T_surf):
            T_surf = self._T0

        V = tape_voltage(I, T_surf, self.n, mode=self.mode,
                         Ic0=self.Ic0, Tc=self.Tc)
        if not np.isfinite(V):
            V = 0.0

        return float(T_surf), float(V)

    def _get_info(self) -> dict:
        return {
            "margin":  float(1.0 - self._I_frac),
            "I_star":  self._I_star,
            "T_star":  self._T_star,
            "T0":      self._T0,
            "lam":     self._lam,
            "I_frac":  self._I_frac,
            "runaway": (1.0 - self._I_frac) <= 0.0,
        }
