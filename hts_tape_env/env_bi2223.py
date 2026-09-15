"""
env_bi2223.py

Gymnasium environment for Bi-2223 HTS tape quench detection.

Mirrors QuenchDetectorEnv (env.py) but uses the Bi-2223 physics from
physics_bi2223.py. Key differences from the REBCO env:

  - No `lam` parameter: cooling is a measured constant (eta_c = 4.56 W/K),
    not a swept parameter.
  - n(T) is temperature-dependent (piecewise-linear from paper Fig 6e),
    not a fixed constructor parameter.
  - Supports mode='no_stab' (default, paper's Sec 3 model) and
    mode='ag_stab' (Ag-matrix current redistribution, Phase 2).

Episode structure, observation/action spaces, and reward are identical to
QuenchDetectorEnv for surrogate architecture compatibility.
"""

from __future__ import annotations

import numpy as np
import gymnasium

from hts_tape_env.physics_bi2223 import (
    solve_runaway,
    stable_surface_T,
    tape_voltage,
    T_X_K,
    IC_ANCHOR_A,
)
from hts_tape_env.thermal_bi2223 import time_margin_to_safe


class Bi2223QuenchDetectorEnv(gymnasium.Env):
    """
    Bi-2223 single-conductor quench-detection environment.

    Parameters
    ----------
    mode : {'no_stab', 'ag_stab'}
        Physics mode. 'no_stab' uses pure power-law E-J (paper Sec 3).
        'ag_stab' adds Ag-matrix current redistribution (Phase 2).
    T0_range : (float, float)
        Uniform sampling range for bath temperature [K].
    Ic0 : float
        Critical current at the anchor temperature [A].
    eta_override : float or None
        Override the measured cooling coefficient [W/K].
    ramp_rate : float
        Current increment per step as a fraction of I*.
    I_start_frac : float
        Starting I/I* at episode onset.
    I_max_frac : float
        I/I* at which the episode truncates.
    noise_std : dict
        Gaussian noise standard deviations: keys 'I' [A], 'V' [V], 'T' [K].
    dump_threshold : float
        Margin below which a dump is considered "timely".
    """

    metadata: dict = {"render_modes": []}

    def __init__(
        self,
        mode: str = "no_stab",
        T0_range: tuple[float, float] = (77.16, 81.0),
        Ic0: float = IC_ANCHOR_A,
        eta_override: float | None = None,
        ramp_rate: float = 0.01,
        I_start_frac: float = 0.30,
        I_max_frac: float = 1.10,
        noise_std: dict | None = None,
        dump_threshold: float = 0.05,
        reward_hit: float = 10.0,
        reward_false_alarm: float = -2.0,
        reward_missed: float = -50.0,
        reward_step: float = 0.01,
        n_branch: int = 200,
    ):
        super().__init__()

        if mode not in ("no_stab", "ag_stab"):
            raise ValueError(f"mode must be 'no_stab' or 'ag_stab', got {mode!r}")
        self.mode = mode
        self.T0_range = T0_range
        self.Ic0 = Ic0
        self.eta_override = eta_override
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

        self._surrogate = None
        self._surrogate_pred: dict | None = None

        obs_lo = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        obs_hi = np.array([200.0, 2000.0, 10.0, 300.0], dtype=np.float32)
        self.observation_space = gymnasium.spaces.Box(obs_lo, obs_hi, dtype=np.float32)
        self.action_space = gymnasium.spaces.Discrete(2)

        self._T0: float = 0.0
        self._I_star: float = 0.0
        self._T_star: float = 0.0
        self._I_frac: float = 0.0
        self._last_obs: np.ndarray = np.zeros(4, dtype=np.float32)
        self._true_I: float = 0.0
        self._true_T_surface: float = 0.0
        self._true_voltage: float = 0.0
        self._time_margin_ms: float = np.nan
        self._rng = np.random.default_rng()

    def set_surrogate(self, surrogate):
        """Attach a trained surrogate (same interface as QuenchDetectorEnv)."""
        self._surrogate = surrogate
        self._surrogate_pred = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        for _ in range(50):
            T0 = float(self._rng.uniform(*self.T0_range))
            I_star, T_star = self._solve(T0)
            if np.isfinite(I_star) and I_star > 0:
                break
        else:
            T0 = 78.0
            I_star, T_star = self._solve(T0)

        return self._begin_episode(T0, I_star, T_star)

    def reset_at(self, T0: float, seed=None):
        """Deterministic reset at a specific T0 (no λ for Bi-2223)."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        I_star, T_star = self._solve(T0)
        return self._begin_episode(T0, I_star, T_star)

    def _solve(self, T0: float):
        return solve_runaway(T0, Ic0=self.Ic0, eta_override=self.eta_override,
                             n_branch=self.n_branch, mode=self.mode)

    def _begin_episode(self, T0: float, I_star: float, T_star: float):
        self._T0 = float(T0)
        self._I_star = float(I_star)
        self._T_star = float(T_star)
        self._I_frac = self.I_start_frac

        tm_s = time_margin_to_safe(I_star, T_star)
        self._time_margin_ms = float(tm_s * 1000.0) if np.isfinite(tm_s) else np.nan

        obs = self._get_obs()
        self._last_obs = obs
        if self._surrogate is not None:
            self._surrogate.reset()
            self._surrogate_pred = self._surrogate.predict_step(self._true_obs_vector())
        return obs, self._get_info()

    def step(self, action: int):
        self._I_frac += self.ramp_rate
        margin = 1.0 - self._I_frac

        obs = self._get_obs()
        self._last_obs = obs
        if self._surrogate is not None:
            self._surrogate_pred = self._surrogate.predict_step(self._true_obs_vector())
        info = self._get_info()

        terminated = False
        truncated = False

        if action == 1:
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
        elif self._I_frac >= self.I_max_frac:
            truncated = True
            reward = self.reward_step
        else:
            reward = self.reward_step

        return obs, float(reward), terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        I = self._I_frac * self._I_star
        T_surf = stable_surface_T(I, self._T0, Ic0=self.Ic0,
                                   eta_override=self.eta_override, T_hi=self._T_star,
                                   mode=self.mode)
        if not np.isfinite(T_surf):
            T_surf = self._T0
        V = tape_voltage(I, T_surf, Ic0=self.Ic0, mode=self.mode)
        if not np.isfinite(V):
            V = 0.0

        self._true_I = float(I)
        self._true_T_surface = float(T_surf)
        self._true_voltage = float(V)

        noise = self._rng.standard_normal(3)
        I_obs = float(I)      + noise[0] * self.noise_std.get("I", 0.5)
        V_obs = float(V)      + noise[1] * self.noise_std.get("V", 1e-6)
        T_obs = float(T_surf) + noise[2] * self.noise_std.get("T", 0.05)

        obs = np.array([self._T0, I_obs, V_obs, T_obs], dtype=np.float32)
        return np.clip(obs, self.observation_space.low, self.observation_space.high)

    def _true_obs_vector(self) -> np.ndarray:
        return np.array([self._T0, self._true_I, self._true_voltage, self._true_T_surface],
                         dtype=np.float32)

    def _get_info(self) -> dict:
        info = {
            "margin":  float(1.0 - self._I_frac),
            "I_star":  self._I_star,
            "T_star":  self._T_star,
            "T0":      self._T0,
            "I_frac":  self._I_frac,
            "runaway": (1.0 - self._I_frac) <= 0.0,
            "time_margin_ms": self._time_margin_ms,
            "I_true_A":         self._true_I,
            "voltage_true_V":   self._true_voltage,
            "T_surface_true_K": self._true_T_surface,
        }
        if self._surrogate_pred is not None:
            info.update(self._surrogate_pred)
        return info
