"""
env_fd1d.py

Gymnasium environment for HTS quench detection backed by the 1D Du Fort-Frankel
finite-difference solver (physics_fd1d.py).  Same observation/action/reward API
as env.py (QuenchDetectorEnv) so surrogates and controllers transfer directly.

Key difference: this env carries a spatially-resolved T(x) field forward between
steps — thermal history matters.  A localized hot-spot from a previous step
persists and evolves, unlike the lumped model which re-solves from scratch.

Observation:  Box(4)  [T_bath, I_A, voltage_V, T_surface_K]
Action:       Discrete(2)  0=continue, 1=dump
"""

from __future__ import annotations

import os
import numpy as np
import gymnasium

from hts_tape_env.materials import (
    TapeGeometry,
    lumped_rebco_tape,
    freefem_rebco_tape,
)
from hts_tape_env.physics_fd1d import (
    Tape1D,
    HeaterPulse,
    CoolingModel,
    find_Istar_fd,
    _solve_current_sharing_vec,
    du_fort_frankel_step,
)

_JAX_AVAILABLE = False
if not os.environ.get("DISABLE_JAX"):
    try:
        import jax.numpy as jnp
        from hts_tape_env.physics_fd1d_jax import (
            make_lumped_params,
            make_settle_fn,
            make_settle_fn_lateral,
            LumpedParams,
        )
        _JAX_AVAILABLE = True
    except ImportError:
        pass


class FD1DQuenchDetectorEnv(gymnasium.Env):
    """
    Quench-detection env backed by the 1D FD solver.

    Parameters
    ----------
    geometry : str or TapeGeometry
        'lumped' (Cu+SS, matches physics.py) or 'freefem' (7-layer) or an
        explicit TapeGeometry instance.
    nx : int
        Number of FD spatial nodes.
    T0_range : (float, float)
        Uniform sampling range for bath temperature [K].
    heater : HeaterPulse or None
        Localized disturbance for I* computation.
    cool_mode : str
        'bath_ends', 'lateral', or 'lumped_equiv'.
    lam_range : (float, float) or None
        Cooling lambda range for 'lumped_equiv' mode.
    ramp_rate : float
        Current increment per step as a fraction of I*.
    n_settle_steps : int
        FD timesteps per env step.
    dt : float
        FD time step [s].
    """

    metadata: dict = {"render_modes": []}

    def __init__(
        self,
        geometry: str | TapeGeometry = "lumped",
        nx: int = 100,
        T0_range: tuple[float, float] = (40.0, 75.0),
        heater: HeaterPulse | None = None,
        cool_mode: str = "bath_ends",
        lam_range: tuple[float, float] | None = None,
        ramp_rate: float = 0.005,
        I_start_frac: float = 0.30,
        I_max_frac: float = 1.10,
        noise_std: dict | None = None,
        dump_threshold: float = 0.05,
        reward_hit: float = 10.0,
        reward_false_alarm: float = -2.0,
        reward_missed: float = -50.0,
        reward_step: float = 0.01,
        n_settle_steps: int = 20,
        dt: float = 1e-4,
        Ic0: float = 500.0,
        Tc: float = 85.0,
        n_value: float = 20.0,
        istar_tol: float = 2.0,
        istar_max_iter: int = 15,
        istar_t_end: float = 0.15,
        backend: str = "numpy",
    ):
        super().__init__()

        # Build geometry
        if isinstance(geometry, str):
            if geometry == "lumped":
                geom = lumped_rebco_tape(Ic0=Ic0, Tc=Tc, n_value=n_value)
            elif geometry == "freefem":
                geom = freefem_rebco_tape()
            else:
                raise ValueError(f"Unknown geometry preset: {geometry!r}")
        else:
            geom = geometry

        self.tape = Tape1D(geom=geom, nx=nx)
        self.heater = heater or HeaterPulse()
        self.cool_mode = cool_mode
        self.lam_range = lam_range
        self.T0_range = T0_range
        self.ramp_rate = ramp_rate
        self.I_start_frac = I_start_frac
        self.I_max_frac = I_max_frac
        self.noise_std = noise_std or {"I": 0.5, "V": 1e-6, "T": 0.05}
        self.dump_threshold = dump_threshold
        self.reward_hit = reward_hit
        self.reward_false_alarm = reward_false_alarm
        self.reward_missed = reward_missed
        self.reward_step = reward_step
        self.n_settle_steps = n_settle_steps
        self.dt = dt
        self.istar_tol = istar_tol
        self.istar_max_iter = istar_max_iter
        self.istar_t_end = istar_t_end
        self.backend = backend

        # JAX backend: build JIT-compiled settle function
        self._jax_settle_fn = None
        self._jax_params = None
        if backend == "jax":
            if not _JAX_AVAILABLE:
                raise ImportError("JAX backend requested but jax is not installed")
            if isinstance(geometry, str) and geometry == "lumped":
                from hts_tape_env import physics
                self._jax_params = make_lumped_params(
                    Ic0=Ic0, Tc=Tc, n_value=n_value,
                    w=geom.w, t_cu=physics.t_cu, t_ss=physics.t_ss, Ec=physics.Ec,
                )
                if cool_mode == "bath_ends":
                    self._jax_settle_fn = make_settle_fn(
                        self._jax_params, self.tape.dx, dt, n_settle_steps,
                    )
                elif cool_mode == "lateral":
                    self._jax_settle_fn = make_settle_fn_lateral(
                        self._jax_params, self.tape.dx, dt, n_settle_steps,
                        h_lateral=0.0, tape_w=geom.w,
                    )
            else:
                raise NotImplementedError(
                    f"JAX backend only supports geometry='lumped', got {geometry!r}"
                )

        self._surrogate = None
        self._surrogate_pred: dict | None = None

        obs_lo = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        obs_hi = np.array([200.0, 2000.0, 10.0, 300.0], dtype=np.float32)
        self.observation_space = gymnasium.spaces.Box(obs_lo, obs_hi, dtype=np.float32)
        self.action_space = gymnasium.spaces.Discrete(2)

        # Episode state
        self._T_bath: float = 0.0
        self._lam: float = 0.0
        self._I_star: float = 0.0
        self._T_star: float = 0.0
        self._I_frac: float = 0.0
        self._T_field: np.ndarray = np.zeros(nx)
        self._T_field_prev: np.ndarray = np.zeros(nx)
        self._true_I: float = 0.0
        self._true_T_surface: float = 0.0
        self._true_voltage: float = 0.0
        self._rng = np.random.default_rng()

    def set_surrogate(self, surrogate):
        """Attach a trained surrogate (same interface as env.py)."""
        self._surrogate = surrogate
        self._surrogate_pred = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        for _ in range(50):
            T_bath = float(self._rng.uniform(*self.T0_range))
            lam = float(self._rng.uniform(*self.lam_range)) if self.lam_range else None

            cool = self._make_cool(T_bath, lam)
            I_star, T_star = find_Istar_fd(
                self.tape, T_bath, heater=self.heater, cool=cool,
                tol=self.istar_tol, max_iter=self.istar_max_iter,
                t_end=self.istar_t_end, dt=self.dt,
            )
            if np.isfinite(I_star) and I_star > 0:
                break
        else:
            T_bath, lam = 50.0, 80.0 if self.lam_range else None
            cool = self._make_cool(T_bath, lam)
            I_star, T_star = find_Istar_fd(
                self.tape, T_bath, heater=self.heater, cool=cool,
                tol=self.istar_tol, max_iter=self.istar_max_iter,
                t_end=self.istar_t_end, dt=self.dt,
            )

        return self._begin_episode(T_bath, lam or 0.0, I_star, T_star)

    def reset_at(self, T_bath: float, lam: float = 0.0, seed=None):
        """Deterministic reset at a specific operating point."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        cool = self._make_cool(T_bath, lam if lam > 0 else None)
        I_star, T_star = find_Istar_fd(
            self.tape, T_bath, heater=self.heater, cool=cool,
            tol=self.istar_tol, max_iter=self.istar_max_iter,
            t_end=self.istar_t_end, dt=self.dt,
        )
        return self._begin_episode(T_bath, lam, I_star, T_star)

    def _begin_episode(self, T_bath, lam, I_star, T_star):
        self._T_bath = T_bath
        self._lam = lam
        self._I_star = float(I_star)
        self._T_star = float(T_star)
        self._I_frac = self.I_start_frac

        # Initialize T field to bath temperature
        self._T_field = np.full(self.tape.nx, T_bath)
        self._T_field_prev = self._T_field.copy()

        obs = self._get_obs()
        if self._surrogate is not None:
            self._surrogate.reset()
            self._surrogate_pred = self._surrogate.predict_step(self._true_obs_vector())
        return obs, self._get_info()

    def step(self, action: int):
        self._I_frac += self.ramp_rate
        margin = 1.0 - self._I_frac

        obs = self._get_obs()
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

    # ── Internal ──────────────────────────────────────────────────────────────

    def _make_cool(self, T_bath: float, lam: float | None) -> CoolingModel:
        cool = CoolingModel(mode=self.cool_mode, T_bath=T_bath)
        if self.cool_mode == "lumped_equiv" and lam is not None:
            cool.lam_override = lam
        return cool

    def _get_obs(self) -> np.ndarray:
        I = self._I_frac * self._I_star

        # Advance the T field by n_settle_steps
        T_surf, V = self._advance_fd(I)

        self._true_I = float(I)
        self._true_T_surface = float(T_surf)
        self._true_voltage = float(V)

        noise = self._rng.standard_normal(3)
        I_obs = float(I) + noise[0] * self.noise_std.get("I", 0.5)
        V_obs = float(V) + noise[1] * self.noise_std.get("V", 1e-6)
        T_obs = float(T_surf) + noise[2] * self.noise_std.get("T", 0.05)

        obs = np.array([self._T_bath, I_obs, V_obs, T_obs], dtype=np.float32)
        return np.clip(obs, self.observation_space.low, self.observation_space.high)

    def _advance_fd(self, I: float) -> tuple[float, float]:
        """Run n_settle_steps of FD time stepping, return (Tmax, V_tape)."""
        if self.backend == "jax" and self._jax_settle_fn is not None:
            return self._advance_fd_jax(I)
        return self._advance_fd_numpy(I)

    def _advance_fd_numpy(self, I: float) -> tuple[float, float]:
        """NumPy settle loop (original)."""
        geom = self.tape.geom
        dx = self.tape.dx
        dt = self.dt
        T_bath = self._T_bath
        cool = self._make_cool(T_bath, self._lam if self._lam > 0 else None)
        cool._tape_w = geom.w
        cool._tape_d = 0.01

        T_curr = self._T_field
        T_prev = self._T_field_prev
        E = np.zeros(self.tape.nx)

        for _ in range(self.n_settle_steps):
            E, Q_joule = _solve_current_sharing_vec(I, T_curr, geom)
            Q_cool = cool.Q_cool(T_curr)
            Q_total = Q_joule - Q_cool

            T_next = du_fort_frankel_step(T_curr, T_prev, Q_total, geom, self.tape.dx, dt, T_bath)
            T_prev = T_curr
            T_curr = T_next

        self._T_field = T_curr
        self._T_field_prev = T_prev

        T_surf = float(np.max(T_curr))
        V_tape = float(np.sum(E * dx))
        return T_surf, V_tape

    def _advance_fd_jax(self, I: float) -> tuple[float, float]:
        """JAX-JIT settle loop — fused via jax.lax.scan."""
        T_curr_j = jnp.array(self._T_field)
        T_prev_j = jnp.array(self._T_field_prev)

        T_final, T_prev_final, Tmax, V_tape = self._jax_settle_fn(
            I, T_curr_j, T_prev_j, self._T_bath,
        )

        self._T_field = np.array(T_final)
        self._T_field_prev = np.array(T_prev_final)
        return float(Tmax), float(V_tape)

    def _true_obs_vector(self) -> np.ndarray:
        return np.array([self._T_bath, self._true_I, self._true_voltage, self._true_T_surface],
                        dtype=np.float32)

    def _get_info(self) -> dict:
        info = {
            "margin": float(1.0 - self._I_frac),
            "I_star": self._I_star,
            "T_star": self._T_star,
            "T0": self._T_bath,
            "lam": self._lam,
            "I_frac": self._I_frac,
            "runaway": (1.0 - self._I_frac) <= 0.0,
            "I_true_A": self._true_I,
            "voltage_true_V": self._true_voltage,
            "T_surface_true_K": self._true_T_surface,
        }
        if self._surrogate_pred is not None:
            info.update(self._surrogate_pred)
        return info
