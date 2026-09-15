"""
physics_fd1d_jax.py

JAX-JIT accelerated version of the 1D Du Fort-Frankel FD solver hot path.

Replaces the per-node Python loops in physics_fd1d.py with pure JAX array ops
that compile to a single XLA program via jax.jit.  The settle loop (n_settle_steps
of current-sharing + DFF step) is fused via jax.lax.scan — no Python overhead.

Material property functions are pure JAX reimplementations of:
  physics.rho_cu, physics.rho_ss, physics.Ic   (physics.py)
  thermal.cp_cu, thermal.cp_ss                  (thermal.py)
  materials.k_eff, C_eff, normal_layers_conductance  (materials.py)

All functions operate on arrays of shape (nx,) and are designed for @jax.jit.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

# ── Constants ──────────────────────────────────────────────────────────────────

L0_WF = 2.44e-8          # Lorenz number [W·Ω/K²]

RHO_CU_300 = 1.6e-8      # Cu resistivity at 300 K [Ω·m]
RRR_CU = 24.0
RHO_CU_0 = RHO_CU_300 / RRR_CU

RHO_SS_300 = 7.0e-7      # SS 304 resistivity at 300 K [Ω·m]

# Mass densities [kg/m³]
RHO_MASS_CU = 8960.0
RHO_MASS_SS = 8000.0

# Specific heat anchor points (log-log interpolation) from thermal.py
_CU_T = jnp.array([4.0, 10.0, 20.0, 50.0, 77.0, 100.0, 300.0])
_CU_CP = jnp.array([0.09, 3.0, 40.0, 160.0, 230.0, 280.0, 385.0])
_LOG_CU_T = jnp.log(_CU_T)
_LOG_CU_CP = jnp.log(_CU_CP)

_SS_T = jnp.array([4.0, 10.0, 20.0, 50.0, 77.0, 100.0, 300.0])
_SS_CP = jnp.array([0.20, 2.0, 10.0, 90.0, 180.0, 250.0, 500.0])
_LOG_SS_T = jnp.log(_SS_T)
_LOG_SS_CP = jnp.log(_SS_CP)


# ── Geometry parameters (frozen for JIT) ──────────────────────────────────────

class LumpedParams(NamedTuple):
    """Frozen parameter set for the lumped 2-layer (Cu+SS) tape."""
    Ic0: float
    Tc: float
    n_value: float
    Ec: float
    A_cu: float           # Cu cross-section [m²]
    A_ss: float           # SS cross-section [m²]
    w: float              # tape width [m]


def make_lumped_params(Ic0: float = 500.0, Tc: float = 85.0,
                       n_value: float = 20.0,
                       w: float = 0.004,
                       t_cu: float = 50e-6, t_ss: float = 50e-6,
                       Ec: float = 1e-4) -> LumpedParams:
    return LumpedParams(
        Ic0=Ic0, Tc=Tc, n_value=n_value, Ec=Ec,
        A_cu=t_cu * w, A_ss=t_ss * w, w=w,
    )


# ── Material property functions (vectorized over nx) ──────────────────────────

@jax.jit
def rho_cu_jax(T: jnp.ndarray) -> jnp.ndarray:
    return RHO_CU_0 + (RHO_CU_300 - RHO_CU_0) * jnp.clip(T / 300.0, 0.0, 1.5) ** 1.7


@jax.jit
def rho_ss_jax(T: jnp.ndarray) -> jnp.ndarray:
    return RHO_SS_300 * (0.85 + 0.15 * jnp.clip(T / 300.0, 0.0, 1.5))


@jax.jit
def k_cu_jax(T: jnp.ndarray) -> jnp.ndarray:
    return L0_WF * T / rho_cu_jax(T)


@jax.jit
def k_ss_jax(T: jnp.ndarray) -> jnp.ndarray:
    return L0_WF * T / rho_ss_jax(T)


@jax.jit
def cp_cu_jax(T: jnp.ndarray) -> jnp.ndarray:
    T_c = jnp.clip(T, 4.0, 300.0)
    return jnp.exp(jnp.interp(jnp.log(T_c), _LOG_CU_T, _LOG_CU_CP))


@jax.jit
def cp_ss_jax(T: jnp.ndarray) -> jnp.ndarray:
    T_c = jnp.clip(T, 4.0, 300.0)
    return jnp.exp(jnp.interp(jnp.log(T_c), _LOG_SS_T, _LOG_SS_CP))


@jax.jit
def Ic_jax(T: jnp.ndarray, Ic0: float, Tc: float) -> jnp.ndarray:
    T_c = jnp.clip(T, 1.0, Tc - 1e-6)
    return jnp.maximum(Ic0 * (1.0 - (T_c / Tc) ** 1.7), 1e-12)


# ── Composite geometry functions ──────────────────────────────────────────────

@jax.jit
def k_eff_jax(T: jnp.ndarray, A_cu: float, A_ss: float) -> jnp.ndarray:
    return k_cu_jax(T) * A_cu + k_ss_jax(T) * A_ss


@jax.jit
def C_eff_jax(T: jnp.ndarray, A_cu: float, A_ss: float) -> jnp.ndarray:
    return RHO_MASS_CU * A_cu * cp_cu_jax(T) + RHO_MASS_SS * A_ss * cp_ss_jax(T)


@jax.jit
def G_norm_jax(T: jnp.ndarray, A_cu: float, A_ss: float) -> jnp.ndarray:
    return A_cu / rho_cu_jax(T) + A_ss / rho_ss_jax(T)


# ── Current sharing solver (vectorized Newton) ───────────────────────────────

def _solve_current_sharing_jax(I: float, T: jnp.ndarray,
                                params: LumpedParams) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Vectorized Newton iteration for per-node E-field, pure JAX.

    Solves: Ic(T) * (E/Ec)^(1/n) + G_norm(T) * E = I  at each node.
    Returns (E, Q_joule) arrays of shape (nx,).
    """
    Ec = params.Ec
    IcT = Ic_jax(T, params.Ic0, params.Tc)
    inv_n = 1.0 / jnp.maximum(params.n_value, 1.01)
    G = G_norm_jax(T, params.A_cu, params.A_ss)

    E = jnp.where(G > 0, I / G, Ec)
    E = jnp.clip(E, 1e-20, 1e3)

    def newton_step(E, _):
        I_sc = IcT * (E / Ec) ** inv_n
        I_norm = G * E
        residual = I_sc + I_norm - I
        dI_sc_dE = IcT * inv_n * (E / Ec) ** (inv_n - 1.0) / Ec
        dI_dE = dI_sc_dE + G
        dE = -residual / jnp.maximum(dI_dE, 1e-30)
        E_new = jnp.clip(E + dE, 1e-20, 1e3)
        return E_new, None

    E, _ = jax.lax.scan(newton_step, E, None, length=30)

    Q = E * I
    return E, Q


# ── Du Fort-Frankel step (pure JAX) ──────────────────────────────────────────

def _du_fort_frankel_step_jax(T_curr: jnp.ndarray, T_prev: jnp.ndarray,
                               Q: jnp.ndarray, params: LumpedParams,
                               dx: float, dt: float,
                               T_bath: float) -> jnp.ndarray:
    """One Du Fort-Frankel timestep, pure JAX array ops."""
    kA = k_eff_jax(T_curr, params.A_cu, params.A_ss)
    C = jnp.maximum(C_eff_jax(T_curr, params.A_cu, params.A_ss), 1e-12)

    A = 2.0 * kA * dt / (C * dx * dx)
    gamma = 2.0 * dt / C

    T_left = jnp.concatenate([jnp.array([T_bath]), T_curr[:-1]])
    T_right = jnp.concatenate([T_curr[1:], jnp.array([T_bath])])

    T_next = (1.0 / (1.0 + A)) * (
        (1.0 - A) * T_prev + A * (T_left + T_right) + gamma * Q
    )

    T_next = T_next.at[0].set(T_bath)
    T_next = T_next.at[-1].set(T_bath)
    return T_next


def _forward_euler_step_jax(T: jnp.ndarray, Q: jnp.ndarray,
                             params: LumpedParams,
                             dx: float, dt: float,
                             T_bath: float) -> jnp.ndarray:
    """Forward Euler step for bootstrapping, pure JAX."""
    kA = k_eff_jax(T, params.A_cu, params.A_ss)
    C = jnp.maximum(C_eff_jax(T, params.A_cu, params.A_ss), 1e-12)

    T_left = jnp.concatenate([jnp.array([T_bath]), T[:-1]])
    T_right = jnp.concatenate([T[1:], jnp.array([T_bath])])

    diffusion = kA * (T_left - 2.0 * T + T_right) / (dx * dx)
    dTdt = (diffusion + Q) / C

    T_new = T + dt * dTdt
    T_new = T_new.at[0].set(T_bath)
    T_new = T_new.at[-1].set(T_bath)
    return T_new


# ── Fused settle loop via jax.lax.scan ────────────────────────────────────────

def make_settle_fn(params: LumpedParams, dx: float, dt: float,
                   n_steps: int, cool_mode: str = "bath_ends"):
    """
    Build a JIT-compiled function that fuses n_steps of
    (current_sharing → DFF step) into one XLA program.

    Returns:
        settle(I, T_curr, T_prev, T_bath) → (T_final, T_prev_final, Tmax, V_tape)
    """

    @jax.jit
    def settle(I, T_curr, T_prev, T_bath):
        def body(carry, _):
            T_c, T_p = carry
            E, Q_joule = _solve_current_sharing_jax(I, T_c, params)
            # bath_ends: Q_cool = 0 (BCs handle cooling)
            Q_total = Q_joule
            T_n = _du_fort_frankel_step_jax(T_c, T_p, Q_total, params, dx, dt, T_bath)
            return (T_n, T_c), (jnp.max(T_n), jnp.sum(E * dx))

        (T_final, T_prev_final), (Tmax_trace, V_trace) = jax.lax.scan(
            body, (T_curr, T_prev), None, length=n_steps
        )
        return T_final, T_prev_final, Tmax_trace[-1], V_trace[-1]

    return settle


def make_settle_fn_lateral(params: LumpedParams, dx: float, dt: float,
                           n_steps: int, h_lateral: float, tape_w: float):
    """Settle function with lateral (Newton) cooling."""

    @jax.jit
    def settle(I, T_curr, T_prev, T_bath):
        perimeter = 2.0 * tape_w

        def body(carry, _):
            T_c, T_p = carry
            E, Q_joule = _solve_current_sharing_jax(I, T_c, params)
            Q_cool = h_lateral * perimeter * jnp.maximum(T_c - T_bath, 0.0)
            Q_total = Q_joule - Q_cool
            T_n = _du_fort_frankel_step_jax(T_c, T_p, Q_total, params, dx, dt, T_bath)
            return (T_n, T_c), (jnp.max(T_n), jnp.sum(E * dx))

        (T_final, T_prev_final), (Tmax_trace, V_trace) = jax.lax.scan(
            body, (T_curr, T_prev), None, length=n_steps
        )
        return T_final, T_prev_final, Tmax_trace[-1], V_trace[-1]

    return settle


# ── Transient simulation (Python loop + JIT steps) ───────────────────────────

def run_transient_jax(nx: int, dx: float, I: float, T_bath: float,
                      params: LumpedParams,
                      heater_power: float = 13.0,
                      heater_center: float | None = None,
                      heater_width: float = 0.002,
                      heater_duration: float = 0.01,
                      dt: float = 1e-4, t_end: float = 0.1,
                      T_runaway: float = 300.0,
                      record_every: int = 10,
                      L: float = 1.0) -> dict:
    """
    Run a transient FD simulation at fixed I.  Outer loop stays Python
    (for early exit on runaway); each step is JIT-compiled.

    Returns dict matching run_transient() signature.
    """
    x = jnp.linspace(0.5 * dx, L - 0.5 * dx, nx)
    T_curr = jnp.full(nx, T_bath)
    n_steps = int(t_end / dt)

    # Heater setup
    hc = heater_center if heater_center is not None else L / 2.0
    mask = jnp.abs(x - hc) <= heater_width / 2.0
    n_on = jnp.maximum(jnp.sum(mask), 1)
    Q_heater_on = jnp.where(mask, heater_power / (n_on * dx), 0.0)

    # JIT-compile single step
    @jax.jit
    def step_fn(T_curr, T_prev, Q_heat, T_bath_val):
        E, Q_joule = _solve_current_sharing_jax(I, T_curr, params)
        Q_total = Q_joule + Q_heat
        T_next = _du_fort_frankel_step_jax(T_curr, T_prev, Q_total, params, dx, dt, T_bath_val)
        return T_next, E

    # Bootstrap
    E_init, Q_j_init = _solve_current_sharing_jax(I, T_curr, params)
    Q_boot = Q_j_init + Q_heater_on
    dt_boot = dt * 0.01
    T_prev = _forward_euler_step_jax(T_curr, Q_boot, params, dx, dt_boot, T_bath)

    rec_t = [0.0]
    rec_Tmax = [float(T_bath)]
    rec_V = [float(jnp.sum(E_init * dx))]
    Tmax_after_heater = None
    runaway = False

    for step_i in range(1, n_steps + 1):
        t = step_i * dt
        Q_heat = Q_heater_on if t < heater_duration else jnp.zeros(nx)
        T_next, E = step_fn(T_curr, T_prev, Q_heat, T_bath)
        T_prev = T_curr
        T_curr = T_next

        Tmax_now = float(jnp.max(T_curr))

        if Tmax_now >= T_runaway:
            runaway = True
            rec_t.append(t)
            rec_Tmax.append(Tmax_now)
            rec_V.append(float(jnp.sum(E * dx)))
            break

        if t > heater_duration + dt:
            Tmax_after_heater = Tmax_now

        if step_i % record_every == 0:
            rec_t.append(t)
            rec_Tmax.append(Tmax_now)
            rec_V.append(float(jnp.sum(E * dx)))

    recovered = False
    if not runaway and len(rec_Tmax) > 2:
        peak_idx = int(np.argmax(rec_Tmax))
        if peak_idx < len(rec_Tmax) - 1:
            recovered = rec_Tmax[-1] < rec_Tmax[peak_idx] - 0.1

    return dict(
        t=np.array(rec_t),
        Tmax=np.array(rec_Tmax),
        V_tape=np.array(rec_V),
        T_final=np.array(T_curr),
        x=np.array(x),
        runaway=runaway,
        recovered=recovered,
    )
