"""
physics_fd1d.py

1D Du Fort-Frankel finite-difference heat equation solver for HTS tape.

Bridges the gap between the lumped analytic model (physics.py) and full 3D FEM.
Captures heat diffusion along the tape length with localized hot-spots, per-node
current sharing, and configurable cooling — all in pure Python/NumPy.

Physics follows Sirois/Breault's RAPDi approach: Levin's two-layer current
sharing + Du Fort-Frankel explicit FD.  The NN correction prefactors are NOT
implemented here (stretch goal); this is the pure FD solver.

Du Fort-Frankel scheme (unconditionally stable, explicit, no matrix solve):

    T_i^{n+1} = 1/(1+A_i) * [(1-A_i)*T_i^{n-1}
                 + A_i*(T_{i+1}^n + T_{i-1}^n)
                 + γ_i * Q_i^n]

    A_i = 2·k_eff(T_i)·dt / (C_eff(T_i)·dx²)
    γ_i = 2·dt / C_eff(T_i)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.optimize import brentq

from hts_tape_env.materials import TapeGeometry


# ── Domain ──────────────────────────────────────────────────────────────────

@dataclass
class Tape1D:
    """1D finite-difference domain along the tape length."""
    geom: TapeGeometry
    nx: int = 200

    def __post_init__(self):
        self.dx = self.geom.L / self.nx
        self.x = np.linspace(0.5 * self.dx, self.geom.L - 0.5 * self.dx, self.nx)


# ── Heater ──────────────────────────────────────────────────────────────────

@dataclass
class HeaterPulse:
    """Localized heater disturbance."""
    power: float = 13.0          # total heater power [W]
    center: Optional[float] = None  # position [m]; None → tape midpoint
    width: float = 0.002         # heater width [m]
    t_start: float = 0.0        # pulse start time [s]
    duration: float = 0.01       # pulse duration [s]

    def Q_heater(self, x: np.ndarray, t: float, L: float) -> np.ndarray:
        """Heater power per unit length [W/m] at each node at time t."""
        c = self.center if self.center is not None else L / 2.0
        if t < self.t_start or t >= self.t_start + self.duration:
            return np.zeros_like(x)
        mask = np.abs(x - c) <= self.width / 2.0
        if not mask.any():
            mask = np.zeros_like(x, dtype=bool)
            mask[np.argmin(np.abs(x - c))] = True
        q = np.zeros_like(x)
        n_on = mask.sum()
        dx = x[1] - x[0] if len(x) > 1 else L
        q[mask] = self.power / (n_on * dx)
        return q


# ── Cooling ─────────────────────────────────────────────────────────────────

@dataclass
class CoolingModel:
    """Cooling model for the 1D tape."""
    mode: str = "bath_ends"      # 'bath_ends', 'lateral', 'lumped_equiv'
    T_bath: float = 77.0
    h_lateral: float = 0.0      # [W/m²/K] surface heat transfer coeff, for 'lateral'
    lam_override: Optional[float] = None  # for 'lumped_equiv'
    _tape_w: float = 0.004      # set by the solver from geometry
    _tape_d: float = 0.01       # cooling depth (brass shim thickness)

    def Q_cool(self, T: np.ndarray) -> np.ndarray:
        """Cooling power per unit length [W/m] removed at each node."""
        dT = np.maximum(T - self.T_bath, 0.0)
        if self.mode == "bath_ends":
            return np.zeros_like(T)
        elif self.mode == "lateral":
            # Newton cooling on both tape faces
            perimeter = 2.0 * self._tape_w
            return self.h_lateral * perimeter * dT
        elif self.mode == "lumped_equiv":
            from hts_tape_env.physics import lambda_brass
            if self.lam_override is not None:
                lam = float(self.lam_override)
            else:
                lam = np.vectorize(lambda t: float(lambda_brass(t)))(T)
            eta_per_m = lam * 2.0 * self._tape_w / self._tape_d
            return eta_per_m * dT
        return np.zeros_like(T)


# ── Per-node current sharing ────────────────────────────────────────────────

def solve_current_sharing_1d(I: float, T: np.ndarray,
                              geom: TapeGeometry) -> tuple[np.ndarray, np.ndarray]:
    """
    At each spatial node, find E-field satisfying current conservation:
        I_sc(E, T_i) + I_normal(E, T_i) = I_total

    Returns (E, Q_joule) arrays of shape (nx,).
    E: local electric field [V/m]
    Q_joule: Joule heating per unit length [W/m]
    """
    nx = len(T)
    E_arr = np.empty(nx)
    Q_arr = np.empty(nx)

    sc = geom.sc
    Ec = sc.Ec

    for i in range(nx):
        Ti = float(T[i])
        IcT = sc.Ic_func(Ti)
        nT = sc.n_func(Ti)
        G_norm = geom.normal_layers_conductance(Ti)

        if I <= 0.0:
            E_arr[i] = 0.0
            Q_arr[i] = 0.0
            continue

        # Current as function of E: I(E) = I_sc(E) + I_normal(E)
        # I_sc = Ic * (E/Ec)^(1/n) for E > 0 (power-law E-J)
        # I_normal = G_norm * E
        inv_n = 1.0 / max(nT, 1.01)

        def current_balance(log_E):
            E = np.exp(log_E)
            I_sc = IcT * (E / Ec) ** inv_n
            I_norm = G_norm * E
            return I_sc + I_norm - I

        # Bracket: at very low E, I(E)→0; at high E, I(E)→∞
        log_E_lo = np.log(1e-20)
        log_E_hi = np.log(max(I / max(G_norm, 1e-30), Ec) * 10.0)

        # Ensure bracket contains root
        for _ in range(20):
            if current_balance(log_E_hi) > 0:
                break
            log_E_hi += 2.0

        try:
            log_E = brentq(current_balance, log_E_lo, log_E_hi,
                           maxiter=80, xtol=1e-10)
        except Exception:
            log_E = np.log(I / max(G_norm, 1e-30))

        E = np.exp(log_E)
        E_arr[i] = E

        # Joule heating per unit length: sum of E*J*A for each layer
        I_sc = IcT * (E / Ec) ** inv_n
        I_norm = G_norm * E
        # Q = E * I_total (since voltage is uniform across parallel layers)
        # But per unit length: q = E * (I_sc + I_norm) = E * I
        Q_arr[i] = E * I

    return E_arr, Q_arr


def _solve_current_sharing_vec(I: float, T: np.ndarray,
                                geom: TapeGeometry) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized approximation for speed — Newton iteration instead of per-node brentq."""
    sc = geom.sc
    Ec = sc.Ec
    nx = len(T)

    IcT = np.array([sc.Ic_func(float(T[i])) for i in range(nx)])
    nT = np.array([sc.n_func(float(T[i])) for i in range(nx)])
    G_norm = np.array([geom.normal_layers_conductance(float(T[i])) for i in range(nx)])

    inv_n = 1.0 / np.maximum(nT, 1.01)

    # Initial guess: all current through normal metal
    E = np.where(G_norm > 0, I / G_norm, Ec)
    E = np.clip(E, 1e-20, 1e3)

    # Newton refinement
    for _ in range(30):
        I_sc = IcT * (E / Ec) ** inv_n
        I_norm = G_norm * E
        residual = I_sc + I_norm - I

        # Derivative: dI/dE
        dI_sc_dE = IcT * inv_n * (E / Ec) ** (inv_n - 1.0) / Ec
        dI_dE = dI_sc_dE + G_norm

        dE = -residual / np.maximum(dI_dE, 1e-30)
        E = np.clip(E + dE, 1e-20, 1e3)

        if np.max(np.abs(residual)) < 1e-6:
            break

    Q = E * I
    return E, Q


# ── Du Fort-Frankel time stepper ────────────────────────────────────────────

def du_fort_frankel_step(T_curr: np.ndarray, T_prev: np.ndarray,
                          Q: np.ndarray, geom: TapeGeometry,
                          dx: float, dt: float,
                          T_bath: float) -> np.ndarray:
    """
    One Du Fort-Frankel timestep.

    T_curr: T^n (shape nx)
    T_prev: T^{n-1} (shape nx)
    Q: net source per unit length [W/m] (shape nx)
    Returns T^{n+1} (shape nx)
    """
    nx = len(T_curr)
    T_next = np.empty(nx)

    # Compute k_eff and C_eff at each node
    kA = np.array([geom.k_eff(float(T_curr[i])) for i in range(nx)])
    C = np.array([max(geom.C_eff(float(T_curr[i])), 1e-12) for i in range(nx)])

    A = 2.0 * kA * dt / (C * dx * dx)
    gamma = 2.0 * dt / C

    # Interior nodes
    T_left = np.empty(nx)
    T_right = np.empty(nx)
    T_left[0] = T_bath
    T_left[1:] = T_curr[:-1]
    T_right[-1] = T_bath
    T_right[:-1] = T_curr[1:]

    T_next = (1.0 / (1.0 + A)) * (
        (1.0 - A) * T_prev + A * (T_left + T_right) + gamma * Q
    )

    # Enforce BCs
    T_next[0] = T_bath
    T_next[-1] = T_bath

    return T_next


def _forward_euler_step(T: np.ndarray, Q: np.ndarray, geom: TapeGeometry,
                         dx: float, dt: float, T_bath: float) -> np.ndarray:
    """Forward Euler step for bootstrapping Du Fort-Frankel."""
    nx = len(T)
    kA = np.array([geom.k_eff(float(T[i])) for i in range(nx)])
    C = np.array([max(geom.C_eff(float(T[i])), 1e-12) for i in range(nx)])

    T_left = np.empty(nx)
    T_right = np.empty(nx)
    T_left[0] = T_bath
    T_left[1:] = T[:-1]
    T_right[-1] = T_bath
    T_right[:-1] = T[1:]

    diffusion = kA * (T_left - 2.0 * T + T_right) / (dx * dx)
    dTdt = (diffusion + Q) / C

    T_new = T + dt * dTdt
    T_new[0] = T_bath
    T_new[-1] = T_bath
    return T_new


# ── Transient simulation ───────────────────────────────────────────────────

def run_transient(tape: Tape1D, I: float, T_bath: float,
                  heater: Optional[HeaterPulse] = None,
                  cool: Optional[CoolingModel] = None,
                  dt: float = 1e-4, t_end: float = 0.1,
                  T_runaway: float = 300.0,
                  record_every: int = 10,
                  vectorized: bool = True) -> dict:
    """
    Run a transient FD simulation at fixed transport current I.

    Returns dict:
        t: recorded time array
        Tmax: max(T(x)) at each recorded time
        V_tape: integrated voltage [V] at each recorded time
        T_final: T(x) at end of simulation
        runaway: bool — did Tmax exceed T_runaway?
        recovered: bool — did Tmax decrease after heater off?
    """
    if cool is None:
        cool = CoolingModel(mode="bath_ends", T_bath=T_bath)
    cool.T_bath = T_bath
    cool._tape_w = tape.geom.w
    cool._tape_d = 0.01  # brass shim depth

    geom = tape.geom
    nx = tape.nx
    dx = tape.dx
    x = tape.x

    current_sharing = _solve_current_sharing_vec if vectorized else solve_current_sharing_1d

    # Initialize
    T_curr = np.full(nx, T_bath)
    n_steps = int(t_end / dt)

    # Bootstrap: Forward Euler half-step
    E_init, Q_joule_init = current_sharing(I, T_curr, geom)
    Q_heat_init = heater.Q_heater(x, 0.0, geom.L) if heater else np.zeros(nx)
    Q_cool_init = cool.Q_cool(T_curr)
    Q_total_init = Q_joule_init + Q_heat_init - Q_cool_init

    dt_boot = dt * 0.01
    T_prev = _forward_euler_step(T_curr, Q_total_init, geom, dx, dt_boot, T_bath)

    # Recording
    rec_t = [0.0]
    rec_Tmax = [T_bath]
    rec_V = [float(np.sum(E_init * dx))]
    Tmax_after_heater = None
    runaway = False

    for step in range(1, n_steps + 1):
        t = step * dt

        # Current sharing
        E, Q_joule = current_sharing(I, T_curr, geom)

        # Sources
        Q_heat = heater.Q_heater(x, t, geom.L) if heater else np.zeros(nx)
        Q_cooling = cool.Q_cool(T_curr)
        Q_total = Q_joule + Q_heat - Q_cooling

        # Du Fort-Frankel step
        T_next = du_fort_frankel_step(T_curr, T_prev, Q_total, geom, dx, dt, T_bath)

        T_prev = T_curr
        T_curr = T_next

        Tmax_now = float(np.max(T_curr))

        # Check runaway
        if Tmax_now >= T_runaway:
            runaway = True
            rec_t.append(t)
            rec_Tmax.append(Tmax_now)
            rec_V.append(float(np.sum(E * dx)))
            break

        # Track recovery after heater off
        if heater and t > heater.t_start + heater.duration + dt:
            if Tmax_after_heater is None:
                Tmax_after_heater = Tmax_now
            else:
                Tmax_after_heater = Tmax_now

        # Record
        if step % record_every == 0:
            rec_t.append(t)
            rec_Tmax.append(Tmax_now)
            rec_V.append(float(np.sum(E * dx)))

    # Determine recovery
    recovered = False
    if heater and not runaway and len(rec_Tmax) > 2:
        peak_idx = int(np.argmax(rec_Tmax))
        if peak_idx < len(rec_Tmax) - 1:
            recovered = rec_Tmax[-1] < rec_Tmax[peak_idx] - 0.1

    return dict(
        t=np.array(rec_t),
        Tmax=np.array(rec_Tmax),
        V_tape=np.array(rec_V),
        T_final=T_curr.copy(),
        x=x.copy(),
        runaway=runaway,
        recovered=recovered,
    )


# ── I* finder ──────────────────────────────────────────────────────────────

def find_Istar_fd(tape: Tape1D, T_bath: float,
                  heater: Optional[HeaterPulse] = None,
                  cool: Optional[CoolingModel] = None,
                  I_lo: float = 10.0, I_hi: Optional[float] = None,
                  tol: float = 1.0, max_iter: int = 20,
                  t_end: float = 0.1, dt: float = 1e-4,
                  warm_start: bool = True,
                  vectorized: bool = True) -> tuple[float, float]:
    """
    Find the runaway current I* by binary search.

    For each candidate I, runs a transient with heater pulse and checks
    if the system recovers (Tmax drops after heater off) or runs away.

    Returns (I_star, T_star) where T_star is peak Tmax at I ≈ I*.
    Returns (nan, nan) if no runaway found in the search range.
    """
    if heater is None:
        heater = HeaterPulse()

    # Warm-start bracket from lumped model
    if warm_start and I_hi is None:
        try:
            from hts_tape_env.physics import solve_runaway as lumped_solve
            I_lumped, _ = lumped_solve(T_bath, 20.0, mode="stab")
            if np.isfinite(I_lumped) and I_lumped > 0:
                I_lo = max(I_lo, I_lumped * 0.5)
                I_hi = I_lumped * 2.0
        except Exception:
            pass

    if I_hi is None:
        IcT = tape.geom.sc.Ic_func(T_bath)
        I_hi = IcT * 1.5

    def is_runaway(I_test):
        result = run_transient(tape, I_test, T_bath, heater=heater, cool=cool,
                               dt=dt, t_end=t_end, vectorized=vectorized)
        return result["runaway"] or not result["recovered"]

    # Verify bracket: I_lo should be safe, I_hi should be runaway
    if is_runaway(I_lo):
        return float(I_lo), T_bath
    if not is_runaway(I_hi):
        # Expand upper bound
        for _ in range(5):
            I_hi *= 1.5
            if is_runaway(I_hi):
                break
        else:
            return np.nan, np.nan

    # Binary search
    best_I = np.nan
    best_Tmax = np.nan
    for _ in range(max_iter):
        if I_hi - I_lo < tol:
            break
        I_mid = 0.5 * (I_lo + I_hi)
        if is_runaway(I_mid):
            I_hi = I_mid
        else:
            I_lo = I_mid

    best_I = 0.5 * (I_lo + I_hi)

    # Get T* from a run at I*
    result = run_transient(tape, best_I, T_bath, heater=heater, cool=cool,
                           dt=dt, t_end=t_end, vectorized=vectorized)
    best_Tmax = float(np.max(result["Tmax"]))

    return best_I, best_Tmax


# ── Convenience wrappers (API parity with physics.py) ──────────────────────

def stable_surface_T_fd(I: float, T_bath: float, tape: Tape1D,
                        cool: Optional[CoolingModel] = None,
                        dt: float = 1e-4, t_end: float = 0.5) -> float:
    """Steady-state max(T) at current I (no heater)."""
    result = run_transient(tape, I, T_bath, heater=None, cool=cool,
                           dt=dt, t_end=t_end)
    return float(np.max(result["T_final"]))


def tape_voltage_fd(I: float, T_bath: float, tape: Tape1D,
                    cool: Optional[CoolingModel] = None,
                    dt: float = 1e-4, t_end: float = 0.5) -> float:
    """Integrated tape voltage at steady state [V]."""
    result = run_transient(tape, I, T_bath, heater=None, cool=cool,
                           dt=dt, t_end=t_end)
    return float(result["V_tape"][-1]) if len(result["V_tape"]) > 0 else 0.0


def time_margin_fd(I: float, T_bath: float, tape: Tape1D,
                   T_target: float = 300.0, dt: float = 1e-4,
                   t_max: float = 10.0) -> float:
    """Adiabatic time margin: time from hot-spot initiation to T_target with no cooling."""
    adiabatic_cool = CoolingModel(mode="bath_ends", T_bath=T_bath)
    heater = HeaterPulse(power=13.0, duration=0.01)
    result = run_transient(tape, I, T_bath, heater=heater, cool=adiabatic_cool,
                           dt=dt, t_end=t_max, T_runaway=T_target)
    if result["runaway"]:
        return float(result["t"][-1])
    return np.nan
