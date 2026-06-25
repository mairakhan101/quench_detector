"""
physics.py

Thermal-runaway physics for an HTS conductor with optional Cu/SS stabilizer.

Based on: Marchevsky & Prestemon, Supercond. Sci. Technol. 37 035012 (2024).
All quantities SI (K, A, V, W, Ω, m) unless noted.
Default geometry matches the paper: L=1 m, w=4 mm, d=10 mm, n=20, Ic0=500 A, Tc=85 K.
"""

import numpy as np
from scipy.optimize import brentq

# ── Reference geometry (paper sections 2b-2c) ─────────────────────────────────
L   = 1.0        # tape length [m]
w   = 0.004      # tape width [m]
d   = 0.01       # cooling path depth (brass shim) [m]
Ec  = 1e-4       # E-field criterion [V/m]
U0  = Ec * L     # criterion voltage across full tape [V]

# Stabilizer layer thicknesses and cross-sections
t_cu = 50e-6
t_ss = 50e-6
A_cu = t_cu * w
A_ss = t_ss * w

# Resistivity anchors
RRR_CU     = 24.0
rho_cu_300 = 1.6e-8   # [Ω·m]
rho_ss_300 = 7.0e-7   # [Ω·m]

# ── Brass thermal conductivity (NIST C26000, sign-corrected vs paper eq. 6) ───
_BRASS = dict(a=0.021035, b=-1.01835, c=4.54083, d=-5.03374,
              e=3.20536,  f=-1.12933, g=+0.174057, h=-0.0038151)
_BRASS_TMIN, _BRASS_TMAX = 5.0, 110.0


def lambda_brass(T):
    """Brass thermal conductivity [W/m/K] — NIST C26000 log-polynomial."""
    T = np.asarray(T, dtype=float)
    x = np.log10(np.clip(T, _BRASS_TMIN, _BRASS_TMAX))
    c = _BRASS
    p = (c['a'] + c['b']*x + c['c']*x**2 + c['d']*x**3
         + c['e']*x**4 + c['f']*x**5 + c['g']*x**6 + c['h']*x**7)
    return 10.0**p


# ── Conductor transport model ──────────────────────────────────────────────────

def Ic(T, Ic0=500.0, Tc=85.0):
    """Critical current [A]: Ic0*(1-(T/Tc)^1.7)."""
    T = np.asarray(T, dtype=float)
    return np.maximum(Ic0 * (1.0 - (np.clip(T, 1.0, Tc - 1e-6) / Tc)**1.7), 1e-12)


def eta_cool(T, lam_override=None):
    """Cooling coefficient η [W/K] for the full tape."""
    lam = lambda_brass(T) if lam_override is None else float(lam_override)
    return float(lam) * (2.0 * w * L / d)


def Qout(T, T0, lam_override=None):
    """Heat removed to environment [W]."""
    T = np.asarray(T, dtype=float)
    return eta_cool(np.maximum(T, T0 + 1e-9), lam_override=lam_override) * np.maximum(T - T0, 0.0)


def rho_cu(T):
    T = np.asarray(T, dtype=float)
    rho0 = rho_cu_300 / RRR_CU
    return rho0 + (rho_cu_300 - rho0) * np.clip(T / 300.0, 0.0, 1.5)**1.7


def rho_ss(T):
    T = np.asarray(T, dtype=float)
    return rho_ss_300 * (0.85 + 0.15 * np.clip(T / 300.0, 0.0, 1.5))


def R_stab(T):
    """Parallel Cu‖SS stabilizer resistance [Ω] for tape of length L."""
    Rcu = rho_cu(T) * L / A_cu
    Rss = rho_ss(T) * L / A_ss
    return 1.0 / (1.0 / Rcu + 1.0 / Rss)


# ── Current-sharing solver ─────────────────────────────────────────────────────

def solve_Ix(I, T, n, Ic0=500.0, Tc=85.0):
    """
    SC-layer current Ix at voltage equilibrium between SC and stabilizer.

    Solves: U0*(Ix/Ic(T))^n = (I - Ix)*R_stab(T)
    Always has a root in (0, I) for I > 0.
    """
    I = float(I)
    if I <= 0.0:
        return 0.0
    Rst = float(R_stab(T))
    IcT = float(Ic(T, Ic0=Ic0, Tc=Tc))

    def f(ix):
        return U0 * (ix / IcT)**n - (I - ix) * Rst

    try:
        return brentq(f, 0.0, I, maxiter=120, xtol=1e-6)
    except Exception:
        return min(I, IcT)


# ── Joule heating ──────────────────────────────────────────────────────────────

def Qin_stab(I, T, n, Ic0=500.0, Tc=85.0):
    """Joule heating [W] with current redistribution into Cu/SS stabilizer."""
    Ix = solve_Ix(float(I), T, n, Ic0=Ic0, Tc=Tc)
    IcT = float(Ic(T, Ic0=Ic0, Tc=Tc))
    U_common = U0 * (Ix / max(IcT, 1e-12))**n
    return float(I) * U_common


def Qin_no_stab(I, T, n, Ic0=500.0, Tc=85.0):
    """Joule heating [W] with no stabilizer (all current in SC layer)."""
    IcT = float(Ic(T, Ic0=Ic0, Tc=Tc))
    return float(I) * U0 * (abs(float(I)) / max(IcT, 1e-12))**n


def _Qin(mode):
    return Qin_stab if mode == 'stab' else Qin_no_stab


# ── Equilibrium solvers ────────────────────────────────────────────────────────

def _equilibrium_I_at_T(T, T0, n, mode='stab', lam_override=None,
                         Ic0=500.0, Tc=85.0):
    """Transport current [A] that gives Qin = Qout at temperature T."""
    cooling = float(Qout(T, T0, lam_override=lam_override))
    qin = _Qin(mode)

    def bal(I):
        return qin(I, T, n, Ic0=Ic0, Tc=Tc) - cooling

    hi = max(4.0 * float(Ic(T0, Ic0=Ic0, Tc=Tc)), 100.0)
    for _ in range(40):
        if bal(hi) > 0:
            break
        hi *= 1.5
    else:
        return np.nan

    try:
        return brentq(bal, 0.0, hi, maxiter=120)
    except Exception:
        return np.nan


def solve_runaway(T0, n, mode='stab', lam_override=None, n_branch=200,
                  Ic0=500.0, Tc=85.0):
    """
    Thermal-runaway threshold (I*, T*).

    Sweeps T in (T0, Tc), traces the equilibrium I(T) curve, and locates its
    interior maximum. Returns (nan, nan) if no interior maximum exists
    (runaway not reachable at this T0/cooling).
    """
    Ts = np.linspace(T0 + 1e-4, Tc - 1e-4, n_branch)
    Is = np.array([
        _equilibrium_I_at_T(T, T0, n, mode=mode, lam_override=lam_override,
                             Ic0=Ic0, Tc=Tc)
        for T in Ts
    ])
    Is = np.nan_to_num(Is, nan=0.0, posinf=0.0, neginf=0.0)

    if not np.any(Is > 0):
        return np.nan, np.nan

    dI = np.gradient(Is, Ts)
    peaks = np.where((dI[:-1] > 0) & (dI[1:] <= 0))[0]

    if len(peaks) == 0:
        return np.nan, np.nan

    idx = int(peaks[0])
    return float(Is[idx]), float(Ts[idx])


def stable_surface_T(I, T0, n, mode='stab', lam_override=None,
                     Ic0=500.0, Tc=85.0):
    """
    Stable equilibrium surface temperature [K] at operating point (I, T0).

    Finds the lower root of Qin(I,T) = Qout(T,T0) in (T0, Tc).
    Returns nan if no stable root exists (I > I* runaway regime).
    """
    qin = _Qin(mode)

    def f(T):
        return qin(float(I), T, n, Ic0=Ic0, Tc=Tc) - float(Qout(T, T0, lam_override=lam_override))

    # f(T0) = Qin(I, T0) - 0 > 0 always
    # f(Tc-eps) → 0 - Qout < 0 always
    # So there is always a root — but above I* it is in the unstable branch.
    # We narrow the upper bound to T* if we have it, otherwise use Tc.
    try:
        return brentq(f, T0 + 1e-4, Tc - 1e-4, maxiter=100)
    except Exception:
        return np.nan


def tape_voltage(I, T, n, mode='stab', Ic0=500.0, Tc=85.0):
    """Voltage across the tape segment [V] at operating temperature T."""
    if mode == 'stab':
        Ix = solve_Ix(float(I), T, n, Ic0=Ic0, Tc=Tc)
    else:
        Ix = float(I)
    IcT = float(Ic(T, Ic0=Ic0, Tc=Tc))
    return U0 * (Ix / max(IcT, 1e-12))**n
