"""
physics_bi2223.py

Thermal-runaway heat-balance physics for a Bi-2223 tape conductor, mirroring
physics.py's REBCO model but using the EXPERIMENTAL characterization from
Section 3 of the same paper (Marchevsky & Prestemon, Supercond. Sci. Technol.
37 035012 (2024)) instead of the theoretical Ic(T) formula and Cu/SS
current-redistribution model used there.

Why a separate module, not a mode flag on physics.py
------------------------------------------------------
The paper itself draws this line: section 2 (physics.py) assumes the REBCO
conductor is "electrically uniform across its length and thickness" (p3) and
carries current re-distribution into a Cu/stainless stabilizer (section 2.3,
equations 7-8). Section 3's Bi-2223 validation conductor has no such
stabilizer model here -- it's Ic(T)/n(T)/eta measured directly and plugged
into the *no-stabilizer* heat balance (their equations 2-4, physics.py's
Qin_no_stab form). The paper's own Discussion (p10) explicitly says ReBCO is
"far more spatially inhomogeneous than the Bi-2223 conductor" -- these are
different conductors with different assumptions, not two settings of the
same model, so they get different code rather than a shared abstraction that
would blur the difference (see also hts_tape_env.fem_env for the
spatially-resolved REBCO complement, which is where that inhomogeneity is
actually addressed -- with real FEM sensor data rather than an invented
Ic(x) model this paper doesn't provide).

Source of every number below (Section 3 / Figure 6 / Figure 7, p7-9)
----------------------------------------------------------------------
    Conductor:  Sumitomo Electric HT-CA type Bi-2223 tape, 40 cm sample.
                Width 4.3 mm, thickness 0.43 mm, non-superconductor
                cross-section 1 mm^2 (Ag/AgMg matrix, not Cu/SS).
    Geometry:   voltage taps 305 mm apart -- the segment length used for the
                E-field criterion voltage U0 = Ec * L (same Ec=1e-4 V/m
                criterion as physics.py).
    I_c(T):     "fitted linearly" (p9); the paper gives only two hard values:
                I_c -> 0 at T_x = 91.35 K, with n = 1 assumed there. The low-T
                anchor (~200 A at the coldest measured point, T=77.16 K) and
                the intermediate n(T) values ARE NOT TABULATED IN THE PAPER
                -- Fig 6(e) is a scatter plot with no published digitization.
                IC_ANCHOR_A is read off the lower panel; n(T) anchors were
                re-digitized from the upper panel with additional points in
                the steep 82-86 K transition zone. Accuracy estimated ~5-10%
                (improved from the original ~10-20% eyeball). The solver is
                validated up to T0 ~ 83 K; above that n(T) < 5 makes the
                equilibrium branch increasingly poorly conditioned.
    eta:        the paper reports TWO measured heat-transfer coefficients:
                eta_h = 7.84 W/K (heater-based, Fig 6c) and eta_c = 4.56 W/K
                (self-heating from the actual transport current, Fig 7a).
                Their own runaway simulation (Fig 7b) found eta_c the better
                fit -- "correctly predicting the runaway current with a
                remarkably small error of just 0.54%" (p8) -- so that's the
                default here. Unlike physics.py's eta_cool (which scales
                W/K/m brass conductivity by 2wL/d), these are ALREADY the
                total conductance the paper measured for their assembly, so
                ETA_DEFAULT_W_PER_K is used as-is, not scaled by L.

No stabilizer heat-capacity / adiabatic time-margin model is provided here
(thermal.py's C_per_m is Cu/SS-specific) -- this module covers the runaway
boundary (I*, T*) only, matching what the paper's Bi-2223 section covers.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

# ── Reference geometry (Marchevsky & Prestemon 2024, Sec. 3, p7-8) ────────────
L = 0.305        # voltage-tap spacing used as the E-field criterion length [m]
w = 0.0043       # tape width [m]
d = 0.01         # thermal barrier (brass block) thickness [m] -- same as physics.py;
                 # not independently re-derived for Bi-2223 (eta is measured directly)
Ec = 1e-4        # E-field criterion [V/m] -- same convention as physics.py
U0 = Ec * L      # criterion voltage across the L=305 mm instrumented segment [V]

# ── Ic(T) / n(T) -- see module docstring: APPROXIMATE, read off Fig 6(e) ──────
T_ANCHOR_K = 77.16     # coldest measured base temperature (self-field LN2 bath)
T_X_K = 91.35           # paper's own number: I_c -> 0 here, n = 1 assumed here
IC_ANCHOR_A = 200.0     # APPROXIMATE -- I_c at T_ANCHOR_K, read off fig 6(e)

# n(T) anchors re-digitized from Fig 6(e) scatter plot using the PDF.
# Measured at the same base temperatures as the V-I curves in Fig 6(d).
# Added intermediate points in the steep 82-86 K transition region for
# better interpolation. The paper's actual fit was a 4th-power polynomial
# (p9) whose coefficients weren't published; this piecewise-linear
# reconstruction is smoother than the original eyeballed version.
_N_ANCHORS_T = np.array([77.16, 78.79, 79.96, 80.5,  81.36, 82.0,  82.42, 83.5,  84.25, 86.04, 88.33, T_X_K])
_N_ANCHORS_N = np.array([15.5,  16.0,  15.5,  15.5,  14.5,  13.0,  12.5,   9.5,   7.5,   3.0,   1.3,   1.0])


def Ic(T, Ic0=IC_ANCHOR_A, T_anchor=T_ANCHOR_K, T_x=T_X_K):
    """Critical current [A]: linear fit from (T_anchor, Ic0) to (T_x, 0),
    per the paper's stated linear I_c(T) fit (p9). Clipped to >=0 outside."""
    T = np.asarray(T, dtype=float)
    slope = -Ic0 / (T_x - T_anchor)
    val = Ic0 + slope * (T - T_anchor)
    return np.maximum(val, 1e-12)


def n_value(T, T_x=T_X_K):
    """Power-law n-value, monotone-interpolated through the approximate
    anchors read off fig 6(e) (see module docstring). Floored at 1 (the
    paper's assumed value at T_x, where the conductor is fully normal)."""
    T = np.asarray(T, dtype=float)
    return np.maximum(np.interp(T, _N_ANCHORS_T, _N_ANCHORS_N, left=_N_ANCHORS_N[0], right=1.0), 1.0)


# ── Cooling ─────────────────────────────────────────────────────────────────
ETA_DEFAULT_W_PER_K = 4.56   # self-heating measurement (fig 7a) -- paper's own
                              # best-fit value for the runaway simulation (p8)
ETA_HEATER_W_PER_K = 7.84    # alternate: heater-based measurement (fig 6c)


def eta_cool(eta_override=None):
    """Heat-transfer coefficient [W/K] for the measured assembly -- a measured
    constant here (not derived from a barrier-material lambda(T)), per the
    paper's own approach in Sec. 3. Pass eta_override to use ETA_HEATER_W_PER_K
    or a different measured value instead of the default self-heating one."""
    return ETA_DEFAULT_W_PER_K if eta_override is None else float(eta_override)


def Qout(T, T0, eta_override=None):
    """Heat removed to the LN2 bath [W]."""
    T = np.asarray(T, dtype=float)
    return eta_cool(eta_override) * np.maximum(T - T0, 0.0)


# ── Ag-matrix current redistribution (Phase 2) ────────────────────────────────
#
# The paper's Sec. 3 covers the Bi-2223 tape with NO stabilizer model.
# This extension adds Ag-matrix redistribution following the same voltage-
# equilibrium approach as physics.py's Cu/SS redistribution for REBCO
# (equations 7-8), but using the Ag/AgMg matrix resistance R_ag(T) from
# thermal_bi2223.py and the T-dependent n(T) from this module.
#
# At voltage equilibrium between SC layer and Ag matrix:
#     U0 * (Ix / Ic(T))^n(T) = (I - Ix) * R_ag(T)
#
# where Ix is the SC-layer current and (I - Ix) flows through the Ag matrix.
# Redistribution raises I* because excess current has a lower-resistance
# parallel path, reducing local Joule heating at any given (I, T).

def solve_Ix_ag(I, T, Ic0=IC_ANCHOR_A):
    """SC-layer current Ix at voltage equilibrium with the Ag/AgMg matrix.

    Solves: U0*(Ix/Ic(T))^n(T) = (I - Ix)*R_ag(T)
    Always has a root in (0, I) for I > 0.
    """
    from hts_tape_env.thermal_bi2223 import R_ag

    I = float(I)
    if I <= 0.0:
        return 0.0
    Rag = float(R_ag(T))
    IcT = float(Ic(T, Ic0=Ic0))
    nT = float(n_value(T))

    def f(ix):
        return U0 * (ix / max(IcT, 1e-12)) ** nT - (I - ix) * Rag

    try:
        return brentq(f, 0.0, I, maxiter=120, xtol=1e-6)
    except Exception:
        return min(I, IcT)


def Qin_stab_ag(I, T, Ic0=IC_ANCHOR_A):
    """Joule heating [W] with Ag-matrix current redistribution."""
    Ix = solve_Ix_ag(float(I), T, Ic0=Ic0)
    IcT = float(Ic(T, Ic0=Ic0))
    nT = float(n_value(T))
    U_common = U0 * (Ix / max(IcT, 1e-12)) ** nT
    return float(I) * U_common


# ── Joule heating ──────────────────────────────────────────────────────────────

def Qin(I, T, Ic0=IC_ANCHOR_A, mode='no_stab'):
    """Joule heating [W]. mode='no_stab' uses pure power-law E-J (paper Sec 3);
    mode='ag_stab' adds Ag-matrix current redistribution."""
    if mode == 'ag_stab':
        return Qin_stab_ag(I, T, Ic0=Ic0)
    IcT = float(Ic(T, Ic0=Ic0))
    nT = float(n_value(T))
    return float(I) * U0 * (abs(float(I)) / max(IcT, 1e-12)) ** nT


# ── Equilibrium solvers (same structure as physics.py) ────────────────────────

def _equilibrium_I_at_T(T, T0, Ic0=IC_ANCHOR_A, eta_override=None, mode='no_stab'):
    """Transport current [A] that gives Qin = Qout at temperature T."""
    cooling = float(Qout(T, T0, eta_override=eta_override))

    def bal(I):
        return Qin(I, T, Ic0=Ic0, mode=mode) - cooling

    hi = max(4.0 * float(Ic(T0, Ic0=Ic0)), 50.0)
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


def solve_runaway(T0, Ic0=IC_ANCHOR_A, eta_override=None, n_branch=200, T_x=T_X_K,
                   sanity_cap_factor=3.0, n_floor=3.0, mode='no_stab'):
    """
    Thermal-runaway threshold (I*, T*) for the Bi-2223 conductor: sweeps T in
    (T0, T_x), traces the equilibrium I(T) curve, and locates its interior
    maximum. Returns (nan, nan) if no interior maximum exists.

    mode='no_stab': pure power-law E-J (paper Sec 3).
    mode='ag_stab': Ag-matrix current redistribution (I* is higher).

    The search is restricted to the region where n(T) >= n_floor (default 3).
    Below this threshold the power-law E-J is too weak to produce a
    physically meaningful equilibrium maximum — the branch either rises
    monotonically or produces a nonphysical spike as n → 1. Any peak above
    sanity_cap_factor * Ic0 within the valid region is still rejected.
    """
    T_hi = T_x - 1e-4
    n_at_hi = float(n_value(T_hi))
    if n_at_hi < n_floor:
        from scipy.optimize import brentq as _brentq
        try:
            T_hi = _brentq(lambda T: float(n_value(T)) - n_floor, T0 + 1e-4, T_x, maxiter=50)
        except Exception:
            pass

    if T_hi <= T0 + 1e-4:
        return np.nan, np.nan

    Ts = np.linspace(T0 + 1e-4, T_hi, n_branch)
    Is = np.array([_equilibrium_I_at_T(T, T0, Ic0=Ic0, eta_override=eta_override, mode=mode)
                    for T in Ts])
    Is = np.nan_to_num(Is, nan=0.0, posinf=0.0, neginf=0.0)

    if not np.any(Is > 0):
        return np.nan, np.nan

    dI = np.gradient(Is, Ts)
    peaks = np.where((dI[:-1] > 0) & (dI[1:] <= 0))[0]

    if len(peaks) > 0:
        idx = int(peaks[0])
    else:
        idx = int(np.argmax(Is))
        if idx == 0 or idx == len(Is) - 1:
            return np.nan, np.nan

    if Is[idx] > sanity_cap_factor * Ic0:
        return np.nan, np.nan
    return float(Is[idx]), float(Ts[idx])


def stable_surface_T(I, T0, Ic0=IC_ANCHOR_A, eta_override=None, T_x=T_X_K,
                      T_hi=None, mode='no_stab'):
    """
    Stable equilibrium surface temperature [K] at operating point (I, T0).

    T_hi : upper search bound; pass the operating point's own T* (from
    solve_runaway) here whenever you have it.
    """
    def f(T):
        return Qin(I, T, Ic0=Ic0, mode=mode) - float(Qout(T, T0, eta_override=eta_override))

    hi = (T_x if T_hi is None else min(float(T_hi), T_x)) - 1e-4
    if hi <= T0 + 1e-4:
        return np.nan
    try:
        return brentq(f, T0 + 1e-4, hi, maxiter=100)
    except Exception:
        return np.nan


def tape_voltage(I, T, Ic0=IC_ANCHOR_A, mode='no_stab'):
    """Voltage across the L=305 mm instrumented segment [V] at temperature T."""
    if mode == 'ag_stab':
        Ix = solve_Ix_ag(float(I), T, Ic0=Ic0)
        IcT = float(Ic(T, Ic0=Ic0))
        nT = float(n_value(T))
        return U0 * (Ix / max(IcT, 1e-12)) ** nT
    IcT = float(Ic(T, Ic0=Ic0))
    nT = float(n_value(T))
    return U0 * (abs(float(I)) / max(IcT, 1e-12)) ** nT
