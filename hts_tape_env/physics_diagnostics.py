"""
physics_diagnostics.py

Diagnostic utilities that expose *why* the runaway boundary is a hard
prediction target, in terms of the underlying physics rather than model
architecture. Used by scripts/analyze_controller_physics.py to explain the
GRU surrogate's controller-level errors physically, not just statistically.

Two structural facts about solve_runaway's equilibrium branch drive most of
the difficulty:

  1. I*(T0, lambda) is, by construction, the *maximum* of the equilibrium
     branch I_eq(T): dI/dT = 0 exactly at the runaway point. So the live
     telemetry (which tracks I, not T directly) is *least* informative right
     where the controller needs it most -- a wide range of T maps to nearly
     the same I near the peak. branch_flatness() quantifies this.

  2. The runaway surface I*(T0, lambda) is steep near Tc (Ic(T) -> 0 there),
     so a fixed absolute error in the surrogate's internal state estimate
     produces a much larger *relative* I* error near Tc than in the middle of
     the T0 range. runaway_sensitivity() quantifies this via the analytic
     surface's own slope.
"""

from __future__ import annotations

import numpy as np

from hts_tape_env.physics import Ic, Tc, _equilibrium_I_at_T, solve_runaway
from hts_tape_env import physics_bi2223


def equilibrium_branch(T0: float, n: float, *, mode: str = "stab", lam_override=None,
                        Ic0: float = 500.0, Tc_: float = Tc, n_branch: int = 400):
    """
    The full I_eq(T) branch solve_runaway searches for its peak in. Exposed
    directly for diagnostics -- solve_runaway only returns the peak.

    Returns (Ts, Is) arrays.
    """
    Ts = np.linspace(T0 + 1e-4, Tc_ - 1e-4, n_branch)
    Is = np.array([
        _equilibrium_I_at_T(T, T0, n, mode=mode, lam_override=lam_override,
                             Ic0=Ic0, Tc=Tc_)
        for T in Ts
    ])
    return Ts, np.nan_to_num(Is, nan=0.0, posinf=0.0, neginf=0.0)


def branch_flatness(T0: float, n: float, *, mode: str = "stab", lam_override=None,
                     Ic0: float = 500.0, Tc_: float = Tc, n_branch: int = 400,
                     frac_below_peak: float = 0.95) -> float:
    """
    How flat is I_eq(T) just *before* the runaway peak? Returns
    |dI/dT| evaluated at the point on the branch where I first reaches
    `frac_below_peak` * I* (approaching from below) -- a proxy for how much
    the current changes for a given temperature change right before runaway.
    Small values mean the telemetry is nearly uninformative there (the whole
    point of "runaway": the branch is flattening toward its maximum).
    """
    Ts, Is = equilibrium_branch(T0, n, mode=mode, lam_override=lam_override,
                                 Ic0=Ic0, Tc_=Tc_, n_branch=n_branch)
    if not np.any(Is > 0):
        return np.nan
    peak_idx = int(np.argmax(Is))
    target = frac_below_peak * Is[peak_idx]
    idx = np.where(Is[:peak_idx] >= target)[0]
    if len(idx) == 0:
        return np.nan
    i = idx[0]
    if i == 0:
        return np.nan
    return float(abs((Is[i] - Is[i - 1]) / (Ts[i] - Ts[i - 1])))


def runaway_sensitivity(T0: float, lam: float, n: float, *, mode: str = "stab",
                         Ic0: float = 500.0, Tc_: float = Tc, n_branch: int = 200,
                         dT0: float = 0.5, dlam: float = 2.0) -> dict:
    """
    Finite-difference slope of the analytic runaway surface itself:
    dI*/dT0 and dI*/dlambda at (T0, lam), plus proximity to Tc (the physical
    regime where the surface is steepest and Ic(T0) is smallest, so absolute
    errors become large relative errors).
    """
    def istar(t0, l):
        Is, _ = solve_runaway(t0, n, mode=mode, lam_override=l, Ic0=Ic0, Tc=Tc_,
                               n_branch=n_branch)
        return Is

    i_hi, i_lo = istar(T0 + dT0, lam), istar(T0 - dT0, lam)
    dIdT0 = (i_hi - i_lo) / (2 * dT0) if np.isfinite(i_hi) and np.isfinite(i_lo) else np.nan

    i_hi, i_lo = istar(T0, lam + dlam), istar(T0, lam - dlam)
    dIdlam = (i_hi - i_lo) / (2 * dlam) if np.isfinite(i_hi) and np.isfinite(i_lo) else np.nan

    return dict(
        dIstar_dT0=float(dIdT0) if np.isfinite(dIdT0) else np.nan,
        dIstar_dlam=float(dIdlam) if np.isfinite(dIdlam) else np.nan,
        Tc_minus_T0=float(Tc_ - T0),
        Ic_T0=float(Ic(T0, Ic0=Ic0, Tc=Tc_)),
    )


def runaway_curvature(T0: float, lam: float, n: float, *, mode: str = "stab",
                       Ic0: float = 500.0, Tc_: float = Tc, n_branch: int = 200,
                       dT0: float = 1.0, dlam: float = 4.0) -> dict:
    """
    Finite-difference *curvature* (second derivatives) of the analytic
    runaway surface I*(T0, lambda) -- the actual nonlinearity metric, as
    opposed to runaway_sensitivity()'s slope. A first derivative can be large
    on a perfectly straight (locally linear) stretch of the surface; only the
    second derivative says the surface is *bending* there, which is what
    makes a point genuinely harder to interpolate/extrapolate.

    Physically this is expected to blow up near Tc (Ic(T0) -> 0 makes the
    I*(T0) curve drop steeply and then flatten near 0) and at low lambda
    (poor cooling narrows and bends the stable branch). Returns NaN where any
    of the 5-point stencil misses an interior runaway.
    """
    def istar(t0, l):
        Is, _ = solve_runaway(t0, n, mode=mode, lam_override=l, Ic0=Ic0, Tc=Tc_,
                               n_branch=n_branch)
        return Is

    i0 = istar(T0, lam)
    i_t_hi, i_t_lo = istar(T0 + dT0, lam), istar(T0 - dT0, lam)
    i_l_hi, i_l_lo = istar(T0, lam + dlam), istar(T0, lam - dlam)

    vals = [i0, i_t_hi, i_t_lo, i_l_hi, i_l_lo]
    if not all(np.isfinite(v) and v > 0 for v in vals):
        return dict(d2Istar_dT02=np.nan, d2Istar_dlam2=np.nan, curvature=np.nan)

    d2I_dT02 = (i_t_hi - 2 * i0 + i_t_lo) / dT0 ** 2
    d2I_dlam2 = (i_l_hi - 2 * i0 + i_l_lo) / dlam ** 2
    curvature = float(np.hypot(d2I_dT02, d2I_dlam2))
    return dict(d2Istar_dT02=float(d2I_dT02), d2Istar_dlam2=float(d2I_dlam2),
                curvature=curvature)


# ── Bi-2223 diagnostics ──────────────────────────────────────────────────────

def bi2223_equilibrium_branch(T0: float, *, Ic0: float = physics_bi2223.IC_ANCHOR_A,
                               eta_override=None, n_branch: int = 400):
    """Bi-2223 equilibrium branch I_eq(T). Returns (Ts, Is) arrays."""
    Ts = np.linspace(T0 + 1e-4, physics_bi2223.T_X_K - 1e-4, n_branch)
    Is = np.array([
        physics_bi2223._equilibrium_I_at_T(T, T0, Ic0=Ic0, eta_override=eta_override)
        for T in Ts
    ])
    return Ts, np.nan_to_num(Is, nan=0.0, posinf=0.0, neginf=0.0)


def bi2223_branch_flatness(T0: float, *, Ic0: float = physics_bi2223.IC_ANCHOR_A,
                            eta_override=None, n_branch: int = 400,
                            frac_below_peak: float = 0.95) -> float:
    """How flat is the Bi-2223 equilibrium branch near the runaway peak."""
    Ts, Is = bi2223_equilibrium_branch(T0, Ic0=Ic0, eta_override=eta_override,
                                        n_branch=n_branch)
    if not np.any(Is > 0):
        return np.nan
    peak_idx = int(np.argmax(Is))
    target = frac_below_peak * Is[peak_idx]
    idx = np.where(Is[:peak_idx] >= target)[0]
    if len(idx) == 0:
        return np.nan
    i = idx[0]
    if i == 0:
        return np.nan
    return float(abs((Is[i] - Is[i - 1]) / (Ts[i] - Ts[i - 1])))


def bi2223_runaway_sensitivity(T0: float, *, Ic0: float = physics_bi2223.IC_ANCHOR_A,
                                eta_override=None, dT0: float = 0.3) -> dict:
    """Finite-difference slope of the Bi-2223 I*(T0) curve."""
    def istar(t0):
        Is, _ = physics_bi2223.solve_runaway(t0, Ic0=Ic0, eta_override=eta_override)
        return Is

    i_hi, i_lo = istar(T0 + dT0), istar(T0 - dT0)
    dIdT0 = (i_hi - i_lo) / (2 * dT0) if np.isfinite(i_hi) and np.isfinite(i_lo) else np.nan

    return dict(
        dIstar_dT0=float(dIdT0) if np.isfinite(dIdT0) else np.nan,
        Tx_minus_T0=float(physics_bi2223.T_X_K - T0),
        Ic_T0=float(physics_bi2223.Ic(T0, Ic0=Ic0)),
    )


def bi2223_runaway_curvature(T0: float, *, Ic0: float = physics_bi2223.IC_ANCHOR_A,
                              eta_override=None, dT0: float = 0.5,
                              mode: str = 'no_stab') -> dict:
    """Finite-difference curvature of the Bi-2223 I*(T0) curve (1D — no λ)."""
    def istar(t0):
        Is, _ = physics_bi2223.solve_runaway(t0, Ic0=Ic0, eta_override=eta_override, mode=mode)
        return Is

    i0 = istar(T0)
    i_hi, i_lo = istar(T0 + dT0), istar(T0 - dT0)

    if not all(np.isfinite(v) and v > 0 for v in [i0, i_hi, i_lo]):
        return dict(d2Istar_dT02=np.nan, curvature=np.nan)

    d2I_dT02 = (i_hi - 2 * i0 + i_lo) / dT0 ** 2
    return dict(d2Istar_dT02=float(d2I_dT02), curvature=float(abs(d2I_dT02)))
