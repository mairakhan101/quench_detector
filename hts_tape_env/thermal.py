"""
thermal.py

Adiabatic time-margin physics for the HTS conductor — the Fig. 4 quantity from
Marchevsky & Prestemon (2024). Ported out of the exploratory notebook so that
dataset generation (datagen.py) has no notebook dependency.

All quantities SI (K, J, kg, W, s). Heat-capacity anchors are log-log
interpolated NIST-style values for Cu (RRR grade) and 304 SS.
"""

import numpy as np

from hts_tape_env.physics import A_cu, A_ss, R_stab

# ── Mass densities [kg/m^3] ───────────────────────────────────────────────────
rho_mass_cu = 8960.0
rho_mass_ss = 8000.0

# ── Cryogenic specific heat anchors [J/kg/K] vs T [K], log-log interpolated ────
_CU_TCP = np.array([4.0, 10.0, 20.0, 50.0, 77.0, 100.0, 300.0])
_CU_CP  = np.array([0.09, 3.0, 40.0, 160.0, 230.0, 280.0, 385.0])

_SS_TCP = _CU_TCP.copy()
_SS_CP  = np.array([0.20, 2.0, 10.0, 90.0, 180.0, 250.0, 500.0])


def _logp(T, Ta, Va):
    """Log-log interpolation of anchor table (Ta, Va), clamped to the anchor range."""
    T = np.asarray(T, dtype=float)
    Tc_ = np.clip(T, Ta[0], Ta[-1])
    return np.exp(np.interp(np.log(Tc_), np.log(Ta), np.log(Va)))


def cp_cu(T):
    """Copper specific heat [J/kg/K]."""
    return _logp(T, _CU_TCP, _CU_CP)


def cp_ss(T):
    """Stainless-steel specific heat [J/kg/K]."""
    return _logp(T, _SS_TCP, _SS_CP)


def C_per_m(T):
    """Volumetric heat capacity of the stabilizer per unit length [J/m/K]."""
    return rho_mass_cu * A_cu * cp_cu(T) + rho_mass_ss * A_ss * cp_ss(T)


def time_margin_to_safe(I, Tstar, T_target=300.0, dt=2e-4, t_max=1.0):
    """
    Fig. 4-style adiabatic time margin [s]: how long from the runaway point
    (I = I*, T = T*) until the conductor reaches T_target with no cooling.

        dT/dt = I^2 * R_stab(T) / C(T)

    Returns nan if T_target is not reached within t_max.
    """
    T = float(Tstar)
    n_steps = int(t_max / dt)

    for k in range(n_steps):
        Pgen = I ** 2 * float(R_stab(T))
        C = max(float(C_per_m(min(T, T_target))), 1e-12)
        T += dt * Pgen / C
        if T >= T_target:
            return (k + 1) * dt

    return np.nan
