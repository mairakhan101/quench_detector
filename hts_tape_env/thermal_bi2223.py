"""
thermal_bi2223.py

Adiabatic time-margin physics for the Bi-2223 conductor with Ag/AgMg matrix —
the Fig. 4 analog for the Bi-2223 tape. Mirrors thermal.py's structure but
uses silver (Ag) material properties and the Bi-2223 conductor geometry from
Marchevsky & Prestemon 2024, Section 3.

The paper doesn't provide this model for Bi-2223 (their Sec. 3 only covers
the runaway boundary I*/T*, not the post-runaway adiabatic transient). This
module supplies it using NIST cryogenic data for silver and the 1 mm^2
Ag/AgMg cross-section stated in the paper (p7).

Material property caveat: the NIST anchors below are for pure annealed
silver (RRR ~ 100). The actual Ag/AgMg alloy has significantly higher
resistivity due to alloy scattering — typically 7-12× pure Ag for
Ag-3%Mg at 77 K.  RHO_AG_MULTIPLIER (default 10.0) scales the NIST
values to approximate this.  Sensitivity analysis shows that the Bi-2223
Ag-stab runaway boundary (I*) only exists when the multiplier exceeds
~6×; at pure-Ag values (1×) the matrix shunts so effectively that no I*
exists, which is unrealistic.  The paper provides no measured
normal-state resistance, so the default 10× is a mid-range estimate.
The thermal properties (cp, density) are close to pure Ag at cryogenic
temperatures since the Mg content is small.
"""

import numpy as np

from hts_tape_env.physics_bi2223 import L  # 0.305 m segment length

# ── Bi-2223 conductor geometry (paper Sec. 3, p7-8) ─────────────────
A_ag = 1.0e-6          # non-SC (Ag/AgMg matrix) cross-section [m^2] = 1 mm^2
rho_mass_ag = 10490.0   # silver mass density [kg/m^3]

# Resistivity multiplier: pure Ag NIST anchors × this factor.
# Pure Ag (1.0) is unrealistically low for the Ag/AgMg alloy matrix.
# Ag-3%Mg at 77 K has ~7-12× the resistivity of pure Ag due to alloy
# scattering.  Default 10.0 is a mid-range estimate; the paper provides
# no measured normal-state resistance for this conductor.
RHO_AG_MULTIPLIER = 10.0

# ── Silver specific-heat anchors [J/kg/K] vs T [K], NIST ────────────
_AG_T_CP = np.array([4.0, 10.0, 20.0, 50.0, 77.0, 100.0, 200.0, 300.0])
_AG_CP   = np.array([0.03, 1.0, 13.0, 100.0, 170.0, 205.0, 225.0, 235.0])

# ── Silver resistivity anchors [Ohm-m] vs T [K], NIST ───────────────
_AG_T_RHO = np.array([4.0, 10.0, 20.0, 50.0, 77.0, 100.0, 200.0, 300.0])
_AG_RHO   = np.array([1.5e-10, 1.5e-10, 1.6e-10, 2.5e-10, 4.2e-10,
                       6.5e-10, 1.15e-8, 1.59e-8])


def _logp(T, Ta, Va):
    """Log-log interpolation of anchor table (Ta, Va), clamped to anchor range."""
    T = np.asarray(T, dtype=float)
    Tc_ = np.clip(T, Ta[0], Ta[-1])
    return np.exp(np.interp(np.log(Tc_), np.log(Ta), np.log(Va)))


def cp_ag(T):
    """Silver specific heat [J/kg/K]."""
    return _logp(T, _AG_T_CP, _AG_CP)


def rho_ag(T):
    """Ag/AgMg matrix resistivity [Ohm-m].

    Returns pure-Ag NIST values scaled by RHO_AG_MULTIPLIER to approximate
    the alloy.  Change the module-level constant or call set_rho_multiplier()
    to adjust.
    """
    return _logp(T, _AG_T_RHO, _AG_RHO) * RHO_AG_MULTIPLIER


def set_rho_multiplier(mult: float):
    """Set the Ag-matrix resistivity multiplier at runtime."""
    global RHO_AG_MULTIPLIER
    RHO_AG_MULTIPLIER = float(mult)


def R_ag(T):
    """Normal-state resistance of the Ag/AgMg matrix [Ohm] for the
    L = 305 mm instrumented segment."""
    return rho_ag(T) * L / A_ag


def C_per_m(T):
    """Heat capacity of the Ag/AgMg matrix per unit length [J/m/K]."""
    return rho_mass_ag * A_ag * cp_ag(T)


def time_margin_to_safe(I, Tstar, T_target=300.0, dt=2e-4, t_max=60.0):
    """
    Adiabatic time margin [s] for the Bi-2223 conductor: how long from the
    runaway point (I = I*, T = T*) until T_target, assuming all current
    flows through the Ag/AgMg matrix with no cooling.

        dT/dt = I^2 * rho_ag(T) / (A_ag * C_per_m(T))

    Uses per-unit-length power dissipation and per-unit-length heat capacity
    (not whole-segment resistance), so the result is independent of segment
    length L.  t_max defaults to 60 s because the Ag matrix heats slowly
    compared to the Cu/SS stabilizer in the REBCO conductor.

    Returns nan if T_target is not reached within t_max.
    """
    T = float(Tstar)
    n_steps = int(t_max / dt)

    for k in range(n_steps):
        Pgen_per_m = I ** 2 * float(rho_ag(T)) / A_ag
        C = max(float(C_per_m(min(T, T_target))), 1e-12)
        T += dt * Pgen_per_m / C
        if T >= T_target:
            return (k + 1) * dt

    return np.nan
