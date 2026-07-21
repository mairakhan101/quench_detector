"""
Thermal runaway physics for REBCO tape.

Implements the model from:
  M. Marchevsky, S. Prestemon, "Thermal runaway criterion as a basis
  for the protection of high-temperature superconductor magnets",
  Supercond. Sci. Technol. 37, 2024.  DOI: 10.1088/1361-6668/ad20fe

Two cases:
  Case 1 – bare SC (no stabilizer): power-law E-J + linear cooling
  Case 2 – with Cu/SS stabilizer and current sharing (Eq 7-8)

The key idea: thermal runaway occurs when no thermal equilibrium
(Q_in = Q_out) exists. The runaway current I* is the maximum current
for which a stable equilibrium temperature T* still exists.
"""

import numpy as np
from scipy.optimize import root_scalar
from dataclasses import dataclass, field
from typing import Optional, Tuple


# ---------------------------------------------------------------
# Material property fits
# ---------------------------------------------------------------
def lambda_brass(T: float) -> float:
    """Brass thermal conductivity [W/m/K] from NIST polynomial fit."""
    a, b, c, d, e, f, g, h = (
        0.021035, -1.01835, 4.54083, -5.03374,
        3.20536, -1.12933, -0.174057, -0.0038151,
    )
    T = np.clip(T, 4.0, 300.0)
    x = np.log10(T)
    log10_lam = a + b*x + c*x**2 + d*x**3 + e*x**4 + f*x**5 + g*x**6 + h*x**7
    log10_lam = np.clip(log10_lam, -2.0, 2.3)
    return float(10**log10_lam)


def Ic_T(T: float, Ic0: float = 500.0, Tc: float = 85.0) -> float:
    """Critical current Ic(T) = Ic0 * (1 - (T/Tc)^1.7).  (Eq. 5 of paper)"""
    T = np.clip(T, 0.0, 300.0)
    val = Ic0 * (1.0 - (T / Tc) ** 1.7)
    return float(max(val, 1e-12))


def eta_from_geometry(
    lmbda: float,
    width: float = 0.004,
    length: float = 1.0,
    d_barrier: float = 0.01,
) -> float:
    """Effective linear cooling coefficient eta = 2*w*L/d * lambda."""
    return (2.0 * width * length / d_barrier) * lmbda


def rho_cu(T: float, RRR: float = 24.0) -> float:
    """Copper resistivity [Ohm·m] — simple monotone approx."""
    rho_300 = 1.6e-8
    rho_0 = rho_300 / RRR
    T_c = np.clip(T, 1e-3, 300.0)
    rho_ph = (rho_300 - rho_0) * (T_c / 300.0) ** 1.5
    return rho_0 + rho_ph


def rho_ss(T: float) -> float:
    """Stainless steel resistivity [Ohm·m] — weakly T-dependent at cryo."""
    rho_300 = 7.0e-7
    rho_20 = 5.5e-7
    T_c = np.clip(T, 1e-3, 300.0)
    a = (rho_300 - rho_20) / (np.log(300.0) - np.log(20.0))
    b = rho_20 - a * np.log(20.0)
    return a * np.log(T_c) + b


# ---------------------------------------------------------------
# Thermal runaway model
# ---------------------------------------------------------------
@dataclass
class ThermalRunawayModel:
    """
    Encapsulates the thermal runaway physics for a single REBCO tape element.

    Parameters match the Marchevsky & Prestemon paper defaults.
    """
    # Superconductor parameters
    Ic0: float = 500.0          # A, critical current at base T
    Tc: float = 85.0            # K, irreversibility temperature
    n_value: float = 20.0       # power-law exponent
    U0: float = 1e-4            # V/m, criterion electric field

    # Geometry
    width: float = 0.004        # m, tape width
    length: float = 1.0         # m, element length
    d_barrier: float = 0.01     # m, thermal barrier thickness

    # Stabilizer layer geometry (Case 2)
    t_sc: float = 1.3e-6        # m, SC layer thickness
    t_cu: float = 50e-6         # m, copper stabilizer
    t_ss: float = 50e-6         # m, stainless steel

    # Cooling
    s_eta: float = 1.0          # cooling scale factor
    C_eff: float = 20.0         # J/(m·K), effective heat capacity per unit length

    # Operating state (mutable)
    T: float = 20.0             # K, current temperature
    T0: float = 20.0            # K, coolant/base temperature
    I_op: float = 0.0           # A, operating current

    # RRR for copper
    RRR: float = 24.0

    def Ic(self, T: Optional[float] = None) -> float:
        if T is None:
            T = self.T
        return Ic_T(T, self.Ic0, self.Tc)

    def eta(self, T: Optional[float] = None) -> float:
        if T is None:
            T = self.T
        lam = lambda_brass(T)
        return eta_from_geometry(
            self.s_eta * lam, self.width, self.length, self.d_barrier
        )

    # --- Case 1: no stabilizer ---
    def R_sc(self, I: float, T: float) -> float:
        """SC resistance per unit length (power-law)."""
        Ic_val = self.Ic(T)
        return (self.U0 / max(I, 1e-15)) * (I / Ic_val) ** self.n_value

    def Qin_case1(self, I: float, T: float) -> float:
        return I**2 * self.R_sc(I, T)

    def Qout(self, T: float) -> float:
        return self.eta(T) * (T - self.T0)

    # --- Case 2: with stabilizer ---
    def _Rst(self, T: float) -> float:
        """Parallel resistance of Cu + SS stabilizer layers."""
        A_cu = self.width * self.t_cu
        A_ss = self.width * self.t_ss
        R_cu = rho_cu(T, self.RRR) / A_cu
        R_ss = rho_ss(T) / A_ss
        return 1.0 / (1.0 / R_cu + 1.0 / R_ss)

    def _solve_Ix(self, I: float, T: float) -> Optional[float]:
        """Solve current sharing: U0*(Ix/Ic)^n = (I-Ix)*Rst  (Eq. 8)."""
        if I <= 0:
            return 0.0
        Ic_val = self.Ic(T)
        Rst_val = self._Rst(T)

        def g(Ix):
            return self.U0 * (Ix / Ic_val) ** self.n_value - (I - Ix) * Rst_val

        g0 = g(0.0)
        gI = g(I)
        if np.isnan(g0) or np.isnan(gI):
            return None
        if g0 > 0:
            return 0.0
        if gI < 0:
            return I
        sol = root_scalar(g, bracket=[0.0, I], method="bisect",
                          xtol=1e-10, rtol=1e-10, maxiter=200)
        return sol.root if sol.converged else None

    def Reff_stabilized(self, I: float, T: float) -> float:
        """Effective parallel resistance of SC + stabilizer."""
        Ix = self._solve_Ix(I, T)
        if Ix is None:
            return np.nan
        if Ix <= 0:
            Rs = np.inf
        else:
            Rs = (self.U0 / Ix) * (Ix / self.Ic(T)) ** self.n_value
        Rst_val = self._Rst(T)
        if np.isinf(Rs):
            return Rst_val
        return (Rs * Rst_val) / (Rs + Rst_val)

    def Qin_case2(self, I: float, T: float) -> float:
        return I**2 * self.Reff_stabilized(I, T)

    # --- Dynamics ---
    def dTdt(self, I: float, T: float, stabilized: bool = True) -> float:
        """Temperature rate of change [K/s]."""
        if stabilized:
            qin = self.Qin_case2(I, T)
        else:
            qin = self.Qin_case1(I, T)
        qout = self.Qout(T)
        return (qin - qout) / self.C_eff

    def step(self, dt: float, stabilized: bool = True) -> float:
        """Explicit Euler step. Returns new temperature."""
        dT = self.dTdt(self.I_op, self.T, stabilized) * dt
        self.T = float(np.clip(self.T + dT, 4.0, 500.0))
        return self.T

    def is_quenched(self, dT_crit: float = 12.0) -> bool:
        """Has temperature risen above T0 + dT_crit?"""
        return self.T >= self.T0 + dT_crit

    def is_runaway(self) -> bool:
        """Has temperature exceeded Tc?"""
        return self.T >= self.Tc

    def margin_to_runaway(self) -> float:
        """I*/Ic(T0) — how close operating current is to runaway."""
        return self.I_op / max(self.Ic(self.T0), 1e-12)

    def reset(self, T0: float = 20.0, I_op: float = 0.0):
        self.T = T0
        self.T0 = T0
        self.I_op = I_op

    # --- Static analysis: find runaway boundary ---
    def find_runaway_current(
        self, T0: float, eta_val: float, stabilized: bool = True
    ) -> Tuple[float, float]:
        """
        Find the runaway current I* and equilibrium temperature T*
        for given base temperature and cooling.

        Returns (I_star, T_star) or (nan, nan) if not found.
        """
        if stabilized:
            balance_fn = lambda T, I: self.Qin_case2(I, T) - eta_val * (T - T0)
            equil_fn = lambda I: self._find_equil(balance_fn, I, T0)
        else:
            balance_fn = lambda T, I: self.Qin_case1(I, T) - eta_val * (T - T0)
            equil_fn = lambda I: self._find_equil(balance_fn, I, T0)

        Ic_base = self.Ic(T0)
        I_low, I_high = 0.0, 5.0 * Ic_base

        while equil_fn(I_high) is not None and I_high < 50 * Ic_base:
            I_high *= 2.0

        if equil_fn(I_high) is not None:
            return np.nan, np.nan

        for _ in range(60):
            I_mid = 0.5 * (I_low + I_high)
            if equil_fn(I_mid) is not None:
                I_low = I_mid
            else:
                I_high = I_mid

        T_star = equil_fn(I_low)
        return I_low, T_star if T_star is not None else np.nan

    def _find_equil(self, balance_fn, I, T0, nscan=400):
        Ts = np.linspace(T0 + 1e-6, self.Tc - 1e-3, nscan)
        fs = np.array([balance_fn(T, I) for T in Ts])
        for k in range(len(Ts) - 1):
            f1, f2 = fs[k], fs[k + 1]
            if np.isnan(f1) or np.isnan(f2):
                continue
            if f1 * f2 < 0:
                try:
                    sol = root_scalar(
                        lambda T: balance_fn(T, I),
                        bracket=[Ts[k], Ts[k + 1]],
                        method="bisect", xtol=1e-8, maxiter=200,
                    )
                    if sol.converged:
                        return sol.root
                except Exception:
                    pass
        return None
