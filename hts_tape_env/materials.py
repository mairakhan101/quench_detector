"""
materials.py

Tape geometry and T-dependent material property abstraction for the 1D FD
solver.  Two factory functions produce a TapeGeometry:

    lumped_rebco_tape()   — two-layer (Cu + SS), matches physics.py constants
    freefem_rebco_tape()  — seven-layer, loads tabulated data from local-FreeFEM

The TapeGeometry computes effective longitudinal thermal conductivity k_eff(T),
heat capacity per unit length C_eff(T), and normal-metal conductance for the
Du Fort-Frankel solver in physics_fd1d.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np


# Lorenz number for Wiedemann-Franz law [W·Ω/K²]
L0_WF = 2.44e-8


@dataclass
class LayerProps:
    """T-dependent properties for one material layer in the tape stack."""
    name: str
    thickness: float                        # [m]
    rho_mass: float                         # mass density [kg/m³]
    cp: Callable[[float], float]            # specific heat [J/kg/K]
    k: Callable[[float], float]             # thermal conductivity [W/m/K]
    rho_e: Optional[Callable[[float], float]] = None  # electrical resistivity [Ω·m]; None = insulator
    is_sc: bool = False


@dataclass
class SCParams:
    """Superconductor transport parameters."""
    Ic_func: Callable[[float], float]       # Ic(T) [A]
    n_func: Callable[[float], float]        # n-value(T)
    Ec: float = 1e-4                        # E-field criterion [V/m]
    A_sc: float = 0.0                       # SC cross-section [m²]


@dataclass
class TapeGeometry:
    """1D tape: a stack of layers extruded along length L."""
    layers: list[LayerProps]
    L: float                                # tape length [m]
    w: float                                # tape width [m]
    sc: Optional[SCParams] = None

    def _A(self, layer: LayerProps) -> float:
        return layer.thickness * self.w

    def k_eff(self, T: float) -> float:
        """Effective longitudinal thermal conductivity × area [W·m/m = W]
        (actually k·A, to multiply by dT/dx and get heat flow [W])."""
        T = float(T)
        return sum(float(ly.k(T)) * self._A(ly) for ly in self.layers)

    def C_eff(self, T: float) -> float:
        """Heat capacity per unit length [J/m/K]."""
        T = float(T)
        return sum(ly.rho_mass * self._A(ly) * float(ly.cp(T)) for ly in self.layers)

    def normal_layers_conductance(self, T: float) -> float:
        """Total electrical conductance of normal-metal layers per unit length [S·m].
        G_per_m = sum(A_i / rho_i(T)).  Multiply by E [V/m] to get I_normal [A]."""
        T = float(T)
        g = 0.0
        for ly in self.layers:
            if ly.rho_e is not None and not ly.is_sc:
                rho = float(ly.rho_e(T))
                if rho > 0:
                    g += self._A(ly) / rho
        return g


# ── Factory: lumped REBCO tape matching physics.py ──────────────────────────

def lumped_rebco_tape(L: float = 1.0, w: float = 0.004,
                      Ic0: float = 500.0, Tc: float = 85.0,
                      n_value: float = 20.0) -> TapeGeometry:
    """Two-layer tape (Cu + SS) reproducing the constants in physics.py / thermal.py."""
    from hts_tape_env import physics, thermal

    t_cu = physics.t_cu   # 50e-6 m
    t_ss = physics.t_ss   # 50e-6 m

    def k_cu(T):
        return L0_WF * float(T) / float(physics.rho_cu(T))

    def k_ss(T):
        return L0_WF * float(T) / float(physics.rho_ss(T))

    cu = LayerProps(
        name="Cu", thickness=t_cu, rho_mass=thermal.rho_mass_cu,
        cp=thermal.cp_cu, k=k_cu, rho_e=physics.rho_cu,
    )
    ss = LayerProps(
        name="SS", thickness=t_ss, rho_mass=thermal.rho_mass_ss,
        cp=thermal.cp_ss, k=k_ss, rho_e=physics.rho_ss,
    )

    A_sc_nominal = 1e-6 * w / 0.004  # scale with width; ~1 µm × w

    def Ic_func(T):
        return float(physics.Ic(T, Ic0=Ic0, Tc=Tc))

    def n_func(_T):
        return n_value

    sc = SCParams(Ic_func=Ic_func, n_func=n_func, Ec=physics.Ec, A_sc=A_sc_nominal)
    return TapeGeometry(layers=[cu, ss], L=L, w=w, sc=sc)


# ── Factory: FreeFEM 7-layer REBCO tape ─────────────────────────────────────

_FREEFEM_DATA_DIR = Path("/data/maira/local-FreeFEM/2D/data")


def _load_table(filepath: Path) -> Callable[[float], float]:
    """Load a two-column (T, value) text file and return np.interp wrapper."""
    data = np.loadtxt(filepath)
    T_tab, V_tab = data[:, 0], data[:, 1]

    def interp(T):
        return float(np.interp(float(T), T_tab, V_tab))

    return interp


def freefem_rebco_tape(L: float = 0.20, w: float = 0.012,
                       data_dir: Optional[str | Path] = None,
                       Jc0: float = 8e10, Tc: float = 90.0,
                       n0: float = 20.0) -> TapeGeometry:
    """Seven-layer REBCO tape matching FreeFEM 2D model geometry and materials."""
    d = Path(data_dir) if data_dir else _FREEFEM_DATA_DIR

    cp_cu = _load_table(d / "cp_cu.txt")
    cp_ag = _load_table(d / "cp_ag.txt")
    cp_hast = _load_table(d / "cp_hast.txt")
    cp_buff = _load_table(d / "cp_buff.txt")
    cp_rebco = _load_table(d / "cp_rebco.txt")

    k_cu = _load_table(d / "k_cu.txt")
    k_ag = _load_table(d / "k_ag.txt")
    k_hast = _load_table(d / "k_hast.txt")
    k_buff = _load_table(d / "k_buff.txt")
    k_rebco = _load_table(d / "k_rebco.txt")

    rho_cu = _load_table(d / "rho_cu.txt")
    rho_ag = _load_table(d / "rho_silver.txt")
    rho_hast = _load_table(d / "rho_hast.txt")

    layers = [
        LayerProps("Cu_bot",    20e-6,  8960,  cp_cu,    k_cu,    rho_cu),
        LayerProps("Ag_bot",    1.8e-6, 10490, cp_ag,    k_ag,    rho_ag),
        LayerProps("Hastelloy", 50e-6,  8890,  cp_hast,  k_hast,  rho_hast),
        LayerProps("Buffer",    0.2e-6, 3600,  cp_buff,  k_buff),  # insulator
        LayerProps("REBCO",     1e-6,   6390,  cp_rebco, k_rebco, is_sc=True),
        LayerProps("Ag_top",    2e-6,   10490, cp_ag,    k_ag,    rho_ag),
        LayerProps("Cu_top",    20e-6,  8960,  cp_cu,    k_cu,    rho_cu),
    ]

    t_sc = 1e-6
    A_sc = t_sc * w

    def Jc_T(T):
        T = float(T)
        if T >= Tc:
            return 1.0
        return Jc0 * (Tc - T) / (Tc - 50.0)

    def Ic_func(T):
        return Jc_T(T) * A_sc

    def n_func(T):
        T = float(T)
        if T >= Tc:
            return 10.0
        return (n0 - 10.0) * (Tc - T) / (Tc - 50.0) + 10.0

    sc = SCParams(Ic_func=Ic_func, n_func=n_func, Ec=1e-4, A_sc=A_sc)
    return TapeGeometry(layers=layers, L=L, w=w, sc=sc)
