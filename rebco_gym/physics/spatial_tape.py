"""
Spatially-resolved REBCO tape: nz segments along tape length, each with
its own temperature, local critical current (set by a sampled Ic dropout
field and the local B-field via `electromagnetic.AnisotropicJcModel`), and
current-sharing dynamics. Longitudinal heat diffusion couples neighboring
segments so a local hot spot can propagate.

Physics pattern (current-sharing relaxation, power-law E-J, longitudinal
diffusion) follows the validated `SpatialTape` prototype in
test_rebco.ipynb section 1; this version replaces the uniform Ic0 with a
per-segment Ic0 driven by `IcDropoutField` and adds field-angle-dependent
Jc via `AnisotropicJcModel` instead of a pure T-only linear Ic(T).
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from .electromagnetic import AnisotropicJcModel, IcDropoutField, self_field


@dataclass
class SpatialTapeParams:
    length: float = 0.20        # m, tape length
    width: float = 12e-3        # m, tape width
    thickness: float = 95e-6    # m, total tape thickness (for heat capacity)
    Ic0: float = 400.0          # A, nominal critical current at T_bath, zero field
    Tc: float = 90.0            # K, critical temperature
    T_bath: float = 20.0        # K, bath/coolant temperature
    n_eff: float = 20.0         # power-law exponent
    Ec: float = 1e-4            # V/m, E-field criterion
    C_vol: float = 2.0e6        # J/(m^3 K), volumetric heat capacity
    kappa: float = 400.0        # W/(m K), longitudinal thermal conductivity
    h_cool: float = 50.0        # W/(m^2 K), surface cooling coefficient
    R_cu: float = 5e-4          # Ohm, Cu-stabilizer parallel resistance per segment
    tau_share: float = 2e-3     # s, current-sharing relaxation time


class SpatialTape:
    """nz-segment electrothermal model of a REBCO tape with EM-coupled Ic."""

    def __init__(
        self,
        nz: int = 40,
        params: Optional[SpatialTapeParams] = None,
        jc_model: Optional[AnisotropicJcModel] = None,
        dropout_field: Optional[IcDropoutField] = None,
        B_ext: float = 0.0,
        B_angle_rad: float = 0.0,
        self_field_coeff: float = 1.0,
        rng: Optional[np.random.Generator] = None,
    ):
        self.p = params or SpatialTapeParams()
        self.jc_model = jc_model or AnisotropicJcModel()
        self.dropout_field = dropout_field or IcDropoutField()
        self.B_ext = B_ext
        self.B_angle_rad = B_angle_rad
        self.self_field_coeff = self_field_coeff

        self.nz = nz
        self.dx = self.p.length / nz
        self.x = np.linspace(self.dx / 2, self.p.length - self.dx / 2, nz)
        self.A_cs = self.p.width * self.p.thickness
        self.C_seg = self.p.C_vol * self.A_cs * self.dx
        self.A_cool = 2.0 * self.p.width * self.dx

        rng = rng or np.random.default_rng()
        self.Ic0_local = self.p.Ic0 * self.dropout_field.sample(nz, self.p.length, rng)

        self.T = np.full(nz, self.p.T_bath)
        self.f_rebco = np.ones(nz)
        self._E = np.zeros(nz)
        self._I_total = 0.0

    def _field_fraction(self, I_total: float) -> float:
        """Jc(B,theta)/Jc(0) factor, shared by all segments (uniform self-field
        approximation: same transport current flows through every cross-section)."""
        B_self = self_field(I_total, self.p.width, self.self_field_coeff)
        B_mag = np.sqrt(B_self**2 + self.B_ext**2)
        B_par, B_perp = self.jc_model.resolve_components(B_mag, self.B_angle_rad)
        return self.jc_model.Jc_fraction(B_par, B_perp)

    def Ic(self, T: np.ndarray, I_total: float) -> np.ndarray:
        """Local critical current [A] per segment at temperature T and given
        operating current (which sets the self-field)."""
        T_frac = np.clip((self.p.Tc - T) / (self.p.Tc - self.p.T_bath), 0.0, 1.0)
        return self.Ic0_local * T_frac * self._field_fraction(I_total)

    def E_field(self, I: float, T: np.ndarray, f_rebco: np.ndarray) -> np.ndarray:
        Ic_arr = self.Ic(T, I)
        ratio = np.abs(I) * f_rebco / np.maximum(Ic_arr, 1e-10)
        E = np.minimum(self.p.Ec * np.power(np.maximum(ratio, 1e-20), self.p.n_eff), 1e3)
        above_Tc = Ic_arr < 1.0
        E = np.where(above_Tc, np.abs(I) * self.p.R_cu / self.dx, E)
        return E

    def step(self, I: float, dt: float) -> None:
        T, f = self.T, self.f_rebco
        Ic_arr = self.Ic(T, I)

        # Current-sharing relaxation toward the steady-state REBCO fraction
        f_ss = np.clip(Ic_arr / max(abs(I), 1e-10), 0.0, 1.0)
        alpha = min(dt / self.p.tau_share, 1.0) if self.p.tau_share > 0 else 1.0
        f = f + (f_ss - f) * alpha

        E = self.E_field(I, T, f)

        P_rebco = E * np.abs(I) * f * self.dx
        P_cu = (np.abs(I) * (1.0 - f)) ** 2 * self.p.R_cu
        P_joule = P_rebco + P_cu
        P_cool = self.p.h_cool * self.A_cool * (T - self.p.T_bath)

        d2T = np.zeros(self.nz)
        d2T[1:-1] = (T[:-2] - 2.0 * T[1:-1] + T[2:]) / self.dx**2
        d2T[0] = (T[1] - T[0]) / self.dx**2
        d2T[-1] = (T[-2] - T[-1]) / self.dx**2
        P_diff = self.p.kappa * self.A_cs * d2T

        dT = (P_joule - P_cool + P_diff) / self.C_seg * dt
        T_new = np.clip(T + dT, 4.0, 500.0)

        self.T = T_new
        self.f_rebco = f
        self._E = E
        self._I_total = I

    def apply_disturbance(self, x_center: float, width_m: float, delta_T: float) -> None:
        mask = np.abs(self.x - x_center) <= width_m / 2.0
        self.T[mask] += delta_T

    def max_temperature(self) -> float:
        return float(self.T.max())

    def min_local_margin(self) -> float:
        """Worst-case I/Ic_local across all segments (>1 means overcritical)."""
        Ic_arr = self.Ic(self.T, self._I_total)
        margin = np.abs(self._I_total) / np.maximum(Ic_arr, 1e-9)
        return float(np.max(margin))

    def terminal_voltage(self) -> float:
        """Total tape voltage: sum of local E-field * segment length."""
        return float(np.sum(self._E * self.dx))

    def qa_signal(self, qa_model, y_offset: float = 0.075) -> float:
        """
        Aggregate quench-antenna voltage response summed over segments,
        weighted by local E-field (proxy for local dissipation / flux
        change). Reuses QuenchAntennaModel.compute_response_at_location
        with each segment mapped to a position along the real axis,
        centered on the tape midpoint.
        """
        Emax = float(np.max(self._E)) if np.max(self._E) > 0 else 1.0
        total = 0.0
        for xi, Ei in zip(self.x, self._E):
            if Ei <= 0:
                continue
            z_q = complex(xi - self.p.length / 2.0, y_offset)
            total += qa_model.compute_response_at_location(z_q, amplitude=Ei / Emax)
        return total

    def reset(self, T_bath: Optional[float] = None, rng: Optional[np.random.Generator] = None):
        if T_bath is not None:
            self.p.T_bath = T_bath
        rng = rng or np.random.default_rng()
        self.Ic0_local = self.p.Ic0 * self.dropout_field.sample(self.nz, self.p.length, rng)
        self.T = np.full(self.nz, self.p.T_bath)
        self.f_rebco = np.ones(self.nz)
        self._E = np.zeros(self.nz)
        self._I_total = 0.0
