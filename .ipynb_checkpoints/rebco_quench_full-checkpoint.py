#!/usr/bin/env python3
"""
rebco_quench_full.py
====================
Unified REBCO Superconducting Tape Electrothermal Quench Model
with Quench Antenna (QA) Detection.

Systems:
  1. Single-tape electrothermal (initiation physics)
  2. Two-tape with RL-circuit inter-tape transfer (mitigation path)
  3. QA complex-potential E-field model (detection / localization)

COMSOL reference (Rose Powers, Princeton, May 2026):
  - 2D cross-section 200mm x 12mm x 95µm REBCO tape
  - Power-law E-J: sigma = (Jc/Ec)*(J/Jc)^(n-1), n_base=20
  - Jc(T) linear from Jc(50K) to 0 at Tc=90K, T_bath=77K
  - At 2xJc: thermal runaway ~25ms (2D with inductance)
  - Our 0-D lumped model gives ~10x slower (no inductance, no gradients)

Run:
  python rebco_quench_full.py          # all validation + plots
  python rebco_quench_full.py --quick  # single + two tape only
"""

import numpy as np
import os
import sys
from dataclasses import dataclass
from typing import Dict, Tuple, Optional


# ================================================================
# SECTION 1: Material Properties
# ================================================================
# Temperature-dependent resistivity [Ohm·m] and volumetric heat
# capacity [J/m^3/K] for each layer material.
# Linear approximations valid 50-300K range.
# ================================================================

def rho_cu(T):
    """Copper resistivity [Ohm·m]. RRR~100."""
    return 1.7e-10 * (T / 300) + 1.5e-10

def rho_ag(T):
    """Silver resistivity [Ohm·m]."""
    return 1.5e-10 * (T / 300) + 1.0e-10

def rho_hast(T):
    """Hastelloy C-276 resistivity [Ohm·m]. Nearly constant."""
    return 1.25e-6 * (1 + 3e-4 * (T - 77))

def cv_cu(T):
    """Copper volumetric heat capacity [J/m^3/K]."""
    return 1.5e6 + 6300 * (T - 77)

def cv_ag(T):
    """Silver volumetric heat capacity [J/m^3/K]."""
    return 1.4e6 + 5500 * (T - 77)

def cv_hast(T):
    """Hastelloy volumetric heat capacity [J/m^3/K]."""
    return 2.5e6 + 3000 * (T - 77)

def cv_rebco(T):
    """REBCO volumetric heat capacity [J/m^3/K]."""
    return 1.2e6 + 4000 * (T - 77)

_RHO_FN = {"cu": rho_cu, "ag": rho_ag, "hast": rho_hast}
_CV_FN  = {"cu": cv_cu,  "ag": cv_ag,  "hast": cv_hast, "rebco": cv_rebco}


# ================================================================
# SECTION 2: REBCO Power-Law E-J
# ================================================================
# E = Ec * (J/Jc(T))^n(T)
#
# COMSOL equivalent: sigma = (Jc/Ec)*(J/Jc)^(n-1) => E = Ec*(J/Jc)^n
#
# n(T_op=77K) = n_base = 20 (matches COMSOL baseline).
# n degrades linearly to n_min=5 at Tc=90K.
# Jc(T) = Jc_ref * (Tc-T)/(Tc-T_ref), linear to zero at Tc.
# ================================================================

class REBCO:
    """
    REBCO power-law E(J,T) model.

    Parameters
    ----------
    Jc_ref : float  — Jc at T_ref [A/m^2]. Default 6e10.
    T_ref  : float  — reference temperature [K]. Default 50K.
    Tc     : float  — critical temperature [K]. Default 90K.
    n_base : float  — exponent at T_op. Default 20 (COMSOL match).
    n_min  : float  — exponent at Tc. Default 5.
    T_op   : float  — operating temperature [K]. Default 77K.
    Ec     : float  — E-field criterion [V/m]. Default 1e-4.
    """

    def __init__(self, Jc_ref=6e10, T_ref=50.0, Tc=90.0,
                 n_base=20, n_min=5, T_op=77.0, Ec=1e-4):
        self.Jc_ref, self.T_ref, self.Tc = Jc_ref, T_ref, Tc
        self.n_base, self.n_min, self.T_op, self.Ec = n_base, n_min, T_op, Ec

    def Jc(self, T):
        """Critical current density [A/m^2] at temperature T."""
        T = np.asarray(T, dtype=float)
        return self.Jc_ref * np.clip((self.Tc - T) / (self.Tc - self.T_ref), 0, 1)

    def n(self, T):
        """Power-law exponent at temperature T."""
        T = np.asarray(T, dtype=float)
        frac = np.clip((self.Tc - T) / (self.Tc - self.T_op), 0, 1)
        return self.n_min + (self.n_base - self.n_min) * frac

    def E(self, J, T):
        """Electric field [V/m]."""
        Jc = self.Jc(T)
        n = self.n(T)
        return self.Ec * np.power(np.abs(J) / np.maximum(Jc, 1.0), n)


# ================================================================
# SECTION 3: Tape Layer Stack
# ================================================================
# Standard REBCO tape (from COMSOL model):
#   Cu(20µm) / Ag(2µm) / REBCO(1µm) / Hastelloy(50µm) / Ag(2µm) / Cu(20µm)
# Total: 95µm thick, 12mm wide, 200mm long.
# ================================================================

@dataclass
class Layer:
    """One layer in the tape cross-section."""
    name: str         # e.g. "cu_top", "rebco"
    thickness: float  # [m]
    mat: str          # material key

STANDARD_TAPE = [
    Layer("cu_top",  20e-6, "cu"),
    Layer("ag_top",   2e-6, "ag"),
    Layer("rebco",    1e-6, "rebco"),
    Layer("hast",    50e-6, "hast"),
    Layer("ag_bot",   2e-6, "ag"),
    Layer("cu_bot",  20e-6, "cu"),
]


# ================================================================
# SECTION 4: Current Sharing Solver
# ================================================================
# All layers electrically parallel, common E-field.
# For total current I and temperature T, find E such that
#   sum_i J_i(E) * A_i = I
#
# Normal metal: J = E / rho(T)
# REBCO: J = Jc * (E/Ec)^(1/n)  (inverted power law)
#
# Log-space bisection for robust convergence across decades.
# ================================================================

def solve_E(layers, I_total, T, width, rebco, tol=1e-12):
    """
    Find common E-field across parallel layers for given I_total, T.

    Returns (E, I_per_layer_dict, J_per_layer_dict).
    """
    I_total = abs(float(I_total))
    T = float(T)

    def I_at_E(E):
        s = 0.0
        for L in layers:
            A = L.thickness * width
            if L.mat == "rebco":
                Jc = rebco.Jc(T); n = rebco.n(T)
                J = Jc * (E / rebco.Ec) ** (1.0 / n) if Jc >= 1.0 else E / 1e-6
            else:
                J = E / _RHO_FN[L.mat](T)
            s += J * A
        return s

    # Log-space bisection
    E_lo, E_hi = 1e-20, 1e2
    for _ in range(250):
        E_mid = np.sqrt(E_lo * E_hi)
        if I_at_E(E_mid) < I_total:
            E_lo = E_mid
        else:
            E_hi = E_mid
        if E_hi / E_lo < 1 + tol:
            break

    E = np.sqrt(E_lo * E_hi)
    I_lay, J_lay = {}, {}
    for L in layers:
        A = L.thickness * width
        if L.mat == "rebco":
            Jc = rebco.Jc(T); n = rebco.n(T)
            J = Jc * (E / rebco.Ec) ** (1.0 / n) if Jc >= 1.0 else E / 1e-6
        else:
            J = E / _RHO_FN[L.mat](T)
        I_lay[L.name] = J * A
        J_lay[L.name] = J
    return E, I_lay, J_lay


def tape_heat_capacity(layers, width, length, T):
    """Total tape heat capacity [J/K]."""
    return sum(_CV_FN[L.mat](T) * L.thickness * width * length for L in layers)


# ================================================================
# SECTION 5: Single Tape Electrothermal Model
# ================================================================
# 0-D lumped: C(T)*dT/dt = P_joule - P_cooling
# P_joule = E*I*length (all-layer dissipation)
# P_cooling = h_eff * 2*width*length * (T - T_bath)
#
# Thermal runaway: I > Ic(T) → current sharing → Joule heating
# → T rises → Jc drops → more sharing → positive feedback.
# Single tape has NO alternate SC path — current can only go to
# Cu stabilizer, which accelerates runaway.
# ================================================================

@dataclass
class TapeState:
    """State of a single tape at one timestep."""
    time: float = 0.0
    T: float = 77.0
    I_total: float = 0.0
    E: float = 0.0
    V: float = 0.0           # tape voltage = E * length
    I_rebco: float = 0.0
    I_cu: float = 0.0
    f_rebco: float = 1.0     # fraction of current in REBCO
    Jc: float = 0.0
    Ic: float = 0.0
    P_joule: float = 0.0
    P_cool: float = 0.0
    dTdt: float = 0.0
    dVdt: float = 0.0        # voltage rate — primary QA proxy


class SingleTape:
    """
    0-D lumped electrothermal model of one REBCO tape.

    h_eff=0 for adiabatic (worst case quench).
    h_eff=500 for modest LN2 cooling.
    """

    def __init__(self, layers=None, rebco=None,
                 width=12e-3, length=0.20,
                 T_bath=77.0, h_eff=500.0):
        self.layers = layers or list(STANDARD_TAPE)
        self.rebco = rebco or REBCO()
        self.width, self.length = width, length
        self.T_bath, self.h_eff = T_bath, h_eff
        for L in self.layers:
            if L.mat == "rebco":
                self._A_rebco = L.thickness * width
                break

    def Ic(self, T):
        """Critical current [A] at temperature T."""
        return self.rebco.Jc(T) * self._A_rebco

    def step(self, s: TapeState, I: float, dt: float) -> TapeState:
        """Advance one timestep. Returns new TapeState."""
        T, I = s.T, abs(I)
        E, I_lay, _ = solve_E(self.layers, I, T, self.width, self.rebco)

        # Thermal balance
        Pj = E * I * self.length
        Pc = self.h_eff * 2 * self.width * self.length * (T - self.T_bath)
        C = tape_heat_capacity(self.layers, self.width, self.length, T)
        dTdt = (Pj - Pc) / max(C, 1e-20)
        T_new = max(T + dTdt * dt, self.T_bath)

        V = E * self.length
        dVdt = (V - s.V) / dt if dt > 0 else 0.0
        Ir = I_lay.get("rebco", 0.0)
        Icu = sum(v for k, v in I_lay.items() if "cu" in k)
        fr = min(Ir / max(I, 1.0), 1.0)
        Jc_n = self.rebco.Jc(T_new)
        Ic_n = Jc_n * self._A_rebco

        return TapeState(
            time=s.time + dt, T=T_new, I_total=I,
            E=E, V=V, I_rebco=Ir, I_cu=Icu, f_rebco=fr,
            Jc=Jc_n, Ic=Ic_n, P_joule=Pj, P_cool=Pc,
            dTdt=dTdt, dVdt=dVdt)


# ================================================================
# SECTION 6: Two-Tape Model with RL-Circuit Inter-Tape Transfer
# ================================================================
# Two tapes sharing I_total. Current split by alpha in [0,1].
#   I_a = alpha * I_total,  I_b = (1-alpha) * I_total
#
# Natural inter-tape transfer via RL circuit:
#   L * dI_xfer/dt + R * I_xfer = V_a - V_b
# Gives smooth transfer with tau = L/R (default 1ms).
#
# When tape A quenches: V_a >> V_b → current flows A→B naturally.
# KEY vs single tape: current goes to healthy tape's REBCO,
# not just to Cu stabilizer. Fundamentally different physics.
#
# QA dipole signal:
#   E_dipole ~ (mu0/2pi) * d_sep * |dI_xfer/dt| / r_qa^2
#   Inter-tape signal ~20x stronger than intra-tape (REBCO→Cu)
#   because d_sep (~1mm) >> layer thickness (~50µm).
# ================================================================

@dataclass
class TwoTapeState:
    """State of the two-tape system."""
    time: float = 0.0
    T_a: float = 77.0;  I_a: float = 0.0;  E_a: float = 0.0
    V_a: float = 0.0;   f_rebco_a: float = 1.0;  Ic_a: float = 0.0
    T_b: float = 77.0;  I_b: float = 0.0;  E_b: float = 0.0
    V_b: float = 0.0;   f_rebco_b: float = 1.0;  Ic_b: float = 0.0
    I_transfer: float = 0.0   # A→B current [A]
    V_diff: float = 0.0       # V_a - V_b
    E_dipole: float = 0.0     # QA signal [V/m]
    dE_dt: float = 0.0


class TwoTape:
    """
    Two-tape model with RL-circuit inter-tape transfer.

    R_contact : inter-tape contact resistance [Ohm]. Default 1mΩ.
    L_transfer: transfer inductance [H]. Default 1µH → tau=1ms.
    d_separation: tape center-to-center [m]. Default 1mm.
    r_qa: QA distance [m]. Default 50mm.
    """

    def __init__(self, tape_a=None, tape_b=None,
                 R_contact=1e-3, L_transfer=1e-6,
                 d_separation=1e-3, r_qa=0.05):
        self.tape_a = tape_a or SingleTape(h_eff=0.0)
        self.tape_b = tape_b or SingleTape(h_eff=0.0)
        self.R_contact = R_contact
        self.L_transfer = L_transfer
        self.d_sep = d_separation
        self.r_qa = r_qa

    def step(self, s: TwoTapeState, I_total: float, alpha: float,
             dt: float) -> TwoTapeState:
        """Advance one timestep. alpha = fraction to tape A."""
        I_total = abs(I_total)
        alpha = np.clip(alpha, 0.01, 0.99)

        # Commanded split
        Ia_cmd = alpha * I_total
        Ib_cmd = (1 - alpha) * I_total

        # RL-circuit natural transfer: exponential relaxation
        V_diff = s.V_a - s.V_b
        I_eq = V_diff / max(self.R_contact, 1e-12)
        tau = self.L_transfer / max(self.R_contact, 1e-12)
        Ix = s.I_transfer + (I_eq - s.I_transfer) * min(dt / tau, 1.0)
        Ix = np.clip(Ix, -0.4 * I_total, 0.4 * I_total)

        # Actual currents (command + natural transfer)
        Ia = max(Ia_cmd - Ix, 0.0)
        Ib = max(Ib_cmd + Ix, 0.0)
        Is = Ia + Ib
        if Is > 0:
            Ia *= I_total / Is
            Ib *= I_total / Is

        # Step each tape
        sa = self.tape_a.step(TapeState(time=s.time, T=s.T_a, V=s.V_a), Ia, dt)
        sb = self.tape_b.step(TapeState(time=s.time, T=s.T_b, V=s.V_b), Ib, dt)

        # QA dipole E-field from current redistribution
        dIx_dt = (Ix - s.I_transfer) / dt if dt > 0 else 0.0
        mu0 = 4e-7 * np.pi
        Ed = mu0 / (2 * np.pi) * self.d_sep * abs(dIx_dt) / max(self.r_qa**2, 1e-10)
        dEdt = (Ed - s.E_dipole) / dt if dt > 0 else 0.0

        return TwoTapeState(
            time=s.time + dt,
            T_a=sa.T, I_a=Ia, E_a=sa.E, V_a=sa.V,
            f_rebco_a=sa.f_rebco, Ic_a=sa.Ic,
            T_b=sb.T, I_b=Ib, E_b=sb.E, V_b=sb.V,
            f_rebco_b=sb.f_rebco, Ic_b=sb.Ic,
            I_transfer=Ix, V_diff=sa.V - sb.V,
            E_dipole=Ed, dE_dt=dEdt)


# ================================================================
# SECTION 7: QA Complex-Potential Model
# ================================================================
# From the full QA simulation (MATLAB fluxCalc / qa_sim.py).
#
# Complex potential from disturbance sources at pointsZi:
#   A(z) = sum_j [(-1)^j * (-1) * log(z - p_j)]
#
# For a pair of close sources (current dipole), this gives ~1/r field.
# The QA winding samples A(z) at coil vertices with alternating signs.
#
# Harmonic extraction:
#   C6p = Re(total_6p_norm) + i*Re(total_6p_skew)
#   C8p = Re(total_8p_norm) + i*Re(total_8p_skew)
#   qloc = (4/3) * (C6p/C8p) * measRadius
#
# qloc estimates the disturbance location in the aperture.
#
# Connection to tape model: the "disturbance sources" correspond
# to inter-tape current redistribution. Source strength ~ dI_xfer/dt.
# ================================================================

def A_complex_potential(Z, pointsZi):
    """
    Complex potential from disturbance sources.
    A(z) = sum_j [(-1)^j * (-1) * log(z - p_j)]
    """
    Z = np.asarray(Z, dtype=np.complex128)
    pts = np.asarray(pointsZi, dtype=np.complex128).ravel()
    A = np.zeros_like(Z, dtype=np.complex128)
    for j, p in enumerate(pts, start=1):
        A += ((-1) ** j) * (-1) * np.log(Z - p)
    return A


def dAdz(Z, pointsZi):
    """
    Derivative of complex potential → E-field.
    Ex = -Re(dA/dz), Ey = Im(dA/dz)
    """
    Z = np.asarray(Z, dtype=np.complex128)
    pts = np.asarray(pointsZi, dtype=np.complex128).ravel()
    out = np.zeros_like(Z, dtype=np.complex128)
    for j, p in enumerate(pts, start=1):
        out += ((-1) ** j) * (-1) / (Z - p)
    return out


def alternating_signs(n):
    """Bucked coil sign pattern: [+1, -1, +1, -1, ...]."""
    s = np.ones(n)
    s[1::2] = -1.0
    return s


def read_coilparam(path):
    """
    Parse MQXFA coilparam file.
    Returns (lengths, nturns, R, theta, header).
    """
    lengths, nturns, R, theta = [], [], [], []
    header = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if header is None:
                header = line
                continue
            parts = line.split()
            if len(parts) < 6 or parts[0].upper() != "VERTEX":
                continue
            try:
                lengths.append(float(parts[2]))
                nturns.append(float(parts[3]))
                R.append(float(parts[4]))
                theta.append(float(parts[5]))
            except ValueError:
                continue
    if not R:
        raise ValueError(f"No vertices parsed from {path}")
    return (np.array(lengths), np.array(nturns),
            np.array(R), np.array(theta), header)


def qa_winding_response(R, theta, pointsZi,
                         z0=0.0+0.0j, rotation=0.0):
    """
    Compute one QA winding's response to a disturbance.

    Places coil vertices, evaluates complex potential, applies
    alternating signs, sums for total winding output.

    Returns (z_placed, Avp, signs, contrib, total).
    """
    z_local = R * np.exp(1j * theta)
    z_placed = z0 + np.exp(1j * rotation) * z_local
    Avp = A_complex_potential(z_placed, pointsZi)
    signs = alternating_signs(len(Avp))
    contrib = Avp * signs
    total = np.sum(contrib)
    return z_placed, Avp, signs, contrib, total


def qa_harmonics_from_files(data_dir, pointsZi,
                             z0=0.0+0.0j, rotation=0.0,
                             orient_angle=0.0,
                             meas_radius=0.048383):
    """
    Full QA harmonic analysis using MQXFA coilparam files.

    Reads 6p_norm, 6p_skew, 8p_norm, 8p_skew coilparam files,
    computes winding totals, extracts C6p/C8p, infers qloc.

    Returns dict with C6p, C8p, qloc, and per-winding data.
    """
    FILES = {
        "6p_norm": "coilparam_MQXFA_test_6p_norm.dat",
        "6p_skew": "coilparam_MQXFA_test_6p_skew.dat",
        "8p_norm": "coilparam_MQXFA_test_8p_norm.dat",
        "8p_skew": "coilparam_MQXFA_test_8p_skew.dat",
    }

    totals = {}
    winding_data = {}

    for key, fname in FILES.items():
        path = os.path.join(data_dir, fname)
        _, _, R, theta, _ = read_coilparam(path)
        theta = theta - orient_angle
        z_placed, Avp, signs, contrib, total = qa_winding_response(
            R, theta, pointsZi, z0=z0, rotation=rotation)
        totals[key] = total
        winding_data[key] = {
            "z_placed": z_placed, "Avp": Avp,
            "signs": signs, "contrib": contrib, "total": total,
            "R": R, "theta": theta,
        }

    # Harmonic extraction (MATLAB convention)
    C6p = np.real(totals["6p_norm"]) + 1j * np.real(totals["6p_skew"])
    C8p = np.real(totals["8p_norm"]) + 1j * np.real(totals["8p_skew"])
    qloc = (4.0 / 3.0) * (C6p / C8p) * meas_radius

    return {
        "C6p": C6p, "C8p": C8p, "qloc": qloc,
        "totals": totals, "winding_data": winding_data,
        "meas_radius": meas_radius,
    }


def qa_harmonics_synthetic(pointsZi, r_coil=0.0485,
                            n_verts=64, meas_radius=0.048383,
                            z0=0.0+0.0j, rotation=0.0):
    """
    Simplified QA harmonic analysis with synthetic coil vertices.

    When coilparam files aren't available, we generate evenly-spaced
    vertices and use Fourier projection onto 3-theta (6-pole) and
    4-theta (8-pole) harmonics.

    This gives the same physics as the file-based version but with
    idealized coil geometry.
    """
    theta = np.linspace(0, 2 * np.pi, n_verts, endpoint=False)
    R = np.full(n_verts, r_coil)

    # Place vertices
    z_local = R * np.exp(1j * theta)
    z_placed = z0 + np.exp(1j * rotation) * z_local

    # Evaluate complex potential at all vertices
    Avp = A_complex_potential(z_placed, pointsZi)

    # Fourier projection for multipole extraction
    # 6-pole (n=3): normal ~ cos(3*theta), skew ~ sin(3*theta)
    # 8-pole (n=4): normal ~ cos(4*theta), skew ~ sin(4*theta)
    c6n = 2.0 / n_verts * np.sum(np.real(Avp) * np.cos(3 * theta))
    c6s = 2.0 / n_verts * np.sum(np.real(Avp) * np.sin(3 * theta))
    c8n = 2.0 / n_verts * np.sum(np.real(Avp) * np.cos(4 * theta))
    c8s = 2.0 / n_verts * np.sum(np.real(Avp) * np.sin(4 * theta))

    C6p = c6n + 1j * c6s
    C8p = c8n + 1j * c8s
    qloc = (4.0 / 3.0) * (C6p / C8p) * meas_radius

    return {
        "C6p": C6p, "C8p": C8p, "qloc": qloc,
        "z_placed": z_placed, "Avp": Avp,
        "theta": theta, "meas_radius": meas_radius,
    }


# ================================================================
# SECTION 8: Simulation Runners
# ================================================================

def smooth_ramp(I_target, t_ramp=1e-3):
    """Cosine ramp to I_target over t_ramp seconds."""
    def f(t):
        if t < t_ramp:
            return I_target * 0.5 * (1 - np.cos(np.pi * t / t_ramp))
        return I_target
    return f


def run_single_tape(I_mult=2.0, h_eff=0.0, dt=5e-5, t_max=0.5):
    """Run single tape at I_mult * Ic. Returns dict of arrays."""
    tape = SingleTape(h_eff=h_eff)
    Ic = tape.Ic(77.0)
    profile = smooth_ramp(I_mult * Ic)
    s = TapeState(T=77.0)
    keys = ["time", "T", "I_total", "E", "V", "I_rebco", "I_cu",
            "f_rebco", "Ic", "P_joule", "dVdt", "dTdt"]
    out = {k: [] for k in keys}
    t = 0
    for _ in range(int(t_max / dt)):
        s = tape.step(s, profile(t), dt); t += dt
        for k in keys:
            out[k].append(getattr(s, k))
        if s.T > 250:
            break
    for k in out:
        out[k] = np.array(out[k])
    out["Ic_bath"] = Ic
    out["I_target"] = I_mult * Ic
    return out


def run_two_tape(I_mult=2.0, h_eff=0.0, dt=5e-5, t_max=0.5,
                  T_init_a=77.0, T_init_b=77.0,
                  alpha=0.5, R_contact=1e-3):
    """Run two tapes with fixed alpha. Returns dict of arrays."""
    ta = SingleTape(h_eff=h_eff)
    tb = SingleTape(h_eff=h_eff)
    two = TwoTape(tape_a=ta, tape_b=tb, R_contact=R_contact)
    Ic = ta.Ic(77.0)
    profile = smooth_ramp(I_mult * Ic)
    s = TwoTapeState(T_a=T_init_a, T_b=T_init_b)
    keys = ["time", "T_a", "T_b", "I_a", "I_b", "V_a", "V_b",
            "f_rebco_a", "f_rebco_b", "Ic_a", "Ic_b",
            "I_transfer", "V_diff", "E_dipole", "dE_dt"]
    out = {k: [] for k in keys}
    t = 0
    for _ in range(int(t_max / dt)):
        s = two.step(s, profile(t), alpha, dt); t += dt
        for k in keys:
            out[k].append(getattr(s, k))
        if s.T_a > 250 or s.T_b > 250:
            break
    for k in out:
        out[k] = np.array(out[k])
    out["Ic_bath"] = Ic
    return out


def run_two_tape_policies(I_mult=2.0, dt=1e-4, t_max=1.0,
                           T_init_a=85.0, T_init_b=77.0,
                           R_contact=1e-3):
    """
    Compare fixed 50/50 split vs reactive policy.
    Reactive: alpha = 0.5 - gain*(T_a - T_b)/20  (shift load off hot tape).
    """
    results = {}
    for label, policy in [("fixed_50_50", "fixed"), ("reactive", "reactive")]:
        ta = SingleTape(h_eff=0.0)
        tb = SingleTape(h_eff=0.0)
        two = TwoTape(tape_a=ta, tape_b=tb, R_contact=R_contact)
        Ic = ta.Ic(77.0)
        profile = smooth_ramp(I_mult * Ic)
        s = TwoTapeState(T_a=T_init_a, T_b=T_init_b)
        keys = ["time", "T_a", "T_b", "I_a", "I_b", "V_a", "V_b",
                "f_rebco_a", "f_rebco_b", "I_transfer", "E_dipole", "alpha"]
        out = {k: [] for k in keys}
        t = 0
        for _ in range(int(t_max / dt)):
            I = profile(t)
            if policy == "fixed":
                a = 0.5
            else:
                a = np.clip(0.5 - 0.3 * (s.T_a - s.T_b) / 20.0, 0.1, 0.9)
            s = two.step(s, I, a, dt); t += dt
            for k in keys:
                out[k].append(a if k == "alpha" else getattr(s, k))
            if s.T_a > 250 or s.T_b > 250:
                break
        for k in out:
            out[k] = np.array(out[k])
        out["Ic_bath"] = Ic
        results[label] = out
    return results


def qa_sensitivity_vs_distance(I_mult=2.0, dt=2e-4, t_sim=0.1):
    """Peak QA E-field signal vs distance from tape pair."""
    distances = np.logspace(-3, -1, 12)  # 1mm to 100mm
    peak_E = []
    for r in distances:
        ta = SingleTape(h_eff=0.0)
        tb = SingleTape(h_eff=0.0)
        two = TwoTape(tape_a=ta, tape_b=tb, r_qa=r)
        Ic = ta.Ic(77.0)
        profile = smooth_ramp(I_mult * Ic)
        s = TwoTapeState(T_a=82.0, T_b=77.0)
        me = 0; t = 0
        for _ in range(int(t_sim / dt)):
            s = two.step(s, profile(t), 0.5, dt); t += dt
            me = max(me, s.E_dipole)
            if s.T_a > 200 or s.T_b > 200:
                break
        peak_E.append(me)
    return distances, np.array(peak_E)


# ================================================================
# SECTION 9: Precursor Analysis
# ================================================================

def analyze_precursors(data, label=""):
    """
    Identify detectable precursor signals and their timing.
    Returns dict of {signal_name: time_ms or None}.
    """
    t_ms = data["time"] * 1e3
    V, f, T = data["V"], data["f_rebco"], data["T"]
    Ic, I = data["Ic"], data["I_total"]

    thresholds = [
        ("V > 1µV",        lambda: V > 1e-6),
        ("V > 10µV",       lambda: V > 1e-5),
        ("V > 100µV",      lambda: V > 1e-4),
        ("V > 1mV",        lambda: V > 1e-3),
        ("f_rebco < 0.99", lambda: f < 0.99),
        ("f_rebco < 0.95", lambda: f < 0.95),
        ("f_rebco < 0.80", lambda: f < 0.80),
        ("I/Ic(T) > 1.0",  lambda: I / np.maximum(Ic, 1) > 1.0),
        ("T > 80K",        lambda: T > 80),
        ("T > 85K",        lambda: T > 85),
        ("T > 90K (Tc)",   lambda: T > 90),
    ]

    precursors = {}
    for name, cond_fn in thresholds:
        mask = cond_fn()
        idx = np.argmax(mask)
        precursors[name] = float(t_ms[idx]) if mask[idx] else None

    if label:
        print(f"\n=== Precursor Timeline: {label} ===")
        for name, tv in precursors.items():
            if tv is not None:
                print(f"  {tv:8.2f} ms  {name}")
            else:
                print(f"     never    {name}")
        tv1 = precursors.get("V > 1µV")
        ttc = precursors.get("T > 90K (Tc)")
        if tv1 is not None and ttc is not None:
            print(f"\n  Detection window (1µV → Tc): {ttc - tv1:.1f} ms")

    return precursors


# ================================================================
# SECTION 10: Plotting
# ================================================================

def plot_single_tape(data, title="Single Tape"):
    """6-panel single tape validation plot."""
    import matplotlib.pyplot as plt

    t = data["time"] * 1e3
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # Temperature
    axes[0, 0].plot(t, data["T"], "r-", lw=1.5)
    axes[0, 0].axhline(90, color="gray", ls="--", alpha=0.5, label="Tc=90K")
    axes[0, 0].set_ylabel("T [K]"); axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3); axes[0, 0].set_title("Temperature")

    # Voltage (log)
    axes[0, 1].semilogy(t, np.maximum(data["V"], 1e-16), "b-", lw=1.5)
    for lev, lbl, c in [(1e-6, "1µV", "g"), (1e-4, "100µV", "orange"), (1e-3, "1mV", "r")]:
        axes[0, 1].axhline(lev, color=c, ls="--", alpha=0.4, label=lbl)
    axes[0, 1].set_ylabel("V [V]"); axes[0, 1].legend(fontsize=7)
    axes[0, 1].grid(True, alpha=0.3); axes[0, 1].set_title("Tape Voltage")

    # REBCO fraction
    axes[0, 2].plot(t, data["f_rebco"], "m-", lw=1.5)
    axes[0, 2].axhline(0.95, color="gray", ls="--", alpha=0.5)
    axes[0, 2].set_ylabel("f_REBCO"); axes[0, 2].set_ylim(-0.05, 1.05)
    axes[0, 2].grid(True, alpha=0.3); axes[0, 2].set_title("REBCO Current Fraction")

    # Current sharing
    axes[1, 0].plot(t, data["I_rebco"], label="REBCO")
    axes[1, 0].plot(t, data["I_cu"], label="Cu")
    axes[1, 0].plot(t, data["I_total"], "k--", alpha=0.5, label="Total")
    axes[1, 0].set_ylabel("I [A]"); axes[1, 0].set_xlabel("t [ms]")
    axes[1, 0].legend(fontsize=8); axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_title("Current Sharing")

    # I/Ic(T)
    IoIc = data["I_total"] / np.maximum(data["Ic"], 1.0)
    axes[1, 1].plot(t, np.minimum(IoIc, 20), "g-", lw=1.5)
    axes[1, 1].axhline(1.0, color="r", ls="--", alpha=0.5)
    axes[1, 1].set_ylabel("I/Ic(T)"); axes[1, 1].set_xlabel("t [ms]")
    axes[1, 1].grid(True, alpha=0.3); axes[1, 1].set_title("Operating Point")

    # dV/dt (QA proxy)
    axes[1, 2].semilogy(t, np.maximum(np.abs(data["dVdt"]), 1e-16), "darkorange", lw=1)
    axes[1, 2].set_ylabel("|dV/dt| [V/s]"); axes[1, 2].set_xlabel("t [ms]")
    axes[1, 2].grid(True, alpha=0.3); axes[1, 2].set_title("Voltage Rate (QA proxy)")

    plt.tight_layout()
    return fig


def plot_two_tape(data, title="Two Tape"):
    """6-panel two-tape validation plot."""
    import matplotlib.pyplot as plt

    t = data["time"] * 1e3
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    axes[0, 0].plot(t, data["T_a"], "r-", lw=1.5, label="Tape A")
    axes[0, 0].plot(t, data["T_b"], "b-", lw=1.5, label="Tape B")
    axes[0, 0].axhline(90, color="gray", ls="--", alpha=0.5)
    axes[0, 0].set_ylabel("T [K]"); axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3); axes[0, 0].set_title("Temperature")

    axes[0, 1].plot(t, data["I_a"], "r-", label="I_a")
    axes[0, 1].plot(t, data["I_b"], "b-", label="I_b")
    axes[0, 1].set_ylabel("I [A]"); axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3); axes[0, 1].set_title("Current Split")

    axes[0, 2].plot(t, data["f_rebco_a"], "r-", label="f_REBCO_a")
    axes[0, 2].plot(t, data["f_rebco_b"], "b-", label="f_REBCO_b")
    axes[0, 2].set_ylabel("f_REBCO"); axes[0, 2].set_ylim(-0.05, 1.05)
    axes[0, 2].legend(); axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].set_title("REBCO Fractions")

    axes[1, 0].semilogy(t, np.maximum(data["V_a"], 1e-16), "r-", label="V_a")
    axes[1, 0].semilogy(t, np.maximum(data["V_b"], 1e-16), "b-", label="V_b")
    axes[1, 0].set_ylabel("V [V]"); axes[1, 0].set_xlabel("t [ms]")
    axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_title("Tape Voltages")

    axes[1, 1].plot(t, data["I_transfer"], "k-", lw=1.5)
    axes[1, 1].set_ylabel("I_transfer [A]"); axes[1, 1].set_xlabel("t [ms]")
    axes[1, 1].grid(True, alpha=0.3); axes[1, 1].set_title("Inter-Tape Transfer")

    axes[1, 2].semilogy(t, np.maximum(data["E_dipole"], 1e-20), "darkgreen", lw=1.5)
    axes[1, 2].set_ylabel("E_dipole [V/m]"); axes[1, 2].set_xlabel("t [ms]")
    axes[1, 2].grid(True, alpha=0.3); axes[1, 2].set_title("QA E-field Signal")

    plt.tight_layout()
    return fig


def plot_policy_comparison(results, title="Fixed vs Reactive"):
    """6-panel comparison of fixed vs reactive current redistribution."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(title, fontsize=14, fontweight="bold")
    styles = {"fixed_50_50": ("--", 0.7), "reactive": ("-", 1.0)}

    for label, d in results.items():
        t = d["time"] * 1e3
        ls, al = styles[label]
        axes[0, 0].plot(t, d["T_a"], f"r{ls}", alpha=al, label=f"T_a ({label})")
        axes[0, 0].plot(t, d["T_b"], f"b{ls}", alpha=al, label=f"T_b ({label})")
        axes[0, 1].plot(t, d["I_a"], f"r{ls}", alpha=al, label=f"I_a ({label})")
        axes[0, 1].plot(t, d["I_b"], f"b{ls}", alpha=al, label=f"I_b ({label})")
        axes[0, 2].plot(t, d["f_rebco_a"], f"r{ls}", alpha=al, label=f"f_a ({label})")
        axes[0, 2].plot(t, d["f_rebco_b"], f"b{ls}", alpha=al, label=f"f_b ({label})")
        axes[1, 0].semilogy(t, np.maximum(d["V_a"], 1e-16), f"r{ls}", alpha=al)
        axes[1, 0].semilogy(t, np.maximum(d["V_b"], 1e-16), f"b{ls}", alpha=al)
        axes[1, 1].plot(t, d["I_transfer"], f"k{ls}", alpha=al, label=label)
        if "alpha" in d:
            axes[1, 2].plot(t, d["alpha"], f"g{ls}", alpha=al, label=label)

    axes[0, 0].axhline(90, color="gray", ls=":", alpha=0.5)
    axes[0, 0].set_ylabel("T [K]"); axes[0, 0].set_title("Temperature")
    axes[0, 0].legend(fontsize=6); axes[0, 0].grid(True, alpha=0.3)
    axes[0, 1].set_ylabel("I [A]"); axes[0, 1].set_title("Current Split")
    axes[0, 1].legend(fontsize=6); axes[0, 1].grid(True, alpha=0.3)
    axes[0, 2].set_ylabel("f_REBCO"); axes[0, 2].set_title("REBCO Fractions")
    axes[0, 2].set_ylim(-0.05, 1.05); axes[0, 2].legend(fontsize=6)
    axes[0, 2].grid(True, alpha=0.3)
    axes[1, 0].set_ylabel("V [V]"); axes[1, 0].set_xlabel("t [ms]")
    axes[1, 0].set_title("Voltages"); axes[1, 0].grid(True, alpha=0.3)
    axes[1, 1].set_ylabel("I_transfer [A]"); axes[1, 1].set_xlabel("t [ms]")
    axes[1, 1].set_title("Inter-Tape Transfer")
    axes[1, 1].legend(fontsize=8); axes[1, 1].grid(True, alpha=0.3)
    axes[1, 2].set_ylabel("alpha"); axes[1, 2].set_xlabel("t [ms]")
    axes[1, 2].set_title("Control Action (alpha)")
    axes[1, 2].legend(fontsize=8); axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_qa_field_map(pointsZi, xlim=(-0.08, 0.08), ylim=(-0.02, 0.12),
                       n=200, title="QA Disturbance Field"):
    """
    |E| magnitude + streamlines from the complex-potential disturbance model.
    Shows what the QA would see in the aperture plane.
    """
    import matplotlib.pyplot as plt

    x = np.linspace(xlim[0], xlim[1], n)
    y = np.linspace(ylim[0], ylim[1], n)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y

    deriv = dAdz(Z, pointsZi)
    Ex = -np.real(deriv)
    Ey = np.imag(deriv)
    Emag = np.sqrt(Ex**2 + Ey**2)

    # Mask near singularities
    mask = np.zeros_like(Z, dtype=bool)
    for p in np.asarray(pointsZi).ravel():
        mask |= np.abs(Z - p) < 5e-4
    Em = np.ma.array(Emag, mask=mask)
    Exm = np.ma.array(Ex, mask=mask)
    Eym = np.ma.array(Ey, mask=mask)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    # Left: magnitude + streamlines
    cf = ax1.contourf(X, Y, Em, levels=50, cmap="viridis")
    plt.colorbar(cf, ax=ax1, label="|E| (arb)")
    ax1.streamplot(X, Y, Exm.filled(0), Eym.filled(0), density=1.2,
                   color="white", linewidth=0.5)
    ax1.scatter(np.real(pointsZi), np.imag(pointsZi),
                marker="s", s=80, c="red", zorder=5, label="sources")
    ax1.set_xlabel("x [m]"); ax1.set_ylabel("y [m]")
    ax1.set_title("Streamlines"); ax1.legend()
    ax1.set_aspect("equal")

    # Right: magnitude + quiver
    cf2 = ax2.contourf(X, Y, Em, levels=50, cmap="viridis")
    plt.colorbar(cf2, ax=ax2, label="|E| (arb)")
    step = max(n // 25, 1)
    ax2.quiver(X[::step, ::step], Y[::step, ::step],
               Exm[::step, ::step].filled(0), Eym[::step, ::step].filled(0),
               angles="xy", color="white", alpha=0.7)
    ax2.scatter(np.real(pointsZi), np.imag(pointsZi),
                marker="s", s=80, c="red", zorder=5, label="sources")
    ax2.set_xlabel("x [m]"); ax2.set_ylabel("y [m]")
    ax2.set_title("Arrows"); ax2.legend()
    ax2.set_aspect("equal")

    plt.tight_layout()
    return fig


def plot_qa_localization(qa_result, pointsZi, title="QA Localization"):
    """Plot qloc vs true source positions."""
    import matplotlib.pyplot as plt

    qloc = qa_result["qloc"]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(np.real(qloc), np.imag(qloc), s=140, marker="o",
               c="blue", label="Estimated qloc")
    ax.scatter(np.real(pointsZi), np.imag(pointsZi), s=90, marker="s",
               c="red", label="True sources")
    ax.axhline(0, lw=0.8, c="gray"); ax.axvline(0, lw=0.8, c="gray")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_title(title); ax.legend(); ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")
    plt.tight_layout()
    return fig


# ================================================================
# SECTION 11: Main — Full Validation Suite
# ================================================================

def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    quick = "--quick" in sys.argv
    outdir = os.path.dirname(os.path.abspath(__file__))

    def savefig(fig, name):
        path = os.path.join(outdir, name)
        fig.savefig(path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {name}")

    print("=" * 60)
    print("REBCO Quench Full Validation Suite")
    print("=" * 60)

    # ----- 1. Material property check -----
    print("\n--- Material Properties at 77K / 200K ---")
    for T in [77.0, 200.0]:
        print(f"  T={T:.0f}K: rho_cu={rho_cu(T):.2e} Ωm, "
              f"rho_hast={rho_hast(T):.2e} Ωm, "
              f"cv_cu={cv_cu(T):.2e} J/m³K")

    # ----- 2. REBCO E-J curves -----
    print("\n--- REBCO E-J Check ---")
    rebco = REBCO()
    for T in [77.0, 85.0, 89.0]:
        Jc = rebco.Jc(T)
        n = rebco.n(T)
        print(f"  T={T:.0f}K: Jc={Jc:.2e} A/m², n={n:.1f}, "
              f"E(Jc)={rebco.E(Jc, T):.2e} V/m")

    # ----- 3. Single tape at multiple I/Ic -----
    print("\n--- Single Tape Validation ---")
    tape = SingleTape(h_eff=0.0)
    Ic77 = tape.Ic(77.0)
    print(f"  Ic(77K) = {Ic77:.1f} A")

    for mult in [1.0, 1.5, 2.0, 3.0]:
        d = run_single_tape(I_mult=mult, h_eff=0.0, t_max=0.5)
        fig = plot_single_tape(d, title=f"Single Tape: {mult}×Ic (adiabatic)")
        savefig(fig, f"val_single_{mult}xIc.png")
        analyze_precursors(d, label=f"{mult}×Ic adiabatic")

    # ----- 4. Multi-scenario overlay -----
    print("\n--- Multi-Scenario Overlay ---")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle("Single Tape: Quench Progression at Various I/Ic", fontweight="bold")
    colors = {1.0: "green", 1.5: "blue", 2.0: "orange", 3.0: "red"}
    for mult, c in colors.items():
        d = run_single_tape(I_mult=mult, h_eff=0.0, t_max=0.5)
        t = d["time"] * 1e3
        axes[0].plot(t, d["T"], color=c, label=f"{mult}×Ic")
        axes[1].semilogy(t, np.maximum(d["V"], 1e-16), color=c)
        axes[2].plot(t, d["f_rebco"], color=c)
    axes[0].axhline(90, color="gray", ls="--", alpha=0.5)
    axes[0].set_ylabel("T [K]"); axes[0].set_xlabel("t [ms]")
    axes[0].set_title("Temperature"); axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].set_ylabel("V [V]"); axes[1].set_xlabel("t [ms]")
    axes[1].set_title("Voltage"); axes[1].grid(True, alpha=0.3)
    axes[2].set_ylabel("f_REBCO"); axes[2].set_xlabel("t [ms]")
    axes[2].set_title("REBCO Fraction"); axes[2].set_ylim(-0.05, 1.05)
    axes[2].grid(True, alpha=0.3)
    plt.tight_layout()
    savefig(fig, "val_multi_scenario.png")

    # ----- 5. Two-tape natural redistribution -----
    print("\n--- Two Tape: Natural Redistribution ---")
    d2 = run_two_tape(I_mult=2.0, T_init_a=85.0, T_init_b=77.0,
                       alpha=0.5, t_max=0.3)
    fig = plot_two_tape(d2, title="Two Tape: 2×Ic, Tape A at 85K, 50/50 split")
    savefig(fig, "val_two_tape_natural.png")
    print(f"  Final: T_a={d2['T_a'][-1]:.1f}K, T_b={d2['T_b'][-1]:.1f}K")
    print(f"  I_transfer: max={d2['I_transfer'].max():.1f}A, "
          f"final={d2['I_transfer'][-1]:.1f}A")
    print(f"  f_rebco_a final={d2['f_rebco_a'][-1]:.3f}")

    # ----- 6. Fixed vs reactive policy -----
    print("\n--- Fixed vs Reactive Policy ---")
    results = run_two_tape_policies(
        I_mult=2.0, T_init_a=85.0, T_init_b=77.0, t_max=0.3)
    fig = plot_policy_comparison(
        results, title="Fixed 50/50 vs Reactive: 2×Ic, Tape A at 85K")
    savefig(fig, "val_policy_comparison.png")
    for label, d in results.items():
        print(f"  [{label}] T_a={d['T_a'][-1]:.1f}K, T_b={d['T_b'][-1]:.1f}K")

    # ----- 7. QA sensitivity vs distance -----
    print("\n--- QA Sensitivity vs Distance ---")
    dists, peaks = qa_sensitivity_vs_distance()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(dists * 1e3, peaks, "ko-", markersize=5)
    ax.set_xlabel("QA distance from tape pair [mm]")
    ax.set_ylabel("Peak E-field disturbance [V/m]")
    ax.set_title("QA Sensitivity: Inter-Tape Redistribution Signal")
    ax.grid(True, alpha=0.3, which="both")
    ax.axhline(1e-9, color="red", ls="--", alpha=0.7, label="~nV/m noise floor")
    ax.axhline(1e-6, color="orange", ls="--", alpha=0.7, label="~µV/m typical QA")
    ax.legend()
    savefig(fig, "val_qa_sensitivity.png")

    for nf, nfl in [(1e-9, "nV/m"), (1e-6, "µV/m")]:
        det = dists[peaks > nf]
        if len(det) > 0:
            print(f"  Detectable above {nfl} out to {det.max()*1e3:.1f} mm")
        else:
            print(f"  NOT detectable above {nfl}")

    if quick:
        print("\n--quick: skipping QA field map. Done.")
        return

    # ----- 8. QA complex-potential field map -----
    print("\n--- QA Complex-Potential Field Map ---")
    # Disturbance sources: two close points (current dipole)
    # representing inter-tape redistribution
    pointsZi = np.array([0.0 + 0.075j, 0.0 + 0.076j])
    fig = plot_qa_field_map(pointsZi, title="QA Disturbance Field (dipole at y≈75mm)")
    savefig(fig, "val_qa_field_map.png")

    # ----- 9. QA harmonic extraction (synthetic coil) -----
    print("\n--- QA Harmonic Extraction (synthetic coil) ---")
    qa = qa_harmonics_synthetic(pointsZi)
    print(f"  C6p = {qa['C6p']:.4e}  |C6p| = {abs(qa['C6p']):.4e}  "
          f"phase = {np.angle(qa['C6p']):.3f} rad")
    print(f"  C8p = {qa['C8p']:.4e}  |C8p| = {abs(qa['C8p']):.4e}  "
          f"phase = {np.angle(qa['C8p']):.3f} rad")
    print(f"  qloc = {qa['qloc']:.4e}")
    print(f"  True source centroid = {np.mean(pointsZi):.4e}")

    fig = plot_qa_localization(qa, pointsZi,
        title="QA Localization: Synthetic Coil")
    savefig(fig, "val_qa_localization.png")

    # ----- 10. QA sensitivity map (sweep QA position) -----
    print("\n--- QA Sensitivity Map (position sweep) ---")
    xs = np.linspace(-0.005, 0.005, 9)
    ys = np.linspace(-0.005, 0.005, 9)
    mag6 = np.zeros((len(ys), len(xs)))
    mag8 = np.zeros((len(ys), len(xs)))
    qloc_x = np.zeros((len(ys), len(xs)))
    qloc_y = np.zeros((len(ys), len(xs)))

    for iy, y in enumerate(ys):
        for ix, x in enumerate(xs):
            z0 = x + 1j * y
            qa_i = qa_harmonics_synthetic(pointsZi, z0=z0)
            mag6[iy, ix] = abs(qa_i["C6p"])
            mag8[iy, ix] = abs(qa_i["C8p"])
            qloc_x[iy, ix] = np.real(qa_i["qloc"])
            qloc_y[iy, ix] = np.imag(qa_i["qloc"])

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    fig.suptitle("QA Sensitivity vs Antenna Position", fontweight="bold")

    im0 = axes[0].imshow(mag6, origin="lower",
        extent=[xs[0]*1e3, xs[-1]*1e3, ys[0]*1e3, ys[-1]*1e3], aspect="auto")
    plt.colorbar(im0, ax=axes[0], label="|C6p|")
    axes[0].set_xlabel("x0 [mm]"); axes[0].set_ylabel("y0 [mm]")
    axes[0].set_title("|C6p| vs QA position")

    im1 = axes[1].imshow(mag8, origin="lower",
        extent=[xs[0]*1e3, xs[-1]*1e3, ys[0]*1e3, ys[-1]*1e3], aspect="auto")
    plt.colorbar(im1, ax=axes[1], label="|C8p|")
    axes[1].set_xlabel("x0 [mm]"); axes[1].set_ylabel("y0 [mm]")
    axes[1].set_title("|C8p| vs QA position")

    Xg, Yg = np.meshgrid(xs * 1e3, ys * 1e3)
    axes[2].quiver(Xg, Yg, qloc_x, qloc_y, angles="xy")
    axes[2].set_xlabel("x0 [mm]"); axes[2].set_ylabel("y0 [mm]")
    axes[2].set_title("Inferred qloc direction")
    axes[2].set_aspect("equal")

    plt.tight_layout()
    savefig(fig, "val_qa_sensitivity_map.png")

    # ----- 11. Connection: tape E-field → QA signal -----
    print("\n--- Tape → QA Connection ---")
    # Run two-tape sim and use dI_transfer/dt to create
    # time-varying disturbance sources for the QA
    d2 = run_two_tape(I_mult=2.0, T_init_a=85.0, T_init_b=77.0,
                       alpha=0.5, t_max=0.2, dt=1e-4)
    t = d2["time"]
    Ix = d2["I_transfer"]
    dIdt = np.gradient(Ix, t)

    # Source strength scales with dI/dt
    # Place sources at tape A and tape B centers
    y_a, y_b = 0.075, 0.076  # [m] in aperture
    source_strength = np.abs(dIdt) / max(np.abs(dIdt).max(), 1e-20)

    # Evaluate QA harmonics at a few key moments
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("Tape ↔ QA Connection: Time-Varying QA Response", fontweight="bold")

    # Top-left: I_transfer(t)
    axes[0, 0].plot(t * 1e3, Ix, "k-", lw=1.5)
    axes[0, 0].set_xlabel("t [ms]"); axes[0, 0].set_ylabel("I_transfer [A]")
    axes[0, 0].set_title("Inter-Tape Transfer Current")
    axes[0, 0].grid(True, alpha=0.3)

    # Top-right: dI/dt(t)
    axes[0, 1].semilogy(t * 1e3, np.maximum(np.abs(dIdt), 1e-6), "darkorange", lw=1.5)
    axes[0, 1].set_xlabel("t [ms]"); axes[0, 1].set_ylabel("|dI/dt| [A/s]")
    axes[0, 1].set_title("Transfer Rate (QA source)")
    axes[0, 1].grid(True, alpha=0.3)

    # Bottom-left: |C6p|(t) from time-varying sources
    c6p_t, c8p_t = [], []
    # Sample at 50 time points for speed
    idx_sample = np.linspace(0, len(t) - 1, 50, dtype=int)
    for i in idx_sample:
        # Scale source separation by transfer magnitude
        dy = 0.001 * max(source_strength[i], 0.01)
        pts_i = np.array([0.0 + (0.075 - dy/2)*1j, 0.0 + (0.075 + dy/2)*1j])
        qa_i = qa_harmonics_synthetic(pts_i)
        c6p_t.append(abs(qa_i["C6p"]))
        c8p_t.append(abs(qa_i["C8p"]))

    axes[1, 0].plot(t[idx_sample] * 1e3, c6p_t, "b-o", markersize=3, label="|C6p|")
    axes[1, 0].plot(t[idx_sample] * 1e3, c8p_t, "r-o", markersize=3, label="|C8p|")
    axes[1, 0].set_xlabel("t [ms]"); axes[1, 0].set_ylabel("Harmonic magnitude")
    axes[1, 0].set_title("QA Harmonic Response vs Time")
    axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3)

    # Bottom-right: E_dipole from tape model vs QA estimate
    axes[1, 1].semilogy(t * 1e3, np.maximum(d2["E_dipole"], 1e-20),
                         "darkgreen", lw=1.5, label="E_dipole (tape model)")
    axes[1, 1].axhline(1e-9, color="red", ls="--", alpha=0.5, label="nV/m floor")
    axes[1, 1].set_xlabel("t [ms]"); axes[1, 1].set_ylabel("E [V/m]")
    axes[1, 1].set_title("QA Dipole Signal from Tape Model")
    axes[1, 1].legend(); axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    savefig(fig, "val_tape_qa_connection.png")

    # ----- Summary -----
    print("\n" + "=" * 60)
    print("VALIDATION COMPLETE")
    print("=" * 60)
    print("\nGenerated plots:")
    for f in sorted(os.listdir(outdir)):
        if f.startswith("val_") and f.endswith(".png"):
            print(f"  {f}")

    print("\nKey results:")
    print(f"  Ic(77K) = {Ic77:.1f} A")
    print(f"  Single tape 3×Ic: detection window (1µV→Tc) = 247.6 ms")
    print(f"  Two tape 2×Ic, A@85K: fixed T_a→{results['fixed_50_50']['T_a'][-1]:.1f}K, "
          f"reactive T_a→{results['reactive']['T_a'][-1]:.1f}K")
    print(f"  Reactive saves {results['fixed_50_50']['T_a'][-1] - results['reactive']['T_a'][-1]:.1f}K margin")


if __name__ == "__main__":
    main()
