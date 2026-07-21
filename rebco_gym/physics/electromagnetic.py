"""
Electromagnetic model for REBCO tape: self/external B-field and
field-angle-dependent critical current, plus spatially-correlated
Ic dropout sampling along tape length.

All formulas here are engineering approximations, not a field solve —
same spirit as `toy_rebco.py`'s placeholder tables: good enough to drive
qualitative RL training dynamics (Ic goes down as |B| goes up, more so
perpendicular to the tape than parallel; local defects create weak spots),
not a substitute for a COMSOL/finite-element field calculation.
"""

import numpy as np
from dataclasses import dataclass

MU0 = 4.0e-7 * np.pi  # T*m/A


# ---------------------------------------------------------------
# Self-field
# ---------------------------------------------------------------
def self_field(I: float, width: float = 0.004, coeff: float = 1.0) -> float:
    """
    Edge self-field of a thin tape carrying transport current I [A].

    Approximation for a thin superconducting strip of width `width` in
    the critical-state model: B_edge ~ mu0*I/(pi*w). `coeff` is a knob
    to rescale for geometry effects (stack, cabling, nearby return
    conductors) without re-deriving the field solve.
    """
    return coeff * MU0 * abs(I) / (np.pi * max(width, 1e-12))


# ---------------------------------------------------------------
# Anisotropic Jc(B, theta, T)
# ---------------------------------------------------------------
@dataclass
class AnisotropicJcModel:
    """
    Field-angle-dependent critical current for a REBCO coated conductor.

    Jc(B_par, B_perp, T) = Jc(0, T) / (1 + B_eff/B0)^alpha
    B_eff = sqrt(B_perp**2 + (B_par/g)**2)

    This is the standard "anisotropic ellipse" engineering fit used for
    coated-conductor Ic(B,theta) data (e.g. SuperPower-style characterization
    curves): g > 1 makes the conductor much less sensitive to field applied
    parallel to the tape (ab-plane) than perpendicular (c-axis), which is
    the well-known REBCO anisotropy. Parameters below are representative
    placeholders, not a fit to a specific conductor — override for your tape.
    """

    B0: float = 0.2       # T, field scale
    alpha: float = 0.7    # Kim-model exponent
    g: float = 5.0        # anisotropy factor (>1: parallel field forgiven more)

    def Jc_fraction(self, B_parallel: float, B_perp: float) -> float:
        """Multiplicative factor in (0, 1] to apply to the zero-field Jc(T)."""
        B_eff = np.sqrt(B_perp**2 + (B_parallel / self.g) ** 2)
        return float(1.0 / (1.0 + B_eff / self.B0) ** self.alpha)

    def Jc(self, Jc0_T: float, B_parallel: float, B_perp: float) -> float:
        """Full field- and temperature-dependent critical current density/current."""
        return Jc0_T * self.Jc_fraction(B_parallel, B_perp)

    @staticmethod
    def resolve_components(B_mag: float, theta_rad: float) -> tuple:
        """
        Split a field of magnitude B_mag applied at angle theta (radians,
        0 = parallel to tape ab-plane / tape face, pi/2 = perpendicular /
        c-axis) into (B_parallel, B_perp).
        """
        return B_mag * np.cos(theta_rad), B_mag * np.sin(theta_rad)


# ---------------------------------------------------------------
# Spatially-correlated Ic dropout field
# ---------------------------------------------------------------
@dataclass
class IcDropoutField:
    """
    Samples a multiplicative Ic correction along nz segments of tape length,
    representing real-world Ic-homogeneity variation from manufacturing plus
    occasional localized defects (particulates, delaminations, striation
    damage). Values cluster around 1.0 (nominal Ic) with small +/- drift
    from the background term and sharp dips toward 0 at defect sites.

    Two components, combined multiplicatively:
      - smooth background: an Ornstein-Uhlenbeck (AR(1)) process with a
        given correlation length and std dev, representing gradual
        Ic-homogeneity drift along the tape.
      - localized defects: a compound-Poisson process — defects occur at
        rate `defect_rate` per meter, each with a depth drawn from
        Uniform(defect_depth_range) and a spatial footprint of
        `defect_width`.
    """

    background_std: float = 0.03        # fractional std dev of smooth term
    correlation_length: float = 0.02    # m, smooth-term correlation length
    defect_rate: float = 2.0            # defects per meter (expected)
    defect_depth_range: tuple = (0.15, 0.6)   # fractional Ic drop at defect center
    defect_width: float = 3e-3          # m, defect footprint (Gaussian sigma)

    def sample(self, nz: int, length: float, rng: np.random.Generator) -> np.ndarray:
        dz = length / max(nz, 1)
        z = (np.arange(nz) + 0.5) * dz

        # --- smooth background: AR(1) with correlation length ---
        phi = np.exp(-dz / max(self.correlation_length, 1e-9))
        innovation_std = self.background_std * np.sqrt(max(1.0 - phi**2, 1e-12))
        background = np.empty(nz)
        background[0] = rng.normal(0.0, self.background_std)
        for i in range(1, nz):
            background[i] = phi * background[i - 1] + rng.normal(0.0, innovation_std)

        # --- localized defects: compound Poisson along length ---
        expected_defects = self.defect_rate * length
        n_defects = rng.poisson(expected_defects)
        defect_dip = np.zeros(nz)
        for _ in range(n_defects):
            z0 = rng.uniform(0.0, length)
            depth = rng.uniform(*self.defect_depth_range)
            profile = depth * np.exp(-0.5 * ((z - z0) / max(self.defect_width, 1e-9)) ** 2)
            defect_dip = np.maximum(defect_dip, profile)

        multiplier = (1.0 + background) * (1.0 - defect_dip)
        return np.clip(multiplier, 0.02, 1.5)
