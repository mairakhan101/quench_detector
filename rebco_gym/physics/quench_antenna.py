"""
Quench Antenna (QA) voltage response model.

Simulates the voltage signal a quench antenna would produce in response
to a flux disturbance in the REBCO tape cross-section. The disturbance
is modeled as a pair of closely-spaced current lines (dipole source)
representing a local normal zone.

Adapted from qa_sim.py which implements the MQXFA-style quench antenna
geometry with 6-pole and 8-pole normal/skew windings.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class QuenchAntennaModel:
    """
    Simplified QA model for gym integration.

    Uses complex potential theory: the disturbance at position z_q
    produces a dipole-like field. The QA winding samples the complex
    potential at its vertex positions and sums with alternating signs.

    For the gym, we don't need the full MQXFA coil files — we use a
    simplified circular winding with configurable pole count.
    """

    # Antenna geometry
    n_poles: int = 6                # number of poles (6 or 8)
    n_vertices_per_pole: int = 4    # vertices per pole segment
    antenna_radius: float = 0.048   # m, measurement radius
    meas_radius: float = 0.048383   # m, normalization radius

    # Placement
    z0: complex = 0.0 + 0.0j       # antenna center position
    rotation: float = 0.0          # radians

    # Disturbance source
    z_quench: complex = 0.0 + 0.075j  # quench location
    dipole_separation: float = 1e-3    # m, separation of dipole pair

    # State
    _voltage: float = 0.0

    def _make_vertices(self) -> np.ndarray:
        """Generate simplified QA winding vertices on a circle."""
        n_total = self.n_poles * self.n_vertices_per_pole
        theta = np.linspace(0, 2 * np.pi, n_total, endpoint=False)
        z_local = self.antenna_radius * np.exp(1j * theta)
        return self.z0 + np.exp(1j * self.rotation) * z_local

    def _dipole_sources(self) -> np.ndarray:
        """Two point sources separated by dipole_separation, centered at z_quench."""
        half = self.dipole_separation / 2.0
        return np.array([
            self.z_quench - half * 1j,
            self.z_quench + half * 1j,
        ], dtype=complex)

    @staticmethod
    def _complex_potential(Z: np.ndarray, sources: np.ndarray) -> np.ndarray:
        """
        A(z) = sum_j [(-1)^j * (-1) * log(z - p_j)]
        Same convention as MATLAB fluxCalc.
        """
        A = np.zeros_like(Z, dtype=complex)
        for j, p in enumerate(sources, start=1):
            A += ((-1) ** j) * (-1) * np.log(Z - p)
        return A

    def compute_voltage(self, amplitude: float = 1.0) -> float:
        """
        Compute the QA winding voltage response to the current disturbance.

        amplitude: scales the disturbance strength (proportional to
                   local dissipation / flux change rate).

        Returns a scalar voltage (magnitude of complex winding sum).
        """
        vertices = self._make_vertices()
        sources = self._dipole_sources()

        Avp = self._complex_potential(vertices, sources)
        signs = np.ones(len(Avp))
        signs[1::2] = -1.0

        total = np.sum(Avp * signs) * amplitude
        self._voltage = float(np.abs(total))
        return self._voltage

    def compute_response_at_location(
        self, z_q: complex, amplitude: float = 1.0
    ) -> float:
        """Compute voltage for an arbitrary quench location."""
        old_zq = self.z_quench
        self.z_quench = z_q
        v = self.compute_voltage(amplitude)
        self.z_quench = old_zq
        return v

    def sensitivity_map(
        self, x_range: Tuple[float, float] = (-0.08, 0.08),
        y_range: Tuple[float, float] = (-0.02, 0.12),
        n_points: int = 50,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute QA voltage sensitivity as function of disturbance position.
        Returns (X, Y, V_magnitude) grids.
        """
        xs = np.linspace(x_range[0], x_range[1], n_points)
        ys = np.linspace(y_range[0], y_range[1], n_points)
        X, Y = np.meshgrid(xs, ys)
        V = np.zeros_like(X)

        for i in range(n_points):
            for j in range(n_points):
                z_q = xs[j] + 1j * ys[i]
                V[i, j] = self.compute_response_at_location(z_q)

        return X, Y, V

    @property
    def voltage(self) -> float:
        return self._voltage
