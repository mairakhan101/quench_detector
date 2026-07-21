"""
Current redistribution model for a split REBCO tape cross-section.

Models current redistribution in a 2D mesh (cross-section of tape)
when a local region transitions from superconducting to normal state.
Uses a resistor-network approach on a rectangular mesh that represents
the COMSOL 0.5mm mesh.

Layers (top to bottom):
  - Cu stabilizer (top)
  - Ag overlayer
  - REBCO (SC layer)
  - Buffer layers
  - Hastelloy substrate
  - Ag overlayer
  - Cu stabilizer (bottom)

The mesh nodes carry current; when a SC node goes normal, current
redistributes through the stabilizer layers.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict


@dataclass
class CurrentRedistributionMesh:
    """
    2D resistor-network model of REBCO tape cross-section.

    The mesh is (ny x nx) where:
      - nx = tape_width / dx  (along width)
      - ny = number of layers (through thickness)

    Each node has a resistivity that depends on which material layer
    it belongs to and the local temperature.
    """

    # Tape geometry
    tape_width: float = 4.0e-3      # m (4 mm)
    dx: float = 0.5e-3              # m (0.5 mm mesh)

    # Layer thicknesses (m) and count
    layer_thicknesses: Dict[str, float] = field(default_factory=lambda: {
        "cu_top":    20e-6,
        "ag_top":    2e-6,
        "rebco":     1.3e-6,
        "buffer":    0.5e-6,
        "hastelloy": 50e-6,
        "ag_bot":    2e-6,
        "cu_bot":    20e-6,
    })

    # Operating parameters
    I_total: float = 400.0      # A, total transport current
    T_base: float = 20.0        # K
    Ic0: float = 500.0          # A
    Tc: float = 85.0            # K
    n_value: float = 20.0

    def __post_init__(self):
        self.nx = max(2, int(round(self.tape_width / self.dx)))
        self.layer_names = list(self.layer_thicknesses.keys())
        self.ny = len(self.layer_names)

        # Temperature field: (ny, nx)
        self.T_field = np.full((self.ny, self.nx), self.T_base)

        # Current density field: (ny, nx) in A/m^2
        self.J_field = np.zeros((self.ny, self.nx))

        # Initialize uniform current in REBCO layer
        self._distribute_current()

    def _resistivity(self, layer_name: str, T: float) -> float:
        """Resistivity [Ohm·m] for each material at temperature T."""
        T = np.clip(T, 4.0, 500.0)

        if layer_name.startswith("cu"):
            rho_300 = 1.6e-8
            rho_0 = rho_300 / 24.0
            return rho_0 + (rho_300 - rho_0) * (T / 300.0) ** 1.5

        elif layer_name.startswith("ag"):
            rho_300 = 1.5e-8
            rho_0 = rho_300 / 20.0
            return rho_0 + (rho_300 - rho_0) * (T / 300.0) ** 1.5

        elif layer_name == "rebco":
            # Power-law resistivity: rho_eff = E0/Jc * (J/Jc)^(n-1)
            # But for the conductance-weighted distribution we need
            # a resistivity that's very low when superconducting.
            Ic = self.Ic0 * max(1.0 - (T / self.Tc) ** 1.7, 0.0)
            if T >= self.Tc or Ic <= 0:
                return 1e-6  # normal state REBCO
            # In the superconducting state, REBCO has effectively zero
            # resistance. Use a very low value to model this.
            # The power-law only kicks in when J approaches Jc.
            return 1e-14  # effectively zero — superconducting

        elif layer_name == "buffer":
            return 1e-2  # effectively insulating

        elif layer_name == "hastelloy":
            return 1.25e-6  # ~constant at cryo

        return 1e-6  # fallback

    def _distribute_current(self):
        """
        Redistribute current across the mesh based on local resistivity.

        Uses a simple conductance-weighted distribution: each column
        (through-thickness) acts as parallel resistors. Current splits
        in proportion to conductance.
        """
        # For each x-position, compute the conductance of each layer
        G = np.zeros((self.ny, self.nx))
        for iy, layer in enumerate(self.layer_names):
            t = self.layer_thicknesses[layer]
            for ix in range(self.nx):
                rho = self._resistivity(layer, self.T_field[iy, ix])
                # Conductance per unit length: A / (rho * L)
                # A = t * dx, L = 1 (unit length)
                G[iy, ix] = (t * self.dx) / max(rho, 1e-20)

        # Total conductance per column
        G_total = G.sum(axis=0)  # (nx,)
        G_total_all = G_total.sum()

        if G_total_all < 1e-30:
            self.J_field[:] = 0.0
            return

        # Current per column (proportional to column conductance)
        I_per_col = self.I_total * (G_total / G_total_all)  # (nx,)

        # Within each column, split by layer conductance
        for ix in range(self.nx):
            col_G = G[:, ix]
            col_sum = col_G.sum()
            if col_sum < 1e-30:
                self.J_field[:, ix] = 0.0
            else:
                I_layer = I_per_col[ix] * (col_G / col_sum)
                for iy in range(self.ny):
                    t = self.layer_thicknesses[self.layer_names[iy]]
                    area = t * self.dx
                    self.J_field[iy, ix] = I_layer[iy] / max(area, 1e-20)

    def set_local_temperature(self, ix: int, iy: int, T: float):
        """Set temperature at a specific mesh node."""
        self.T_field[iy, ix] = T

    def set_hotspot(self, x_center: float, width: float, T_hot: float):
        """
        Create a hotspot centered at x_center with given width.
        Affects all layers at those x positions.
        """
        x_positions = np.linspace(0, self.tape_width, self.nx)
        mask = np.abs(x_positions - x_center) <= width / 2.0
        self.T_field[:, mask] = T_hot

    def step(self, dt: float):
        """
        Advance one time step using a lumped thermal model per column.

        Each x-column is treated as thermally well-connected through
        thickness (valid for thin tapes). Temperature is uniform per
        column. Current redistribution uses conductance weighting.
        """
        # Step 1: redistribute current
        self._distribute_current()

        # Step 2: compute total Joule heating per column, update column T
        rebco_idx = self.layer_names.index("rebco")

        for ix in range(self.nx):
            T_col = self.T_field[0, ix]  # uniform per column

            # Total Joule heating in this column [W/m]
            Q_joule = 0.0
            for iy, layer in enumerate(self.layer_names):
                t_layer = self.layer_thicknesses[layer]
                J = self.J_field[iy, ix]

                if layer == "rebco":
                    Ic_local = self.Ic0 * max(1.0 - (T_col / self.Tc) ** 1.7, 0.0)
                    Jc = Ic_local / max(self.tape_width * t_layer, 1e-20)
                    if Jc > 0 and abs(J) > 0:
                        ratio = abs(J) / Jc
                        E0 = 1e-4
                        if ratio < 1.0:
                            E_local = E0 * ratio ** self.n_value
                        else:
                            rho_ff = 1e-6
                            E_local = E0 + rho_ff * Jc * (ratio - 1.0)
                        Q_joule += E_local * abs(J) * t_layer * self.dx
                    # else: zero
                else:
                    rho = self._resistivity(layer, T_col)
                    Q_joule += J**2 * rho * t_layer * self.dx

            # Cooling [W/m]: linear to bath
            lambda_eff = 10.0   # W/m/K
            d_cool = 5e-3       # m
            total_thickness = sum(self.layer_thicknesses.values())
            Q_cool = (lambda_eff / d_cool) * (T_col - self.T_base) * 2.0 * self.dx

            # Effective heat capacity of the column [J/(m·K)]
            # Dominated by Hastelloy and Cu layers
            C_col = 0.0
            for layer in self.layer_names:
                t_layer = self.layer_thicknesses[layer]
                # Volume per unit length: t * dx
                vol = t_layer * self.dx
                # C_vol ~ 2e6 J/(m^3·K) at 20K, ~3.4e6 at 300K
                C_300 = 3.4e6
                C_vol = C_300 * min((max(T_col, 4.0) / 150.0) ** 2, 1.0)
                C_vol = max(C_vol, 5e4)
                C_col += C_vol * vol

            # Temperature update
            if C_col > 0:
                dT = (Q_joule - Q_cool) / C_col * dt
                T_new = T_col + dT
                T_new = np.clip(T_new, 4.0, 500.0)
                self.T_field[:, ix] = T_new  # uniform column T

        self.T_field = np.clip(self.T_field, 4.0, 500.0)

    def get_rebco_current_fraction(self) -> np.ndarray:
        """
        Fraction of current carried by the REBCO layer at each x-position.
        Returns array of shape (nx,).
        """
        rebco_idx = self.layer_names.index("rebco")
        t_rebco = self.layer_thicknesses["rebco"]

        I_rebco = self.J_field[rebco_idx, :] * t_rebco * self.dx
        I_total_per_col = np.zeros(self.nx)
        for iy, layer in enumerate(self.layer_names):
            t = self.layer_thicknesses[layer]
            I_total_per_col += self.J_field[iy, :] * t * self.dx

        with np.errstate(divide="ignore", invalid="ignore"):
            frac = np.where(I_total_per_col > 0,
                            I_rebco / I_total_per_col, 0.0)
        return frac

    def get_E_field(self) -> np.ndarray:
        """Electric field magnitude at each mesh node [V/m]."""
        E = np.zeros((self.ny, self.nx))
        for iy, layer in enumerate(self.layer_names):
            for ix in range(self.nx):
                rho = self._resistivity(layer, self.T_field[iy, ix])
                E[iy, ix] = self.J_field[iy, ix] * rho
        return E

    def get_Jc_field(self) -> np.ndarray:
        """Critical current density at REBCO layer nodes [A/m^2]."""
        rebco_idx = self.layer_names.index("rebco")
        t_rebco = self.layer_thicknesses["rebco"]
        Jc = np.zeros(self.nx)
        for ix in range(self.nx):
            T = self.T_field[rebco_idx, ix]
            Ic_local = self.Ic0 * max(1.0 - (T / self.Tc) ** 1.7, 1e-12)
            Jc[ix] = Ic_local / (self.tape_width * t_rebco)
        return Jc

    def get_Ic_distribution(self) -> np.ndarray:
        """Local Ic at each x-position based on REBCO layer temperature."""
        rebco_idx = self.layer_names.index("rebco")
        Ic = np.zeros(self.nx)
        for ix in range(self.nx):
            T = self.T_field[rebco_idx, ix]
            Ic[ix] = self.Ic0 * max(1.0 - (T / self.Tc) ** 1.7, 1e-12)
        return Ic

    def max_temperature(self) -> float:
        return float(self.T_field.max())

    def reset(self, T_base: Optional[float] = None, I_total: Optional[float] = None):
        if T_base is not None:
            self.T_base = T_base
        if I_total is not None:
            self.I_total = I_total
        self.T_field[:] = self.T_base
        self._distribute_current()
