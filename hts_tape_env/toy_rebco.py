"""
toy_rebco.py

Phase 1 toy model:
- Mesh-faithful 2D tape geometry
- Placeholder REBCO Jc(T), n(T)
- Power-law E-J relation
- Current sweep from 0 to 400 A
- Plots:
    1. material map
    2. resistive V-I curves
    3. J/Jc vs current

Assumptions:
- All current flows only in the REBCO layer.
- Current density is uniform along tape length.
- Temperature is fixed per curve.
- No thermal feedback.
- No heater.
- No current redistribution.
- No COMSOL import yet.
"""

import numpy as np
import matplotlib.pyplot as plt


class TapeMesh2D:
    """
    Structured 2D Cartesian mesh for the tape cross-section.

    Field convention:
        field[y, x]

    Material IDs:
        1 = Cu
        2 = Ag
        3 = REBCO
        4 = Hastelloy
    """

    def __init__(self, nx=2000, length_m=0.20, width_m=12e-3):
        self.nx = nx
        self.length_m = length_m
        self.width_m = width_m
        self.dx = length_m / nx

        self.layers = [
            {"name": "cu_top", "n": 25, "dy": 0.8e-6, "mat": 1},
            {"name": "ag_top", "n": 4, "dy": 0.5e-6, "mat": 2},
            {"name": "rebco", "n": 1, "dy": 1.0e-6, "mat": 3},
            {
                "name": "hastelloy",
                "n": 10,
                "edges": np.array([0, 0.5, 1, 2, 5, 10, 15, 18, 19, 19.5, 20]) * 1e-6,
                "mat": 4,
            },
            {"name": "ag_bottom", "n": 4, "dy": 0.5e-6, "mat": 2},
            {"name": "cu_bottom", "n": 25, "dy": 0.8e-6, "mat": 1},
        ]

        self._build()

    def _build(self):
        dy = []
        layer_names = []
        material_ids = []

        for layer in self.layers:
            if "edges" in layer:
                layer_dy = np.diff(layer["edges"])
            else:
                layer_dy = np.full(layer["n"], layer["dy"])

            for i in range(len(layer_dy)):
                dy.append(layer_dy[i])
                layer_names.append(layer["name"])
                material_ids.append(layer["mat"])

        self.dy = np.array(dy)
        self.layer_names = np.array(layer_names)
        self.material_y = np.array(material_ids)
        self.ny = len(self.dy)

        self.x_edges = np.linspace(0, self.length_m, self.nx + 1)
        self.x_centers = 0.5 * (self.x_edges[:-1] + self.x_edges[1:])

        self.y_edges = np.concatenate([[0.0], np.cumsum(self.dy)])
        self.y_centers = 0.5 * (self.y_edges[:-1] + self.y_edges[1:])

        self.material_id = np.repeat(self.material_y[:, None], self.nx, axis=1)

    def rows(self, name):
        return np.where(self.layer_names == name)[0]

    def empty_field(self, val=0.0):
        return np.full((self.ny, self.nx), val)

    def summary(self):
        print("Mesh summary")
        print("------------")
        print("nx =", self.nx)
        print("ny =", self.ny)
        print("dx [mm] =", self.dx * 1e3)
        print("length [m] =", self.length_m)
        print("width [mm] =", self.width_m * 1e3)
        print("thickness [um] =", self.y_edges[-1] * 1e6)
        print("field shape = (%d, %d)" % (self.ny, self.nx))
        print("")

        for layer in self.layers:
            r = self.rows(layer["name"])
            print(
                "%-12s rows %2d-%2d, cells=%2d, thickness=%.2f um"
                % (
                    layer["name"],
                    r[0],
                    r[-1],
                    len(r),
                    np.sum(self.dy[r]) * 1e6,
                )
            )

    def plot_materials(self):
        plt.figure(figsize=(9, 3))
        plt.imshow(
            self.material_id,
            aspect="auto",
            origin="upper",
            extent=[0, self.length_m * 1000, self.y_edges[-1] * 1e6, 0],
        )
        plt.xlabel("x position [mm]")
        plt.ylabel("y position [um]")
        plt.title("Material map")
        plt.colorbar(label="material id")
        plt.tight_layout()
        plt.show()


class REBCOModel:
    """
    Placeholder REBCO electromagnetic model.

    Uses:
        Jc(T) table interpolation
        n(T) table interpolation
        E = Ec * (J/Jc)^n

    These tables are placeholders.
    Replace with measured/vendor/COMSOL tables later.
    """

    def __init__(self):
        self.T_table = np.array([77, 80, 82, 85, 88, 90, 91, 92], dtype=float)

        self.Jc_table = np.array(
            [3e10, 2.4e10, 2e10, 1.4e10, 8e9, 3.5e9, 1.5e9, 2e8],
            dtype=float,
        )

        self.n_table = np.array(
            [30, 27, 25, 21, 16, 11, 7, 4],
            dtype=float,
        )

        # Electric field criterion.
        # 1e-4 V/m = 1 microvolt/cm.
        self.Ec = 1e-4

    def Jc(self, T):
        return np.interp(T, self.T_table, self.Jc_table)

    def n(self, T):
        return np.interp(T, self.T_table, self.n_table)

    def E(self, J, T):
        Jc = self.Jc(T)
        n_val = self.n(T)

        ratio = np.abs(J) / np.maximum(Jc, 1e-12)

        # Numerical floor for plotting only.
        # Prevents extreme underflow like 1e-300 on semilog plots.
        ratio = np.maximum(ratio, 1e-3)

        return self.Ec * ratio ** n_val

    def J_over_Jc(self, J, T):
        return np.abs(J) / np.maximum(self.Jc(T), 1e-12)


def plot_vi_curves(mesh, rebco, I_vals, T_vals):
    rebco_row = mesh.rows("rebco")[0]
    A_rebco = mesh.width_m * mesh.dy[rebco_row]

    plt.figure(figsize=(7, 5))

    for T0 in T_vals:
        J_vals = I_vals / A_rebco
        E_vals = rebco.E(J_vals, T0)

        # Approximate resistive voltage:
        # V = integral E dx.
        # Here E is uniform along x, so V = E * L.
        V_vals = E_vals * mesh.length_m

        plt.semilogy(I_vals, V_vals, label=str(T0) + " K")

    plt.xlabel("Transport current I [A]")
    plt.ylabel("Resistive tape voltage V [V]")
    plt.title("Toy REBCO V-I curves from power-law E-J")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_j_over_jc(mesh, rebco, I_vals, T_vals):
    rebco_row = mesh.rows("rebco")[0]
    A_rebco = mesh.width_m * mesh.dy[rebco_row]

    plt.figure(figsize=(7, 5))

    for T0 in T_vals:
        Ic = rebco.Jc(T0) * A_rebco
        print(str(T0) + " K Ic =", round(Ic, 1), "A")

        J_vals = I_vals / A_rebco
        ratio = rebco.J_over_Jc(J_vals, T0)

        plt.plot(I_vals, ratio, label=str(T0) + " K")
        plt.axvline(Ic, linestyle="--", alpha=0.3)

    plt.axhline(1.0, color="black", linestyle="--", label="J/Jc = 1")
    plt.axhline(0.95, color="gray", linestyle=":", label="control threshold = 0.95")

    plt.xlabel("Transport current I [A]")
    plt.ylabel("J / Jc(T)")
    plt.title("Critical-current fraction vs transport current")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main():
    mesh = TapeMesh2D()
    rebco = REBCOModel()

    mesh.summary()
    mesh.plot_materials()

    I_vals = np.linspace(0, 400, 401)
    T_vals = [77, 80, 85, 90]

    plot_vi_curves(mesh, rebco, I_vals, T_vals)
    plot_j_over_jc(mesh, rebco, I_vals, T_vals)


if __name__ == "__main__":
    main()
