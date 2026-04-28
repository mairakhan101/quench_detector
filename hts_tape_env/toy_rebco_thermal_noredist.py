import numpy as np
import matplotlib.pyplot as plt


class TapeMesh2D:
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
        return np.full((self.ny, self.nx), val, dtype=float)


class REBCOModel:
    def __init__(self):
        self.T_table = np.array([77, 80, 82, 85, 88, 90, 91, 92], dtype=float)
        self.Jc_table = np.array(
            [3e10, 2.4e10, 2e10, 1.4e10, 8e9, 3.5e9, 1.5e9, 2e8],
            dtype=float,
        )
        self.n_table = np.array([30, 27, 25, 21, 16, 11, 7, 4], dtype=float)

        self.Ec = 1e-4
        self.max_ratio_for_E = 5.0

    def Jc(self, T):
        return np.interp(T, self.T_table, self.Jc_table)

    def n(self, T):
        return np.interp(T, self.T_table, self.n_table)

    def E(self, J, T):
        Jc = self.Jc(T)
        n_val = self.n(T)

        ratio = np.abs(J) / np.maximum(Jc, 1e-12)
        ratio = np.clip(ratio, 1e-3, self.max_ratio_for_E)

        return self.Ec * ratio ** n_val

    def J_over_Jc(self, J, T):
        return np.abs(J) / np.maximum(self.Jc(T), 1e-12)


class ToyElectroThermalTape:
    """
    Toy electrothermal tape model.

    Still no current redistribution.

    Important assumptions:
    - All transport current is in REBCO.
    - Heater is applied at the top surface.
    - Heat diffuses in x and y.
    - Added crude vertical thermal mixing so top-surface heat reaches REBCO.
    - Ends are clamped to 77 K.
    - Temperature is clipped for toy stability.
    """

    def __init__(self, mesh, rebco):
        self.mesh = mesh
        self.rebco = rebco

        self.T_bath = 77.0
        self.T_max_clip = 120.0

        self.T = mesh.empty_field(self.T_bath)
        self.J = mesh.empty_field(0.0)
        self.E = mesh.empty_field(0.0)
        self.Q = mesh.empty_field(0.0)

        self.time = 0.0
        self.I_total = 0.0

        self.rebco_row = mesh.rows("rebco")[0]
        self.A_rebco = mesh.width_m * mesh.dy[self.rebco_row]

        # Toy thermal constants
        self.heat_capacity_eff = 3.0e6
        self.k_x_eff = 10.0
        self.k_y_eff = 10.0
        self.cooling_rate = 0.2

        # Scale down EM heat for toy stability.
        self.joule_heat_scale = 0.01

        # Crude vertical thermal coupling/mixing.
        # Increase if REBCO still stays too cold.
        self.vertical_mix_strength = 0.25

    def reset(self, T0=77.0):
        self.T[:, :] = T0
        self.J[:, :] = 0.0
        self.E[:, :] = 0.0
        self.Q[:, :] = 0.0
        self.time = 0.0
        self.I_total = 0.0

    def apply_current(self, I_total):
        self.I_total = I_total
        self.J[:, :] = 0.0

        J_rebco = I_total / self.A_rebco
        self.J[self.rebco_row, :] = J_rebco

    def update_electric_field(self):
        T_rebco = self.T[self.rebco_row, :]
        J_rebco = self.J[self.rebco_row, :]

        self.E[:, :] = 0.0
        self.E[self.rebco_row, :] = self.rebco.E(J_rebco, T_rebco)

    def update_joule_heating(self):
        self.Q[:, :] = 0.0
        self.Q[self.rebco_row, :] = (
            self.joule_heat_scale
            * self.J[self.rebco_row, :]
            * self.E[self.rebco_row, :]
        )

    def add_heater(self, heater_W_m2=0.0, center_m=0.10, length_m=2e-3):
        if heater_W_m2 <= 0:
            return

        x = self.mesh.x_centers
        mask = np.abs(x - center_m) <= length_m / 2.0

        top_row = 0
        dy_top = self.mesh.dy[top_row]

        self.Q[top_row, mask] += heater_W_m2 / dy_top

    def diffuse_x(self, T, dt):
        T_new = T.copy()

        alpha = self.k_x_eff / self.heat_capacity_eff
        coeff = alpha * dt / (self.mesh.dx ** 2)
        coeff = min(coeff, 0.45)

        T_new[:, 1:-1] = (
            T[:, 1:-1]
            + coeff * (T[:, 2:] - 2.0 * T[:, 1:-1] + T[:, :-2])
        )

        return T_new

    def diffuse_y(self, T, dt):
        T_new = T.copy()

        alpha = self.k_y_eff / self.heat_capacity_eff

        for j in range(1, self.mesh.ny - 1):
            dy_up = self.mesh.y_centers[j] - self.mesh.y_centers[j - 1]
            dy_dn = self.mesh.y_centers[j + 1] - self.mesh.y_centers[j]
            dy_eff = 0.5 * (dy_up + dy_dn)

            coeff = alpha * dt / (dy_eff ** 2)
            coeff = min(coeff, 0.25)

            T_new[j, :] = (
                T[j, :]
                + coeff * (T[j + 1, :] - 2.0 * T[j, :] + T[j - 1, :])
            )

        return T_new

    def mix_y(self, T):
        """
        Crude vertical thermal mixing.

        This is not a physical COMSOL replacement.
        It is a toy closure so that top-surface heater energy can influence
        the REBCO row before we implement/import a better thermal model.
        """
        T_mean_y = np.mean(T, axis=0, keepdims=True)
        return (
            (1.0 - self.vertical_mix_strength) * T
            + self.vertical_mix_strength * T_mean_y
        )

    def thermal_step(self, dt):
        dTdt_source = self.Q / self.heat_capacity_eff
        dTdt_cooling = -self.cooling_rate * (self.T - self.T_bath)

        self.T = self.T + dt * (dTdt_source + dTdt_cooling)

        self.T = self.diffuse_x(self.T, dt)
        self.T = self.diffuse_y(self.T, dt)
        self.T = self.mix_y(self.T)

        self.T[:, 0] = self.T_bath
        self.T[:, -1] = self.T_bath

        self.T = np.clip(self.T, self.T_bath, self.T_max_clip)

    def step(self, I_total, dt, heater_W_m2=0.0):
        self.apply_current(I_total)
        self.update_electric_field()
        self.update_joule_heating()
        self.add_heater(heater_W_m2=heater_W_m2)
        self.thermal_step(dt)

        self.time += dt
        return self.measure()

    def measure(self):
        T_rebco = self.T[self.rebco_row, :]
        J_rebco = self.J[self.rebco_row, :]
        E_rebco = self.E[self.rebco_row, :]

        J_ratio = self.rebco.J_over_Jc(J_rebco, T_rebco)
        V_tape = np.trapz(E_rebco, self.mesh.x_centers)

        cell_volume = self.mesh.dx * self.mesh.dy[:, None] * self.mesh.width_m
        P_total = np.sum(self.Q * cell_volume)

        return {
            "time": self.time,
            "I_total": self.I_total,
            "Tmax": float(np.max(self.T)),
            "Tmax_REBCO": float(np.max(T_rebco)),
            "Emax": float(np.max(self.E)),
            "V_tape": float(V_tape),
            "P_total": float(P_total),
            "max_J_over_Jc": float(np.max(J_ratio)),
            "exceeds_critical_current": bool(np.max(J_ratio) > 1.0),
        }
