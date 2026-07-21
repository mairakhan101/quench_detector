"""
CurrentRedistEnv-v0

A Gymnasium environment for current redistribution control in a split
REBCO tape cross-section, using a 0.5mm mesh model with quench antenna
voltage feedback.

This environment models the COMSOL-style mesh where the agent can
observe Ic, E-field, and Jc distributions across the tape width,
along with the QA voltage signal.

Observation space (Box):
  Flattened vector of:
  - Ic distribution across tape width (nx values)
  - E-field at REBCO layer (nx values)
  - Jc distribution (nx values)
  - Temperature profile at REBCO layer (nx values)
  - QA voltage signal (1 value)
  - Total current (1 value)
  - Max temperature (1 value)

Action space (Box):
  For now: single continuous action in [0, 1] representing current
  setpoint fraction (0 = full ramp-down, 1 = maintain).
  Future: per-mesh-element current redistribution from MOSFET controls.

Reward:
  Combines:
  - Operational reward for maintaining current
  - Penalty for high local temperatures
  - Large penalty for quench
  - Bonus for smooth current profiles (uniform Jc utilization)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from ..physics.current_redistribution import CurrentRedistributionMesh
from ..physics.quench_antenna import QuenchAntennaModel
from ..physics.thermal import Ic_T


class CurrentRedistEnv(gym.Env):
    """Current redistribution environment with mesh-level observability."""

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 10}

    def __init__(
        self,
        render_mode=None,
        # Mesh parameters
        tape_width: float = 4.0e-3,
        dx: float = 0.5e-3,
        # Operating conditions
        Ic0: float = 500.0,
        Tc: float = 85.0,
        n_value: float = 20.0,
        T0_range: tuple = (15.0, 50.0),
        I_op_fraction_range: tuple = (0.6, 0.95),
        # Disturbance
        disturbance_prob: float = 0.03,
        hotspot_width_range: tuple = (0.5e-3, 2.0e-3),
        hotspot_dT_range: tuple = (1.0, 8.0),
        # Episode
        dt: float = 0.001,
        max_steps: int = 1000,
        dT_crit: float = 12.0,
    ):
        super().__init__()
        self.render_mode = render_mode

        self.tape_width = tape_width
        self.dx = dx
        self.nx = max(2, int(round(tape_width / dx)))
        self.Ic0 = Ic0
        self.Tc = Tc
        self.n_value = n_value
        self.T0_range = T0_range
        self.I_op_fraction_range = I_op_fraction_range
        self.disturbance_prob = disturbance_prob
        self.hotspot_width_range = hotspot_width_range
        self.hotspot_dT_range = hotspot_dT_range
        self.dt = dt
        self.max_steps = max_steps
        self.dT_crit = dT_crit

        # Action: continuous current setpoint [0, 1]
        self.action_space = spaces.Box(
            low=np.array([0.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
        )

        # Observation: Ic(nx) + E(nx) + Jc(nx) + T(nx) + V_qa + I_total + T_max
        obs_dim = 4 * self.nx + 3
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_dim,), dtype=np.float32,
        )

        # Models
        self.mesh: CurrentRedistributionMesh = None
        self.qa: QuenchAntennaModel = None
        self._step_count = 0
        self._T0 = 20.0
        self._I_nominal = 400.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        T0 = self.np_random.uniform(*self.T0_range)
        I_frac = self.np_random.uniform(*self.I_op_fraction_range)
        Ic_at_T0 = Ic_T(T0, self.Ic0, self.Tc)
        I_op = I_frac * Ic_at_T0

        self._T0 = T0
        self._I_nominal = I_op

        self.mesh = CurrentRedistributionMesh(
            tape_width=self.tape_width,
            dx=self.dx,
            I_total=I_op,
            T_base=T0,
            Ic0=self.Ic0,
            Tc=self.Tc,
            n_value=self.n_value,
        )

        self.qa = QuenchAntennaModel()
        self._step_count = 0

        return self._get_obs(), self._get_info()

    def step(self, action):
        self._step_count += 1

        # Apply action: scale current
        current_fraction = float(np.clip(action[0], 0.0, 1.0))
        self.mesh.I_total = self._I_nominal * current_fraction

        # Random disturbance injection (incremental heating, not absolute)
        if self.np_random.random() < self.disturbance_prob:
            x_center = self.np_random.uniform(0, self.tape_width)
            width = self.np_random.uniform(*self.hotspot_width_range)
            dT = self.np_random.uniform(*self.hotspot_dT_range)
            # Add heat incrementally rather than setting absolute temperature
            x_positions = np.linspace(0, self.tape_width, self.nx)
            mask = np.abs(x_positions - x_center) <= width / 2.0
            self.mesh.T_field[:, mask] += dT

        # Physics step
        self.mesh.step(self.dt)

        # QA signal
        T_max = self.mesh.max_temperature()
        dT = T_max - self._T0
        qa_amp = max(dT / self.dT_crit, 0.0)
        self.qa.compute_voltage(amplitude=qa_amp)

        # --- Reward ---
        reward = 0.0

        # Operational reward: proportional to current maintained
        reward += 1.0 * current_fraction

        # Temperature penalty: penalize high local temperatures
        T_rebco = self.mesh.T_field[self.mesh.layer_names.index("rebco"), :]
        T_excess = np.maximum(T_rebco - self._T0 - self.dT_crit * 0.5, 0.0)
        reward -= 0.1 * np.sum(T_excess)

        # Uniformity bonus: reward uniform Jc utilization
        Ic_dist = self.mesh.get_Ic_distribution()
        if Ic_dist.std() > 0:
            uniformity = 1.0 - np.std(Ic_dist) / (np.mean(Ic_dist) + 1e-12)
            reward += 0.5 * uniformity

        # Quench check
        terminated = False
        truncated = False

        if T_max >= self._T0 + self.dT_crit:
            reward -= 100.0
            truncated = True

        if self._step_count >= self.max_steps:
            truncated = True

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def _get_obs(self) -> np.ndarray:
        Ic_dist = self.mesh.get_Ic_distribution()       # (nx,)
        E_field = self.mesh.get_E_field()
        rebco_idx = self.mesh.layer_names.index("rebco")
        E_rebco = E_field[rebco_idx, :]                 # (nx,)
        Jc_dist = self.mesh.get_Jc_field()              # (nx,)
        T_rebco = self.mesh.T_field[rebco_idx, :]       # (nx,)

        V_qa = self.qa.voltage
        I_total = self.mesh.I_total
        T_max = self.mesh.max_temperature()

        obs = np.concatenate([
            Ic_dist, E_rebco, Jc_dist, T_rebco,
            [V_qa, I_total, T_max],
        ]).astype(np.float32)

        return obs

    def _get_info(self) -> dict:
        rebco_frac = self.mesh.get_rebco_current_fraction()
        return {
            "step": self._step_count,
            "T_max": self.mesh.max_temperature(),
            "I_total": self.mesh.I_total,
            "rebco_current_fraction": rebco_frac.tolist(),
            "T0": self._T0,
            "Ic_distribution": self.mesh.get_Ic_distribution().tolist(),
        }

    def render(self):
        if self.render_mode == "ansi":
            T_max = self.mesh.max_temperature()
            I = self.mesh.I_total
            Ic_mean = np.mean(self.mesh.get_Ic_distribution())
            frac = np.mean(self.mesh.get_rebco_current_fraction())
            dT = T_max - self._T0
            bar_len = int(min(dT / self.dT_crit * 30, 30))
            bar = "█" * bar_len + "░" * (30 - bar_len)
            return (
                f"[Step {self._step_count:4d}] I={I:.0f}A Ic_avg={Ic_mean:.0f}A "
                f"Tmax={T_max:.1f}K REBCO_frac={frac:.2f}\n"
                f"  ΔT margin: [{bar}] {dT:.1f}/{self.dT_crit:.0f}K"
            )
