"""
ThermalRunawayEnv-v0

A Gymnasium environment for learning quench protection policy on a
single REBCO tape element using the Marchevsky & Prestemon thermal
runaway criterion.

Observation space (Box):
  [T, T0, I_op, Ic(T), dT/dt, V_qa, margin]
  - T:        current tape temperature [K]
  - T0:       coolant base temperature [K]
  - I_op:     operating current [A]
  - Ic_T:     critical current at current T [A]
  - dTdt:     temperature rate of change [K/s]
  - V_qa:     quench antenna voltage signal [V]
  - margin:   I_op / Ic(T) ratio (>1 means overcritical)

Action space (Discrete(2)):
  0 = HOLD   — maintain current, keep operating
  1 = RAMP_DOWN — initiate protective ramp-down

Reward:
  Each timestep at full current: +1.0 (operational reward)
  Successful ramp-down before quench: +5.0
  Quench (T > T0 + dT_crit): -100.0 (magnet damage)
  Unnecessary ramp-down (no quench threat): -10.0
  Ramp-down in progress: +0.1 per step (safe operation)

Episode ends when:
  - Quench occurs (T > T0 + dT_crit)  → truncated
  - Successful ramp-down completed     → terminated
  - Max steps reached                  → truncated

The environment injects random flux disturbances that cause local
heating. The agent must learn when dT/dt or V_qa signals indicate
an approaching thermal runaway and trigger ramp-down in time.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from ..physics.thermal import ThermalRunawayModel
from ..physics.quench_antenna import QuenchAntennaModel


class ThermalRunawayEnv(gym.Env):
    """Thermal runaway protection environment for a single REBCO tape element."""

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 10}

    def __init__(
        self,
        render_mode=None,
        # Physics parameters
        Ic0: float = 500.0,
        Tc: float = 85.0,
        n_value: float = 20.0,
        T0_range: tuple = (15.0, 50.0),
        I_op_fraction_range: tuple = (0.6, 0.95),
        # Disturbance parameters
        disturbance_prob: float = 0.05,
        disturbance_dT_range: tuple = (2.0, 20.0),
        # Episode parameters
        dt: float = 0.005,
        max_steps: int = 500,
        dT_crit: float = 12.0,
        ramp_down_steps: int = 20,
        # Reward shaping
        reward_operate: float = 1.0,
        reward_safe_rampdown: float = 5.0,
        reward_quench: float = -100.0,
        reward_unnecessary_rampdown: float = -10.0,
    ):
        super().__init__()
        self.render_mode = render_mode

        # Store config
        self.Ic0 = Ic0
        self.Tc = Tc
        self.n_value = n_value
        self.T0_range = T0_range
        self.I_op_fraction_range = I_op_fraction_range
        self.disturbance_prob = disturbance_prob
        self.disturbance_dT_range = disturbance_dT_range
        self.dt = dt
        self.max_steps = max_steps
        self.dT_crit = dT_crit
        self.ramp_down_steps = ramp_down_steps

        # Rewards
        self.reward_operate = reward_operate
        self.reward_safe_rampdown = reward_safe_rampdown
        self.reward_quench = reward_quench
        self.reward_unnecessary_rampdown = reward_unnecessary_rampdown

        # Action: 0=hold, 1=ramp_down
        self.action_space = spaces.Discrete(2)

        # Observation: [T, T0, I_op, Ic(T), dTdt, V_qa, margin]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, -1e4, 0.0, 0.0], dtype=np.float32),
            high=np.array([500.0, 100.0, 2000.0, 2000.0, 1e4, 100.0, 10.0], dtype=np.float32),
        )

        # Physics models (created on reset)
        self.thermal: ThermalRunawayModel = None
        self.qa: QuenchAntennaModel = None

        # Episode state
        self._step_count = 0
        self._ramping_down = False
        self._ramp_steps_remaining = 0
        self._disturbance_active = False
        self._disturbance_magnitude = 0.0
        self._initial_I_op = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Randomize operating conditions
        T0 = self.np_random.uniform(*self.T0_range)
        I_frac = self.np_random.uniform(*self.I_op_fraction_range)

        self.thermal = ThermalRunawayModel(
            Ic0=self.Ic0, Tc=self.Tc, n_value=self.n_value,
        )
        self.thermal.reset(T0=T0)
        Ic_at_T0 = self.thermal.Ic(T0)
        I_op = I_frac * Ic_at_T0
        self.thermal.I_op = I_op
        self._initial_I_op = I_op

        self.qa = QuenchAntennaModel()

        self._step_count = 0
        self._ramping_down = False
        self._ramp_steps_remaining = 0
        self._disturbance_active = False
        self._disturbance_magnitude = 0.0

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action):
        assert self.action_space.contains(action), f"Invalid action {action}"
        self._step_count += 1

        reward = 0.0
        terminated = False
        truncated = False

        # --- Handle ramp-down logic ---
        if action == 1 and not self._ramping_down:
            self._ramping_down = True
            self._ramp_steps_remaining = self.ramp_down_steps

            # Was this ramp-down necessary?
            # Check if we were actually heading toward runaway
            dTdt = self.thermal.dTdt(self.thermal.I_op, self.thermal.T)
            margin = self.thermal.I_op / max(self.thermal.Ic(), 1e-12)
            if dTdt > 0.5 or margin > 0.85 or self._disturbance_active:
                reward += self.reward_safe_rampdown
            else:
                reward += self.reward_unnecessary_rampdown

        # --- Inject disturbances ---
        if not self._ramping_down and not self._disturbance_active:
            if self.np_random.random() < self.disturbance_prob:
                self._disturbance_active = True
                self._disturbance_magnitude = self.np_random.uniform(
                    *self.disturbance_dT_range
                )

        # --- Apply disturbance as local heating ---
        if self._disturbance_active and not self._ramping_down:
            self.thermal.T += self._disturbance_magnitude * self.dt
            self.thermal.T = float(np.clip(self.thermal.T, 4.0, 500.0))

        # --- Physics step ---
        if self._ramping_down:
            # Linearly ramp current to zero
            frac = self._ramp_steps_remaining / self.ramp_down_steps
            self.thermal.I_op = self._initial_I_op * frac
            self._ramp_steps_remaining -= 1
            reward += 0.1  # small reward for safe operation

            if self._ramp_steps_remaining <= 0:
                terminated = True  # safe shutdown complete
        else:
            reward += self.reward_operate

        self.thermal.step(self.dt)

        # --- QA voltage signal ---
        # Scale QA response by temperature rise (proxy for flux disturbance)
        dT = self.thermal.T - self.thermal.T0
        qa_amplitude = max(dT / self.dT_crit, 0.0)
        self.qa.compute_voltage(amplitude=qa_amplitude)

        # --- Check quench ---
        if self.thermal.is_quenched(self.dT_crit):
            reward += self.reward_quench
            truncated = True

        # --- Check max steps ---
        if self._step_count >= self.max_steps:
            truncated = True

        obs = self._get_obs()
        info = self._get_info()

        return obs, reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        T = self.thermal.T
        T0 = self.thermal.T0
        I_op = self.thermal.I_op
        Ic_T = self.thermal.Ic()
        dTdt = self.thermal.dTdt(I_op, T)
        V_qa = self.qa.voltage
        margin = I_op / max(Ic_T, 1e-12)

        return np.array([T, T0, I_op, Ic_T, dTdt, V_qa, margin],
                        dtype=np.float32)

    def _get_info(self) -> dict:
        return {
            "step": self._step_count,
            "ramping_down": self._ramping_down,
            "disturbance_active": self._disturbance_active,
            "disturbance_magnitude": self._disturbance_magnitude,
            "T": self.thermal.T,
            "T0": self.thermal.T0,
            "I_op": self.thermal.I_op,
            "Ic": self.thermal.Ic(),
            "dT": self.thermal.T - self.thermal.T0,
        }

    def render(self):
        if self.render_mode == "ansi":
            T = self.thermal.T
            T0 = self.thermal.T0
            I = self.thermal.I_op
            Ic = self.thermal.Ic()
            dT = T - T0
            bar_len = int(min(dT / self.dT_crit * 40, 40))
            bar = "█" * bar_len + "░" * (40 - bar_len)
            status = "RAMP↓" if self._ramping_down else "RUN  "
            return (
                f"[{status}] T={T:.1f}K T0={T0:.1f}K I={I:.0f}A "
                f"Ic={Ic:.0f}A ΔT={dT:.2f}K\n"
                f"  Margin: [{bar}] {dT/self.dT_crit*100:.0f}%"
            )
        elif self.render_mode == "human":
            print(self.render())
