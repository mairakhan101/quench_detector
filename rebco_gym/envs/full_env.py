"""
REBCOFullEnv-v0

Coupled thermal + electromagnetic Gymnasium environment for a spatially
resolved REBCO tape. Each episode poses a current ramp-up task: a
*requested* ramp rate (A/s) defines how fast the operator wants the magnet
charged, under randomly sampled field conditions (self-field + external
field at some angle) and a randomly sampled Ic dropout map (manufacturing
weak spots along the tape, from `physics.electromagnetic.IcDropoutField`).

The agent's control is how much of that requested ramp rate to actually
apply at each step — i.e. it can load current more slowly than requested
to keep local heating/current-sharing margin in check, trading ramp speed
for safety.

Observation space (Box, instrument-style signals — matches the four-sensor
philosophy of the notebook this ports from: bath thermometer, current
shunt, voltage taps, surface thermometer):
  [I_op, I_requested_target, T0, T_surface, terminal_voltage, V_qa,
   dTdt, worst_local_margin, B_self, B_ext]
Ground-truth-only quantities (per-segment Ic, dropout map) are exposed via
`info`, not observation, so the agent works from realistic sensors.

Action space (Box(1)): fraction in [0, 1] of the requested ramp rate to
apply this step. 0 = hold, 1 = track the full requested ramp.

Reward:
  + progress toward the ramp target
  - penalty for falling behind the requested ramp (unjustified throttling)
  - penalty for local overheating margin
  - large penalty on quench
  + bonus for reaching and holding the ramp target safely

Episode ends when:
  - Quench (T_max > T0 + dT_crit)          -> truncated
  - Target current reached and held stable  -> terminated
  - Max steps reached                       -> truncated
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from ..physics.spatial_tape import SpatialTape, SpatialTapeParams
from ..physics.electromagnetic import AnisotropicJcModel, IcDropoutField, self_field
from ..physics.quench_antenna import QuenchAntennaModel


class REBCOFullEnv(gym.Env):
    """Thermal + electromagnetic ramp-control environment for a REBCO tape."""

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 10}

    def __init__(
        self,
        render_mode=None,
        # Tape / physics parameters
        nz: int = 30,
        Ic0: float = 400.0,
        Tc: float = 90.0,
        T0_range: tuple = (15.0, 50.0),
        I_target_fraction_range: tuple = (0.6, 0.9),
        ramp_rate_range: tuple = (500.0, 4000.0),   # A/s, requested charge rate
        # EM / dropout parameters
        B_ext_range: tuple = (0.0, 0.15),
        B_angle_range: tuple = (0.0, np.pi / 2.0),
        dropout_severity_range: tuple = (0.5, 1.5),  # scales background_std & defect_rate
        # Episode parameters
        dt: float = 2e-4,
        max_steps: int = 800,
        dT_crit: float = 15.0,
        dT_hold_margin: float = 3.0,
        hold_steps_required: int = 15,
        # Reward shaping
        reward_progress: float = 20.0,
        reward_lag_penalty: float = 5.0,
        reward_overheat: float = 0.5,
        reward_quench: float = -100.0,
        reward_complete: float = 20.0,
    ):
        super().__init__()
        self.render_mode = render_mode

        self.nz = nz
        self.Ic0 = Ic0
        self.Tc = Tc
        self.T0_range = T0_range
        self.I_target_fraction_range = I_target_fraction_range
        self.ramp_rate_range = ramp_rate_range
        self.B_ext_range = B_ext_range
        self.B_angle_range = B_angle_range
        self.dropout_severity_range = dropout_severity_range

        self.dt = dt
        self.max_steps = max_steps
        self.dT_crit = dT_crit
        self.dT_hold_margin = dT_hold_margin
        self.hold_steps_required = hold_steps_required

        self.reward_progress = reward_progress
        self.reward_lag_penalty = reward_lag_penalty
        self.reward_overheat = reward_overheat
        self.reward_quench = reward_quench
        self.reward_complete = reward_complete

        # Action: fraction of requested ramp rate to apply, in [0, 1]
        self.action_space = spaces.Box(
            low=np.array([0.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
        )

        # Observation: [I_op, I_requested_target, T0, T_surface, V_terminal,
        #               V_qa, dTdt, worst_local_margin, B_self, B_ext]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32,
        )

        self.tape: SpatialTape = None
        self.qa: QuenchAntennaModel = None

        self._step_count = 0
        self._T0 = 20.0
        self._I_target_final = 0.0
        self._requested_rate = 0.0
        self._I_requested_target = 0.0
        self._I_op = 0.0
        self._prev_Tmax = 20.0
        self._hold_counter = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        T0 = float(self.np_random.uniform(*self.T0_range))
        I_target_frac = float(self.np_random.uniform(*self.I_target_fraction_range))
        rate = float(self.np_random.uniform(*self.ramp_rate_range))
        B_ext = float(self.np_random.uniform(*self.B_ext_range))
        B_angle = float(self.np_random.uniform(*self.B_angle_range))
        severity = float(self.np_random.uniform(*self.dropout_severity_range))

        params = SpatialTapeParams(Ic0=self.Ic0, Tc=self.Tc, T_bath=T0)
        jc_model = AnisotropicJcModel()
        dropout = IcDropoutField(
            background_std=0.03 * severity,
            defect_rate=2.0 * severity,
        )
        rng = np.random.default_rng(int(self.np_random.integers(0, 2**31 - 1)))

        self.tape = SpatialTape(
            nz=self.nz, params=params, jc_model=jc_model, dropout_field=dropout,
            B_ext=B_ext, B_angle_rad=B_angle, rng=rng,
        )
        self.qa = QuenchAntennaModel()

        self._T0 = T0
        self._I_target_final = I_target_frac * self.Ic0
        self._requested_rate = rate
        self._I_requested_target = 0.0
        self._I_op = 0.0
        self._prev_Tmax = T0
        self._step_count = 0
        self._hold_counter = 0

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action):
        self._step_count += 1
        frac = float(np.clip(action[0], 0.0, 1.0))

        self._I_requested_target = min(
            self._I_requested_target + self._requested_rate * self.dt,
            self._I_target_final,
        )
        delta_I = frac * self._requested_rate * self.dt
        prev_I_op = self._I_op
        self._I_op = float(np.clip(self._I_op + delta_I, 0.0, self._I_target_final))

        self.tape.step(self._I_op, self.dt)

        T_max = self.tape.max_temperature()
        dTdt = (T_max - self._prev_Tmax) / self.dt
        self._prev_Tmax = T_max
        worst_margin = self.tape.min_local_margin()

        # --- Reward ---
        reward = 0.0
        actual_progress = self._I_op - prev_I_op
        reward += self.reward_progress * (actual_progress / max(self._I_target_final, 1e-9))

        lag = max(self._I_requested_target - self._I_op, 0.0)
        reward -= self.reward_lag_penalty * (lag / max(self._I_target_final, 1e-9))

        dT = T_max - self._T0
        overheat = max(dT - self.dT_hold_margin, 0.0)
        reward -= self.reward_overheat * overheat

        terminated = False
        truncated = False

        if dT >= self.dT_crit:
            reward += self.reward_quench
            truncated = True

        at_target = self._I_op >= 0.999 * self._I_target_final
        stable = dT < self.dT_hold_margin
        if at_target and stable and not truncated:
            self._hold_counter += 1
        else:
            self._hold_counter = 0

        if self._hold_counter >= self.hold_steps_required:
            reward += self.reward_complete
            terminated = True

        if self._step_count >= self.max_steps:
            truncated = True

        obs = self._get_obs()
        info = self._get_info()
        return obs, reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        T_max = self.tape.max_temperature()
        dTdt = (T_max - self._prev_Tmax) / self.dt if self._step_count > 0 else 0.0
        worst_margin = self.tape.min_local_margin()
        V_terminal = self.tape.terminal_voltage()
        V_qa = self.tape.qa_signal(self.qa)
        B_self = self_field(self._I_op, self.tape.p.width, self.tape.self_field_coeff)

        return np.array([
            self._I_op, self._I_requested_target, self._T0, T_max,
            V_terminal, V_qa, dTdt, worst_margin, B_self, self.tape.B_ext,
        ], dtype=np.float32)

    def _get_info(self) -> dict:
        return {
            "step": self._step_count,
            "T0": self._T0,
            "T_max": self.tape.max_temperature(),
            "I_op": self._I_op,
            "I_requested_target": self._I_requested_target,
            "I_target_final": self._I_target_final,
            "requested_rate": self._requested_rate,
            "worst_local_margin": self.tape.min_local_margin(),
            "Ic0_local": self.tape.Ic0_local.tolist(),
            "hold_counter": self._hold_counter,
        }

    def render(self):
        if self.render_mode == "ansi":
            T_max = self.tape.max_temperature()
            dT = T_max - self._T0
            bar_len = int(min(dT / self.dT_crit * 40, 40))
            bar = "█" * bar_len + "░" * (40 - bar_len)
            return (
                f"[Step {self._step_count:4d}] I_op={self._I_op:.0f}A "
                f"target={self._I_target_final:.0f}A req={self._I_requested_target:.0f}A "
                f"Tmax={T_max:.1f}K margin={self.tape.min_local_margin():.2f}\n"
                f"  ΔT: [{bar}] {dT:.1f}/{self.dT_crit:.0f}K  hold={self._hold_counter}"
            )
        elif self.render_mode == "human":
            print(self.render())
