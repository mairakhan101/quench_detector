"""
REBCO Gym — Farama Gymnasium environments for REBCO tape quench protection.

Environments:
  ThermalRunawayEnv-v0   : Binary ramp-down decision based on thermal state
  CurrentRedistEnv-v0    : Current redistribution from mesh + QA signal
  REBCOFullEnv-v0        : Spatially-resolved ramp control with thermal +
                            electromagnetic (B-field, Ic dropout) coupling
"""

from gymnasium.envs.registration import register

register(
    id="ThermalRunawayEnv-v0",
    entry_point="rebco_gym.envs.thermal_runaway_env:ThermalRunawayEnv",
)

register(
    id="CurrentRedistEnv-v0",
    entry_point="rebco_gym.envs.current_redist_env:CurrentRedistEnv",
)

register(
    id="REBCOFullEnv-v0",
    entry_point="rebco_gym.envs.full_env:REBCOFullEnv",
)
