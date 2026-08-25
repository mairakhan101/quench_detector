"""
test_fem_env.py -- FemReplayQuenchEnv (fem_env.py). Skipped if the FEM sweep
isn't reachable, same as test_fem_data.py.
"""

import os

import numpy as np
import pytest

from hts_tape_env.fem_data import DEFAULT_PATH

pytestmark = pytest.mark.skipif(
    not os.path.exists(DEFAULT_PATH),
    reason=f"FEM sweep not reachable at {DEFAULT_PATH} (network mount)",
)


@pytest.fixture(scope="module")
def env():
    from hts_tape_env.fem_env import FemReplayQuenchEnv
    return FemReplayQuenchEnv()


def test_six_episodes(env):
    assert env.n_episodes == 6


def test_observation_layout_fixed_length(env):
    from hts_tape_env.fem_data import RTD_CHANNELS, VTAP_CHANNELS
    assert env.observation_space.shape == (1 + len(RTD_CHANNELS) + len(VTAP_CHANNELS),)
    assert len(env.rtd_order) == len(RTD_CHANNELS)
    assert len(env.vtap_order) == len(VTAP_CHANNELS)


def test_reset_at_specific_run(env):
    obs, info = env.reset_at("r0p99", seed=0)
    assert env.observation_space.contains(obs)
    assert info["ratio_label"] == "r0p99"
    assert info["step"] == 0


def test_runaway_episode_reaches_negative_margin_without_dump(env):
    obs, info = env.reset_at("r0p99", seed=0)
    terminated = truncated = False
    while not (terminated or truncated):
        obs, reward, terminated, truncated, info = env.step(0)
    assert terminated and not truncated
    assert info["runaway"]
    assert reward == env.reward_missed


def test_stable_episode_truncates_safely(env):
    obs, info = env.reset_at("r0p5", seed=0)
    terminated = truncated = False
    while not (terminated or truncated):
        obs, reward, terminated, truncated, info = env.step(0)
    assert truncated and not terminated
    assert not info["runaway"]


def test_margin_sign_matches_runaway_flag(env):
    for label in ("r0p5", "r0p99"):
        obs, info = env.reset_at(label, seed=0)
        terminated = truncated = False
        while not (terminated or truncated):
            obs, reward, terminated, truncated, info = env.step(0)
            assert info["runaway"] == (info["margin"] <= 0.0)
