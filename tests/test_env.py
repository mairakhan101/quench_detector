"""
test_env.py -- QuenchDetectorEnv (env.py).
"""

import numpy as np
import pytest

from hts_tape_env.env import QuenchDetectorEnv


def test_reset_obs_in_bounds():
    env = QuenchDetectorEnv()
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    assert info["margin"] == pytest.approx(1.0 - env.I_start_frac)
    assert not info["runaway"]


def test_reset_at_deterministic():
    env = QuenchDetectorEnv()
    obs1, info1 = env.reset_at(40.0, 80.0, seed=0)
    obs2, info2 = env.reset_at(40.0, 80.0, seed=0)
    assert info1["I_star"] == info2["I_star"]
    assert info1["T_star"] == info2["T_star"]


def test_dump_action_terminates_immediately():
    env = QuenchDetectorEnv()
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(1)
    assert terminated
    assert not truncated


def test_continue_ramps_margin_down():
    env = QuenchDetectorEnv(ramp_rate=0.01)
    obs, info = env.reset(seed=0)
    m0 = info["margin"]
    obs, reward, terminated, truncated, info = env.step(0)
    assert info["margin"] == pytest.approx(m0 - 0.01, abs=1e-6)


def test_episode_terminates_or_truncates_eventually():
    """Always-continue should reach a terminal state (runaway) or truncate
    (safe ramp completed) within a bounded number of steps -- not hang."""
    env = QuenchDetectorEnv(ramp_rate=0.02)
    obs, info = env.reset(seed=1)
    for _ in range(1000):
        obs, reward, terminated, truncated, info = env.step(0)
        if terminated or truncated:
            return
    pytest.fail("episode did not terminate/truncate within 1000 steps")


def test_missed_quench_reward_is_the_worst_outcome():
    """Running straight through I* with no dump should score reward_missed,
    the most negative of the four reward branches."""
    env = QuenchDetectorEnv(ramp_rate=0.05, I_max_frac=1.5)
    obs, info = env.reset_at(40.0, 80.0, seed=0)
    terminated = truncated = False
    reward = None
    while not (terminated or truncated):
        obs, reward, terminated, truncated, info = env.step(0)
    assert info["runaway"]
    assert reward == env.reward_missed


def test_env_checker():
    """Gymnasium's own API-compliance checker (regression guard for the
    reset()/step() contract)."""
    gym_check = pytest.importorskip("gymnasium.utils.env_checker")
    env = QuenchDetectorEnv()
    gym_check.check_env(env, skip_render_check=True)


def test_true_channels_used_for_stab_bug_regression():
    """physics.py's stable_surface_T bracket-bug fix (see its docstring):
    T_surface_true_K should not collapse to T0 for the large majority of
    steps across many episodes' full ramps -- a loose regression guard, not
    a tight bound (some early-ramp steps genuinely have negligible
    self-heating and T_surface==T0 is the physically correct answer there)."""
    env = QuenchDetectorEnv()
    n_fallback, n_total = 0, 0
    for ep in range(30):
        obs, info = env.reset(seed=ep)
        terminated = truncated = False
        while not (terminated or truncated):
            obs, reward, terminated, truncated, info = env.step(0)
            n_total += 1
            n_fallback += int(info["T_surface_true_K"] == info["T0"])
    assert n_fallback / n_total < 0.35
