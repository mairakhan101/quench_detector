"""Tests for continuous current control: CurrentControlEnv + SupervisedCurrentController."""

import numpy as np
import torch
import pytest

from hts_tape_env.env import QuenchDetectorEnv
from hts_tape_env.controllers import (
    CurrentControlEnv,
    SupervisedCurrentController,
    run_current_episode,
    evaluate_current_controller,
)


class TestCurrentControlEnv:
    def test_action_space(self):
        inner = QuenchDetectorEnv(mode="no_stab")
        env = CurrentControlEnv(inner)
        assert env.action_space.shape == (1,)
        assert float(env.action_space.low[0]) == 0.0
        assert float(env.action_space.high[0]) == 1.0

    def test_observation_space(self):
        inner = QuenchDetectorEnv(mode="no_stab")
        env = CurrentControlEnv(inner)
        assert env.observation_space.shape == (7,)

    def test_reset_returns_7d_obs(self):
        inner = QuenchDetectorEnv(mode="no_stab")
        env = CurrentControlEnv(inner)
        obs, info = env.reset(seed=42)
        assert obs.shape == (7,)
        assert "margin" in info

    def test_step_returns_correct_shapes(self):
        inner = QuenchDetectorEnv(mode="no_stab")
        env = CurrentControlEnv(inner)
        env.reset(seed=42)
        action = np.array([0.5], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == (7,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert "margin" in info
        assert "I_frac" in info
        assert "integrated_current_As" in info
        assert info["integrated_current_As"] > 0

    def test_current_fraction_respected(self):
        inner = QuenchDetectorEnv(mode="no_stab")
        env = CurrentControlEnv(inner)
        env.reset(seed=42)
        target_frac = 0.4
        action = np.array([target_frac], dtype=np.float32)
        _, _, _, _, info = env.step(action)
        assert abs(info["I_frac"] - target_frac) < 1e-6

    def test_runaway_on_high_current(self):
        inner = QuenchDetectorEnv(mode="no_stab")
        env = CurrentControlEnv(inner)
        env.reset(seed=42)
        terminated = False
        for _ in range(200):
            action = np.array([1.05], dtype=np.float32)
            _, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        assert terminated or truncated


class TestSupervisedCurrentController:
    def test_forward_shape(self):
        ctrl = SupervisedCurrentController(obs_dim=7, hidden=32)
        x = torch.randn(4, 7)
        out = ctrl(x)
        assert out.shape == (4, 1)
        assert (out >= 0).all() and (out <= 1).all()

    def test_act_returns_scalar(self):
        ctrl = SupervisedCurrentController(obs_dim=7, hidden=32)
        obs = np.random.randn(7).astype(np.float32)
        action = ctrl.act(obs, {})
        assert isinstance(action, (float, np.floating, np.ndarray))

    def test_reset_is_noop(self):
        ctrl = SupervisedCurrentController()
        ctrl.reset()


class TestEvaluateCurrentController:
    def test_run_episode(self):
        inner = QuenchDetectorEnv(mode="no_stab")
        env = CurrentControlEnv(inner)
        ctrl = SupervisedCurrentController(obs_dim=7, hidden=16)
        result = run_current_episode(env, ctrl, seed=42)
        assert "reward" in result
        assert "mean_utilization" in result
        assert "integrated_current_As" in result
        assert "steps" in result
        assert result["steps"] > 0
        assert result["integrated_current_As"] >= 0

    def test_evaluate(self):
        inner = QuenchDetectorEnv(mode="no_stab")
        env = CurrentControlEnv(inner)
        ctrl = SupervisedCurrentController(obs_dim=7, hidden=16)
        result = evaluate_current_controller(env, ctrl, n_episodes=3, seed0=0)
        assert "mean_reward" in result
        assert "mean_utilization" in result
        assert "mean_integrated_current_As" in result
        assert "mean_integrated_current_normed" in result
        assert "safety_rate" in result
        assert "mean_steps" in result
        assert 0 <= result["safety_rate"] <= 1
        assert result["mean_integrated_current_As"] >= 0
