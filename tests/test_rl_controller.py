"""Tests for the RL current controller (PPO)."""

import numpy as np
import torch
import pytest

from hts_tape_env.env import QuenchDetectorEnv
from hts_tape_env.controllers import CurrentControlEnv
from hts_tape_env.current_policy import (
    CurrentPolicy,
    RolloutBuffer,
    compute_gae,
    PPOTrainer,
)


class TestCurrentPolicy:
    def test_forward_shapes(self):
        policy = CurrentPolicy(obs_dim=7, hidden=32)
        obs = torch.randn(4, 7)
        mean, std, value = policy(obs)
        assert mean.shape == (4, 1)
        assert std.shape == (4, 1)
        assert value.shape == (4,)

    def test_mean_in_01(self):
        policy = CurrentPolicy(obs_dim=7, hidden=32)
        obs = torch.randn(10, 7)
        mean, _, _ = policy(obs)
        assert (mean >= 0).all() and (mean <= 1).all()

    def test_get_action(self):
        policy = CurrentPolicy(obs_dim=7, hidden=32)
        obs = torch.randn(1, 7)
        action, log_prob, value = policy.get_action(obs)
        assert action.shape == (1, 1)
        assert log_prob.shape == (1,)
        assert value.shape == (1,)
        assert (action >= 0).all() and (action <= 1).all()

    def test_evaluate_actions(self):
        policy = CurrentPolicy(obs_dim=7, hidden=32)
        obs = torch.randn(4, 7)
        actions = torch.rand(4, 1)
        log_prob, value, entropy = policy.evaluate_actions(obs, actions)
        assert log_prob.shape == (4,)
        assert value.shape == (4,)
        assert entropy.shape == (4,)

    def test_act(self):
        policy = CurrentPolicy(obs_dim=7, hidden=32)
        obs = np.random.randn(7).astype(np.float32)
        action = policy.act(obs, {})
        assert isinstance(action, (float, np.floating, np.ndarray))

    def test_reset_is_noop(self):
        policy = CurrentPolicy()
        policy.reset()


class TestComputeGAE:
    def test_shape(self):
        rewards = torch.tensor([1.0, 2.0, 3.0])
        values = torch.tensor([0.5, 1.0, 1.5])
        dones = torch.tensor([0.0, 0.0, 1.0])
        advantages, returns = compute_gae(rewards, values, dones)
        assert advantages.shape == (3,)
        assert returns.shape == (3,)

    def test_terminal_advantage(self):
        rewards = torch.tensor([1.0])
        values = torch.tensor([0.5])
        dones = torch.tensor([1.0])
        advantages, _ = compute_gae(rewards, values, dones)
        assert abs(advantages[0].item() - 0.5) < 1e-5


class TestPPOTrainer:
    def _make_env(self):
        inner = QuenchDetectorEnv(mode="no_stab")
        return CurrentControlEnv(inner, max_steps=50)

    def test_collect_rollout(self):
        env = self._make_env()
        policy = CurrentPolicy(obs_dim=7, hidden=16)
        trainer = PPOTrainer(env, policy, n_steps=100, verbose=False)
        buf = trainer.collect_rollout()
        assert len(buf) == 100

    def test_one_update(self):
        env = self._make_env()
        policy = CurrentPolicy(obs_dim=7, hidden=16)
        trainer = PPOTrainer(env, policy, n_steps=100, n_epochs=2,
                             batch_size=32, verbose=False)
        buf = trainer.collect_rollout()
        stats = trainer.update(buf)
        assert "pg_loss" in stats
        assert "vf_loss" in stats
        assert "mean_entropy" in stats
        assert "mean_reward" in stats

    def test_training_smoke(self):
        env = self._make_env()
        policy = CurrentPolicy(obs_dim=7, hidden=16)
        trainer = PPOTrainer(env, policy, n_steps=100, n_epochs=2,
                             batch_size=32, verbose=False)
        trained = trainer.train(total_timesteps=200)
        assert isinstance(trained, CurrentPolicy)
