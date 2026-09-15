"""
current_policy.py

Minimal PPO for continuous current control.  No external RL dependencies —
a self-contained actor-critic with GAE, clipped surrogate objective, and
value-function loss.

The policy observes the 7-dim augmented observation from CurrentControlEnv
(4 telemetry channels + 3 surrogate predictions) and outputs a Gaussian
action in [0, 1] (desired I/I* fraction).

NOTE: the GRU surrogate was trained on fixed-ramp trajectories.  An RL
policy that chooses non-monotonic currents is out-of-distribution for the
surrogate and may elicit unreliable predictions.  This module is
experimental — the surrogate may need retraining on diverse current
schedules for robust RL control.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


class CurrentPolicy(nn.Module):
    """Actor-critic for continuous current control.

    Actor:  obs → (mean, log_std) → Normal → sample → clip to [0, 1]
    Critic: obs → scalar state value V(s)
    """

    def __init__(self, obs_dim: int = 7, hidden: int = 64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.actor_mean = nn.Linear(hidden, 1)
        self.actor_log_std = nn.Parameter(torch.zeros(1))
        self.critic = nn.Linear(hidden, 1)
        self.name = "rl_current"

    def forward(self, obs: torch.Tensor):
        h = self.shared(obs)
        mean = torch.sigmoid(self.actor_mean(h))
        std = self.actor_log_std.exp().expand_as(mean)
        value = self.critic(h).squeeze(-1)
        return mean, std, value

    def get_action(self, obs: torch.Tensor):
        mean, std, value = self(obs)
        dist = Normal(mean, std)
        action = dist.sample()
        action_clamped = action.clamp(0.0, 1.0)
        log_prob = dist.log_prob(action).sum(-1)
        return action_clamped, log_prob, value

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        mean, std, value = self(obs)
        dist = Normal(mean, std)
        log_prob = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_prob, value, entropy

    def reset(self) -> None:
        pass

    @torch.no_grad()
    def act(self, obs: np.ndarray, info: dict) -> np.ndarray:
        x = torch.from_numpy(np.asarray(obs, dtype=np.float32)).unsqueeze(0)
        mean, _, _ = self(x)
        return mean.squeeze().cpu().numpy()


@dataclass
class RolloutBuffer:
    obs: list
    actions: list
    log_probs: list
    rewards: list
    values: list
    dones: list

    @classmethod
    def empty(cls):
        return cls([], [], [], [], [], [])

    def add(self, obs, action, log_prob, reward, value, done):
        self.obs.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def __len__(self):
        return len(self.obs)

    def to_tensors(self):
        return (
            torch.stack(self.obs),
            torch.stack(self.actions),
            torch.stack(self.log_probs),
            torch.tensor(self.rewards, dtype=torch.float32),
            torch.stack(self.values),
            torch.tensor(self.dones, dtype=torch.float32),
        )


def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """Generalized Advantage Estimation."""
    T = len(rewards)
    advantages = torch.zeros(T)
    last_gae = 0.0
    for t in reversed(range(T)):
        next_value = values[t + 1] if t + 1 < T else 0.0
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        advantages[t] = last_gae = delta + gamma * lam * (1 - dones[t]) * last_gae
    returns = advantages + values
    return advantages, returns


class PPOTrainer:
    """Minimal PPO trainer for CurrentControlEnv.

    Parameters
    ----------
    env : CurrentControlEnv
        Environment with Box(1) action space and Box(7) observation space.
    policy : CurrentPolicy
        Actor-critic network.
    lr : float
        Learning rate.
    gamma, gae_lam : float
        Discount factor and GAE lambda.
    clip_eps : float
        PPO clipping parameter.
    n_steps : int
        Rollout length before each PPO update.
    n_epochs : int
        Number of optimization epochs per rollout.
    batch_size : int
        Minibatch size for PPO updates.
    ent_coef : float
        Entropy bonus coefficient.
    vf_coef : float
        Value function loss coefficient.
    """

    def __init__(
        self,
        env,
        policy: CurrentPolicy,
        *,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lam: float = 0.95,
        clip_eps: float = 0.2,
        n_steps: int = 2048,
        n_epochs: int = 10,
        batch_size: int = 64,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        verbose: bool = True,
    ):
        self.env = env
        self.policy = policy
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
        self.gamma = gamma
        self.gae_lam = gae_lam
        self.clip_eps = clip_eps
        self.n_steps = n_steps
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.verbose = verbose

    def collect_rollout(self) -> RolloutBuffer:
        buf = RolloutBuffer.empty()
        obs, _ = self.env.reset()
        obs_t = torch.from_numpy(obs)

        for _ in range(self.n_steps):
            action, log_prob, value = self.policy.get_action(obs_t.unsqueeze(0))
            action_np = action.squeeze(0).detach().numpy()
            next_obs, reward, terminated, truncated, info = self.env.step(action_np)

            buf.add(obs_t, action.squeeze(0).detach(), log_prob.squeeze().detach(),
                    reward, value.squeeze().detach(), float(terminated or truncated))

            if terminated or truncated:
                next_obs, _ = self.env.reset()
            obs_t = torch.from_numpy(next_obs)

        return buf

    def update(self, buf: RolloutBuffer) -> dict:
        obs, actions, old_log_probs, rewards, values, dones = buf.to_tensors()
        advantages, returns = compute_gae(rewards, values, dones, self.gamma, self.gae_lam)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        n = len(obs)
        total_pg, total_vf, total_ent = 0.0, 0.0, 0.0
        n_updates = 0

        for _ in range(self.n_epochs):
            perm = torch.randperm(n)
            for start in range(0, n, self.batch_size):
                idx = perm[start:start + self.batch_size]
                log_probs_new, values_new, entropy = self.policy.evaluate_actions(
                    obs[idx], actions[idx])

                ratio = (log_probs_new - old_log_probs[idx]).exp()
                surr1 = ratio * advantages[idx]
                surr2 = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * advantages[idx]
                pg_loss = -torch.min(surr1, surr2).mean()
                vf_loss = nn.functional.mse_loss(values_new, returns[idx])
                ent_loss = -entropy.mean()

                loss = pg_loss + self.vf_coef * vf_loss + self.ent_coef * ent_loss
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()

                total_pg += pg_loss.item()
                total_vf += vf_loss.item()
                total_ent += entropy.mean().item()
                n_updates += 1

        return {
            "pg_loss": total_pg / max(n_updates, 1),
            "vf_loss": total_vf / max(n_updates, 1),
            "mean_entropy": total_ent / max(n_updates, 1),
            "mean_reward": float(rewards.mean()),
        }

    def train(self, total_timesteps: int = 100_000, seed: int = 42) -> CurrentPolicy:
        """Run PPO training for total_timesteps steps."""
        n_iterations = max(1, total_timesteps // self.n_steps)
        for iteration in range(n_iterations):
            buf = self.collect_rollout()
            stats = self.update(buf)
            if self.verbose and (iteration % 5 == 0 or iteration == n_iterations - 1):
                print(f"  iter {iteration:3d}/{n_iterations}  "
                      f"reward={stats['mean_reward']:.3f}  "
                      f"pg={stats['pg_loss']:.4f}  "
                      f"vf={stats['vf_loss']:.4f}  "
                      f"ent={stats['mean_entropy']:.4f}")
        return self.policy
