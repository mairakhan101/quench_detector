"""
controllers.py

Decision-making policies over QuenchDetectorEnv's Discrete(2) action space
(0=continue, 1=dump). These are rule-based ("bang-bang"/thermostat-style)
controllers, not RL-trained agents -- the two simplest baselines the sequence
surrogate is meant to be judged against:

    oracle_controller()      Dumps using the environment's ground-truth
                              margin (info["margin"], derived from I*_true).
                              This is "prevent I from ever going above
                              I*_true": it has perfect knowledge and is the
                              best any controller sharing this action/reward
                              structure can do.

    surrogate_controller()   Dumps using the trained sequence surrogate's
                              margin_pred (info["margin_pred"]) -- the
                              realistic, deployable controller: it never
                              observes I*_true, only the live telemetry.

    surrogate_prob_controller()  Same realistic setting, but dumps on the
                              surrogate's dedicated breach_prob_pred
                              classifier head instead of thresholding the
                              margin regression -- trained directly against
                              the decision boundary (BCE), so typically the
                              better of the two realistic controllers.

All three are the same BangBangController shape (dump iff signal < threshold,
or > for the probability one), just pointed at a different signal, so
they're directly comparable via evaluate().

Usage
-----
    from hts_tape_env.env import QuenchDetectorEnv
    from hts_tape_env.seq_surrogates import SequenceSurrogateAdapter, load_seq_surrogate
    from hts_tape_env.controllers import oracle_controller, surrogate_controller, evaluate

    env = QuenchDetectorEnv()
    print(evaluate(env, oracle_controller(env.dump_threshold), n_episodes=200, seed0=0))

    env2 = QuenchDetectorEnv()
    env2.set_surrogate(SequenceSurrogateAdapter(load_seq_surrogate("models/seq_gru.pt")))
    print(evaluate(env2, surrogate_controller(env2.dump_threshold), n_episodes=200, seed0=0))
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import gymnasium


@dataclass
class BangBangController:
    """Binary on/off controller: dump (action=1) iff signal_fn(info) OP threshold,
    where OP is '<' (direction='below', default) or '>' (direction='above')."""

    signal_fn: Callable[[dict], float]
    threshold: float
    direction: str = "below"
    name: str = "bang_bang"

    def reset(self) -> None:
        pass  # stateless -- the signal itself may come from a stateful surrogate

    def act(self, obs, info: dict) -> int:
        signal = self.signal_fn(info)
        return int(signal < self.threshold if self.direction == "below" else signal > self.threshold)


def oracle_controller(threshold: float = 0.05) -> BangBangController:
    """Dumps on the ground-truth margin -- 'never let I exceed I*_true', with
    `threshold` as the same safety buffer the env's own reward function uses
    to call a dump "timely" (needs to be at least one ramp_rate wide, since
    the env applies one more ramp step in the same call that executes the
    dump; the default 0.05 comfortably covers the default ramp_rate=0.005)."""
    return BangBangController(signal_fn=lambda info: info["margin"], threshold=threshold,
                               name="oracle (true I*)")


def surrogate_controller(threshold: float = 0.05) -> BangBangController:
    """Dumps on the trained sequence surrogate's margin_pred -- requires
    env.set_surrogate() to have been called; info["margin_pred"] otherwise
    won't exist."""
    return BangBangController(signal_fn=lambda info: info["margin_pred"], threshold=threshold,
                               name="surrogate (margin_pred)")


def surrogate_prob_controller(threshold: float = 0.5) -> BangBangController:
    """Dumps on the surrogate's breach_prob_pred classifier head -- trained
    directly against 'margin < dump_threshold' via BCE, rather than inferred
    from the margin regression. Requires env.set_surrogate()."""
    return BangBangController(signal_fn=lambda info: info["breach_prob_pred"], threshold=threshold,
                               direction="above", name="surrogate (breach_prob_pred)")


def run_episode(env, controller: BangBangController, seed: int) -> dict:
    """Roll out one episode under `controller`, return its outcome summary."""
    obs, info = env.reset(seed=seed)
    controller.reset()
    while True:
        action = controller.act(obs, info)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            return dict(reward=float(reward), final_margin=info["margin"],
                        runaway=bool(info["runaway"]), dumped=(action == 1))


def evaluate(env, controller: BangBangController, n_episodes: int, seed0: int) -> dict:
    """Run `controller` over n_episodes fresh episodes, aggregate outcome stats."""
    rewards, final_margins = [], []
    hits = misses = false_alarms = 0
    for k in range(n_episodes):
        out = run_episode(env, controller, seed0 + k)
        rewards.append(out["reward"])
        final_margins.append(out["final_margin"])
        if out["dumped"] and 0.0 < out["final_margin"] < env.dump_threshold:
            hits += 1
        elif out["runaway"] and not out["dumped"]:
            misses += 1
        elif out["dumped"]:
            false_alarms += 1
    return dict(
        controller=controller.name,
        mean_reward=float(np.mean(rewards)),
        hit_rate=hits / n_episodes,
        miss_rate=misses / n_episodes,
        false_alarm_rate=false_alarms / n_episodes,
        mean_final_margin=float(np.mean(final_margins)),
    )


# ── Continuous current control ────────────────────────────────────────────────

class CurrentControlEnv(gymnasium.Wrapper):
    """Wraps QuenchDetectorEnv to accept continuous current fraction as action.

    Action: Box(1) — desired I/I* fraction in [0, I_max_frac].
    Observation: Box(7) — original [T0, I, V, T_surface] + [margin_pred, I_star_pred, breach_prob].

    The wrapper overrides the env's internal I_frac before each step so the
    physics engine advances at the controller's chosen current, not the
    default linear ramp.

    Primary metric: **integrated current before degradation** ∫I·dt [A·s],
    the total useful transport current delivered before thermal runaway.
    This is what HTS magnet designers optimize — maximizing current
    utilization while staying below the runaway boundary.

    Reward = I_frac per safe step (∝ ∫I·dt), large penalty on runaway.
    """

    def __init__(self, env, surrogate_result=None, max_steps: int = 200,
                 dt_step: float | None = None, runaway_penalty: float = -50.0):
        super().__init__(env)
        self.max_steps = max_steps
        self.runaway_penalty = runaway_penalty
        self._step_count = 0
        self._integrated_current = 0.0

        # Physical time per env step [s].
        # FD1d: n_settle_steps * dt.  Analytic: quasi-static, use 1.0 (A·steps).
        if dt_step is not None:
            self.dt_step = dt_step
        elif hasattr(env.unwrapped, "n_settle_steps") and hasattr(env.unwrapped, "dt"):
            self.dt_step = env.unwrapped.n_settle_steps * env.unwrapped.dt
        else:
            self.dt_step = 1.0

        self.action_space = gymnasium.spaces.Box(
            low=np.float32(0.0), high=np.float32(1.0), shape=(1,), dtype=np.float32)
        obs_lo = np.zeros(7, dtype=np.float32)
        obs_hi = np.array([200.0, 2000.0, 10.0, 300.0, 2.0, 2000.0, 1.0], dtype=np.float32)
        self.observation_space = gymnasium.spaces.Box(obs_lo, obs_hi, dtype=np.float32)

        if surrogate_result is not None:
            from hts_tape_env.seq_surrogates import SequenceSurrogateAdapter
            env.set_surrogate(SequenceSurrogateAdapter(surrogate_result))

    def reset(self, **kwargs):
        self._step_count = 0
        self._integrated_current = 0.0
        obs, info = self.env.reset(**kwargs)
        info["integrated_current_As"] = 0.0
        return self._augment_obs(obs, info), info

    def step(self, action):
        a = np.asarray(action, dtype=np.float32).ravel()
        desired_frac = float(np.clip(a[0], 0.0, self.env.unwrapped.I_max_frac))
        ramp = self.env.unwrapped.ramp_rate
        self.env.unwrapped._I_frac = desired_frac - ramp

        obs, _, terminated, truncated, info = self.env.step(0)
        self._step_count += 1

        margin = info["margin"]
        I_A = info["I_true_A"]
        self._integrated_current += I_A * self.dt_step
        info["integrated_current_As"] = self._integrated_current

        if margin <= 0:
            reward = self.runaway_penalty
        else:
            # Reward ∝ current delivered this step — directly incentivises ∫I·dt
            reward = float(info["I_frac"])

        if not terminated and self._step_count >= self.max_steps:
            truncated = True

        return self._augment_obs(obs, info), reward, terminated, truncated, info

    def _augment_obs(self, obs: np.ndarray, info: dict) -> np.ndarray:
        return np.concatenate([obs, [
            info.get("margin_pred", 0.0),
            info.get("I_star_pred", 0.0),
            info.get("breach_prob_pred", 0.0),
        ]], dtype=np.float32)


class SupervisedCurrentController(nn.Module):
    """MLP that maps augmented observation → desired I/I* fraction in [0, 1]."""

    def __init__(self, obs_dim: int = 7, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )
        self.name = "supervised_current"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def reset(self) -> None:
        pass

    @torch.no_grad()
    def act(self, obs: np.ndarray, info: dict) -> np.ndarray:
        x = torch.from_numpy(np.asarray(obs, dtype=np.float32)).unsqueeze(0)
        return self.forward(x).squeeze().cpu().numpy()


def train_supervised_controller(
    df,
    surrogate_result,
    *,
    safety_buffer: float = 0.05,
    safety_penalty: float = 10.0,
    epochs: int = 100,
    batch_size: int = 256,
    lr: float = 1e-3,
    hidden: int = 64,
    seed: int = 42,
    verbose: bool = True,
) -> SupervisedCurrentController:
    """Train a supervised current controller from rollout data.

    Expert target: the ramp's I_frac capped at (1 - safety_buffer), so the
    controller learns the ramp but never suggests current above I*.
    """
    from hts_tape_env.seq_data import TRUE_FEATURES, Normalizer
    from hts_tape_env.seq_surrogates import SequenceSurrogateAdapter, predict_episode

    torch.manual_seed(seed)

    norm = Normalizer.fit(df, columns=TRUE_FEATURES)
    obs_rows, target_rows = [], []

    for _, g in df.sort_values("step").groupby("episode_id", sort=False):
        istar = float(g["Istar_A"].iloc[0])
        for _, row in g.iterrows():
            obs_4 = np.array([row["T0_K"], row["I_true_A"],
                              row["voltage_true_V"], row["T_surface_true_K"]], dtype=np.float32)
            margin_gt = row["margin_norm"]
            i_frac = 1.0 - margin_gt
            target_frac = min(i_frac, 1.0 - safety_buffer)
            obs_7 = np.concatenate([obs_4, [margin_gt, istar, 0.0]], dtype=np.float32)
            obs_rows.append(obs_7)
            target_rows.append(target_frac)

    X = torch.from_numpy(np.array(obs_rows, dtype=np.float32))
    y = torch.from_numpy(np.array(target_rows, dtype=np.float32)).unsqueeze(1)

    controller = SupervisedCurrentController(obs_dim=7, hidden=hidden)
    opt = torch.optim.Adam(controller.parameters(), lr=lr)

    n = len(X)
    for epoch in range(epochs):
        perm = torch.randperm(n)
        total_loss = 0.0
        n_batches = 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            Xb, yb = X[idx], y[idx]
            pred = controller(Xb)
            mse = nn.functional.mse_loss(pred, yb)
            overshoot = torch.clamp(pred - (1.0 - safety_buffer), min=0.0)
            penalty = safety_penalty * (overshoot ** 2).mean()
            loss = mse + penalty
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_batches += 1
        if verbose and (epoch % 20 == 0 or epoch == epochs - 1):
            print(f"  epoch {epoch:3d}  loss={total_loss / n_batches:.4f}")

    controller.eval()
    return controller


def run_current_episode(env: CurrentControlEnv, controller, seed: int) -> dict:
    """Roll out one episode under a current controller."""
    obs, info = env.reset(seed=seed)
    controller.reset()
    utilizations, rewards_list = [], []
    steps = 0
    while True:
        action = controller.act(obs, info)
        if not isinstance(action, np.ndarray):
            action = np.array([action], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        utilizations.append(info["I_frac"])
        rewards_list.append(reward)
        steps += 1
        if terminated or truncated:
            return dict(
                reward=float(np.sum(rewards_list)),
                mean_utilization=float(np.mean(utilizations)),
                integrated_current_As=info["integrated_current_As"],
                steps=steps,
                runaway=bool(info["runaway"]),
                final_margin=info["margin"],
                I_star=info["I_star"],
            )


def evaluate_current_controller(env: CurrentControlEnv, controller,
                                 n_episodes: int, seed0: int) -> dict:
    """Evaluate a current controller over multiple episodes.

    Primary metric: mean integrated current before degradation [A·s] — the
    total useful transport current delivered before thermal runaway.
    """
    rewards, utilizations, step_counts = [], [], []
    integrated_currents, integrated_normed = [], []
    runaways = 0
    for k in range(n_episodes):
        out = run_current_episode(env, controller, seed0 + k)
        rewards.append(out["reward"])
        utilizations.append(out["mean_utilization"])
        step_counts.append(out["steps"])
        integrated_currents.append(out["integrated_current_As"])
        # Normalise by I* for cross-operating-point comparison
        if out["I_star"] > 0:
            integrated_normed.append(out["integrated_current_As"] / out["I_star"])
        runaways += int(out["runaway"])
    return dict(
        controller=getattr(controller, "name", "unknown"),
        mean_reward=float(np.mean(rewards)),
        mean_utilization=float(np.mean(utilizations)),
        mean_integrated_current_As=float(np.mean(integrated_currents)),
        mean_integrated_current_normed=float(np.mean(integrated_normed)) if integrated_normed else 0.0,
        safety_rate=1.0 - runaways / n_episodes,
        mean_steps=float(np.mean(step_counts)),
        n_episodes=n_episodes,
    )
