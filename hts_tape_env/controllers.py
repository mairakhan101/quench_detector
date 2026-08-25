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
