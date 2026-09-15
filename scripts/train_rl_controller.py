#!/usr/bin/env python
"""
train_rl_controller.py

Train an RL (PPO) current controller for quench protection.  The policy
outputs continuous I/I* fractions; the GRU surrogate validates safety.

NOTE: the surrogate was trained on fixed-ramp trajectories.  An RL policy
choosing non-monotonic currents is out-of-distribution.  This is
experimental — results should be interpreted carefully.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
from pathlib import Path

import torch

from hts_tape_env.seq_surrogates import load_seq_surrogate
from hts_tape_env.controllers import CurrentControlEnv, evaluate_current_controller
from hts_tape_env.current_policy import CurrentPolicy, PPOTrainer

CONDUCTOR_ENV_KWARGS = {
    "rebco_nostab": {"env_cls": "QuenchDetectorEnv", "kwargs": {"mode": "no_stab"}},
    "rebco_stab": {"env_cls": "QuenchDetectorEnv", "kwargs": {"mode": "stab"}},
    "bi2223_nostab": {"env_cls": "Bi2223QuenchDetectorEnv", "kwargs": {"mode": "no_stab"}},
    "bi2223_agstab": {"env_cls": "Bi2223QuenchDetectorEnv", "kwargs": {"mode": "ag_stab", "T0_range": (77.16, 79.0)}},
}


def _make_env(conductor: str):
    cfg = CONDUCTOR_ENV_KWARGS[conductor]
    if cfg["env_cls"] == "Bi2223QuenchDetectorEnv":
        from hts_tape_env.env_bi2223 import Bi2223QuenchDetectorEnv
        return Bi2223QuenchDetectorEnv(**cfg["kwargs"])
    else:
        from hts_tape_env.env import QuenchDetectorEnv
        return QuenchDetectorEnv(**cfg["kwargs"])


def main():
    ap = argparse.ArgumentParser(description="Train RL current controller (PPO)")
    ap.add_argument("--surrogate-checkpoint", required=True)
    ap.add_argument("--conductor", default="rebco_nostab",
                    choices=list(CONDUCTOR_ENV_KWARGS))
    ap.add_argument("--total-timesteps", type=int, default=200_000)
    ap.add_argument("--n-steps", type=int, default=2048)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval-episodes", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="models/controllers")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── 1. Setup ──
    print(f"\n=== 1. Loading surrogate from {args.surrogate_checkpoint} ===")
    surrogate = load_seq_surrogate(args.surrogate_checkpoint)

    inner = _make_env(args.conductor)
    env = CurrentControlEnv(inner, surrogate_result=surrogate)

    policy = CurrentPolicy(obs_dim=7, hidden=args.hidden)
    trainer = PPOTrainer(env, policy, lr=args.lr, n_steps=args.n_steps)

    # ── 2. Train ──
    print(f"\n=== 2. PPO training ({args.total_timesteps} timesteps) ===")
    trained_policy = trainer.train(total_timesteps=args.total_timesteps, seed=args.seed)

    policy_path = out / f"{args.conductor}_rl_policy.pt"
    torch.save(trained_policy.state_dict(), policy_path)
    print(f"  Saved policy to {policy_path}")

    # ── 3. Evaluate ──
    print(f"\n=== 3. Evaluating over {args.eval_episodes} episodes ===")
    trained_policy.eval()
    eval_inner = _make_env(args.conductor)
    eval_env = CurrentControlEnv(eval_inner, surrogate_result=surrogate)
    result = evaluate_current_controller(eval_env, trained_policy, args.eval_episodes, seed0=args.seed)
    print(f"  Integrated I·dt:  {result['mean_integrated_current_As']:.1f} A·s")
    print(f"  Integrated I/I*:  {result['mean_integrated_current_normed']:.2f} (normed)")
    print(f"  Mean utilization: {result['mean_utilization']:.2%}")
    print(f"  Safety rate:      {result['safety_rate']:.2%}")
    print(f"  Mean steps:       {result['mean_steps']:.1f}")
    print(f"  Mean reward:      {result['mean_reward']:.2f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
