#!/usr/bin/env python
"""
train_supervised_controller.py

Train a supervised current controller from rollout data.  The controller
outputs continuous I/I* suggestions; the GRU surrogate validates safety.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
from pathlib import Path

import torch

from hts_tape_env.env import QuenchDetectorEnv
from hts_tape_env.rollout import generate_rollout_dataset
from hts_tape_env.seq_surrogates import load_seq_surrogate
from hts_tape_env.controllers import (
    CurrentControlEnv,
    train_supervised_controller,
    evaluate_current_controller,
)

CONDUCTOR_ENV_KWARGS = {
    "rebco_nostab": {"mode": "no_stab"},
    "rebco_stab": {"mode": "stab"},
}


def main():
    ap = argparse.ArgumentParser(description="Train supervised current controller")
    ap.add_argument("--surrogate-checkpoint", required=True,
                    help="Path to trained GRU surrogate .pt checkpoint")
    ap.add_argument("--conductor", default="rebco_nostab", choices=list(CONDUCTOR_ENV_KWARGS))
    ap.add_argument("--n-episodes", type=int, default=2000,
                    help="Rollout episodes for training data")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--safety-buffer", type=float, default=0.05)
    ap.add_argument("--eval-episodes", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="models/controllers")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    env_kwargs = CONDUCTOR_ENV_KWARGS[args.conductor]

    # ── 1. Load surrogate ──
    print(f"\n=== 1. Loading surrogate from {args.surrogate_checkpoint} ===")
    surrogate = load_seq_surrogate(args.surrogate_checkpoint)
    print(f"  arch={surrogate.arch}, I* MAE={surrogate.metrics.get('istar_mae_A', '?')}")

    # ── 2. Generate rollout data ──
    print(f"\n=== 2. Generating {args.n_episodes} rollout episodes ===")
    df = generate_rollout_dataset(args.n_episodes, env_kwargs=env_kwargs, seed=args.seed)
    print(f"  {len(df)} steps, {df.groupby('episode_id').ngroups} episodes")

    # ── 3. Train supervised controller ──
    print(f"\n=== 3. Training supervised controller (epochs={args.epochs}) ===")
    controller = train_supervised_controller(
        df, surrogate,
        safety_buffer=args.safety_buffer,
        epochs=args.epochs,
        hidden=args.hidden,
        seed=args.seed,
    )

    ctrl_path = out / f"{args.conductor}_supervised_controller.pt"
    torch.save(controller.state_dict(), ctrl_path)
    print(f"  Saved controller to {ctrl_path}")

    # ── 4. Evaluate ──
    print(f"\n=== 4. Evaluating over {args.eval_episodes} episodes ===")
    inner = QuenchDetectorEnv(**env_kwargs)
    env = CurrentControlEnv(inner, surrogate_result=surrogate)
    result = evaluate_current_controller(env, controller, args.eval_episodes, seed0=args.seed)
    print(f"  Integrated I·dt:  {result['mean_integrated_current_As']:.1f} A·s")
    print(f"  Integrated I/I*:  {result['mean_integrated_current_normed']:.2f} (normed)")
    print(f"  Mean utilization: {result['mean_utilization']:.2%}")
    print(f"  Safety rate:      {result['safety_rate']:.2%}")
    print(f"  Mean steps:       {result['mean_steps']:.1f}")
    print(f"  Mean reward:      {result['mean_reward']:.2f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
