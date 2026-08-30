#!/usr/bin/env python
"""
train_distill.py

End-to-end teacher-student distillation for quench surrogates.

The teacher sees 5 features (4 measurable channels + lambda), the student
sees only the 4 measurable channels. The student trains on a weighted
combination of hard physics targets and soft targets from the teacher.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
from pathlib import Path

from hts_tape_env.rollout import generate_rollout_dataset
from hts_tape_env.seq_data import TRUE_FEATURES, TEACHER_FEATURES
from hts_tape_env.seq_surrogates import (
    train_seq_surrogate,
    train_distilled_surrogate,
    load_seq_surrogate,
)

CONDUCTOR_ENV_KWARGS = {
    "rebco_nostab": {"mode": "no_stab"},
    "rebco_stab": {"mode": "stab"},
}


def main():
    ap = argparse.ArgumentParser(description="Teacher-student distillation for quench surrogates")
    ap.add_argument("--conductor", default="rebco_nostab", choices=list(CONDUCTOR_ENV_KWARGS))
    ap.add_argument("--teacher-arch", default="gru")
    ap.add_argument("--student-arch", default="gru")
    ap.add_argument("--teacher-epochs", type=int, default=400)
    ap.add_argument("--student-epochs", type=int, default=400)
    ap.add_argument("--n-episodes", type=int, default=3000)
    ap.add_argument("--alpha", type=float, default=0.5,
                    help="Hard/soft weight: loss = alpha*hard + (1-alpha)*soft")
    ap.add_argument("--temperature", type=float, default=2.0,
                    help="Temperature for breach_logit KL divergence")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="models/distill")
    ap.add_argument("--teacher-checkpoint", default=None,
                    help="Skip teacher training; load from this path instead")
    ap.add_argument("--model-kwargs", default=None,
                    help='JSON model kwargs for both teacher and student')
    args = ap.parse_args()

    extra_kwargs = json.loads(args.model_kwargs) if args.model_kwargs else {}
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    env_kwargs = CONDUCTOR_ENV_KWARGS[args.conductor]

    # ── 1. Generate rollout data ──
    print(f"\n=== 1. Generating {args.n_episodes} episodes ({args.conductor}) ===")
    df = generate_rollout_dataset(args.n_episodes, env_kwargs=env_kwargs, seed=args.seed)
    print(f"  {len(df)} rows, {df.groupby('episode_id').ngroups} episodes")

    # ── 2. Train teacher (5 features: 4 channels + lambda) ──
    teacher_path = out / f"{args.conductor}_teacher_{args.teacher_arch}.pt"
    if args.teacher_checkpoint:
        print(f"\n=== 2. Loading teacher from {args.teacher_checkpoint} ===")
        teacher = load_seq_surrogate(args.teacher_checkpoint)
    else:
        print(f"\n=== 2. Training teacher ({args.teacher_arch}, 5 features) ===")
        teacher_kwargs = {**extra_kwargs, "n_features": 5}
        teacher = train_seq_surrogate(
            df, arch=args.teacher_arch, epochs=args.teacher_epochs,
            feature_cols=TEACHER_FEATURES,
            model_kwargs=teacher_kwargs, seed=args.seed,
        )
        teacher.save(teacher_path)
        print(f"  Saved teacher to {teacher_path}")
        print(f"  Teacher I* MAE: {teacher.metrics['istar_mae_A']:.2f} A, "
              f"rel err: {teacher.metrics['istar_median_rel_err']:.2%}")

    # ── 3. Train distilled student (4 features) ──
    print(f"\n=== 3. Distilling student ({args.student_arch}, 4 features, "
          f"alpha={args.alpha}, tau={args.temperature}) ===")
    student = train_distilled_surrogate(
        df, teacher,
        arch=args.student_arch, epochs=args.student_epochs,
        alpha=args.alpha, temperature=args.temperature,
        student_feature_cols=TRUE_FEATURES,
        model_kwargs=extra_kwargs if extra_kwargs else None,
        teacher_path=str(teacher_path),
        seed=args.seed,
    )
    student_path = out / f"{args.conductor}_student_{args.student_arch}.pt"
    student.save(student_path)
    print(f"  Saved student to {student_path}")

    # ── 4. Baseline: train from scratch (4 features, no distillation) ──
    print(f"\n=== 4. Training baseline ({args.student_arch}, 4 features, no distillation) ===")
    baseline = train_seq_surrogate(
        df, arch=args.student_arch, epochs=args.student_epochs,
        feature_cols=TRUE_FEATURES,
        model_kwargs=extra_kwargs if extra_kwargs else None,
        seed=args.seed,
    )
    baseline_path = out / f"{args.conductor}_baseline_{args.student_arch}.pt"
    baseline.save(baseline_path)
    print(f"  Saved baseline to {baseline_path}")

    # ── 5. Comparison ──
    print("\n=== 5. Comparison ===")
    print(f"  {'Model':<25} {'I* MAE [A]':>12} {'I* rel err':>12} {'T* MAE [K]':>12}")
    print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*12}")
    for name, r in [("Teacher (5-feat)", teacher),
                    ("Student (distilled)", student),
                    ("Baseline (scratch)", baseline)]:
        m = r.metrics
        print(f"  {name:<25} {m['istar_mae_A']:>12.2f} {m['istar_median_rel_err']:>11.2%} "
              f"{m['tstar_mae_K']:>12.2f}")

    print(f"\nDone. Outputs in {out}/")


if __name__ == "__main__":
    main()
