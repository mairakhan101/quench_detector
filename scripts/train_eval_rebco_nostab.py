#!/usr/bin/env python
"""
train_eval_rebco_nostab.py

End-to-end training and benchmarking for REBCO no-redistribution surrogates.
Uses the existing physics.py with mode='no_stab' (pure power-law E-J,
no Cu/SS stabilizer current sharing).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from hts_tape_env.datagen import generate_dataset
from hts_tape_env.surrogates import SPECS, train_spec
from hts_tape_env.seq_surrogates import train_seq_surrogate, predict_episode
from hts_tape_env.rollout import generate_rollout_dataset
from hts_tape_env.benchmark import (
    benchmark_point_surrogates,
    benchmark_seq_surrogates,
    make_benchmark_report,
    make_diagnostic_figure,
    plot_error_surfaces,
)
from hts_tape_env.physics import solve_runaway
from hts_tape_env.physics_diagnostics import runaway_curvature

REBCO_SPECS = {
    "A": SPECS["A"],
    "two_d": SPECS["two_d"],
    "B": SPECS["B"],
    "raw": SPECS["raw"],
}

NONLINEAR_BAND = (70.0, 80.0)
CONDUCTOR = "rebco_nostab"
N_FIXED = 20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-samples", type=int, default=8000)
    ap.add_argument("--n-episodes", type=int, default=1500)
    ap.add_argument("--seq-epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="models")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── 1. Static data ──────────────────────────────────────────────────
    print("\n=== 1. Generating static dataset ===")
    df = generate_dataset(
        args.n_samples,
        mode="no_stab",
        lam_range=(20, 200),
        seed=args.seed,
    )
    print(f"  {len(df)} rows, {df.runaway.mean():.0%} unsafe")

    # ── 2. Train point surrogates ───────────────────────────────────────
    print("\n=== 2. Training point surrogates ===")
    point_models = {}
    for name in REBCO_SPECS:
        try:
            result = train_spec(name, df, random_state=args.seed)
            point_models[name] = result.model
            print(f"  {name}: MAE={result.metrics['mae']:.4f}, "
                  f"R²={result.metrics['r2']:.6f}, "
                  f"med_rel_err={result.metrics['median_rel_err']:.4%}")
            result.save(out / f"{CONDUCTOR}_surr_{name}.joblib")
        except Exception as e:
            print(f"  {name}: FAILED — {e}")

    # ── 3. Generate trajectory data ─────────────────────────────────────
    print("\n=== 3. Generating trajectory dataset ===")
    df_traj = generate_rollout_dataset(
        args.n_episodes,
        env_kwargs={"mode": "no_stab"},
        seed=args.seed,
    )

    # ── 4. Train sequence surrogates ────────────────────────────────────
    print("\n=== 4. Training sequence surrogates ===")
    seq_results = {}
    for arch in ["gru", "transformer"]:
        print(f"\n  Training {arch}...")
        try:
            result = train_seq_surrogate(
                df_traj, arch=arch, epochs=args.seq_epochs, seed=args.seed,
            )
            seq_results[arch] = result
            result.save(out / f"{CONDUCTOR}_seq_{arch}.pt")
            print(f"  Saved {out / f'{CONDUCTOR}_seq_{arch}.pt'}")
        except Exception as e:
            print(f"  {arch}: FAILED — {e}")

    # ── 5. Evaluation ───────────────────────────────────────────────────
    print("\n=== 5. Benchmarking ===")

    point_metrics = benchmark_point_surrogates(
        df, REBCO_SPECS, point_models, nonlinear_band=NONLINEAR_BAND,
    )

    # Analytic I*(T0) at fixed lam=100 for evaluation
    T0_eval = np.linspace(10, 75, 50)
    lam_eval = 100.0
    Istar_analytic = np.array([
        solve_runaway(T0, N_FIXED, mode='no_stab', lam_override=lam_eval)[0]
        for T0 in T0_eval
    ])

    # Seq surrogate I* from trajectory data
    seq_metrics = {}
    Istar_pred_dict = {}
    for arch, sr in seq_results.items():
        Istar_pred_arch = []
        valid_T0 = []
        for ep_id in sorted(df_traj["episode_id"].unique()):
            ep = df_traj[df_traj["episode_id"] == ep_id]
            feats = sr.meta.get("features",
                                ["T0_K", "I_true_A", "voltage_true_V", "T_surface_true_K"])
            X = ep[feats].values
            preds = predict_episode(sr, X)
            Istar_pred_arch.append(preds["istar_pred_A"])
            valid_T0.append(ep["T0_K"].iloc[0])

        Istar_pred = np.array(Istar_pred_arch)
        valid_T0 = np.array(valid_T0)

        Istar_analytic_aligned = np.array([
            solve_runaway(T0, N_FIXED, mode='no_stab',
                          lam_override=100.0)[0]
            for T0 in valid_T0
        ])

        seq_metrics[arch] = benchmark_seq_surrogates(
            sr.metrics, valid_T0, Istar_analytic_aligned, Istar_pred,
            nonlinear_band=NONLINEAR_BAND,
        )
        Istar_pred_dict[f"seq_{arch}"] = (valid_T0, Istar_pred)

    # Point surrogate two_d: I* on the eval grid
    if "two_d" in point_models:
        X_eval = np.column_stack([T0_eval, np.full_like(T0_eval, lam_eval)])
        pred_2d = point_models["two_d"].predict(X_eval)
        Istar_pred_dict["point_two_d"] = (T0_eval, pred_2d)

    # ── 6. Plots ────────────────────────────────────────────────────────
    print("\n=== 6. Generating plots ===")

    diag_pred = {}
    for label, (t0s, preds) in Istar_pred_dict.items():
        if np.array_equal(t0s, T0_eval):
            diag_pred[label] = preds

    if diag_pred:
        make_diagnostic_figure(
            CONDUCTOR, T0_eval, Istar_analytic, diag_pred,
            NONLINEAR_BAND, out_dir=out,
        )

    curvature_vals = np.array([
        runaway_curvature(T0, lam_eval, N_FIXED).get("curvature", np.nan)
        for T0 in T0_eval
    ])

    if "point_two_d" in Istar_pred_dict:
        _, pred_2d = Istar_pred_dict["point_two_d"]
        plot_error_surfaces(
            CONDUCTOR, T0_eval, Istar_analytic, pred_2d,
            NONLINEAR_BAND, curvature_vals=curvature_vals, out_dir=out,
        )

    # ── 7. Report ───────────────────────────────────────────────────────
    report_path = make_benchmark_report(
        CONDUCTOR, point_metrics, seq_metrics, out_dir=out,
    )
    print(f"\nReport: {report_path}")


if __name__ == "__main__":
    main()
