#!/usr/bin/env python
"""
train_eval_bi2223_nostab.py

End-to-end training and benchmarking for Bi-2223 no-stabilizer surrogates.
Uses the re-digitized n(T), extended T0 range, and 1.5×Ic overcurrent.
Produces models, metrics JSON, and diagnostic/error-surface plots.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from hts_tape_env.datagen_bi2223 import generate_dataset
from hts_tape_env.surrogates import SPECS, train_spec
from hts_tape_env.seq_surrogates import train_seq_surrogate, predict_episode
from hts_tape_env.rollout_bi2223_env import generate_grid_dataset
from hts_tape_env.benchmark import (
    benchmark_point_surrogates,
    benchmark_seq_surrogates,
    make_benchmark_report,
    make_diagnostic_figure,
    plot_error_surfaces,
)
from hts_tape_env.physics_bi2223 import solve_runaway
from hts_tape_env.physics_diagnostics import bi2223_runaway_curvature

BI2223_POINT_SPECS = {
    "A": SPECS["A"],
    "B": SPECS["B"],
    "raw": SPECS["raw"],
}

NONLINEAR_BAND = (79.5, 81.0)
CONDUCTOR = "bi2223_nostab"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-samples", type=int, default=5000)
    ap.add_argument("--n-eval-points", type=int, default=30)
    ap.add_argument("--seq-epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="models")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── 1. Static (input-based) data ────────────────────────────────────
    print("\n=== 1. Generating static dataset ===")
    df = generate_dataset(
        args.n_samples,
        T0_range=(77.16, 81.0),
        I_max_Ic_frac=1.5,
        seed=args.seed,
    )
    print(f"  {len(df)} rows, {df.runaway.mean():.0%} unsafe")

    # ── 2. Train point surrogates ───────────────────────────────────────
    print("\n=== 2. Training point surrogates ===")
    point_models = {}
    for name in BI2223_POINT_SPECS:
        try:
            result = train_spec(name, df, random_state=args.seed)
            point_models[name] = result.model
            print(f"  {name}: MAE={result.metrics['mae']:.4f}, "
                  f"R²={result.metrics['r2']:.6f}, "
                  f"med_rel_err={result.metrics['median_rel_err']:.4%}")
            result.save(out / f"{CONDUCTOR}_surr_{name}.joblib")
        except Exception as e:
            print(f"  {name}: FAILED — {e}")

    # ── 3. Generate trajectory data via Gym env ─────────────────────────
    print("\n=== 3. Generating trajectory dataset ===")
    T0_grid = np.linspace(77.5, 81.0, args.n_eval_points)
    df_traj = generate_grid_dataset(T0_grid, seed=args.seed)

    # ── 4. Train sequence surrogates ────────────────────────────────────
    print("\n=== 4. Training sequence surrogates ===")
    seq_results = {}
    for arch in ["gru", "transformer", "ssm"]:
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

    # 5a. Point surrogate benchmark on held-out static data
    point_metrics = benchmark_point_surrogates(
        df, BI2223_POINT_SPECS, point_models, nonlinear_band=NONLINEAR_BAND,
    )

    # 5b. Analytic I*(T0) on the eval grid
    Istar_analytic = np.array([solve_runaway(T0)[0] for T0 in T0_grid])

    # 5c. Sequence surrogate I* predictions via predict_episode on grid rollouts
    seq_metrics = {}
    Istar_pred_dict = {}
    for arch, sr in seq_results.items():
        Istar_pred_arch = []
        for ep_id in sorted(df_traj["episode_id"].unique()):
            ep = df_traj[df_traj["episode_id"] == ep_id]
            feats = sr.meta.get("features", ["T0_K", "I_true_A", "voltage_true_V", "T_surface_true_K"])
            X = ep[feats].values
            preds = predict_episode(sr, X)
            Istar_pred_arch.append(preds["istar_pred_A"])
        Istar_pred = np.array(Istar_pred_arch)

        # Align: grid may have skipped some T0 points with no runaway
        valid_T0_idx = df_traj.groupby("episode_id")["T0_K"].first().values
        Istar_analytic_aligned = np.array([solve_runaway(T0)[0] for T0 in valid_T0_idx])

        seq_metrics[arch] = benchmark_seq_surrogates(
            sr.metrics, valid_T0_idx, Istar_analytic_aligned, Istar_pred,
            nonlinear_band=NONLINEAR_BAND,
        )
        Istar_pred_dict[f"seq_{arch}"] = (valid_T0_idx, Istar_pred)

    # 5d. Point surrogate A: I* predictions on the eval grid
    if "A" in point_models:
        pred_A = point_models["A"].predict(T0_grid.reshape(-1, 1))
        Istar_pred_dict["point_A"] = (T0_grid, pred_A)

    # ── 6. Diagnostic and error surface plots ───────────────────────────
    print("\n=== 6. Generating plots ===")

    # Build aligned dicts for make_diagnostic_figure (expects T0-aligned arrays)
    diag_pred = {}
    for label, (t0s, preds) in Istar_pred_dict.items():
        if np.array_equal(t0s, T0_grid):
            diag_pred[label] = preds

    if diag_pred:
        make_diagnostic_figure(
            CONDUCTOR, T0_grid, Istar_analytic, diag_pred,
            NONLINEAR_BAND, out_dir=out,
        )

    # Curvature for error surfaces
    curvature_vals = np.array([
        bi2223_runaway_curvature(T0).get("curvature", np.nan) for T0 in T0_grid
    ])

    if "point_A" in Istar_pred_dict:
        _, pred_A = Istar_pred_dict["point_A"]
        plot_error_surfaces(
            CONDUCTOR, T0_grid, Istar_analytic, pred_A,
            NONLINEAR_BAND, curvature_vals=curvature_vals, out_dir=out,
        )

    # ── 7. Report ───────────────────────────────────────────────────────
    report_path = make_benchmark_report(
        CONDUCTOR, point_metrics, seq_metrics, out_dir=out,
    )
    print(f"\nReport: {report_path}")
    print(f"Valid T0 range: 77.16 — 81.0 K")
    print(f"Nonlinear band: {NONLINEAR_BAND}")


if __name__ == "__main__":
    main()
