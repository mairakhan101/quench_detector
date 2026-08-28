#!/usr/bin/env python
"""
train_eval_bi2223_agstab.py

End-to-end training and benchmarking for Bi-2223 Ag-matrix stabilizer
surrogates.  Uses realistic AgMg resistivity (RHO_AG_MULTIPLIER=10 in
thermal_bi2223.py) which restores the runaway boundary that was
previously absent with pure-Ag values.

Valid T0 range is narrower than no-stab (~77.16–79.0 K) because the Ag
matrix still partially suppresses runaway at higher T0.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401 — path & thread setup

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

T0_LO = 77.16
T0_HI = 79.0
NONLINEAR_BAND = (78.0, 79.0)
CONDUCTOR = "bi2223_agstab"
MODE = "ag_stab"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-samples", type=int, default=5000)
    ap.add_argument("--n-eval-points", type=int, default=30)
    ap.add_argument("--seq-episodes", type=int, default=500)
    ap.add_argument("--seq-epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="models")
    ap.add_argument("--rho-mult", type=float, default=10.0,
                    help="Ag-matrix resistivity multiplier (× pure Ag NIST)")
    args = ap.parse_args()

    from hts_tape_env.thermal_bi2223 import set_rho_multiplier
    set_rho_multiplier(args.rho_mult)
    print(f"Using RHO_AG_MULTIPLIER = {args.rho_mult}")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── 1. Static (input-based) data ────────────────────────────────────
    print("\n=== 1. Generating static dataset ===")
    df = generate_dataset(
        args.n_samples,
        T0_range=(T0_LO, T0_HI),
        I_max_Ic_frac=1.5,
        mode=MODE,
        seed=args.seed,
    )
    print(f"  {len(df)} rows, {df.runaway.mean():.0%} unsafe")

    if len(df) < 50:
        print(f"\nERROR: Too few valid rows ({len(df)}). I* may not exist in "
              f"the T0 range [{T0_LO}, {T0_HI}] K at rho_mult={args.rho_mult}.")
        return

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
    T0_grid = np.linspace(T0_LO + 0.1, T0_HI, args.n_eval_points)
    df_traj = generate_grid_dataset(
        T0_grid, env_kwargs={"mode": MODE}, seed=args.seed,
    )

    # ── 4. Train sequence surrogates ────────────────────────────────────
    print("\n=== 4. Training sequence surrogates ===")
    seq_results = {}

    # Also generate random rollouts for sequence training
    from hts_tape_env.rollout_bi2223_env import generate_rollout_dataset
    df_rollout = generate_rollout_dataset(
        args.seq_episodes,
        env_kwargs={"mode": MODE, "T0_range": (T0_LO, T0_HI - 0.2)},
        seed=args.seed + 1000,
    )

    for arch in ["gru", "transformer", "ssm"]:
        print(f"\n  Training {arch}...")
        try:
            result = train_seq_surrogate(
                df_rollout, arch=arch, epochs=args.seq_epochs, seed=args.seed,
            )
            seq_results[arch] = result
            result.save(out / f"{CONDUCTOR}_seq_{arch}.pt")
            print(f"  Saved {out / f'{CONDUCTOR}_seq_{arch}.pt'}")
        except Exception as e:
            print(f"  {arch}: FAILED — {e}")

    # ── 5. Evaluation ───────────────────────────────────────────────────
    print("\n=== 5. Benchmarking ===")

    point_metrics = benchmark_point_surrogates(
        df, BI2223_POINT_SPECS, point_models, nonlinear_band=NONLINEAR_BAND,
    )

    Istar_analytic = np.array([solve_runaway(T0, mode=MODE)[0] for T0 in T0_grid])

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

        valid_T0_idx = df_traj.groupby("episode_id")["T0_K"].first().values
        Istar_analytic_aligned = np.array([solve_runaway(T0, mode=MODE)[0] for T0 in valid_T0_idx])

        seq_metrics[arch] = benchmark_seq_surrogates(
            sr.metrics, valid_T0_idx, Istar_analytic_aligned, Istar_pred,
            nonlinear_band=NONLINEAR_BAND,
        )
        Istar_pred_dict[f"seq_{arch}"] = (valid_T0_idx, Istar_pred)

    if "A" in point_models:
        pred_A = point_models["A"].predict(T0_grid.reshape(-1, 1))
        Istar_pred_dict["point_A"] = (T0_grid, pred_A)

    # ── 6. Diagnostic and error surface plots ───────────────────────────
    print("\n=== 6. Generating plots ===")

    diag_pred = {}
    for label, (t0s, preds) in Istar_pred_dict.items():
        if np.array_equal(t0s, T0_grid):
            diag_pred[label] = preds

    if diag_pred:
        make_diagnostic_figure(
            CONDUCTOR, T0_grid, Istar_analytic, diag_pred,
            NONLINEAR_BAND, out_dir=out,
        )

    curvature_vals = np.array([
        bi2223_runaway_curvature(T0, mode='ag_stab').get("curvature", np.nan)
        for T0 in T0_grid
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
    print(f"Valid T0 range: {T0_LO} — {T0_HI} K")
    print(f"Nonlinear band: {NONLINEAR_BAND}")
    print(f"Ag-matrix rho multiplier: {args.rho_mult}×")


if __name__ == "__main__":
    main()
