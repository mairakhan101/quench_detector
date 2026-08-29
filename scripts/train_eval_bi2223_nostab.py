#!/usr/bin/env python
"""
train_eval_bi2223_nostab.py

End-to-end training and benchmarking for Bi-2223 no-stabilizer surrogates.
Uses the re-digitized n(T), extended T0 range, and 1.5×Ic overcurrent.
Produces models, metrics JSON, and diagnostic/error-surface plots.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
from pathlib import Path

import numpy as np

from hts_tape_env.datagen_bi2223 import generate_dataset
from hts_tape_env.surrogates import SPECS, train_spec
from hts_tape_env.seq_surrogates import train_seq_surrogate, predict_episode
from hts_tape_env.seq_data import FEATURES, TRUE_FEATURES
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
from hts_tape_env.noise_models import NOISE_CONFIGS, apply_noise_to_rollout

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
    ap.add_argument("--adaptive", action="store_true",
                    help="Use boundary-aware adaptive sampling")
    ap.add_argument("--blend", type=float, default=0.5,
                    help="Curvature/uniform T0 mix ratio (0=uniform, 1=curvature)")
    ap.add_argument("--archs", nargs="+", default=["gru", "transformer", "ssm"],
                    help="Sequence architectures to train (gru transformer ssm diffusion)")
    ap.add_argument("--noise", nargs="+", default=["noiseless"],
                    help="Noise configs to sweep (noiseless gauss_0.5x gauss_1x gauss_2x gauss_5x realistic)")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── 1. Static (input-based) data ────────────────────────────────────
    print("\n=== 1. Generating static dataset ===")
    if args.adaptive:
        from hts_tape_env.adaptive_sampling import generate_adaptive_dataset_bi2223
        df = generate_adaptive_dataset_bi2223(
            args.n_samples, mode="no_stab", blend=args.blend, seed=args.seed,
        )
    else:
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
    if args.adaptive:
        from hts_tape_env.adaptive_sampling import curvature_weighted_T0
        _solve_nostab = lambda T0: solve_runaway(T0, mode="no_stab")
        rng = np.random.default_rng(args.seed)
        T0_grid = np.sort(curvature_weighted_T0(
            rng, args.n_eval_points, (77.5, 81.0),
            _solve_nostab, blend=args.blend,
        ))
    else:
        T0_grid = np.linspace(77.5, 81.0, args.n_eval_points)
    df_traj = generate_grid_dataset(T0_grid, seed=args.seed)

    # ── 4. Train sequence surrogates ────────────────────────────────────
    print("\n=== 4. Training sequence surrogates ===")
    seq_results = {}
    for noise_name in args.noise:
        if noise_name == "noiseless":
            df_train = df_traj
            feat_cols = TRUE_FEATURES
        else:
            print(f"\n  Applying noise: {noise_name}")
            df_train = apply_noise_to_rollout(df_traj, NOISE_CONFIGS[noise_name], seed=args.seed)
            feat_cols = FEATURES

        for arch in args.archs:
            tag = arch if noise_name == "noiseless" else f"{noise_name}/{arch}"
            print(f"\n  Training {tag}...")
            try:
                if arch == "diffusion":
                    from hts_tape_env.diffusion import train_diffusion_surrogate
                    result = train_diffusion_surrogate(
                        df_train, epochs=args.seq_epochs, seed=args.seed,
                        feature_cols=feat_cols,
                    )
                else:
                    result = train_seq_surrogate(
                        df_train, arch=arch, epochs=args.seq_epochs, seed=args.seed,
                        feature_cols=feat_cols,
                    )
                key = tag
                seq_results[key] = result
                fname = f"{CONDUCTOR}_seq_{arch}.pt" if noise_name == "noiseless" \
                    else f"{CONDUCTOR}_{noise_name}_seq_{arch}.pt"
                result.save(out / fname)
                print(f"  Saved {out / fname}")
            except Exception as e:
                print(f"  {tag}: FAILED — {e}")

    # ── 5. Evaluation ───────────────────────────────────────────────────
    print("\n=== 5. Benchmarking ===")

    # 5a. Point surrogate benchmark on held-out static data
    point_metrics = benchmark_point_surrogates(
        df, BI2223_POINT_SPECS, point_models, nonlinear_band=NONLINEAR_BAND,
    )

    # 5b. Analytic I*(T0) on the eval grid
    Istar_analytic = np.array([solve_runaway(T0)[0] for T0 in T0_grid])

    seq_metrics = {}
    Istar_pred_dict = {}
    _eval_noise_cache = {}
    for key, sr in seq_results.items():
        noise_name = key.split("/")[0] if "/" in key else "noiseless"
        if noise_name not in _eval_noise_cache:
            if noise_name == "noiseless":
                _eval_noise_cache[noise_name] = df_traj
            else:
                _eval_noise_cache[noise_name] = apply_noise_to_rollout(
                    df_traj, NOISE_CONFIGS[noise_name], seed=args.seed + 999)
        df_eval = _eval_noise_cache[noise_name]

        Istar_pred_arch = []
        for ep_id in sorted(df_eval["episode_id"].unique()):
            ep = df_eval[df_eval["episode_id"] == ep_id]
            feats = sr.meta.get("features", ["T0_K", "I_true_A", "voltage_true_V", "T_surface_true_K"])
            X = ep[feats].values
            preds = predict_episode(sr, X)
            Istar_pred_arch.append(preds["istar_pred_A"])
        Istar_pred = np.array(Istar_pred_arch)

        valid_T0_idx = df_eval.groupby("episode_id")["T0_K"].first().values
        Istar_analytic_aligned = np.array([solve_runaway(T0)[0] for T0 in valid_T0_idx])

        seq_metrics[key] = benchmark_seq_surrogates(
            sr.metrics, valid_T0_idx, Istar_analytic_aligned, Istar_pred,
            nonlinear_band=NONLINEAR_BAND,
        )
        Istar_pred_dict[f"seq_{key}"] = (valid_T0_idx, Istar_pred)

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
