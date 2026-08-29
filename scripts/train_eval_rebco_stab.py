#!/usr/bin/env python
"""
train_eval_rebco_stab.py

End-to-end training and benchmarking for REBCO with Cu/SS redistribution.
Uses physics.py with mode='stab' (default — voltage-equilibrium current
sharing between SC layer and Cu||SS stabilizer).
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
from pathlib import Path

import numpy as np

from hts_tape_env.datagen import generate_dataset
from hts_tape_env.surrogates import SPECS, train_spec
from hts_tape_env.seq_surrogates import train_seq_surrogate, predict_episode
from hts_tape_env.seq_data import FEATURES, TRUE_FEATURES
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
from hts_tape_env.noise_models import NOISE_CONFIGS, apply_noise_to_rollout

REBCO_SPECS = {
    "A": SPECS["A"],
    "two_d": SPECS["two_d"],
    "B": SPECS["B"],
    "raw": SPECS["raw"],
}

NONLINEAR_BAND = (70.0, 80.0)
CONDUCTOR = "rebco_stab"
N_FIXED = 20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-samples", type=int, default=8000)
    ap.add_argument("--n-episodes", type=int, default=1500)
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

    # ── 1. Static data ──────────────────────────────────────────────────
    print("\n=== 1. Generating static dataset ===")
    if args.adaptive:
        from hts_tape_env.adaptive_sampling import generate_adaptive_dataset_rebco
        df = generate_adaptive_dataset_rebco(
            args.n_samples, mode="stab", blend=args.blend, seed=args.seed,
        )
    else:
        df = generate_dataset(
            args.n_samples,
            mode="stab",
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
    if args.adaptive:
        from hts_tape_env.adaptive_sampling import (
            generate_adaptive_rollout_dataset, curvature_weighted_T0,
        )
        from hts_tape_env.physics import solve_runaway as _solve
        from hts_tape_env.env import QuenchDetectorEnv
        from functools import partial
        _solve_stab = lambda T0: _solve(T0, 20.0, mode="stab", lam_override=100.0)
        _T0_sampler = partial(curvature_weighted_T0, T0_range=(10.0, 75.0),
                              solve_fn=_solve_stab, blend=args.blend)
        env = QuenchDetectorEnv(mode="stab")
        df_traj = generate_adaptive_rollout_dataset(
            args.n_episodes, env=env, T0_sampler=_T0_sampler, seed=args.seed,
        )
    else:
        df_traj = generate_rollout_dataset(
            args.n_episodes,
            env_kwargs={"mode": "stab"},
            seed=args.seed,
        )

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

    point_metrics = benchmark_point_surrogates(
        df, REBCO_SPECS, point_models, nonlinear_band=NONLINEAR_BAND,
    )

    T0_eval = np.linspace(10, 75, 50)
    lam_eval = 100.0
    Istar_analytic = np.array([
        solve_runaway(T0, N_FIXED, mode='stab', lam_override=lam_eval)[0]
        for T0 in T0_eval
    ])

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
        valid_T0 = []
        for ep_id in sorted(df_eval["episode_id"].unique()):
            ep = df_eval[df_eval["episode_id"] == ep_id]
            feats = sr.meta.get("features",
                                ["T0_K", "I_true_A", "voltage_true_V", "T_surface_true_K"])
            X = ep[feats].values
            preds = predict_episode(sr, X)
            Istar_pred_arch.append(preds["istar_pred_A"])
            valid_T0.append(ep["T0_K"].iloc[0])

        Istar_pred = np.array(Istar_pred_arch)
        valid_T0 = np.array(valid_T0)

        Istar_analytic_aligned = np.array([
            solve_runaway(T0, N_FIXED, mode='stab', lam_override=100.0)[0]
            for T0 in valid_T0
        ])

        seq_metrics[key] = benchmark_seq_surrogates(
            sr.metrics, valid_T0, Istar_analytic_aligned, Istar_pred,
            nonlinear_band=NONLINEAR_BAND,
        )

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
