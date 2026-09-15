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
    ap.add_argument("--adaptive", action="store_true",
                    help="Use boundary-aware adaptive sampling")
    ap.add_argument("--blend", type=float, default=0.5,
                    help="Curvature/uniform T0 mix ratio (0=uniform, 1=curvature)")
    ap.add_argument("--archs", nargs="+", default=["gru", "transformer", "ssm"],
                    help="Sequence architectures to train (gru transformer ssm diffusion)")
    ap.add_argument("--noise", nargs="+", default=["noiseless"],
                    help="Noise configs to sweep (noiseless gauss_0.5x gauss_1x gauss_2x gauss_5x realistic)")
    ap.add_argument("--model-kwargs", default=None,
                    help='JSON string of extra model kwargs, e.g. \'{"pos_emb": "sinusoidal"}\'')
    args = ap.parse_args()

    import json as _json
    extra_model_kwargs = _json.loads(args.model_kwargs) if args.model_kwargs else {}

    from hts_tape_env.thermal_bi2223 import set_rho_multiplier
    set_rho_multiplier(args.rho_mult)
    print(f"Using RHO_AG_MULTIPLIER = {args.rho_mult}")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── 1. Static (input-based) data ────────────────────────────────────
    print("\n=== 1. Generating static dataset ===")
    if args.adaptive:
        from hts_tape_env.adaptive_sampling import generate_adaptive_dataset_bi2223
        df = generate_adaptive_dataset_bi2223(
            args.n_samples, mode=MODE, T0_range=(T0_LO, T0_HI),
            blend=args.blend, seed=args.seed,
        )
    else:
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
    if args.adaptive:
        from hts_tape_env.adaptive_sampling import curvature_weighted_T0
        _solve_ag = lambda T0: solve_runaway(T0, mode=MODE)
        rng = np.random.default_rng(args.seed)
        T0_grid = np.sort(curvature_weighted_T0(
            rng, args.n_eval_points, (T0_LO + 0.1, T0_HI),
            _solve_ag, blend=args.blend,
        ))
    else:
        T0_grid = np.linspace(T0_LO + 0.1, T0_HI, args.n_eval_points)
    df_traj = generate_grid_dataset(
        T0_grid, env_kwargs={"mode": MODE}, seed=args.seed,
    )

    # ── 4. Train sequence surrogates ────────────────────────────────────
    print("\n=== 4. Training sequence surrogates ===")
    seq_results = {}

    # Also generate random rollouts for sequence training
    if args.adaptive:
        from hts_tape_env.adaptive_sampling import (
            generate_adaptive_rollout_dataset, curvature_weighted_T0,
        )
        from hts_tape_env.env_bi2223 import Bi2223QuenchDetectorEnv
        from functools import partial
        _solve_ag_fn = lambda T0: solve_runaway(T0, mode=MODE)
        _T0_sampler = partial(curvature_weighted_T0,
                              T0_range=(T0_LO, T0_HI - 0.2),
                              solve_fn=_solve_ag_fn, blend=args.blend)
        _env = Bi2223QuenchDetectorEnv(mode=MODE, T0_range=(T0_LO, T0_HI - 0.2))
        df_rollout = generate_adaptive_rollout_dataset(
            args.seq_episodes, env=_env, T0_sampler=_T0_sampler,
            seed=args.seed + 1000,
        )
    else:
        from hts_tape_env.rollout_bi2223_env import generate_rollout_dataset
        df_rollout = generate_rollout_dataset(
            args.seq_episodes,
            env_kwargs={"mode": MODE, "T0_range": (T0_LO, T0_HI - 0.2)},
            seed=args.seed + 1000,
        )

    for noise_name in args.noise:
        if noise_name == "noiseless":
            df_train = df_rollout
            feat_cols = TRUE_FEATURES
        else:
            print(f"\n  Applying noise: {noise_name}")
            df_train = apply_noise_to_rollout(df_rollout, NOISE_CONFIGS[noise_name], seed=args.seed)
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
                        model_kwargs=extra_model_kwargs if extra_model_kwargs else None,
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
        df, BI2223_POINT_SPECS, point_models, nonlinear_band=NONLINEAR_BAND,
    )

    Istar_analytic = np.array([solve_runaway(T0, mode=MODE)[0] for T0 in T0_grid])

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
        Istar_analytic_aligned = np.array([solve_runaway(T0, mode=MODE)[0] for T0 in valid_T0_idx])

        seq_metrics[key] = benchmark_seq_surrogates(
            sr.metrics, valid_T0_idx, Istar_analytic_aligned, Istar_pred,
            nonlinear_band=NONLINEAR_BAND,
        )
        Istar_pred_dict[f"seq_{key}"] = (valid_T0_idx, Istar_pred)

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
