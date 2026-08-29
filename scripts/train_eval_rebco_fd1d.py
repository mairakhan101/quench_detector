#!/usr/bin/env python
"""
train_eval_rebco_fd1d.py

End-to-end training and benchmarking for REBCO using the 1D Du Fort-Frankel
finite-difference solver.  Follows the same pipeline as train_eval_rebco_stab.py
but generates data from the FD env rather than the lumped analytic model.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from hts_tape_env.env_fd1d import FD1DQuenchDetectorEnv
from hts_tape_env.materials import lumped_rebco_tape
from hts_tape_env.physics_fd1d import (
    Tape1D, HeaterPulse, CoolingModel,
    find_Istar_fd, stable_surface_T_fd, tape_voltage_fd,
    _solve_current_sharing_vec,
)
from hts_tape_env.rollout import rollout_episode
from hts_tape_env.surrogates import SPECS, train_spec
from hts_tape_env.seq_surrogates import train_seq_surrogate, predict_episode
from hts_tape_env.seq_data import FEATURES, TRUE_FEATURES
from hts_tape_env.benchmark import (
    benchmark_point_surrogates,
    benchmark_seq_surrogates,
    make_benchmark_report,
    make_diagnostic_figure,
)
from hts_tape_env.noise_models import NOISE_CONFIGS, apply_noise_to_rollout

CONDUCTOR = "rebco_fd1d"
NONLINEAR_BAND = (70.0, 80.0)
REBCO_SPECS = {
    "A": SPECS["A"],
    "two_d": SPECS["two_d"],
    "B": SPECS["B"],
    "raw": SPECS["raw"],
}


def _make_tape(nx: int = 100) -> Tape1D:
    geom = lumped_rebco_tape(L=0.2, w=0.004)
    return Tape1D(geom=geom, nx=nx)


def generate_fd_dataset(
    n_samples: int,
    *,
    tape: Tape1D,
    T0_range: tuple = (10.0, 75.0),
    frac_unsafe: float = 0.30,
    seed: int = 7,
    dt: float = 5e-5,
    t_end: float = 0.10,
) -> pd.DataFrame:
    """Sweep (T0, I) using the FD solver to generate labelled training rows."""
    rng = np.random.default_rng(seed)
    heater = HeaterPulse(power=13.0, duration=0.01)
    cool = CoolingModel(mode="bath_ends")

    rows = []
    T0_vals = rng.uniform(*T0_range, size=n_samples)
    n_done = 0

    for idx, T0 in enumerate(T0_vals):
        if idx and idx % 200 == 0:
            print(f"  {idx}/{n_samples} samples generated")

        cool_i = CoolingModel(mode="bath_ends", T_bath=T0)
        Istar, Tstar = find_Istar_fd(
            tape, T0, heater=heater, cool=cool_i,
            tol=2.0, max_iter=12, t_end=t_end, dt=dt,
        )
        if not np.isfinite(Istar) or Istar <= 0:
            continue

        # Sample I: mix of safe and unsafe
        if rng.random() < frac_unsafe:
            I = rng.uniform(Istar * 0.95, Istar * 1.10)
        else:
            I = rng.uniform(0.1 * Istar, Istar * 0.95)

        # Fast approximation: with bath_ends cooling and I < ~I*, the steady
        # state T field is nearly uniform at T_bath.  Solve current sharing
        # once instead of running two full transients.
        T_uniform = np.full(tape.nx, T0)
        E_arr, _ = _solve_current_sharing_vec(I, T_uniform, tape.geom)
        V = float(np.sum(E_arr * tape.dx))
        T_surf = T0  # negligible self-heating below I*
        margin = (Istar - I) / Istar

        rows.append(dict(
            T0_K=T0,
            I_A=I,
            voltage_V=V,
            T_surface_K=T_surf,
            lam=0.0,
            Istar_A=Istar,
            Tstar_K=Tstar,
            margin_norm=margin,
            runaway=int(I > Istar),
            time_margin_ms=np.nan,
        ))
        n_done += 1

    df = pd.DataFrame(rows)
    print(f"Generated {len(df)} rows, {df.runaway.mean():.0%} unsafe")
    return df


def generate_fd_rollout_dataset(
    n_episodes: int,
    *,
    env: FD1DQuenchDetectorEnv,
    seed: int = 7,
) -> pd.DataFrame:
    """Roll out always-continue episodes in the FD env."""
    rows = []
    n_runaway = 0

    for ep in range(n_episodes):
        if ep and ep % 50 == 0:
            print(f"  rolled out {ep}/{n_episodes} episodes, {len(rows)} steps")

        obs, info = env.reset(seed=seed + ep)
        ep_rows = rollout_episode(env, ep, obs=obs, info=info)
        rows.extend(ep_rows)
        n_runaway += int(ep_rows[-1]["runaway"])

    df = pd.DataFrame(rows)
    print(f"Generated {len(df)} steps over {n_episodes} episodes "
          f"({n_runaway}/{n_episodes} reached runaway)")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-samples", type=int, default=2000,
                    help="Static dataset size (FD is slower than lumped)")
    ap.add_argument("--n-episodes", type=int, default=200,
                    help="Trajectory episodes")
    ap.add_argument("--seq-epochs", type=int, default=60)
    ap.add_argument("--nx", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="models")
    ap.add_argument("--skip-static", action="store_true",
                    help="Skip static dataset and point surrogates")
    ap.add_argument("--skip-seq", action="store_true",
                    help="Skip trajectory generation and sequence surrogates")
    ap.add_argument("--backend", default="numpy", choices=["numpy", "jax"],
                    help="FD solver backend (numpy or jax)")
    ap.add_argument("--adaptive", action="store_true",
                    help="Use boundary-aware adaptive sampling (future — FD datagen not yet wired)")
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

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    tape = _make_tape(nx=args.nx)
    point_models = {}
    point_metrics = {}

    # ── 1. Static data + point surrogates ──────────────────────────────
    if not args.skip_static:
        print("\n=== 1. Generating FD static dataset ===")
        df = generate_fd_dataset(
            args.n_samples, tape=tape, seed=args.seed,
            T0_range=(10.0, 75.0),
        )
        df.to_csv(out / f"{CONDUCTOR}_static.csv", index=False)

        print("\n=== 2. Training point surrogates ===")
        for name in REBCO_SPECS:
            try:
                result = train_spec(name, df, random_state=args.seed)
                point_models[name] = result.model
                print(f"  {name}: MAE={result.metrics['mae']:.4f}, "
                      f"R²={result.metrics['r2']:.6f}")
                result.save(out / f"{CONDUCTOR}_surr_{name}.joblib")
            except Exception as e:
                print(f"  {name}: FAILED — {e}")

        point_metrics = benchmark_point_surrogates(
            df, REBCO_SPECS, point_models, nonlinear_band=NONLINEAR_BAND,
        ) if point_models else {}

    # ── 2. Trajectory data + sequence surrogates ───────────────────────
    seq_metrics = {}
    if not args.skip_seq:
        print("\n=== 3. Generating FD trajectory dataset ===")
        env = FD1DQuenchDetectorEnv(
            geometry="lumped", nx=args.nx,
            T0_range=(40.0, 75.0),
            cool_mode="bath_ends",
            ramp_rate=0.005,
            n_settle_steps=10,
            dt=1e-4,
            istar_tol=3.0,
            istar_max_iter=12,
            istar_t_end=0.10,
            backend=args.backend,
        )
        df_traj = generate_fd_rollout_dataset(
            args.n_episodes, env=env, seed=args.seed,
        )
        df_traj.to_csv(out / f"{CONDUCTOR}_rollout.csv", index=False)

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

        _eval_noise_cache = {}
        for key, sr in seq_results.items():
            noise_name = key.split("/")[0] if "/" in key else "noiseless"
            if noise_name not in _eval_noise_cache:
                if noise_name == "noiseless":
                    _eval_noise_cache[noise_name] = df_traj
                else:
                    _eval_noise_cache[noise_name] = apply_noise_to_rollout(
                        df_traj, NOISE_CONFIGS[noise_name], seed=args.seed + 999)
            df_eval_noise = _eval_noise_cache[noise_name]

            Istar_preds = []
            valid_T0 = []
            for ep_id in sorted(df_eval_noise["episode_id"].unique()):
                ep = df_eval_noise[df_eval_noise["episode_id"] == ep_id]
                feats = sr.meta.get("features",
                                    ["T0_K", "I_true_A", "voltage_true_V", "T_surface_true_K"])
                X = ep[feats].values
                preds = predict_episode(sr, X)
                Istar_preds.append(preds["istar_pred_A"])
                valid_T0.append(ep["T0_K"].iloc[0])

            Istar_pred = np.array(Istar_preds)
            valid_T0_arr = np.array(valid_T0)

            # FD ground truth for these T0 values
            heater = HeaterPulse(power=13.0, duration=0.01)
            Istar_fd = np.array([
                find_Istar_fd(
                    tape, float(T0), heater=heater,
                    cool=CoolingModel(mode="bath_ends", T_bath=float(T0)),
                    tol=3.0, max_iter=10, t_end=0.10, dt=1e-4,
                )[0]
                for T0 in valid_T0_arr[:50]  # cap to keep eval fast
            ])
            valid_T0_eval = valid_T0_arr[:50]
            Istar_pred_eval = Istar_pred[:50]

            seq_metrics[key] = benchmark_seq_surrogates(
                sr.metrics, valid_T0_eval, Istar_fd, Istar_pred_eval,
                nonlinear_band=NONLINEAR_BAND,
            )

    # ── 3. Report ──────────────────────────────────────────────────────
    print("\n=== 5. Report ===")
    report_path = make_benchmark_report(
        CONDUCTOR, point_metrics, seq_metrics, out_dir=out,
    )
    print(f"\nReport: {report_path}")


if __name__ == "__main__":
    main()
