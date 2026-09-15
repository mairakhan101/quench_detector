#!/usr/bin/env python
"""
plot_3d_surfaces.py

Generate Genesis-style 3D I*(T₀, λ) error surfaces for a trained sequence
surrogate checkpoint. For 2-parameter REBCO conductors, produces analytic vs
surrogate surfaces plus training density overlay. Falls back to 1D plots for
Bi-2223 conductors.

Usage:
    python scripts/plot_3d_surfaces.py \
        --checkpoint models/benchmark_prod_uniform/rebco_nostab/seq_gru.pt \
        --conductor rebco_nostab \
        --out-dir models/figures/

Optional:
    --rollout-csv data/rebco_nostab_rollouts.csv   # for training density overlay
    --T0-lo 10 --T0-hi 80 --n-T0 30
    --lam-lo 20 --lam-hi 400 --n-lam 20
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Conductor configs (T₀ range, λ range, nonlinear band)
CONDUCTOR_CONFIGS = {
    "rebco_nostab": {
        "T0_range": (10.0, 80.0),
        "lam_range": (20.0, 400.0),
        "nonlinear_band": (70.0, 80.0),
        "env_kwargs": {"mode": "no_stab"},
    },
    "rebco_stab": {
        "T0_range": (10.0, 80.0),
        "lam_range": (20.0, 400.0),
        "nonlinear_band": (70.0, 80.0),
        "env_kwargs": {"mode": "stab"},
    },
    "rebco_fd1d": {
        "T0_range": (10.0, 75.0),
        "lam_range": None,  # FD1d is single-λ
        "nonlinear_band": (70.0, 80.0),
        "env_kwargs": {},
    },
    "bi2223_nostab": {
        "T0_range": (77.5, 81.0),
        "lam_range": None,
        "nonlinear_band": (79.5, 81.0),
        "env_kwargs": {"mode": "no_stab"},
    },
    "bi2223_agstab": {
        "T0_range": (77.26, 79.0),
        "lam_range": None,
        "nonlinear_band": (78.0, 79.0),
        "env_kwargs": {"mode": "ag_stab"},
    },
}


def main():
    parser = argparse.ArgumentParser(description="Generate 3D error surface plots")
    parser.add_argument("--checkpoint", required=True, help="Path to seq surrogate .pt")
    parser.add_argument("--conductor", required=True, choices=list(CONDUCTOR_CONFIGS.keys()))
    parser.add_argument("--rollout-csv", default=None, help="Rollout CSV for density overlay")
    parser.add_argument("--out-dir", default="models/figures")
    parser.add_argument("--T0-lo", type=float, default=None)
    parser.add_argument("--T0-hi", type=float, default=None)
    parser.add_argument("--n-T0", type=int, default=25)
    parser.add_argument("--lam-lo", type=float, default=None)
    parser.add_argument("--lam-hi", type=float, default=None)
    parser.add_argument("--n-lam", type=int, default=20)
    args = parser.parse_args()

    from hts_tape_env.seq_surrogates import load_seq_surrogate, predict_runaway_surface
    from hts_tape_env.benchmark import plot_error_surfaces_3d, plot_error_surfaces

    cfg = CONDUCTOR_CONFIGS[args.conductor]
    T0_lo = args.T0_lo or cfg["T0_range"][0]
    T0_hi = args.T0_hi or cfg["T0_range"][1]
    T0_grid = np.linspace(T0_lo, T0_hi, args.n_T0)

    result = load_seq_surrogate(args.checkpoint)
    print(f"Loaded {result.arch} checkpoint from {args.checkpoint}")
    print(f"  Metrics: {result.metrics}")

    # Extract training density if rollout CSV provided
    train_T0, train_lam = None, None
    if args.rollout_csv:
        df = pd.read_csv(args.rollout_csv)
        ep_df = df.groupby("episode_id").first()
        train_T0 = ep_df["T0_K"].values
        if "lam" in ep_df.columns:
            train_lam = ep_df["lam"].values
        print(f"  Loaded {len(ep_df)} training episodes for density overlay")

    is_2param = cfg["lam_range"] is not None

    if is_2param:
        lam_lo = args.lam_lo or cfg["lam_range"][0]
        lam_hi = args.lam_hi or cfg["lam_range"][1]
        lam_grid = np.linspace(lam_lo, lam_hi, args.n_lam)

        print(f"Computing surfaces: {args.n_T0} × {args.n_lam} = {args.n_T0 * args.n_lam} grid points...")
        surfaces = predict_runaway_surface(
            result, T0_grid, lam_grid,
            env_kwargs=cfg["env_kwargs"],
        )

        path = plot_error_surfaces_3d(
            f"{args.conductor} ({result.arch})",
            T0_grid, lam_grid,
            surfaces["Istar_analytic"], surfaces["Istar_pred"],
            train_T0=train_T0, train_lam=train_lam,
            nonlinear_band=cfg["nonlinear_band"],
            time_margin_analytic=surfaces.get("time_margin_analytic_ms"),
            time_margin_pred=surfaces.get("time_margin_pred_ms"),
            out_dir=args.out_dir,
            title_suffix=f" ({result.arch})",
        )
    else:
        # 1-parameter conductor — use existing 1D plot with single lam
        lam_grid = np.array([100.0])  # dummy single λ
        print(f"Computing 1D surface: {args.n_T0} points...")
        surfaces = predict_runaway_surface(
            result, T0_grid, lam_grid,
            env_kwargs=cfg["env_kwargs"],
        )

        path = plot_error_surfaces(
            f"{args.conductor} ({result.arch})",
            T0_grid,
            surfaces["Istar_analytic"][0],  # squeeze lam dimension
            surfaces["Istar_pred"][0],
            cfg["nonlinear_band"],
            time_margin_analytic=surfaces.get("time_margin_analytic_ms", [None])[0],
            time_margin_pred=surfaces.get("time_margin_pred_ms", [None])[0],
            out_dir=args.out_dir,
        )

    print(f"Saved to {path}")


if __name__ == "__main__":
    main()
