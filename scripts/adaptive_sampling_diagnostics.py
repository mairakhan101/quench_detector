#!/usr/bin/env python
"""
adaptive_sampling_diagnostics.py

Side-by-side comparison of uniform vs boundary-aware adaptive sampling for
each conductor family.  Produces diagnostic plots showing:

  1. Sample distribution in (T₀, I/I*) space — uniform vs adaptive
  2. Fraction of samples near boundary — bar chart comparison
  3. Curvature weight function overlaid on I*(T₀)
  4. Margin histograms — uniform vs adaptive
"""

import _bootstrap  # noqa: F401

import argparse
from pathlib import Path
from functools import partial

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hts_tape_env.adaptive_sampling import (
    boundary_I_fractions,
    curvature_weighted_T0,
    _build_curvature_cdf,
    generate_adaptive_dataset_rebco,
    generate_adaptive_dataset_bi2223,
)

# ── Conductor configs ────────────────────────────────────────────────────────

def _rebco_solve(mode):
    from hts_tape_env.physics import solve_runaway
    return lambda T0: solve_runaway(T0, 20.0, mode=mode, lam_override=100.0)


def _bi2223_solve(mode):
    from hts_tape_env.physics_bi2223 import solve_runaway
    return lambda T0: solve_runaway(T0, mode=mode)


CONDUCTORS = {
    "rebco_nostab": dict(
        T0_range=(5.0, 75.0), solve_fn=partial(_rebco_solve, "no_stab"),
        gen_uniform="rebco_uniform_nostab",
        gen_adaptive=partial(generate_adaptive_dataset_rebco, mode="no_stab"),
        has_lambda=True,
    ),
    "rebco_stab": dict(
        T0_range=(5.0, 75.0), solve_fn=partial(_rebco_solve, "stab"),
        gen_uniform="rebco_uniform_stab",
        gen_adaptive=partial(generate_adaptive_dataset_rebco, mode="stab"),
        has_lambda=True,
    ),
    "bi2223_nostab": dict(
        T0_range=(77.16, 81.0), solve_fn=partial(_bi2223_solve, "no_stab"),
        gen_uniform="bi2223_uniform_nostab",
        gen_adaptive=partial(generate_adaptive_dataset_bi2223, mode="no_stab"),
        has_lambda=False,
    ),
    "bi2223_agstab": dict(
        T0_range=(77.16, 79.0), solve_fn=partial(_bi2223_solve, "ag_stab"),
        gen_uniform="bi2223_uniform_agstab",
        gen_adaptive=partial(generate_adaptive_dataset_bi2223, mode="ag_stab",
                             T0_range=(77.16, 79.0)),
        has_lambda=False,
    ),
}


def _gen_uniform_rebco(n, mode, seed):
    from hts_tape_env.datagen import generate_dataset
    return generate_dataset(n, mode=mode, lam_range=(20, 200), seed=seed, verbose=False)


def _gen_uniform_bi2223(n, mode, seed, T0_range=(77.16, 81.0)):
    from hts_tape_env.datagen_bi2223 import generate_dataset
    return generate_dataset(n, mode=mode, T0_range=T0_range, seed=seed, verbose=False)


# ── Plot functions ───────────────────────────────────────────────────────────

def plot_T0_I_distribution(name, df_unif, df_adapt, out_dir):
    """T₀ vs I/I* scatter — uniform vs adaptive."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for ax, df, label in [(axes[0], df_unif, "Uniform"),
                           (axes[1], df_adapt, "Adaptive")]:
        i_frac = df["I_A"] / df["Istar_A"]
        colors = np.where(df["runaway"] == 1, "red", "steelblue")
        ax.scatter(df["T0_K"], i_frac, c=colors, s=4, alpha=0.4)
        ax.axhline(1.0, color="black", ls="--", lw=1, alpha=0.5)
        ax.set_xlabel("T₀ (K)")
        ax.set_ylabel("I / I*")
        ax.set_title(f"{label} ({len(df)} pts)")
        ax.grid(True, alpha=0.2)

        near = (df["margin_norm"].abs() < 0.05).mean()
        ax.text(0.02, 0.98, f"{near:.0%} within 5% of boundary",
                transform=ax.transAxes, va="top", fontsize=9,
                bbox=dict(boxstyle="round", fc="white", alpha=0.8))

    fig.suptitle(f"{name}: Sample Distribution (red = runaway)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / f"adaptive_dist_{name}.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {out_dir / f'adaptive_dist_{name}.png'}")


def plot_margin_histograms(name, df_unif, df_adapt, out_dir):
    """Margin histograms — shows boundary concentration."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    bins = np.linspace(-0.3, 0.5, 60)
    axes[0].hist(df_unif["margin_norm"], bins=bins, alpha=0.7, color="steelblue",
                 label="Uniform", density=True)
    axes[0].hist(df_adapt["margin_norm"], bins=bins, alpha=0.7, color="coral",
                 label="Adaptive", density=True)
    axes[0].axvline(0, color="black", ls="--", lw=1)
    axes[0].set_xlabel("Margin (I*−I)/I*")
    axes[0].set_ylabel("Density")
    axes[0].set_title(f"{name}: Margin Distribution")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Zoom near boundary
    bins_zoom = np.linspace(-0.1, 0.1, 40)
    axes[1].hist(df_unif["margin_norm"], bins=bins_zoom, alpha=0.7, color="steelblue",
                 label="Uniform", density=True)
    axes[1].hist(df_adapt["margin_norm"], bins=bins_zoom, alpha=0.7, color="coral",
                 label="Adaptive", density=True)
    axes[1].axvline(0, color="black", ls="--", lw=1)
    axes[1].set_xlabel("Margin (zoomed)")
    axes[1].set_ylabel("Density")
    axes[1].set_title(f"{name}: Near-Boundary Zoom")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / f"adaptive_margin_{name}.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {out_dir / f'adaptive_margin_{name}.png'}")


def plot_curvature_weights(name, T0_range, solve_fn, out_dir):
    """Curvature weight function overlaid on I*(T₀)."""
    T0_grid, cdf = _build_curvature_cdf(T0_range, solve_fn(), n_grid=200)
    weights = np.diff(np.concatenate([[0], cdf]))

    # Also compute I*(T₀) for overlay
    Istar = np.array([solve_fn()(T0)[0] for T0 in T0_grid])

    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    color1 = "steelblue"
    ax1.plot(T0_grid, Istar, "-", color=color1, lw=2, label="I*(T₀)")
    ax1.set_xlabel("T₀ (K)")
    ax1.set_ylabel("I* (A)", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)

    ax2 = ax1.twinx()
    color2 = "coral"
    ax2.fill_between(T0_grid, weights / weights.max(), alpha=0.4, color=color2,
                     label="Curvature weight (normalized)")
    ax2.set_ylabel("Curvature weight", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)

    fig.suptitle(f"{name}: I*(T₀) and Curvature Sampling Weight", fontsize=12)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=9)
    ax1.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_dir / f"adaptive_curvature_{name}.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {out_dir / f'adaptive_curvature_{name}.png'}")


def plot_summary_bars(all_results, out_dir):
    """Cross-conductor boundary concentration comparison."""
    names = list(all_results.keys())
    n = len(names)
    x = np.arange(n)
    w = 0.35

    near_unif = [all_results[k]["near_unif"] for k in names]
    near_adapt = [all_results[k]["near_adapt"] for k in names]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w/2, near_unif, w, label="Uniform", color="steelblue", alpha=0.8)
    ax.bar(x + w/2, near_adapt, w, label="Adaptive", color="coral", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("Fraction within 5% of boundary")
    ax.set_title("Boundary Concentration: Uniform vs Adaptive Sampling")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    for i in range(n):
        improvement = near_adapt[i] / max(near_unif[i], 1e-6)
        ax.text(x[i] + w/2, near_adapt[i] + 0.005,
                f"{improvement:.1f}×", ha="center", fontsize=9, fontweight="bold")

    fig.tight_layout()
    fig.savefig(out_dir / "adaptive_summary_bars.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {out_dir / 'adaptive_summary_bars.png'}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-samples", type=int, default=1000)
    ap.add_argument("--blend", type=float, default=0.5)
    ap.add_argument("--sigma", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="models")
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    conductors = CONDUCTORS
    if args.only:
        conductors = {k: v for k, v in conductors.items() if k in args.only}

    all_results = {}

    for name, cfg in conductors.items():
        print(f"\n{'='*60}")
        print(f"  {name}")
        print(f"{'='*60}")

        # Curvature weight plot (fast — doesn't need full dataset)
        print("  Computing curvature weights...")
        plot_curvature_weights(name, cfg["T0_range"], cfg["solve_fn"], out)

        # Generate uniform dataset
        print(f"  Generating uniform dataset ({args.n_samples} samples)...")
        if cfg["has_lambda"]:
            mode = name.split("_", 1)[1]
            if mode == "nostab":
                mode = "no_stab"
            df_unif = _gen_uniform_rebco(args.n_samples, mode, args.seed)
        else:
            mode = name.split("_", 1)[1]
            if mode == "nostab":
                mode = "no_stab"
            elif mode == "agstab":
                mode = "ag_stab"
                from hts_tape_env.thermal_bi2223 import set_rho_multiplier
                set_rho_multiplier(10.0)
            T0_range = cfg["T0_range"]
            df_unif = _gen_uniform_bi2223(args.n_samples, mode, args.seed,
                                          T0_range=T0_range)

        # Generate adaptive dataset
        print(f"  Generating adaptive dataset ({args.n_samples} samples)...")
        df_adapt = cfg["gen_adaptive"](
            args.n_samples, blend=args.blend, sigma=args.sigma,
            seed=args.seed, verbose=True,
        )

        if len(df_unif) < 50 or len(df_adapt) < 50:
            print(f"  SKIP — too few rows (unif={len(df_unif)}, adapt={len(df_adapt)})")
            continue

        # Stats
        near_unif = (df_unif["margin_norm"].abs() < 0.05).mean()
        near_adapt = (df_adapt["margin_norm"].abs() < 0.05).mean()
        print(f"  Uniform: {len(df_unif)} rows, {near_unif:.0%} near boundary")
        print(f"  Adaptive: {len(df_adapt)} rows, {near_adapt:.0%} near boundary")

        all_results[name] = dict(
            near_unif=near_unif, near_adapt=near_adapt,
            df_unif=df_unif, df_adapt=df_adapt,
        )

        # Per-conductor plots
        print("  Generating plots...")
        plot_T0_I_distribution(name, df_unif, df_adapt, out)
        plot_margin_histograms(name, df_unif, df_adapt, out)

    # Summary
    if len(all_results) > 1:
        plot_summary_bars(all_results, out)

    # Print table
    print("\n" + "="*70)
    print("ADAPTIVE SAMPLING SUMMARY")
    print("="*70)
    print(f"{'Conductor':<20} {'Unif rows':>10} {'Adapt rows':>11} "
          f"{'Unif <5%':>9} {'Adapt <5%':>10} {'Improvement':>12}")
    print("-"*70)
    for name, r in all_results.items():
        ratio = r["near_adapt"] / max(r["near_unif"], 1e-6)
        print(f"{name:<20} {len(r['df_unif']):>10} {len(r['df_adapt']):>11} "
              f"{r['near_unif']:>9.0%} {r['near_adapt']:>10.0%} "
              f"{ratio:>11.1f}×")
    print("="*70)


if __name__ == "__main__":
    main()
