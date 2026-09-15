#!/usr/bin/env python
"""
robustness_analysis.py

Quantify how sequence surrogates degrade under sensor noise.
Loads benchmark JSONs from the noise benchmark output directory,
extracts metrics by (conductor, noise_level, architecture), and produces:

  1. Degradation curves: metric vs noise level, one line per architecture
  2. Robustness summary table: relative degradation at each noise level
  3. Consolidated CSV for downstream analysis

Usage:
    # From production results
    python scripts/robustness_analysis.py \
        --results-dir /eagle/fanc_ldrd/magnets_fem/noise_bench_uniform

    # Compare uniform vs adaptive
    python scripts/robustness_analysis.py \
        --results-dir /eagle/fanc_ldrd/magnets_fem/noise_bench_uniform \
        --results-dir-2 /eagle/fanc_ldrd/magnets_fem/noise_bench_adaptive

    # From local models dir (debug)
    python scripts/robustness_analysis.py --results-dir models
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CONDUCTORS = [
    "rebco_nostab", "rebco_stab", "rebco_fd1d",
    "bi2223_nostab", "bi2223_agstab",
]

NOISE_ORDER = [
    "noiseless", "gauss_0.5x", "gauss_1x", "gauss_2x", "gauss_5x", "realistic",
]

ARCHS = ["gru", "transformer", "ssm", "diffusion"]

ARCH_COLORS = {
    "gru": "tab:blue",
    "transformer": "tab:orange",
    "ssm": "tab:red",
    "diffusion": "tab:purple",
}

ARCH_MARKERS = {
    "gru": "^",
    "transformer": "v",
    "ssm": "D",
    "diffusion": "o",
}

KEY_METRICS = [
    ("istar_mae_A", "I* MAE [A]", False),
    ("istar_median_rel_err", "I* median rel err", False),
    ("tstar_mae_K", "T* MAE [K]", False),
    ("breach_accuracy", "Breach accuracy", True),
    ("margin_near_boundary_mae", "Margin MAE near boundary", False),
]


def load_benchmark_results(results_dir: Path) -> pd.DataFrame:
    """Load all *_eval_metrics.json files and parse into a flat DataFrame."""
    rows = []

    for conductor in CONDUCTORS:
        json_path = results_dir / f"{conductor}_eval_metrics.json"
        if not json_path.exists():
            continue

        with open(json_path) as f:
            report = json.load(f)

        seq = report.get("seq_surrogates", {})
        for key, metrics in seq.items():
            if "/" in key:
                noise_name, arch = key.split("/", 1)
            else:
                noise_name = "noiseless"
                arch = key

            if not isinstance(metrics, dict):
                continue

            row = {
                "conductor": conductor,
                "noise": noise_name,
                "arch": arch,
            }
            for metric_key, _, _ in KEY_METRICS:
                row[metric_key] = metrics.get(metric_key, np.nan)

            rows.append(row)

    return pd.DataFrame(rows)


def compute_robustness_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Compute relative degradation from noiseless baseline per (conductor, arch)."""
    rows = []
    for (conductor, arch), group in df.groupby(["conductor", "arch"]):
        baseline = group[group["noise"] == "noiseless"]
        if baseline.empty:
            continue
        baseline = baseline.iloc[0]

        for _, row in group.iterrows():
            entry = {
                "conductor": conductor,
                "arch": arch,
                "noise": row["noise"],
            }
            for metric_key, label, higher_better in KEY_METRICS:
                base_val = baseline[metric_key]
                noise_val = row[metric_key]
                if np.isnan(base_val) or np.isnan(noise_val) or base_val == 0:
                    entry[f"{metric_key}_degradation"] = np.nan
                elif higher_better:
                    entry[f"{metric_key}_degradation"] = (base_val - noise_val) / base_val
                else:
                    entry[f"{metric_key}_degradation"] = (noise_val - base_val) / max(abs(base_val), 1e-10)
            rows.append(entry)

    return pd.DataFrame(rows)


def plot_degradation_curves(
    df: pd.DataFrame,
    out_dir: Path,
    tag: str = "",
):
    """One figure per conductor: metric vs noise level, one line per arch."""
    noise_x = {n: i for i, n in enumerate(NOISE_ORDER)}

    for conductor in df["conductor"].unique():
        cond_df = df[df["conductor"] == conductor]
        n_metrics = len(KEY_METRICS)
        fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 4.5))
        if n_metrics == 1:
            axes = [axes]

        for ax, (metric_key, label, higher_better) in zip(axes, KEY_METRICS):
            for arch in ARCHS:
                arch_df = cond_df[cond_df["arch"] == arch].copy()
                if arch_df.empty:
                    continue

                arch_df["x"] = arch_df["noise"].map(noise_x)
                arch_df = arch_df.dropna(subset=[metric_key, "x"]).sort_values("x")

                if arch_df.empty:
                    continue

                ax.plot(
                    arch_df["x"], arch_df[metric_key],
                    marker=ARCH_MARKERS.get(arch, "s"),
                    color=ARCH_COLORS.get(arch, "gray"),
                    label=arch, linewidth=1.5, markersize=6,
                )

            ax.set_xticks(range(len(NOISE_ORDER)))
            ax.set_xticklabels(NOISE_ORDER, rotation=45, ha="right", fontsize=8)
            ax.set_ylabel(label, fontsize=9)
            ax.set_title(label, fontsize=10)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

            if higher_better:
                ax.set_ylim(bottom=0)
            else:
                ax.set_ylim(bottom=0)

        fig.suptitle(f"Noise robustness: {conductor}{' (' + tag + ')' if tag else ''}",
                     fontsize=12)
        fig.tight_layout()
        suffix = f"_{tag}" if tag else ""
        path = out_dir / f"robustness_{conductor}{suffix}.png"
        fig.savefig(path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {path}")


def plot_cross_arch_summary(df: pd.DataFrame, out_dir: Path, tag: str = ""):
    """Summary heatmap: mean I* degradation by (arch, noise) across all conductors."""
    scores = compute_robustness_scores(df)
    metric_key = "istar_median_rel_err_degradation"

    if metric_key not in scores.columns:
        print("  Skipping cross-arch summary — no degradation data")
        return

    pivot = scores.pivot_table(
        index="arch", columns="noise", values=metric_key,
        aggfunc="mean",
    )

    noise_cols = [n for n in NOISE_ORDER if n in pivot.columns]
    arch_rows = [a for a in ARCHS if a in pivot.index]
    if not noise_cols or not arch_rows:
        print("  Skipping cross-arch summary — insufficient data")
        return

    pivot = pivot.loc[arch_rows, noise_cols]

    fig, ax = plt.subplots(figsize=(8, 3.5))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn_r")
    ax.set_xticks(range(len(noise_cols)))
    ax.set_xticklabels(noise_cols, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(arch_rows)))
    ax.set_yticklabels(arch_rows, fontsize=10)

    for i in range(len(arch_rows)):
        for j in range(len(noise_cols)):
            val = pivot.values[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:+.0%}", ha="center", va="center",
                        fontsize=8, color="white" if abs(val) > 0.5 else "black")

    fig.colorbar(im, ax=ax, label="I* rel err degradation from noiseless")
    ax.set_title(f"Mean I* degradation by architecture × noise{' (' + tag + ')' if tag else ''}",
                 fontsize=11)
    fig.tight_layout()
    suffix = f"_{tag}" if tag else ""
    path = out_dir / f"robustness_summary{suffix}.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def print_robustness_table(df: pd.DataFrame):
    """Print a compact robustness summary to stdout."""
    print(f"\n{'='*80}")
    print("  Noise Robustness Summary — I* median relative error")
    print(f"{'='*80}")

    metric = "istar_median_rel_err"
    header = f"  {'Conductor':<18} {'Arch':<13}"
    for n in NOISE_ORDER:
        header += f" {n:>10}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for conductor in CONDUCTORS:
        cond_df = df[df["conductor"] == conductor]
        if cond_df.empty:
            continue
        for arch in ARCHS:
            arch_df = cond_df[cond_df["arch"] == arch]
            if arch_df.empty:
                continue
            line = f"  {conductor:<18} {arch:<13}"
            for noise in NOISE_ORDER:
                val = arch_df[arch_df["noise"] == noise][metric]
                if val.empty or np.isnan(val.iloc[0]):
                    line += f" {'—':>10}"
                else:
                    line += f" {val.iloc[0]:>10.2%}"
            print(line)

    print(f"{'='*80}\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", type=Path, required=True,
                    help="Directory with *_eval_metrics.json files")
    ap.add_argument("--results-dir-2", type=Path, default=None,
                    help="Optional second directory (e.g., adaptive) for comparison")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Output directory for plots (defaults to results-dir)")
    args = ap.parse_args()

    out = args.out_dir or args.results_dir
    out.mkdir(parents=True, exist_ok=True)

    print("=== Loading benchmark results ===")
    df = load_benchmark_results(args.results_dir)
    print(f"  Loaded {len(df)} entries from {args.results_dir}")

    if df.empty:
        print("  No results found. Check that *_eval_metrics.json files exist.")
        return

    tag = "uniform" if args.results_dir_2 else ""

    print("\n=== Degradation curves ===")
    plot_degradation_curves(df, out, tag=tag)

    print("\n=== Cross-architecture summary ===")
    plot_cross_arch_summary(df, out, tag=tag)

    print_robustness_table(df)

    csv_path = out / f"robustness_results{'_' + tag if tag else ''}.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Saved {csv_path}")

    scores = compute_robustness_scores(df)
    if not scores.empty:
        scores_path = out / f"robustness_degradation{'_' + tag if tag else ''}.csv"
        scores.to_csv(scores_path, index=False)
        print(f"  Saved {scores_path}")

    if args.results_dir_2:
        print(f"\n=== Loading comparison results from {args.results_dir_2} ===")
        df2 = load_benchmark_results(args.results_dir_2)
        print(f"  Loaded {len(df2)} entries")

        if not df2.empty:
            plot_degradation_curves(df2, out, tag="adaptive")
            plot_cross_arch_summary(df2, out, tag="adaptive")

            df2.to_csv(out / "robustness_results_adaptive.csv", index=False)
            print_robustness_table(df2)


if __name__ == "__main__":
    main()
