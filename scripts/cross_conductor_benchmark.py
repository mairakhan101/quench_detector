#!/usr/bin/env python
"""
cross_conductor_benchmark.py

Loads benchmark JSONs from all 4 conductor pipelines and produces:
  1. A comparative table (printed + saved as CSV)
  2. A 4-panel summary figure comparing point and seq surrogate accuracy
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


CONDUCTORS = [
    ("bi2223_nostab",  "Bi-2223 no-stab"),
    ("bi2223_agstab",  "Bi-2223 Ag-stab"),
    ("rebco_nostab",   "REBCO no-redist"),
    ("rebco_stab",     "REBCO Cu/SS redist"),
]


def load_reports(models_dir: Path) -> dict:
    reports = {}
    for key, label in CONDUCTORS:
        path = models_dir / f"{key}_eval_metrics.json"
        if path.exists():
            with open(path) as f:
                reports[key] = json.load(f)
        else:
            reports[key] = None
    return reports


def format_pct(val):
    if val is None or not np.isfinite(val):
        return "—"
    return f"{val * 100:.2f}%"


def print_table(reports: dict):
    print("\n" + "=" * 100)
    print("  Cross-Conductor Benchmark Summary")
    print("=" * 100)

    header = (
        f"{'Conductor':<25s}"
        f"{'Best Pt Err':>12s}"
        f"{'Pt Spec':>8s}"
        f"{'Pt Band Err':>12s}"
        f"{'GRU I* Err':>12s}"
        f"{'Xfmr I* Err':>12s}"
        f"{'Xfmr T* MAE':>12s}"
        f"{'Xfmr Band':>12s}"
    )
    print(header)
    print("-" * 100)

    rows = []
    for key, label in CONDUCTORS:
        r = reports.get(key)
        if r is None:
            print(f"  {label:<25s}  (no report)")
            rows.append([label, "—", "—", "—", "—", "—", "—", "—"])
            continue

        pt = r.get("point_surrogates", {})
        best_pt_err = None
        best_pt_name = "—"
        best_pt_band = None
        for name, m in pt.items():
            err = m.get("median_rel_err")
            if err is not None and (best_pt_err is None or err < best_pt_err):
                best_pt_err = err
                best_pt_name = name
                best_pt_band = m.get("band_median_rel_err")

        seq = r.get("seq_surrogates", {})
        gru = seq.get("gru", {})
        xfmr = seq.get("transformer", {})

        gru_istar = gru.get("istar_median_rel_err")
        xfmr_istar = xfmr.get("istar_median_rel_err")
        xfmr_tstar = xfmr.get("tstar_mae_K")
        xfmr_band = xfmr.get("band_median_rel_err")

        tstar_str = f"{xfmr_tstar:.2f} K" if xfmr_tstar is not None else "—"

        print(
            f"  {label:<25s}"
            f"{format_pct(best_pt_err):>12s}"
            f"{best_pt_name:>8s}"
            f"{format_pct(best_pt_band):>12s}"
            f"{format_pct(gru_istar):>12s}"
            f"{format_pct(xfmr_istar):>12s}"
            f"{tstar_str:>12s}"
            f"{format_pct(xfmr_band):>12s}"
        )

        rows.append([
            label, format_pct(best_pt_err), best_pt_name,
            format_pct(best_pt_band), format_pct(gru_istar),
            format_pct(xfmr_istar), tstar_str, format_pct(xfmr_band),
        ])

    print("=" * 100)
    return rows


def save_csv(rows, out_dir: Path):
    header = "Conductor,Best Pt Err,Pt Spec,Pt Band Err,GRU I* Err,Xfmr I* Err,Xfmr T* MAE,Xfmr Band Err"
    lines = [header]
    for row in rows:
        lines.append(",".join(row))
    path = out_dir / "cross_conductor_benchmark.csv"
    path.write_text("\n".join(lines) + "\n")
    print(f"\nSaved {path}")


def make_figure(reports: dict, out_dir: Path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    labels = []
    pt_errs = []
    pt_band_errs = []
    gru_errs = []
    xfmr_errs = []
    xfmr_tstar = []
    xfmr_band = []

    for key, label in CONDUCTORS:
        r = reports.get(key)
        labels.append(label.replace(" ", "\n"))
        if r is None:
            pt_errs.append(np.nan)
            pt_band_errs.append(np.nan)
            gru_errs.append(np.nan)
            xfmr_errs.append(np.nan)
            xfmr_tstar.append(np.nan)
            xfmr_band.append(np.nan)
            continue

        pt = r.get("point_surrogates", {})
        best = min((m["median_rel_err"] for m in pt.values()), default=np.nan)
        best_band = min(
            (m.get("band_median_rel_err", np.nan) for m in pt.values()),
            default=np.nan,
        )
        pt_errs.append(best * 100)
        pt_band_errs.append(best_band * 100)

        seq = r.get("seq_surrogates", {})
        gru_errs.append(seq.get("gru", {}).get("istar_median_rel_err", np.nan) * 100)
        xfmr_errs.append(seq.get("transformer", {}).get("istar_median_rel_err", np.nan) * 100)
        xfmr_tstar.append(seq.get("transformer", {}).get("tstar_mae_K", np.nan))
        xfmr_band.append(seq.get("transformer", {}).get("band_median_rel_err", np.nan) * 100)

    x = np.arange(len(labels))
    w = 0.35

    # Panel 1: Point surrogate median rel err (global vs band)
    ax = axes[0, 0]
    bars1 = ax.bar(x - w / 2, pt_errs, w, label="Global", color="#4C72B0")
    bars2 = ax.bar(x + w / 2, pt_band_errs, w, label="NL Band", color="#DD8452")
    ax.set_ylabel("Median Rel Error (%)")
    ax.set_title("Best Point Surrogate")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.legend(fontsize=8)
    ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.3)

    # Panel 2: Seq I* median rel err (GRU vs Transformer)
    ax = axes[0, 1]
    ax.bar(x - w / 2, gru_errs, w, label="GRU", color="#4C72B0")
    ax.bar(x + w / 2, xfmr_errs, w, label="Transformer", color="#DD8452")
    ax.set_ylabel("I* Median Rel Error (%)")
    ax.set_title("Sequence Surrogates (I*)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # Panel 3: Transformer T* MAE
    ax = axes[1, 0]
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    bars = ax.bar(x, xfmr_tstar, 0.5, color=colors)
    ax.set_ylabel("T* MAE (K)")
    ax.set_title("Transformer T* Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # Panel 4: Transformer NL band err
    ax = axes[1, 1]
    bars = ax.bar(x, xfmr_band, 0.5, color=colors)
    ax.set_ylabel("I* Band Median Rel Err (%)")
    ax.set_title("Transformer — Nonlinear Band")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Cross-Conductor Surrogate Benchmark", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = out_dir / "cross_conductor_benchmark.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def main():
    models_dir = Path("models")
    reports = load_reports(models_dir)

    available = sum(1 for v in reports.values() if v is not None)
    print(f"\nLoaded {available}/{len(CONDUCTORS)} conductor reports.")

    if available == 0:
        print("No reports found. Run the training scripts first.")
        return

    rows = print_table(reports)
    save_csv(rows, models_dir)
    make_figure(reports, models_dir)


if __name__ == "__main__":
    main()
