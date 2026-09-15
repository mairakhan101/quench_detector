#!/usr/bin/env python3
"""Generate publication-quality analysis plots for the talk."""

import json
import pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

REPO = pathlib.Path(__file__).resolve().parent.parent
PROD_DIR = REPO / "models" / "benchmark_prod_uniform"
NOISE_DIR = REPO / "models" / "noise_bench_uniform"
OUT_DIR = REPO / "models"

CONDUCTORS = ["bi2223_agstab", "bi2223_nostab", "rebco_nostab", "rebco_stab"]
COND_LABELS = {
    "bi2223_agstab": "Bi-2223\nAg-stab",
    "bi2223_nostab": "Bi-2223\nno-stab",
    "rebco_nostab": "REBCO\nno-stab",
    "rebco_stab": "REBCO\nCu/SS stab",
}
COND_LABELS_INLINE = {
    "bi2223_agstab": "Bi-2223 Ag-stab",
    "bi2223_nostab": "Bi-2223 no-stab",
    "rebco_nostab": "REBCO no-stab",
    "rebco_stab": "REBCO Cu/SS stab",
}

ARCHS = ["gru", "transformer", "ssm"]
ARCH_COLORS = {"gru": "#2563eb", "transformer": "#c4873b", "ssm": "#16a34a", "diffusion": "#9ca3af"}
ARCH_LABELS = {"gru": "GRU", "transformer": "Transformer", "ssm": "SSM"}

try:
    from matplotlib import font_manager
    font_manager.fontManager.addfont("/usr/share/fonts/ibm-plex-sans/IBMPlexSans-Regular.ttf")
    plt.rcParams["font.family"] = "IBM Plex Sans"
except Exception:
    plt.rcParams["font.family"] = "sans-serif"

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.facecolor": "white",
    "figure.facecolor": "white",
})


def load_prod(cond):
    with open(PROD_DIR / f"{cond}_eval_metrics.json") as f:
        return json.load(f)


def load_noise(cond):
    with open(NOISE_DIR / f"{cond}_eval_metrics.json") as f:
        return json.load(f)


def best_point_err(data):
    pts = data.get("point_surrogates", {})
    best = 1.0
    for spec, m in pts.items():
        if spec == "B":
            continue
        err = m.get("median_rel_err", 1.0)
        if err < best:
            best = err
    return best * 100


def plot_cross_conductor_bar():
    fig, ax = plt.subplots(figsize=(11, 6))

    n_cond = len(CONDUCTORS)
    n_arch = len(ARCHS)
    bar_w = 0.22
    x_base = np.arange(n_cond)

    for i, arch in enumerate(ARCHS):
        vals = []
        for cond in CONDUCTORS:
            d = load_prod(cond)
            seqs = d["seq_surrogates"]
            vals.append(seqs[arch]["istar_median_rel_err"] * 100)
        offset = (i - (n_arch - 1) / 2) * bar_w
        bars = ax.bar(x_base + offset, vals, bar_w, label=ARCH_LABELS[arch],
                      color=ARCH_COLORS[arch], edgecolor="white", linewidth=0.5, zorder=3)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v * 1.08,
                    f"{v:.1f}%", ha="center", va="bottom", fontsize=8.5, fontweight="medium")

    for j, cond in enumerate(CONDUCTORS):
        d = load_prod(cond)
        pt_err = best_point_err(d)
        span = bar_w * n_arch / 2 + 0.04
        ax.hlines(pt_err, j - span, j + span, colors="#71717a", linestyles="--",
                  linewidth=1.2, zorder=4)
        ax.text(j + span + 0.02, pt_err, f"point: {pt_err:.2f}%",
                fontsize=8, color="#71717a", va="center")

    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.get_major_formatter().set_scientific(False)
    ax.set_ylabel("I* median relative error (%)")
    ax.set_xticks(x_base)
    ax.set_xticklabels([COND_LABELS[c] for c in CONDUCTORS])
    ax.set_title("Sequence Surrogate I* Error by Conductor (Production Benchmark)")
    ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#e5e5e5")
    ax.set_ylim(bottom=0.05)
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)

    fig.savefig(OUT_DIR / "talk_cross_conductor_bar.png")
    plt.close(fig)
    print(f"  Saved talk_cross_conductor_bar.png")


def plot_architecture_ranking():
    rows = []
    for cond in CONDUCTORS:
        d = load_prod(cond)
        seqs = d["seq_surrogates"]
        for arch in ARCHS:
            err = seqs[arch]["istar_median_rel_err"] * 100
            rows.append((err, cond, arch))
    rows.sort(key=lambda r: r[0])
    top = rows[:10]
    top.reverse()

    fig, ax = plt.subplots(figsize=(10, 5.5))
    y_pos = np.arange(len(top))
    labels = [f"{ARCH_LABELS[arch]}  —  {COND_LABELS_INLINE[cond]}" for _, cond, arch in top]
    vals = [r[0] for r in top]
    colors = [ARCH_COLORS[r[2]] for r in top]

    bars = ax.barh(y_pos, vals, color=colors, edgecolor="white", linewidth=0.5, height=0.65)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("I* median relative error (%)")
    ax.set_title("Top 10 Surrogate Configurations by I* Accuracy")
    ax.invert_yaxis()

    for bar, v in zip(bars, vals):
        ax.text(v + 0.15, bar.get_y() + bar.get_height() / 2,
                f"{v:.2f}%", va="center", fontsize=10, fontweight="medium")

    ax.set_xlim(right=max(vals) * 1.25)
    ax.grid(axis="x", alpha=0.25, linewidth=0.5)

    fig.savefig(OUT_DIR / "talk_architecture_ranking.png")
    plt.close(fig)
    print(f"  Saved talk_architecture_ranking.png")


def plot_point_vs_seq():
    fig, ax = plt.subplots(figsize=(8, 7))

    pts_list = []
    seq_list = []
    labels = []
    for cond in CONDUCTORS:
        d = load_prod(cond)
        pt = best_point_err(d)
        seqs = d["seq_surrogates"]
        best_seq = min(seqs[a]["istar_median_rel_err"] * 100 for a in ARCHS)
        pts_list.append(pt)
        seq_list.append(best_seq)
        labels.append(COND_LABELS_INLINE[cond])

    lo = min(min(pts_list), min(seq_list)) * 0.5
    hi = max(max(pts_list), max(seq_list)) * 2.0
    ax.plot([lo, hi], [lo, hi], "--", color="#a1a1aa", linewidth=1, zorder=1, label="Parity")

    cond_markers = {"bi2223_agstab": "o", "bi2223_nostab": "s", "rebco_nostab": "D", "rebco_stab": "^"}
    cond_colors = {"bi2223_agstab": "#7c3aed", "bi2223_nostab": "#db2777",
                   "rebco_nostab": "#0891b2", "rebco_stab": "#059669"}

    for i, cond in enumerate(CONDUCTORS):
        ax.scatter(pts_list[i], seq_list[i], s=120, zorder=5,
                   marker=cond_markers[cond], color=cond_colors[cond], edgecolors="white", linewidth=1.2)
        dx, dy = 0.15, 0.08
        if cond == "rebco_stab":
            dy = -0.15
        ax.annotate(labels[i], (pts_list[i], seq_list[i]),
                    textcoords="offset points", xytext=(10, 8 if dy >= 0 else -14),
                    fontsize=10, color=cond_colors[cond], fontweight="medium")

    ax.set_xscale("log")
    ax.set_yscale("log")
    for a in (ax.xaxis, ax.yaxis):
        a.set_major_formatter(ScalarFormatter())
        a.get_major_formatter().set_scientific(False)
    ax.set_xlabel("Best point surrogate I* error (%)")
    ax.set_ylabel("Best sequence surrogate I* error (%)")
    ax.set_title("Point vs Sequence Surrogate Accuracy")

    ax.fill_between([lo, hi], [lo, hi], [hi, hi], alpha=0.04, color="#ef4444", zorder=0)
    ax.fill_between([lo, hi], [lo, lo], [lo, hi], alpha=0.04, color="#22c55e", zorder=0)
    ax.text(hi * 0.7, lo * 1.3, "seq wins", fontsize=9, color="#16a34a", alpha=0.6, style="italic")
    ax.text(lo * 1.1, hi * 0.5, "point wins", fontsize=9, color="#dc2626", alpha=0.6, style="italic")

    ax.grid(True, alpha=0.2, linewidth=0.5)
    fig.savefig(OUT_DIR / "talk_point_vs_seq.png")
    plt.close(fig)
    print(f"  Saved talk_point_vs_seq.png")


def plot_noise_robustness():
    combos = []
    for cond in CONDUCTORS:
        prod = load_prod(cond)
        noise = load_noise(cond)
        noise_seqs = noise.get("seq_surrogates", {})
        for arch in ARCHS:
            noiseless_err = prod["seq_surrogates"][arch]["istar_median_rel_err"] * 100
            noisy_key = None
            for k in noise_seqs:
                if k.endswith(f"/{arch}"):
                    noisy_key = k
                    break
            if noisy_key is None:
                continue
            noisy_err = noise_seqs[noisy_key]["istar_median_rel_err"] * 100
            noise_level = noisy_key.split("/")[0]
            combos.append({
                "cond": cond, "arch": arch, "noiseless": noiseless_err,
                "noisy": noisy_err, "noise_level": noise_level,
            })

    combos.sort(key=lambda r: r["noiseless"])

    fig, ax = plt.subplots(figsize=(12, 6))
    n = len(combos)
    x = np.arange(n)
    w = 0.35

    noiseless_vals = [c["noiseless"] for c in combos]
    noisy_vals = [c["noisy"] for c in combos]
    bar_colors = [ARCH_COLORS[c["arch"]] for c in combos]
    noisy_colors = [matplotlib.colors.to_rgba(ARCH_COLORS[c["arch"]], 0.45) for c in combos]

    ax.bar(x - w / 2, noiseless_vals, w, color=bar_colors, edgecolor="white",
           linewidth=0.5, label="Noiseless", zorder=3)
    ax.bar(x + w / 2, noisy_vals, w, color=noisy_colors, edgecolor=[ARCH_COLORS[c["arch"]] for c in combos],
           linewidth=1, linestyle="--", label="Noisy", zorder=3)

    xlabels = []
    for c in combos:
        noise_tag = c["noise_level"]
        xlabels.append(f"{ARCH_LABELS[c['arch']]}\n{COND_LABELS_INLINE[c['cond']]}\n({noise_tag})")
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=8)

    ax.set_ylabel("I* median relative error (%)")
    ax.set_title("Noise Robustness: Noiseless vs Noisy Sequence Surrogates")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.get_major_formatter().set_scientific(False)
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#555555", label="Noiseless"),
        Patch(facecolor="white", edgecolor="#555555", linestyle="--", label="Noisy"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", frameon=True,
              facecolor="white", edgecolor="#e5e5e5")

    fig.savefig(OUT_DIR / "talk_noise_robustness.png")
    plt.close(fig)
    print(f"  Saved talk_noise_robustness.png")


if __name__ == "__main__":
    print("Generating talk analysis plots...")
    plot_cross_conductor_bar()
    plot_architecture_ranking()
    plot_point_vs_seq()
    plot_noise_robustness()
    print("Done.")
