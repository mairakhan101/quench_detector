"""
plot_fem_runaway.py

Slide figures for the FEA time-to-runaway models (scripts/train_fem_runaway.py),
all on the held-out test runs:

  fem_runaway_architectures.png  headline metrics per architecture; mean, spread
                                 and individual seeds when fem_runaway_{arch}_s*_metrics.json
                                 exist alongside the default (seed 42) run
  fem_runaway_vs_current.png     per-run performance vs transport current I/Ic,
                                 from the default fem_runaway_{arch}.pt checkpoints

Usage
-----
    python scripts/plot_fem_runaway.py --models-dir models --path /path/to/sweep.h5
"""

import _bootstrap  # noqa: F401  (first import: caps BLAS threads, fixes sys.path)

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from hts_tape_env.fem_runaway_data import load_runaway_sweep
from hts_tape_env.runaway_seq import (
    add_model_features,
    load_runaway_model,
    predict_episodes,
    split_by_ratio_blocks,
)

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
# Reference categorical slots 1-3, documented as passing all-pairs CVD checks in light mode.
ARCHS = {"gru": ("GRU", "#2a78d6"), "transformer": ("Transformer", "#eb6834"), "ssm": ("S4D", "#1baf7a")}
ALARM_THRESHOLD = 0.5

ARCH_PANELS = [
    ("false_alarm_rate", 100, "{:.1f}%", "False alarms", "% of test runs that did not run away"),
    ("ttr_mae_s_last_1s", 1000, "{:.0f} ms", "Countdown error, final second", "mean absolute error (ms)"),
    ("ttr_median_rel_err", 100, "{:.1f}%", "Countdown error, whole run", "median relative error (%)"),
]


def _style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 12,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.linewidth": 1.0, "axes.labelcolor": INK2,
        "axes.titlecolor": INK, "axes.titlesize": 13, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.titlepad": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
        "legend.frameon": False, "legend.labelcolor": INK2, "text.color": INK,
    })


def _header(fig, title, subtitle):
    fig.text(0.012, 0.975, title, ha="left", va="top", fontsize=17, fontweight="bold", color=INK)
    fig.text(0.012, 0.905, subtitle, ha="left", va="top", fontsize=12, color=INK2, linespacing=1.4)


def _arch_legend(fig, dots: bool = False):
    style = dict(ls="none", marker="o", ms=10, mec=SURFACE, mew=1.5) if dots else dict(lw=2.5)
    handles = [Line2D([], [], color=c, label=n, **style) for n, c in ARCHS.values()]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.995, 0.985), ncol=3,
               fontsize=12, handlelength=1.8, columnspacing=1.4)


def load_arch_runs(models_dir: Path) -> dict:
    runs = {}
    for arch in ARCHS:
        files = [models_dir / f"fem_runaway_{arch}_metrics.json"]
        files += sorted(models_dir.glob(f"fem_runaway_{arch}_s[0-9]*_metrics.json"))
        loaded = [json.loads(f.read_text()) for f in files if f.exists()]
        if loaded:
            runs[arch] = loaded
    return runs


def false_alarm_ratio_range(models_dir: Path):
    ratios = []
    for arch in ARCHS:
        for f in [models_dir / f"fem_runaway_{arch}_test_episodes.csv",
                  *models_dir.glob(f"fem_runaway_{arch}_s[0-9]*_test_episodes.csv")]:
            if f.exists():
                ep = pd.read_csv(f)
                ratios += ep.loc[ep["false_alarm"], "ratio"].tolist()
    return (min(ratios), max(ratios)) if ratios else None


def fig_architectures(runs: dict, fa_range, out: Path):
    archs = list(runs)
    test = {a: [r["test"] for r in runs[a]] for a in archs}
    horizon = runs[archs[0]][0]["args"]["horizon"]
    first = test[archs[0]][0]
    warn = np.median([m["median_warning_s"] for a in archs for m in test[a]])
    detected = all(m["detection_rate"] == 1.0 for a in archs for m in test[a])

    fig, axes = plt.subplots(1, 3, figsize=(13.33, 4.6), sharey=True)
    fig.subplots_adjust(left=0.105, right=0.985, top=0.70, bottom=0.2, wspace=0.18)
    y = np.arange(len(archs))[::-1]
    for ax, (key, scale, fmt, title, xlabel) in zip(axes, ARCH_PANELS):
        right_edge = 0.0
        for yi, arch in zip(y, archs):
            color = ARCHS[arch][1]
            v = np.array([m[key] for m in test[arch]], dtype=float) * scale
            mean = v.mean()
            if len(v) > 1:
                sd = v.std(ddof=1)
                ax.plot([max(mean - sd, 0), mean + sd], [yi, yi], color=color, lw=2,
                        solid_capstyle="round", zorder=2)
                ax.scatter(v, np.full_like(v, yi), s=26, color=color, alpha=0.35, linewidths=0, zorder=2)
            ax.plot(mean, yi, "o", ms=11, color=color, mec=SURFACE, mew=2, zorder=3)
            end = max(v.max(), mean + (v.std(ddof=1) if len(v) > 1 else 0))
            ax.text(end, yi, "   " + fmt.format(mean), va="center", ha="left", fontsize=12, color=INK)
            right_edge = max(right_edge, end)
        ax.set_xlim(0, right_edge * 1.45 if right_edge > 0 else 1)
        ax.set_ylim(-0.6, len(archs) - 0.4)
        ax.grid(axis="x")
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
        ax.set_title(title)
        ax.set_xlabel(xlabel, fontsize=11)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([f"{ARCHS[a][0]}\n{len(test[a])} seed{'s' if len(test[a]) > 1 else ''}"
                             for a in archs], fontsize=12)
    axes[0].spines["left"].set_visible(False)
    for ax in axes[1:]:
        ax.spines["left"].set_visible(False)

    subtitle = (f"Held-out test set: {first['n_runaway_episodes']} runs that ran away and "
                f"{first['n_censored_episodes']} that did not. "
                + (f"Every model caught every runaway, about {warn:.1f} s ahead (alarm horizon {horizon:g} s)."
                   if detected else ""))
    _arch_legend(fig, dots=True)
    _header(fig, "Runaway prediction from FEA telemetry: architecture comparison", subtitle)
    notes = ["Dot: mean over seeds; line: ±1 standard deviation; faint dots: individual seeds."]
    if fa_range:
        notes.append(f"All false alarms are runs at {fa_range[0]:.3f}–{fa_range[1]:.3f} I/Ic: "
                     "just below the runaway boundary and still heating when the simulation ended.")
    fig.text(0.012, 0.03, "   ".join(notes), fontsize=10.5, color=MUTED, ha="left", va="bottom")
    fig.savefig(out, dpi=200)
    plt.close(fig)


def _broken(x, y, gap=0.0015):
    """Insert NaN between held-out blocks so lines don't bridge untested currents."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    cut = np.flatnonzero(np.diff(x) > gap) + 1
    return np.insert(x, cut, np.nan), np.insert(y, cut, np.nan)


def per_run_table(result, test_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for e in predict_episodes(result, test_df):
        if e["censored"]:
            safe = e["y_valid"] > 0
            rows.append(dict(ratio=e["ratio"], runaway=False,
                             peak_safe_prob=float(e["prob"][safe].max()) if safe.any() else np.nan))
        else:
            alarm = np.flatnonzero(e["prob"] >= ALARM_THRESHOLD)
            near = e["ttr_s"] <= 1.0
            rows.append(dict(
                ratio=e["ratio"], runaway=True,
                warning_s=float(e["ttr_s"][alarm[0]]) if len(alarm) else np.nan,
                ttr_mae_ms=1000 * float(np.mean(np.abs(e["ttr_pred_s"][near] - e["ttr_s"][near])))))
    return pd.DataFrame(rows).sort_values("ratio")


def fig_vs_current(tables: dict, df: pd.DataFrame, horizon: float, out: Path):
    ep = df.groupby("episode_id").first()
    boundary = ep.loc[ep["runaway"] == 1, "ratio"].min()
    ramp_start = ep.loc[ep["status"] == "runaway_before_pulse", "ratio"].min()

    fig, axes = plt.subplots(1, 3, figsize=(13.33, 4.9), gridspec_kw={"width_ratios": [1.25, 1, 1]})
    fig.subplots_adjust(left=0.075, right=0.985, top=0.70, bottom=0.14, wspace=0.28)
    ax_fa, ax_warn, ax_err = axes

    for arch, t in tables.items():
        color = ARCHS[arch][1]
        style = dict(color=color, lw=2, solid_capstyle="round", solid_joinstyle="round")
        c = t[~t["runaway"]]
        r = t[t["runaway"]]
        ax_fa.plot(*_broken(c["ratio"], c["peak_safe_prob"]), **style)
        ax_warn.plot(*_broken(r["ratio"], r["warning_s"]), **style)
        ax_err.plot(*_broken(r["ratio"], r["ttr_mae_ms"]), **style)

    ax_fa.set_yscale("log")
    ax_fa.axhline(ALARM_THRESHOLD, color=MUTED, lw=1)
    ax_fa.text(ax_fa.get_xlim()[0], ALARM_THRESHOLD * 1.25, " alarm threshold", color=INK2, fontsize=10.5,
               va="bottom", ha="left")
    ax_fa.set_title("False-alarm margin, runs that did not run away")
    ax_fa.set_ylabel("peak probability\nwhile known safe")

    ax_warn.axhline(horizon, color=MUTED, lw=1)
    ax_warn.text(ramp_start, horizon * 0.97, f"horizon {horizon:g} s  ", color=INK2, fontsize=10.5,
                 va="top", ha="right")
    ax_warn.set_ylim(0, horizon * 1.6)
    ax_warn.set_title("Warning before runaway")
    ax_warn.set_ylabel("alarm lead time (s)")

    ax_err.set_ylim(0, None)
    ax_err.set_title("Countdown error, final second")
    ax_err.set_ylabel("mean absolute error (ms)")

    for ax in (ax_warn, ax_err):
        ax.axvline(ramp_start, color=AXIS, lw=1, zorder=0)
        top = ax.get_ylim()[1]
        ax.text(ramp_start, top, "steady state  ", ha="right", va="top", fontsize=10, color=MUTED)
        ax.text(ramp_start, top, "  pre-ramp runaway", ha="left", va="top", fontsize=10, color=MUTED)

    # Runaway boundary: right edge of the non-runaway panel, left edge of the other two;
    # the axes are widened slightly so the rotated label sits in empty space beside the line.
    ax_fa.set_xlim(ax_fa.get_xlim()[0], boundary + 0.035)
    for ax in (ax_warn, ax_err):
        ax.set_xlim(boundary - 0.02, ax.get_xlim()[1])
    for ax, side in ((ax_fa, 1), (ax_warn, -1), (ax_err, -1)):
        ax.axvline(boundary, color=MUTED, lw=1, zorder=1)
        ax.annotate(f"runaway boundary {boundary:.3f} Ic", xy=(boundary, 0.5),
                    xycoords=("data", "axes fraction"), xytext=(5 * side, 0),
                    textcoords="offset points", rotation=90, va="center",
                    ha="left" if side > 0 else "right", fontsize=10, color=INK2)
    for ax in axes:
        ax.grid(axis="y")
        ax.set_axisbelow(True)
        ax.set_xlabel("transport current  I / Ic", fontsize=11)

    _arch_legend(fig)
    _header(fig, "Runaway prediction vs transport current",
            f"Curves trace held-out FEA runs, one every 0.001 Ic; gaps are current ranges used for "
            f"training and validation.\nRunaway boundary: {boundary:.3f} Ic. "
            f"Runs above {ramp_start:.3f} Ic ran away before the current ramp finished.")
    fig.savefig(out, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--path", default=None, help="sweep.h5 (default: the path stored in the checkpoint)")
    ap.add_argument("--out-dir", default="plots/fem_runaway")
    args = ap.parse_args()

    _style()
    models_dir, out_dir = Path(args.models_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = load_arch_runs(models_dir)
    fig_architectures(runs, false_alarm_ratio_range(models_dir), out_dir / "fem_runaway_architectures.png")
    print(f"saved {out_dir / 'fem_runaway_architectures.png'}")

    results = {a: load_runaway_model(models_dir / f"fem_runaway_{a}.pt") for a in ARCHS
               if (models_dir / f"fem_runaway_{a}.pt").exists()}
    meta = next(iter(results.values())).meta
    df = add_model_features(load_runaway_sweep(args.path or meta["data_path"], dt_s=meta["dt_s"],
                                               threshold_K=meta["threshold_K"]))
    test_df = split_by_ratio_blocks(df, block_width=meta["block_width"])["test"]
    tables = {a: per_run_table(r, test_df) for a, r in results.items()}
    fig_vs_current(tables, df, meta["horizon_s"], out_dir / "fem_runaway_vs_current.png")
    print(f"saved {out_dir / 'fem_runaway_vs_current.png'}")


if __name__ == "__main__":
    main()
