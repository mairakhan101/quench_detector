"""
fem_surrogate_inference.py

Run a sequence surrogate TRAINED ONLY on QuenchDetectorEnv's simulated,
analytic-physics trajectories (models/seq_gru.pt by default) as inference
against the real FEM sweep (//eagle/fanc_ldrd/magnets_fem/sweep.h5) it has
never seen -- a sim-to-FEM generalization check, not a retrain. For each of
the 6 FEM runs, feeds the whole (T0_K, I_true_A, voltage_true_V,
T_surface_true_K) trajectory through the model in one batched forward pass
(hts_tape_env.seq_surrogates.predict_runaway_surface does the same thing for
grid points of the analytic env; this is the FEM equivalent) and compares:

  - breach_prob_pred crossing 0.5  vs  the FEM run's actual quench_step
    (first step Tmax >= fem_data.QUENCH_T_K) for the two runaway runs, and
    "never crosses" for the four that settle
  - I_star_pred / T_star_pred, the surrogate's inferred runaway point, vs the
    FEM run's actual plateau current / peak Tmax -- NOTE these are not the
    same physical quantity (see fem_data.py's module docstring: this sweep
    has no analytic I* to compare against), reported only as a rough sanity
    check on whether the model's inferred scale is in the right ballpark.

Usage
-----
    python scripts/fem_surrogate_inference.py --checkpoint models/seq_gru.pt
"""

import _bootstrap  # noqa: F401

import argparse

import numpy as np

from hts_tape_env.fem_data import DEFAULT_PATH, QUENCH_T_K, load_fem_sweep
from hts_tape_env.seq_data import TRUE_FEATURES
from hts_tape_env.seq_surrogates import load_seq_surrogate, predict_episode


def run_episode_inference(result, g, feature_cols) -> dict:
    """One FEM episode's dataframe rows (sorted by step) -> predict_episode()'s
    per-step margin_pred/breach_prob plus final istar/tstar/time_margin."""
    X = g[feature_cols].to_numpy(dtype=np.float32)
    return predict_episode(result, X)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default="models/seq_gru.pt")
    ap.add_argument("--fem-path", default=DEFAULT_PATH)
    ap.add_argument("--plot", default="models/fem_surrogate_inference.png")
    args = ap.parse_args()

    result = load_seq_surrogate(args.checkpoint)
    feature_cols = result.meta.get("features", TRUE_FEATURES)
    print(f"Loaded {args.checkpoint} (arch={result.arch}, trained on {feature_cols})")
    print(f"Simulated-env held-out metrics at train time: {result.metrics}")

    df = load_fem_sweep(args.fem_path)
    print(f"\nLoaded FEM sweep: {df['episode_id'].nunique()} episodes from {args.fem_path}\n")

    header = f"{'ratio_label':<10}{'status':<22}{'quench_step':<12}{'alarm_step':<12}{'lead_steps':<12}{'I*_pred_A':<12}{'T*_pred_K'}"
    print(header)
    print("-" * len(header))

    results = {}
    for episode_id, g in df.sort_values("step").groupby("episode_id", sort=True):
        label = g["ratio_label"].iloc[0]
        status = g["status"].iloc[0]
        quench_step = g["quench_step"].iloc[0]
        r = run_episode_inference(result, g, feature_cols)
        results[label] = r

        alarm = r["first_alarm_step"]
        if np.isfinite(quench_step) and alarm is not None:
            lead = f"{quench_step - alarm:+.0f}"
        elif np.isfinite(quench_step):
            lead = "MISSED"
        elif alarm is not None:
            lead = "FALSE ALARM"
        else:
            lead = "-"
        qs = f"{quench_step:.0f}" if np.isfinite(quench_step) else "never"
        al = f"{alarm}" if alarm is not None else "never"
        print(f"{label:<10}{status:<22}{qs:<12}{al:<12}{lead:<12}{r['istar_pred_A']:<12.1f}{r['tstar_pred_K']:.1f}")

    print(f"\n(quench_step/alarm_step are FEM output-step indices -- dt is adaptive, not fixed, see t_s;")
    print(f" QUENCH_T_K={QUENCH_T_K} defines quench_step;")
    print(" 'lead_steps' > 0 means the surrogate flagged risk before the FEM run actually crossed threshold --")
    print(" a sim-trained model has no guarantee of this on real/FEM data, which is exactly what this checks.)")

    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharex=False)
        for ax, (label, r) in zip(axes.flat, results.items()):
            g = df[df["ratio_label"] == label].sort_values("step")
            t = g["t_s"].to_numpy()
            ax2 = ax.twinx()
            ax.plot(t, r["margin_pred"], color="tab:blue", label="margin_pred")
            ax.axhline(0, color="gray", lw=0.5)
            ax2.plot(t, r["breach_prob"], color="tab:red", alpha=0.6, label="breach_prob")
            ax2.axhline(0.5, color="tab:red", lw=0.5, ls="--")
            qs = g["quench_step"].iloc[0]
            if np.isfinite(qs):
                ax.axvline(t[int(qs)], color="black", ls=":", label="FEM quench")
            ax.set_title(f"{label} ({g['status'].iloc[0]})")
            ax.set_xlabel("t [s]")
            ax.set_ylabel("margin_pred", color="tab:blue")
            ax2.set_ylabel("breach_prob", color="tab:red")
        fig.suptitle("Sim-trained surrogate vs FEM ground truth (never trained on FEM data)")
        fig.tight_layout()
        fig.savefig(args.plot, dpi=130)
        print(f"\nSaved {args.plot}")
    except ImportError:
        print("\n(matplotlib not available -- skipped plot)")


if __name__ == "__main__":
    main()
