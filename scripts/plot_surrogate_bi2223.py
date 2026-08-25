"""
plot_surrogate_bi2223.py

Same sim-to-other-physics generalization check as fem_surrogate_inference.py,
against physics_bi2223.py's Bi-2223 trajectories instead of the FEM sweep:
run the REBCO-trained sequence surrogate (models/seq_gru.pt) as pure
inference over Bi-2223 ramp trajectories it has never seen, and plot
predicted vs analytic I*(T0)/T*(T0), and the per-step margin_pred/breach_prob
trace against the analytic margin, over physics_bi2223's validated T0 range
(see its module docstring -- roughly 77.16-80.5 K; outside that
solve_runaway returns NaN and those points are skipped here too).

Usage
-----
    python scripts/plot_surrogate_bi2223.py --checkpoint models/seq_gru.pt
"""

import _bootstrap  # noqa: F401

import argparse

import numpy as np

from hts_tape_env.bi2223_rollout import generate_bi2223_grid_dataset
from hts_tape_env.seq_data import TRUE_FEATURES
from hts_tape_env.seq_surrogates import load_seq_surrogate, predict_episode


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default="models/seq_gru.pt")
    ap.add_argument("--t0-lo", type=float, default=77.16)
    ap.add_argument("--t0-hi", type=float, default=80.5)
    ap.add_argument("--n-points", type=int, default=12)
    ap.add_argument("--plot", default="models/surrogate_bi2223_surface.png")
    args = ap.parse_args()

    result = load_seq_surrogate(args.checkpoint)
    feature_cols = result.meta.get("features", TRUE_FEATURES)
    print(f"Loaded {args.checkpoint} (arch={result.arch}, trained on REBCO trajectories only, "
          f"features={feature_cols})")

    T0_grid = np.linspace(args.t0_lo, args.t0_hi, args.n_points)
    df = generate_bi2223_grid_dataset(T0_grid)
    print(f"Generated {df['episode_id'].nunique()} Bi-2223 episodes over T0 in "
          f"[{args.t0_lo}, {args.t0_hi}] K ({len(df)} steps)")

    header = f"{'T0_K':<10}{'I*_analytic_A':<16}{'I*_pred_A':<14}{'rel_err':<10}{'T*_analytic_K':<16}{'T*_pred_K'}"
    print(header)
    print("-" * len(header))

    T0s, Ia, Ip, Ta, Tp = [], [], [], [], []
    per_episode = {}
    for episode_id, g in df.sort_values("step").groupby("episode_id", sort=True):
        T0 = g["T0_K"].iloc[0]
        istar_a, tstar_a = g["Istar_A"].iloc[0], g["Tstar_K"].iloc[0]
        X = g[feature_cols].to_numpy(dtype=np.float32)
        r = predict_episode(result, X)
        per_episode[T0] = (g, r)

        rel_err = abs(r["istar_pred_A"] - istar_a) / istar_a
        print(f"{T0:<10.2f}{istar_a:<16.1f}{r['istar_pred_A']:<14.1f}{rel_err:<10.2%}"
              f"{tstar_a:<16.2f}{r['tstar_pred_K']:.2f}")
        T0s.append(T0); Ia.append(istar_a); Ip.append(r["istar_pred_A"])
        Ta.append(tstar_a); Tp.append(r["tstar_pred_K"])

    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

        axes[0].plot(T0s, Ia, "o-", label="analytic (physics_bi2223)", color="tab:orange")
        axes[0].plot(T0s, Ip, "s--", label="REBCO-trained surrogate", color="tab:blue")
        axes[0].set_xlabel("T0 [K]"); axes[0].set_ylabel("I* [A]")
        axes[0].set_title("Runaway current I*(T0)"); axes[0].legend()

        axes[1].plot(T0s, Ta, "o-", label="analytic", color="tab:orange")
        axes[1].plot(T0s, Tp, "s--", label="surrogate", color="tab:blue")
        axes[1].set_xlabel("T0 [K]"); axes[1].set_ylabel("T* [K]")
        axes[1].set_title("Runaway temperature T*(T0)"); axes[1].legend()

        # Margin trace for the coldest (most-runaway-current) episode, as a
        # representative example -- same style as fem_surrogate_inference.py.
        T0_example = T0s[0]
        g, r = per_episode[T0_example]
        step = g["step"].to_numpy()
        axes[2].plot(step, g["margin_norm"].to_numpy(), label="analytic margin", color="tab:orange")
        axes[2].plot(step, r["margin_pred"], label="surrogate margin_pred", color="tab:blue")
        axes[2].axhline(0, color="gray", lw=0.5)
        axes[2].set_xlabel("step"); axes[2].set_ylabel("margin")
        axes[2].set_title(f"Margin trace, T0={T0_example:.2f} K"); axes[2].legend()

        fig.suptitle("REBCO-trained sequence surrogate vs Bi-2223 analytic ground truth "
                      "(never trained on Bi-2223)")
        fig.tight_layout()
        fig.savefig(args.plot, dpi=130)
        print(f"\nSaved {args.plot}")
    except ImportError:
        print("\n(matplotlib not available -- skipped plot)")


if __name__ == "__main__":
    main()
