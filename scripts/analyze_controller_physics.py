"""
analyze_controller_physics.py

Physics-grounded failure analysis for the trained sequence surrogate and the
controllers built on it: not just aggregate error, but *where in the
(T0, lambda) solution space* the surrogate does worse, tied to actual
nonlinearity metrics of the analytic runaway surface (physics_diagnostics.py)
rather than just step count or noise.

Produces:
  1. All three runaway surfaces (I*, T*, time-margin) -- solver vs surrogate.
  2. Nonlinearity metrics (curvature, branch flatness, slope) on the same
     grid, and their correlation with surrogate error -- directly answers
     "does it do worse in more nonlinear regions?".
  3. Controller comparison (oracle / margin_pred / breach_prob_pred) broken
     down by T0 tertile, lambda tertile, and *curvature* tertile.

Usage
-----
    python scripts/analyze_controller_physics.py --checkpoint models/seq_gru.pt
"""

import _bootstrap  # noqa: F401

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr

from hts_tape_env.controllers import (
    evaluate,
    oracle_controller,
    run_episode,
    surrogate_controller,
    surrogate_prob_controller,
)
from hts_tape_env.env import QuenchDetectorEnv
from hts_tape_env.physics_diagnostics import branch_flatness, runaway_curvature, runaway_sensitivity
from hts_tape_env.seq_surrogates import SequenceSurrogateAdapter, load_seq_surrogate, predict_runaway_surface

N_FIXED, IC0_FIXED = 20.0, 500.0


def rel_err(pred, true):
    return np.abs(pred - true) / np.maximum(np.abs(true), 1e-6)


def tertile_labels(x: np.ndarray) -> np.ndarray:
    """0/1/2 = low/mid/high tertile, NaN-safe."""
    valid = np.isfinite(x)
    labels = np.full(x.shape, -1, dtype=int)
    q1, q2 = np.nanpercentile(x[valid], [33.3, 66.7])
    labels[valid & (x <= q1)] = 0
    labels[valid & (x > q1) & (x <= q2)] = 1
    labels[valid & (x > q2)] = 2
    return labels


def section(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default="models/seq_gru.pt")
    ap.add_argument("--grid-size", type=int, default=12)
    ap.add_argument("--controller-episodes", type=int, default=400)
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--out-dir", default="models")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)

    result = load_seq_surrogate(args.checkpoint)
    dump_threshold = result.meta.get("dump_threshold", 0.05)

    # ── 1+2. Surfaces + nonlinearity metrics on the same grid ──────────────
    section("1. All three runaway surfaces: solver vs. GRU sequence surrogate")

    T0_grid = np.linspace(8.0, 72.0, args.grid_size)
    lam_grid = np.linspace(24.0, 190.0, args.grid_size)
    surf = predict_runaway_surface(result, T0_grid, lam_grid, seed=args.seed)

    istar_re = rel_err(surf["Istar_pred"], surf["Istar_analytic"])
    tstar_ae = np.abs(surf["Tstar_pred"] - surf["Tstar_analytic"])
    tm_re = rel_err(surf["time_margin_pred_ms"], surf["time_margin_analytic_ms"])

    print(f"I*         median rel err: {np.nanmedian(istar_re):.2%}  95th pct: {np.nanpercentile(istar_re, 95):.2%}")
    print(f"T*         median abs err: {np.nanmedian(tstar_ae):.2f} K  95th pct: {np.nanpercentile(tstar_ae, 95):.2f} K")
    print(f"time-margin median rel err: {np.nanmedian(tm_re):.2%}  95th pct: {np.nanpercentile(tm_re, 95):.2%}  "
          f"(n valid={np.isfinite(surf['time_margin_analytic_ms']).sum()}/{istar_re.size})")

    section("2. Nonlinearity of the analytic surface vs. surrogate error")

    curvature = np.full_like(istar_re, np.nan)
    flatness = np.full_like(istar_re, np.nan)
    slope = np.full_like(istar_re, np.nan)
    tc_gap = np.full_like(istar_re, np.nan)
    for i, T0 in enumerate(T0_grid):
        for j, lam in enumerate(lam_grid):
            if not np.isfinite(surf["Istar_analytic"][j, i]):
                continue
            c = runaway_curvature(T0, lam, N_FIXED, Ic0=IC0_FIXED)
            s = runaway_sensitivity(T0, lam, N_FIXED, Ic0=IC0_FIXED)
            curvature[j, i] = c["curvature"]
            slope[j, i] = abs(s["dIstar_dT0"])
            tc_gap[j, i] = s["Tc_minus_T0"]
            flatness[j, i] = branch_flatness(T0, N_FIXED, lam_override=lam, Ic0=IC0_FIXED)

    def corr_report(name, x, y):
        valid = np.isfinite(x) & np.isfinite(y)
        if valid.sum() < 5:
            print(f"  {name}: not enough valid points")
            return
        pr, _ = pearsonr(x[valid], y[valid])
        sr, _ = spearmanr(x[valid], y[valid])
        print(f"  I* rel-err  vs {name:<22} Pearson r={pr:+.3f}   Spearman rho={sr:+.3f}")

    print("Correlation of surrogate I* relative error with nonlinearity metrics "
          "(positive => error grows where the metric is larger):")
    corr_report("curvature (2nd deriv)", curvature, istar_re)
    corr_report("|dI*/dT0| (slope)", slope, istar_re)
    corr_report("branch flatness", flatness, istar_re)
    corr_report("Tc - T0 (distance from Tc)", tc_gap, istar_re)

    # Binned comparison: does error concentrate in the high-curvature tertile?
    curv_tert = tertile_labels(curvature)
    print("\nI* median relative error by curvature tertile (low/mid/high nonlinearity):")
    for t, label in enumerate(["low", "mid", "high"]):
        m = curv_tert == t
        if m.sum():
            print(f"  {label:<5} curvature (n={m.sum():3d}): median rel err = {np.nanmedian(istar_re[m]):.2%}")

    # Plots
    TT, LL = np.meshgrid(T0_grid, lam_grid)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, Z, title in [
        (axes[0], istar_re, "I* relative error"),
        (axes[1], curvature, "analytic curvature (nonlinearity)"),
        (axes[2], flatness, "branch flatness (|dI/dT| near peak)"),
    ]:
        im = ax.pcolormesh(TT, LL, Z, shading="auto", cmap="viridis")
        ax.set_xlabel("T0 (K)"); ax.set_ylabel("lambda (W/m/K)"); ax.set_title(title, fontsize=10)
        fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_dir / "nonlinearity_vs_error.png", dpi=140)
    print(f"\nSaved heatmaps to {out_dir / 'nonlinearity_vs_error.png'}")

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    valid = np.isfinite(curvature) & np.isfinite(istar_re)
    ax.scatter(curvature[valid], istar_re[valid], s=14, alpha=0.6)
    ax.set_xlabel("analytic curvature (nonlinearity)")
    ax.set_ylabel("I* relative error")
    ax.set_title("Surrogate error vs. surface nonlinearity")
    fig.tight_layout()
    fig.savefig(out_dir / "error_vs_curvature_scatter.png", dpi=140)
    print(f"Saved scatter to {out_dir / 'error_vs_curvature_scatter.png'}")

    # ── 3. Controller comparison, broken down by physical regime ───────────
    section("3. Controllers: aggregate + broken down by regime")

    env_oracle = QuenchDetectorEnv(dump_threshold=dump_threshold)
    env_margin = QuenchDetectorEnv(dump_threshold=dump_threshold)
    env_margin.set_surrogate(SequenceSurrogateAdapter(result))
    env_prob = QuenchDetectorEnv(dump_threshold=dump_threshold)
    env_prob.set_surrogate(SequenceSurrogateAdapter(result))

    controllers = {
        "oracle": (env_oracle, oracle_controller(dump_threshold)),
        "surrogate margin_pred": (env_margin, surrogate_controller(dump_threshold)),
        "surrogate breach_prob": (env_prob, surrogate_prob_controller(0.5)),
    }

    print(f"{'controller':<26}{'mean_reward':<14}{'hit_rate':<11}{'miss_rate':<11}{'false_alarm_rate'}")
    agg = {}
    for name, (env, ctrl) in controllers.items():
        m = evaluate(env, ctrl, n_episodes=args.controller_episodes, seed0=args.seed)
        agg[name] = m
        print(f"{name:<26}{m['mean_reward']:<14.3f}{m['hit_rate']:<11.3f}{m['miss_rate']:<11.3f}{m['false_alarm_rate']:.3f}")

    # Per-episode T0/lambda + curvature regime, for the two realistic controllers
    print("\nBreaking down the two realistic controllers by physical regime "
          f"({args.controller_episodes} episodes each)...")
    for name in ["surrogate margin_pred", "surrogate breach_prob"]:
        env, ctrl = controllers[name]
        T0s, lams, curvs, outcomes = [], [], [], []
        for k in range(args.controller_episodes):
            obs0, info0 = env.reset(seed=args.seed + k)
            ctrl.reset()
            T0s.append(info0["T0"]); lams.append(info0["lam"])
            c = runaway_curvature(info0["T0"], info0["lam"], N_FIXED, Ic0=IC0_FIXED)["curvature"]
            curvs.append(c)
            out = run_episode(env, ctrl, args.seed + k)  # re-reset with same seed -> same episode
            if out["dumped"] and 0.0 < out["final_margin"] < dump_threshold:
                outcomes.append("hit")
            elif out["runaway"] and not out["dumped"]:
                outcomes.append("miss")
            elif out["dumped"]:
                outcomes.append("false_alarm")
            else:
                outcomes.append("other")
        T0s, lams, curvs = np.array(T0s), np.array(lams), np.array(curvs)
        outcomes = np.array(outcomes)

        print(f"\n  [{name}] outcome rate by T0 tertile (low/mid/high T0):")
        for t, label in enumerate(["low", "mid", "high"]):
            m = tertile_labels(T0s) == t
            if m.sum():
                vc = {o: (outcomes[m] == o).mean() for o in ("hit", "miss", "false_alarm")}
                print(f"    {label:<5} T0 (n={m.sum():3d}): " +
                      "  ".join(f"{k}={v:.2%}" for k, v in vc.items()))

        print(f"  [{name}] outcome rate by curvature tertile (low/mid/high nonlinearity):")
        for t, label in enumerate(["low", "mid", "high"]):
            m = tertile_labels(curvs) == t
            if m.sum():
                vc = {o: (outcomes[m] == o).mean() for o in ("hit", "miss", "false_alarm")}
                print(f"    {label:<5} curvature (n={m.sum():3d}): " +
                      "  ".join(f"{k}={v:.2%}" for k, v in vc.items()))


if __name__ == "__main__":
    main()
