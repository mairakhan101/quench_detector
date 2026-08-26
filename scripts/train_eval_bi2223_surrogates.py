"""
train_eval_bi2223_surrogates.py

End-to-end training and evaluation of Bi-2223 thermal-runaway surrogates:

  1. Input-based surrogates (A, B, raw) on swept Bi-2223 operating points
  2. Sequence surrogates (GRU and/or Transformer) on Bi-2223 ramp trajectories
  3. Diagnostic plots:
     a. I*(T0) / T*(T0) / time margin vs T0 — analytic vs all surrogate types
     b. Sample trajectories at several T0 values (I, V, T_surface vs step)
     c. Nonlinear region diagnostics (error vs T0, n(T) overlay, residuals
        near the T0=80.5 K edge where n(T)->1)
  4. Metrics JSON for downstream notebooks

Usage
-----
    python scripts/train_eval_bi2223_surrogates.py
    python scripts/train_eval_bi2223_surrogates.py --epochs 80
    python scripts/train_eval_bi2223_surrogates.py --arch gru
    python scripts/train_eval_bi2223_surrogates.py --arch gru transformer
"""

import _bootstrap  # noqa: F401

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from hts_tape_env.bi2223_rollout import (  # noqa: E402
    generate_bi2223_grid_dataset,
    rollout_bi2223_episode,
)
from hts_tape_env.datagen_bi2223 import generate_dataset as generate_bi2223_dataset  # noqa: E402
from hts_tape_env.physics_bi2223 import (  # noqa: E402
    IC_ANCHOR_A,
    T_ANCHOR_K,
    T_X_K,
    n_value,
    solve_runaway,
)
from hts_tape_env.seq_data import TRUE_FEATURES  # noqa: E402
from hts_tape_env.seq_surrogates import predict_episode, train_seq_surrogate  # noqa: E402
from hts_tape_env.surrogates import train_spec  # noqa: E402
from hts_tape_env.thermal_bi2223 import time_margin_to_safe  # noqa: E402

SEQ_COLORS = {"gru": "tab:blue", "transformer": "tab:orange"}
SEQ_MARKERS = {"gru": "^", "transformer": "v"}


def _make_main_figure(T0_eval, Istar_analytic, Tstar_analytic, tm_analytic_ms,
                      Istar_input_A, Istar_input_raw, seq_eval, trajectory_data,
                      out_dir):
    """3x3 diagnostic figure."""
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))

    # ── Row 0, Col 0: I*(T0) ────────────────────────────────────────
    ax = axes[0, 0]
    ax.plot(T0_eval, Istar_analytic, "k-", lw=2, label="analytic")
    ax.plot(T0_eval, Istar_input_A, "s", ms=5, color="tab:green", label="input (spec A)")
    for arch, ev in seq_eval.items():
        ax.plot(T0_eval, ev["Istar"], SEQ_MARKERS[arch], ms=5,
                color=SEQ_COLORS[arch], label=f"seq ({arch})")
    ax.set_xlabel("T0 [K]")
    ax.set_ylabel("I* [A]")
    ax.set_title("Runaway current I*(T0)")
    ax.legend(fontsize=8)

    # ── Row 0, Col 1: Time margin ───────────────────────────────────
    ax = axes[0, 1]
    valid_a = np.isfinite(tm_analytic_ms)
    ax.plot(T0_eval[valid_a], tm_analytic_ms[valid_a], "k-", lw=2, label="analytic")
    for arch, ev in seq_eval.items():
        valid_s = np.isfinite(ev["tm_ms"])
        if valid_s.any():
            ax.plot(T0_eval[valid_s], ev["tm_ms"][valid_s], SEQ_MARKERS[arch],
                    ms=5, color=SEQ_COLORS[arch], label=f"seq ({arch})")
    ax.set_xlabel("T0 [K]")
    ax.set_ylabel("Time margin [ms]")
    ax.set_title("Adiabatic time margin at runaway")
    ax.legend(fontsize=8)

    # ── Row 0, Col 2: T*(T0) ────────────────────────────────────────
    ax = axes[0, 2]
    ax.plot(T0_eval, Tstar_analytic, "k-", lw=2, label="analytic")
    for arch, ev in seq_eval.items():
        ax.plot(T0_eval, ev["Tstar"], SEQ_MARKERS[arch], ms=5,
                color=SEQ_COLORS[arch], label=f"seq ({arch})")
    ax.set_xlabel("T0 [K]")
    ax.set_ylabel("T* [K]")
    ax.set_title("Runaway temperature T*(T0)")
    ax.legend(fontsize=8)

    # ── Row 1: Sample trajectories ──────────────────────────────────
    sample_T0s = [77.5, 79.0, 80.0]
    for col_idx, T0_target in enumerate(sample_T0s):
        ax = axes[1, col_idx]
        T0_key = min(trajectory_data.keys(), key=lambda t: abs(t - T0_target))
        g = trajectory_data[T0_key]
        step = g["step"].to_numpy()

        ax.plot(step, g["I_true_A"], color="tab:blue", lw=1.5, label="I [A]")
        istar_ep = g["Istar_A"].iloc[0]
        ax.axhline(istar_ep, color="gray", ls=":", lw=1, label=f"I*={istar_ep:.0f} A")
        ax.set_xlabel("step")
        ax.set_ylabel("I [A]", color="tab:blue")
        ax.set_title(f"T0 = {T0_key:.2f} K", fontsize=10)

        ax2 = ax.twinx()
        ax2.plot(step, g["voltage_true_V"] * 1e6, color="tab:red", ls="--",
                 lw=1.5, label="V [µV]")
        ax2.set_ylabel("V [µV]", color="tab:red")

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="upper left")

    # ── Row 2, Col 0: Relative error vs T0 ──────────────────────────
    ax = axes[2, 0]
    denom = np.maximum(np.abs(Istar_analytic), 1e-6)
    rel_err_A = np.abs(Istar_input_A - Istar_analytic) / denom
    rel_err_raw = np.abs(Istar_input_raw - Istar_analytic) / denom
    ax.plot(T0_eval, rel_err_A * 100, "s-", ms=4, color="tab:green", label="input A")
    ax.plot(T0_eval, rel_err_raw * 100, "d-", ms=4, color="tab:purple", label="input raw")
    for arch, ev in seq_eval.items():
        rel_err_seq = np.abs(ev["Istar"] - Istar_analytic) / denom
        ax.plot(T0_eval, rel_err_seq * 100, f"{SEQ_MARKERS[arch]}-", ms=4,
                color=SEQ_COLORS[arch], label=f"seq {arch}")
    ax.axvspan(80.0, 80.5, color="gray", alpha=0.2, label="unreliable n(T)")
    ax.set_xlabel("T0 [K]")
    ax.set_ylabel("I* relative error [%]")
    ax.set_title("Surrogate error vs T0")
    ax.legend(fontsize=7)

    # ── Row 2, Col 1: n(T) overlay ──────────────────────────────────
    ax = axes[2, 1]
    T_range = np.linspace(T_ANCHOR_K, T_X_K, 200)
    n_vals = n_value(T_range)
    ax.plot(T_range, n_vals, "k-", lw=2, label="n(T)")
    ax.set_xlabel("T [K]")
    ax.set_ylabel("n-value", color="black")
    ax.set_title("n(T) and I*(T0) — nonlinearity source")

    ax2 = ax.twinx()
    valid_i = np.isfinite(Istar_analytic)
    ax2.plot(T0_eval[valid_i], Istar_analytic[valid_i], color="tab:orange",
             lw=2, label="I*(T0)")
    ax2.set_ylabel("I* [A]", color="tab:orange")

    cap_line = 3.0 * IC_ANCHOR_A
    ax2.axhline(cap_line, color="red", ls=":", lw=1,
                label=f"sanity cap ({cap_line:.0f} A)")

    for T0_v in [77.5, 79.0, 80.0, 80.5]:
        ax.axvline(T0_v, color="gray", ls="--", lw=0.5, alpha=0.5)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="center left")

    # ── Row 2, Col 2: Residuals near boundary ───────────────────────
    ax = axes[2, 2]
    mask_near = T0_eval >= 79.5
    if mask_near.any() and np.any(np.isfinite(Istar_analytic[mask_near])):
        residual_A = Istar_input_A[mask_near] - Istar_analytic[mask_near]
        ax.plot(T0_eval[mask_near], residual_A, "s-", ms=5, color="tab:green",
                label="input A")
        for arch, ev in seq_eval.items():
            residual_seq = ev["Istar"][mask_near] - Istar_analytic[mask_near]
            ax.plot(T0_eval[mask_near], residual_seq, f"{SEQ_MARKERS[arch]}-",
                    ms=5, color=SEQ_COLORS[arch], label=f"seq {arch}")
        ax.axhline(0, color="gray", lw=0.5)
    ax.set_xlabel("T0 [K]")
    ax.set_ylabel("I* residual [A]")
    ax.set_title("Residuals near boundary (T0 >= 79.5 K)")
    ax.legend(fontsize=8)

    fig.suptitle("Bi-2223 thermal runaway surrogates — training, evaluation, diagnostics",
                 fontsize=13, y=1.01)
    fig.tight_layout()
    path = out_dir / "bi2223_surrogates_full.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def _make_trajectory_figure(out_dir):
    """2x2 detailed trajectory plots at 4 T0 values."""
    T0_vals = [77.5, 78.5, 79.5, 80.2]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for idx, T0 in enumerate(T0_vals):
        ax = axes[idx // 2, idx % 2]
        rows = rollout_bi2223_episode(T0, episode_id=idx)
        if not rows:
            ax.set_title(f"T0 = {T0:.1f} K — no valid runaway")
            continue

        g = pd.DataFrame(rows)
        step = g["step"].to_numpy()
        Istar_ep = g["Istar_A"].iloc[0]

        ax.plot(step, g["I_true_A"] / Istar_ep, color="tab:blue", lw=1.5,
                label="I / I*")
        ax.axhline(1.0, color="gray", ls=":", lw=1, label="I = I*")

        breach_steps = g[g["margin_norm"] < 0.05]["step"]
        if len(breach_steps):
            ax.axvline(breach_steps.iloc[0], color="tab:orange", ls="--",
                       lw=1, label="margin < 0.05")

        ax.set_xlabel("step")
        ax.set_ylabel("I / I*", color="tab:blue")
        ax.set_ylim(0.2, 1.15)

        ax2 = ax.twinx()
        V_uv = g["voltage_true_V"].to_numpy() * 1e6
        V_uv_clipped = np.clip(V_uv, 1e-3, None)
        ax2.semilogy(step, V_uv_clipped, color="tab:red", ls="--", lw=1.5,
                     label="V [µV]")
        ax2.set_ylabel("V [µV]", color="tab:red")

        ax.set_title(f"T0 = {T0:.1f} K  (I* = {Istar_ep:.0f} A)", fontsize=10)
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="upper left")

    fig.suptitle("Bi-2223 trajectory detail — ramp behavior across T0 range",
                 fontsize=12)
    fig.tight_layout()
    path = out_dir / "bi2223_trajectory_detail.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def _eval_seq(seq_result, T0_eval, df_eval):
    """Evaluate one sequence surrogate on the eval grid."""
    Istar = np.full(len(T0_eval), np.nan)
    Tstar = np.full(len(T0_eval), np.nan)
    tm_ms = np.full(len(T0_eval), np.nan)

    for _episode_id, g in df_eval.sort_values("step").groupby("episode_id", sort=True):
        T0 = g["T0_K"].iloc[0]
        X = g[list(TRUE_FEATURES)].to_numpy(dtype=np.float32)
        r = predict_episode(seq_result, X)
        idx = np.argmin(np.abs(T0_eval - T0))
        Istar[idx] = r["istar_pred_A"]
        Tstar[idx] = r["tstar_pred_K"]
        tm_ms[idx] = r["time_margin_pred_ms"]
    return {"Istar": Istar, "Tstar": Tstar, "tm_ms": tm_ms}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-datagen", type=int, default=3000)
    ap.add_argument("--n-rollout-points", type=int, default=20)
    ap.add_argument("--arch", nargs="+", choices=["gru", "transformer"],
                    default=["gru", "transformer"])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--n-eval-points", type=int, default=25)
    ap.add_argument("--t0-lo", type=float, default=77.16)
    ap.add_argument("--t0-hi", type=float, default=80.5)
    ap.add_argument("--out-dir", default="models")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_metrics = {}

    # ── Phase 1: Input-based surrogates ──────────────────────────────
    print("=" * 70)
    print("Phase 1: Input-based surrogates (A, B, raw) on Bi-2223 data")
    print("=" * 70)

    df_input = generate_bi2223_dataset(args.n_datagen, seed=args.seed)
    print(f"  Bi-2223 datagen: {len(df_input)} rows")

    input_results = {}
    for spec_name in ("A", "B", "raw"):
        try:
            result = train_spec(spec_name, df_input)
            result.save(out_dir / f"bi2223_surr_{spec_name}.joblib")
            input_results[spec_name] = result
            all_metrics[f"input_{spec_name}"] = result.metrics
            print(f"  {spec_name}: {result.metrics}")
        except (KeyError, ValueError) as exc:
            print(f"  {spec_name}: SKIPPED — {exc}")

    # ── Phase 2: Sequence surrogates ─────────────────────────────────
    T0_train = np.linspace(args.t0_lo, args.t0_hi, args.n_rollout_points)
    df_seq = generate_bi2223_grid_dataset(T0_train)
    print(f"\n  Rollout dataset: {len(df_seq)} rows, "
          f"{df_seq['episode_id'].nunique()} episodes")

    seq_results = {}
    for arch in args.arch:
        print("\n" + "=" * 70)
        print(f"Phase 2: Sequence surrogate ({arch}) on Bi-2223 trajectories")
        print("=" * 70)
        seq_result = train_seq_surrogate(
            df_seq, arch=arch, epochs=args.epochs, seed=args.seed,
        )
        seq_result.save(out_dir / f"bi2223_seq_{arch}.pt")
        seq_results[arch] = seq_result
        all_metrics[f"seq_{arch}"] = seq_result.metrics

    # ── Phase 3: Evaluation ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Phase 3: Evaluation against analytic ground truth")
    print("=" * 70)

    T0_eval = np.linspace(args.t0_lo, args.t0_hi, args.n_eval_points)

    Istar_analytic = np.full(len(T0_eval), np.nan)
    Tstar_analytic = np.full(len(T0_eval), np.nan)
    tm_analytic_ms = np.full(len(T0_eval), np.nan)

    for i, T0 in enumerate(T0_eval):
        Is, Ts = solve_runaway(T0)
        if np.isfinite(Is) and Is > 0:
            Istar_analytic[i] = Is
            Tstar_analytic[i] = Ts
            tm_s = time_margin_to_safe(Is, Ts)
            if np.isfinite(tm_s):
                tm_analytic_ms[i] = tm_s * 1000.0

    # Input-based: spec A
    Istar_input_A = np.full(len(T0_eval), np.nan)
    if "A" in input_results:
        Istar_input_A = input_results["A"].model.predict(T0_eval.reshape(-1, 1))

    # Input-based: spec raw
    Istar_input_raw = np.full(len(T0_eval), np.nan)
    if "raw" in input_results:
        for i, T0 in enumerate(T0_eval):
            ep = rollout_bi2223_episode(T0, episode_id=0)
            if not ep:
                continue
            mid = len(ep) // 2
            r = ep[mid]
            X_raw = np.array([[r["T0_K"], r["I_true_A"],
                               r["voltage_true_V"], r["T_surface_true_K"]]])
            Istar_input_raw[i] = input_results["raw"].model.predict(X_raw)[0]

    # Sequence: evaluate each arch
    df_eval = generate_bi2223_grid_dataset(T0_eval)
    seq_eval = {}
    trajectory_data = {}
    for arch, sr in seq_results.items():
        seq_eval[arch] = _eval_seq(sr, T0_eval, df_eval)

    for _episode_id, g in df_eval.sort_values("step").groupby("episode_id", sort=True):
        T0 = g["T0_K"].iloc[0]
        trajectory_data[T0] = g

    # Eval metrics for each model
    eval_metrics = {}
    valid_A = np.isfinite(Istar_analytic) & np.isfinite(Istar_input_A)
    if valid_A.any():
        rel_err_A = np.abs(Istar_input_A[valid_A] - Istar_analytic[valid_A]) / Istar_analytic[valid_A]
        eval_metrics["input_A"] = {
            "istar_median_rel_err": float(np.median(rel_err_A)),
            "istar_95th_rel_err": float(np.percentile(rel_err_A, 95)),
        }
        print(f"  Input A I* median rel err: {np.median(rel_err_A):.2%}")

    for arch, ev in seq_eval.items():
        valid = np.isfinite(Istar_analytic) & np.isfinite(ev["Istar"])
        if valid.any():
            rel_err = np.abs(ev["Istar"][valid] - Istar_analytic[valid]) / Istar_analytic[valid]
            eval_metrics[f"seq_{arch}"] = {
                "istar_median_rel_err": float(np.median(rel_err)),
                "istar_95th_rel_err": float(np.percentile(rel_err, 95)),
            }
            print(f"  Seq {arch} I* median rel err: {np.median(rel_err):.2%}")
            print(f"  Seq {arch} I* 95th pctl rel err: {np.percentile(rel_err, 95):.2%}")

    all_metrics["eval"] = eval_metrics

    # Save metrics JSON
    metrics_path = out_dir / "bi2223_eval_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n  Metrics saved to {metrics_path}")

    # ── Phase 4: Plots ───────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Phase 4: Diagnostic plots")
    print("=" * 70)

    _make_main_figure(
        T0_eval, Istar_analytic, Tstar_analytic, tm_analytic_ms,
        Istar_input_A, Istar_input_raw, seq_eval, trajectory_data, out_dir,
    )
    _make_trajectory_figure(out_dir)

    print(f"\nAll outputs saved to {out_dir}/")


if __name__ == "__main__":
    main()
