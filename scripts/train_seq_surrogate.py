"""
train_seq_surrogate.py

End-to-end: roll out QuenchDetectorEnv into a trajectory dataset, train a
sequence-model surrogate (GRU or Transformer) to predict the running margin
and the runaway current I* from telemetry alone, evaluate it against the
analytic (T0, lambda) -> I* runaway surface (the notebook's Fig. 3a
reproduction), and save the model + a comparison plot.

Usage
-----
    python scripts/train_seq_surrogate.py --episodes 1500 --epochs 60 --arch gru
    python scripts/train_seq_surrogate.py --episodes 1500 --epochs 60 --arch transformer
"""

import _bootstrap  # noqa: F401  (first import: caps BLAS threads, fixes sys.path)

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from hts_tape_env.rollout import generate_rollout_dataset
from hts_tape_env.seq_surrogates import predict_runaway_surface, train_seq_surrogate


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodes", type=int, default=1500)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--arch", choices=["gru", "transformer"], default="gru")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--grid-size", type=int, default=18, help="T0 x lambda eval grid resolution")
    ap.add_argument("--out-dir", default="models")
    ap.add_argument("--data-dir", default="data")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"Rolling out {args.episodes} episodes from QuenchDetectorEnv...")
    df = generate_rollout_dataset(args.episodes, seed=args.seed)
    data_path = data_dir / "rollout_main.parquet"
    df.to_parquet(data_path)
    print(f"Saved rollout dataset to {data_path} ({len(df)} steps)")

    print(f"\nTraining {args.arch} sequence surrogate...")
    result = train_seq_surrogate(df, arch=args.arch, epochs=args.epochs, seed=args.seed)
    ckpt_path = result.save(out_dir / f"seq_{args.arch}.pt")
    print(f"Saved checkpoint to {ckpt_path}")
    print(f"Held-out metrics: {result.metrics}")

    print("\nEvaluating against the analytic runaway surface (T0, lambda) -> I*...")
    T0_grid = np.linspace(8.0, 72.0, args.grid_size)
    lam_grid = np.linspace(24.0, 190.0, args.grid_size)
    surfaces = predict_runaway_surface(result, T0_grid, lam_grid, seed=args.seed + 1000)
    Istar_analytic, Istar_pred = surfaces["Istar_analytic"], surfaces["Istar_pred"]

    rel_err = np.abs(Istar_pred - Istar_analytic) / np.maximum(Istar_analytic, 1e-6)
    print(f"I* surface median rel err: {np.nanmedian(rel_err):.2%}, "
          f"95th pct: {np.nanpercentile(rel_err, 95):.2%}")
    print("(T*/time-margin surfaces and nonlinearity-vs-error breakdowns: "
          "see scripts/analyze_controller_physics.py)")

    TT, LL = np.meshgrid(T0_grid, lam_grid)
    fig = plt.figure(figsize=(14, 5))
    for i, (Z, title) in enumerate([
        (Istar_analytic, "Analytic I* (brentq solver)"),
        (Istar_pred, f"{args.arch.upper()} sequence surrogate I*\n(lambda inferred from telemetry, not given)"),
    ]):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        ax.plot_surface(TT, LL, Z, cmap="turbo", edgecolor="k", linewidth=0.2)
        ax.set_xlabel("T0 (K)")
        ax.set_ylabel("lambda (W/m/K)")
        ax.set_zlabel("I* (A)")
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig_path = out_dir / f"seq_{args.arch}_surface.png"
    fig.savefig(fig_path, dpi=140)
    print(f"Saved surface comparison to {fig_path}")


if __name__ == "__main__":
    main()
